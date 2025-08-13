# Copyright (c) 2023, Albert Gu, Tri Dao.

import math
from functools import partial
import json
import os
import copy

from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchtune.modules import RMSNorm as _RMSNorm # EMRAN implemented
from einops import repeat

from ray.rllib.utils.framework import try_import_tf
_, tf, _ = try_import_tf()

from policies.drama.mamba_ssm.models.config_mamba import MambaConfig
from policies.drama.mamba_ssm.modules.mamba_simple import Mamba
# from policies.drama.mamba_ssm.modules.mamba2 import Mamba2
from policies.drama.mamba_ssm.modules.mamba2_simple import Mamba2Simple
from policies.drama.mamba_ssm.modules.mha import MHA
from policies.drama.mamba_ssm.modules.mlp import MLP
from policies.drama.mamba_ssm.modules.block import Block
from policies.drama.mamba_ssm.utils.generation import GenerationMixin
from policies.drama.mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
from policies.drama.mamba_ssm.utils.models import (
    get_model_units,
    get_n_layer,
    get_intermediate_units,
    get_stoch_dim,
)

# EMRAN figure out what to do with this
try:
    from policies.drama.mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

# RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

def create_block(
    model_size,
    ssm_cfg=None,
    attn_layer_idx=None,
    attn_cfg=None,
    pff_cfg=None,
    dropout_p=0.0,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    if attn_layer_idx is None:
        attn_layer_idx = []
    if attn_cfg is None:
        attn_cfg = {}
    if pff_cfg is None:
        pff_cfg = {}        
    factory_kwargs = {} # {"device": device, "dtype": dtype}
    if layer_idx not in attn_layer_idx:
        # Create a copy of the config to modify
        ssm_cfg = copy.deepcopy(ssm_cfg) if ssm_cfg is not None else {}
        ssm_layer = ssm_cfg.pop("layer", "Mamba1")
        if ssm_layer not in ["Mamba1", "Mamba2"]:
            raise ValueError(f"Invalid ssm_layer: {ssm_layer}, only support Mamba1 and Mamba2")
        mixer_cls = partial(
            # EMRAN changed from Mamba2 to Mamba2Simple
            Mamba2Simple if ssm_layer == "Mamba2" else Mamba,
            layer_idx=layer_idx,
            **ssm_cfg,
            **factory_kwargs
        )
    else:
        mixer_cls = partial(MHA, layer_idx=layer_idx, **attn_cfg, **factory_kwargs)
    norm_cls = tf.keras.layers.LayerNormalization
    # EMRAN not using triton LayerNormalization
    # partial(
    #     tf.keras.layers.LayerNormalization if not rms_norm else RMSNorm, epsilon=norm_epsilon, **factory_kwargs
    # )
    if get_intermediate_units(model_size) == 0:
        mlp_cls = tf.keras.layers.Activation('linear')
    else:
        mlp_cls = partial(
            MLP, hidden_features=get_intermediate_units(model_size) , out_features=get_model_units(model_size) , **pff_cfg, **factory_kwargs
        )
    block = Block(
        get_model_units(model_size) ,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        dropout_p=dropout_p
    )
    block.layer_idx = layer_idx
    return block


# # https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
# def _init_weights(
#     module,
#     n_layer,
#     initializer_range=0.02,  # Now only used for embedding layer.
#     rescale_prenorm_residual=True,
#     n_residuals_per_layer=1,  # Change to 2 if we have MLP
# ):
#     #EMRAN come back to this later 
#     if isinstance(module, tf.keras.layers.Dense):
#         if module.bias is not None:
#             if not getattr(module.bias, "_no_reinit", False):
#                 nn.init.zeros_(module.bias)
#     elif isinstance(module, tf.keras.layers.Embedding):
#         nn.init.normal_(module.weight, std=initializer_range)

#     if rescale_prenorm_residual:
#         # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
#         #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
#         #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
#         #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
#         #
#         # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
#         for name, p in module.named_parameters():
#             if name in ["out_proj.weight", "fc2.weight"]:
#                 # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
#                 # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
#                 # We need to reinit p since this code could be called multiple times
#                 # Having just p *= scale would repeatedly scale it down
#                 nn.init.kaiming_uniform_(p, a=math.sqrt(5))
#                 # EMRAN used to be torch.nograd() here
#                 p /= math.sqrt(n_residuals_per_layer * n_layer)

# EMRAN Replaced above
def _init_weights(
    layer,
    n_layer,
    initializer_range=0.02,
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,
):
    if not layer.built: # EMRAN could potentially lead to weights never being initialised if model is not called
        return

    elif isinstance(layer, tf.keras.layers.Dense):
        if layer.use_bias and hasattr(layer, 'bias') and layer.bias is not None:
            # Optional: skip reinit if `_no_reinit` was set as custom attribute
            if not getattr(layer.bias, "_no_reinit", False):
                layer.bias.assign(tf.zeros_like(layer.bias))

    elif isinstance(layer, tf.keras.layers.Embedding):
        if hasattr(layer, 'embeddings') and layer.embeddings is not None:
            normal_init = tf.keras.initializers.RandomNormal(stddev=initializer_range)
            layer.embeddings.assign(normal_init(shape=layer.embeddings.shape))

    if rescale_prenorm_residual:
        # Rescale specific weights if their names match expected GPT-2 keys
        # TensorFlow layers don't have named parameters like PyTorch, so we check variable names
        for var in layer.trainable_variables:
            name = var.name
            if ("out_proj" in name or "fc2" in name) and "kernel" in name:
                # Equivalent to kaiming_uniform_
                fan_in = var.shape[-2] if len(var.shape) >= 2 else 1
                limit = math.sqrt(6 / fan_in)
                kaiming_init = tf.random.uniform(var.shape, minval=-limit, maxval=limit)
                scale = 1.0 / math.sqrt(n_residuals_per_layer * n_layer)
                var.assign(kaiming_init * scale)


class _RMSNorm(tf.keras.layers.Layer):
    def __init__(self, dim, epsilon=1e-8):
        super().__init__()
        self.dim = dim
        self.epsilon = epsilon
        self.scale = tf.Variable(
            initial_value=tf.ones(shape=(dim,)),
            trainable=True,
            name="rms_scale"
        )

    def call(self, x):
        # x shape: [batch, seq, dim]
        rms = tf.sqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + self.epsilon)
        normed = x / rms
        return normed * self.scale

class PositionalEncoding1D(tf.keras.layers.Layer):
    def __init__(
        self,
        max_length: int,
        embed_dim: int,
        device=None,
        dtype=None
    ):
        factory_kwargs = {} # {"device": device, "dtype": dtype}
        super().__init__()
        self.max_length = max_length
        self.embed_dim = embed_dim

        self.pos_emb = tf.keras.layers.Embedding(input_dim=self.max_length, output_dim=embed_dim, **factory_kwargs)

    def call(self, feat):
        pos_emb = self.pos_emb(tf.range(self.max_length))
        pos_emb = repeat(pos_emb, "L D -> B L D", B=feat.shape[0])

        feat = feat + pos_emb[:, :feat.shape[1], :]
        return feat

    def forward_with_position(self, feat, position):
        assert feat.shape[1] == 1
        pos_emb = self.pos_emb(tf.range(self.max_length))
        pos_emb = repeat(pos_emb, "L D -> B L D", B=feat.shape[0])

        feat = feat + pos_emb[:, position:position+1, :]
        return feat


class MixerModel(tf.keras.Model):
    def __init__(
        self,
        model_size: str,
        action_dim: int,
        ssm_cfg=None,
        attn_layer_idx=None,
        attn_cfg=None,
        pff_cfg=None,
        dropout_p: float = 0.0,        
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {} # {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        self.action_dim = action_dim
        self.feat_dim = get_model_units(model_size)

        # self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        self.stem = [# tf.keras.Sequential([
            tf.keras.layers.Dense(units=get_model_units(model_size), use_bias=True, kernel_initializer='glorot_uniform', **factory_kwargs),
            tf.keras.layers.LayerNormalization(epsilon=norm_epsilon),
            # RMSNorm(get_model_units(model_size), epsilon=norm_epsilon, **factory_kwargs), # EMRAN replaced RMSNorm with LayerNormalization
            tf.keras.layers.Activation('swish'),
            # nn.Linear(stoch_dim+action_dim, d_model, bias=True, **factory_kwargs),
            # nn.modules.normalization.RMSNorm(d_model, **factory_kwargs),
        ]# ])
     
        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")
            else:
                self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=norm_epsilon)
                self.norm_dropout = tf.keras.layers.Dropout(dropout_p)
        else:
            self.dropout = tf.keras.layers.Dropout(dropout_p) # "Attention is all you need sec 5.4 dropout"
        self.dropout_p = dropout_p

        # EMRAN Used to be nn.Modulelist, now just a normal list
        self.blocks = [
            create_block(
                model_size,
                ssm_cfg=ssm_cfg,
                attn_layer_idx=attn_layer_idx,
                attn_cfg=attn_cfg,
                pff_cfg=pff_cfg,
                dropout_p=dropout_p,
                norm_epsilon=norm_epsilon,
                rms_norm=rms_norm,
                residual_in_fp32=residual_in_fp32,
                fused_add_norm=fused_add_norm,
                layer_idx=i,
                **factory_kwargs,
            )
            for i in range(get_n_layer(model_size))
        ]

        self.norm_f = (tf.keras.layers.LayerNormalization if not rms_norm else RMSNorm)(
            get_model_units(model_size), epsilon=norm_epsilon, **factory_kwargs
        )

        # self.apply(
        #     partial(
        #         _init_weights,
        #         n_layer=get_n_layer(model_size),
        #         **(initializer_cfg if initializer_cfg is not None else {}),
        #         n_residuals_per_layer=1 if get_intermediate_units(model_size) == 0 else 2,  # 2 if we have MLP
        #     )
        # )

        for layer in self.submodules:
            _init_weights(
                layer,
                n_layer=get_n_layer(model_size),
                **(initializer_cfg if initializer_cfg is not None else {}),
                n_residuals_per_layer=1 if get_intermediate_units(model_size) == 0 else 2,  # 2 if we have MLP
            )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.blocks)
        }

    def call(self, samples, action, inference_params=None, **mixer_kwargs):
        data = tf.concat([samples, action], axis=-1)
        hidden_states = data
        for idx, block in enumerate(self.stem):
            print(f"{idx+1}/{len(self.stem)}) Using layer {block} to process {hidden_states}")
            hidden_states = block(data)
            
        residual = None
        for idx, block in enumerate(self.blocks):
            print(f"{idx+1}/{len(self.blocks)}) Using block {block} to process {hidden_states}")
            hidden_states, residual = block(
                hidden_states, residual, inference_params=inference_params, **mixer_kwargs
            )
        if not self.fused_add_norm:
            print("No fused_add_norm...")
            hidden_states = self.dropout(hidden_states)
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(tf.cast(residual, self.norm_f.weight.dtype))
        else:
            print("Yes fused_add_norm...")
            # EMRAN using LayerNormalization (for now)
            # Set prenorm=False here since we don't need the residual
            # hidden_states = layer_norm(
            #     hidden_states,
            #     self.norm_f.weight,
            #     self.norm_f.bias,
            #     epsilon=self.norm_f.epsilon,
            #     residual=residual,
            #     prenorm=False,
            #     residual_in_fp32=self.residual_in_fp32,
            #     is_rms_norm=isinstance(self.norm_f, RMSNorm),
            #     dropout_p=self.dropout_p if self.training else 0.0
            # )
            hidden_states = self.layer_norm(hidden_states)
            hidden_states = self.norm_dropout(hidden_states)
        return hidden_states


class MambaWrapperModel(tf.keras.Model, GenerationMixin):

    def __init__(
        self,
        config: MambaConfig,
        initializer_cfg=None,
        device=None,
        dtype=None,
    ) -> None:
        self.config = config
        model_size = config.model_size
        action_dim = config.action_dim
        ssm_cfg = config.ssm_cfg
        attn_layer_idx = config.attn_layer_idx
        attn_cfg = config.attn_cfg
        pff_cfg = config.pff_cfg 
        dropout_p = config.dropout_p
        rms_norm = config.rms_norm
        residual_in_fp32 = config.residual_in_fp32
        fused_add_norm = config.fused_add_norm
        factory_kwargs = {} # {"device": device, "dtype": dtype}

        super().__init__()
        self.backbone = MixerModel(
            model_size=model_size,
            action_dim=action_dim,
            ssm_cfg=ssm_cfg,
            pff_cfg = pff_cfg,         
            attn_layer_idx=attn_layer_idx,
            attn_cfg=attn_cfg,
            dropout_p=dropout_p,
            rms_norm=rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            **factory_kwargs,
        )
        # self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)

        # Initialize weights and apply final processing
        # self.apply(
        #     partial(
        #         _init_weights,
        #         n_layer=get_n_layer(model_size),
        #         **(initializer_cfg if initializer_cfg is not None else {}),
        #     )
        # )
        for layer in self.submodules:
            _init_weights(
                layer,
                n_layer=get_n_layer(model_size),
                **(initializer_cfg if initializer_cfg is not None else {}),
            )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def call(self, samples, action, inference_params=None, num_last_tokens=0, **mixer_kwargs):
        """
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        hidden_states = self.backbone(samples, action, inference_params=inference_params, **mixer_kwargs)
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        # lm_logits = self.lm_head(hidden_states)
        return hidden_states