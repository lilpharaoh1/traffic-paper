# Copyright (c) 2024, Tri Dao, Albert Gu.
from typing import Optional

import torch
from torch import nn, Tensor

from ray.rllib.utils.framework import try_import_tf
_, tf, _ = try_import_tf()

from policies.drama.mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn


class Block(tf.keras.layers.Layer):
    def __init__(
        self, dim, mixer_cls, mlp_cls, norm_cls=tf.keras.layers.LayerNormalization, fused_add_norm=False, residual_in_fp32=False, dropout_p=0.0
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.norm = norm_cls(axis=-1)
        self.mixer = mixer_cls(dim)
        if not isinstance(mlp_cls, tf.keras.layers.Activation):
            self.norm2 = norm_cls(axis=-1)
            self.mlp = mlp_cls(dim)
        else:
            self.mlp = None
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (tf.keras.layers.LayerNormalization, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"
        else:
            self.dropout = tf.keras.layers.Dropout(dropout_p) # "Attention is all you need sec 5.4 dropout"
        self.dropout_p = dropout_p

    def call(
            self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None, training=False, **mixer_kwargs
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            hidden_states = self.dropout(hidden_states)
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            hidden_states = self.norm(hidden_states) # EMRAN using LayerNormalization instead (for now)
        #     hidden_states, residual = layer_norm_fn(
        #         hidden_states,
        #         self.norm.weight,
        #         self.norm.bias,
        #         residual=residual,
        #         prenorm=True,
        #         residual_in_fp32=self.residual_in_fp32,
        #         epsilon=self.norm.epsilon,
        #         is_rms_norm=isinstance(self.norm, RMSNorm),
        #         dropout_p=self.dropout_p if training else 0.0 # EMRAN need to confirm training is correctly passed by Keras
        #     )
        hidden_states = self.mixer(hidden_states, **mixer_kwargs) # EMRAN No longer passing inference_params, must be some remenant of Mamba1

        if self.mlp is not None:
            if not self.fused_add_norm:
                hidden_states = self.dropout(hidden_states)
                residual = hidden_states + residual
                hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
            else:
                hidden_states = self.norm2(hidden_states) # EMRAN using LayerNormalization instead (for now)
                # hidden_states, residual = layer_norm_fn(
                #     hidden_states,
                #     self.norm2.weight,
                #     self.norm2.bias,
                #     residual=residual,
                #     prenorm=True,
                #     residual_in_fp32=self.residual_in_fp32,
                #     epsilon=self.norm2.epsilon,
                #     is_rms_norm=isinstance(self.norm2, RMSNorm),
                #     dropout_p=self.dropout_p if training else 0.0 # EMRAN need to confirm training is correctly passed by Keras
                # )
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
