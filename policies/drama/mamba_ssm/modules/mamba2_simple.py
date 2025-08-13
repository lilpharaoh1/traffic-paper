# Copyright (c) 2024, Tri Dao, Albert Gu.

import math
import torch
import torch.nn as nn

from ray.rllib.utils.framework import try_import_tf
_, tf, _ = try_import_tf()

import torch.nn.functional as F

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None

try:
    from policies.drama.mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated, LayerNorm
except ImportError:
    RMSNormGated, LayerNorm = None, None

from policies.drama.mamba_ssm.ops.triton.ssd_combined_tf import mamba_chunk_scan_combined_fwd_tf
from policies.drama.mamba_ssm.ops.triton.ssd_combined_tf import mamba_split_conv1d_scan_combined_tf


class Mamba2Simple(tf.keras.layers.Layer): # EMRAN or potentiall tf.keras.Model instead
    def __init__(
        self,
        d_model,
        d_state=64,
        d_conv=4,
        conv_init=None,
        expand=2,
        headdim=128,
        ngroups=1,
        A_init_range=(1, 16),
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        learnable_init_states=False,
        activation="swish",
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=True,
        layer_idx=None,  # Absorb kwarg for general module
        device=None,
        dtype=None,
    ):
        factory_kwargs = {} # {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        self.headdim = headdim
        self.ngroups = ngroups
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        self.activation = activation
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = tf.keras.layers.Dense(
            units=d_in_proj, 
            use_bias=bias, 
            kernel_initializer='glorot_uniform',
            **factory_kwargs)

        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv1d = tf.keras.layers.Conv1D( # EMRAN nchw for torch, nhwc for tf, I didn't change anything about the input so could be wrong
            filters=conv_dim,
            kernel_size=d_conv,
            strides=1,
            padding='causal', # used to be "d_conv - 1" # EMRAN idk
            use_bias=conv_bias,
            groups=conv_dim,
            kernel_initializer='glorot_uniform',
            **factory_kwargs,
        )
        if self.conv_init is not None:
            initializer = tf.keras.initializers.RandomUniform(minval=-self.conv_init, maxval=self.conv_init)
            raise "Come back to if important"
            # self.conv1d.weight = tf.Variable(initializer(shape), trainable=True)
        # self.conv1d.weight._no_weight_decay = True

        if self.learnable_init_states:
            self.init_states = tf.Variable(tf.zeros([self.nheads, self.headdim, self.d_state], **factory_kwargs), trainable=True)
            self.init_states._no_weight_decay = True

        self.act = tf.keras.layers.Activation('swish')

        # Initialize log dt bias
        dt = tf.exp(
            tf.random.uniform([self.nheads], **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = tf.maximum(dt, dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + tf.math.log(-tf.math.expm1(-dt))
        self.dt_bias = tf.Variable(inv_dt, trainable=True)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        # A parameter
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = tf.random.uniform([self.nheads], minval=A_init_range[0], maxval=A_init_range[1], dtype=tf.float32)
        A_log = tf.math.log(A)
        if not dtype is None:
            A_log = tf.cast(A_log, dtype)
        self.A_log = tf.Variable(A_log, trainable=True)
        # self.register_buffer("A_log", torch.zeros(self.nheads, dtype=torch.float32, device=device), persistent=True)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = tf.Variable(tf.ones(self.nheads), trainable=True)
        self.D._no_weight_decay = True

        # EMRAN commenting out again and just using normal normalization layers (for now)
        # # Extra normalization layer right before output projection
        # assert RMSNormGated is not None
        # self.norm = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=False, **factory_kwargs)
        self.norm = tf.keras.layers.LayerNormalization()

        self.out_proj = tf.keras.layers.Dense(
            units=self.d_model, 
            use_bias=bias, 
            kernel_initializer='glorot_uniform', 
            **factory_kwargs
        )

    def call(self, u, seq_idx=None):
        """
        u: (B, L, D)
        Returns: same shape as u
        """
        batch, seqlen, dim = u.shape
        
        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)
        A = -tf.exp(self.A_log)  # (nheads) or (d_inner, d_state)
        initial_states=repeat(self.init_states, "... -> b ...", b=batch) if self.learnable_init_states else None
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

        if self.use_mem_eff_path:
            # Fully fused path
            out = mamba_split_conv1d_scan_combined_tf(
                zxbcdt,
                self.conv1d,    # pass the Conv1D layer itself
                getattr(self.conv1d, "bias", None) if getattr(self.conv1d, "use_bias", False) else None,    # bias arg is ignored when passing layer
                self.dt_bias,
                A, self.D,
                chunk_size=self.chunk_size,
                seq_idx=seq_idx,
                activation=self.activation,
                rmsnorm_weight=getattr(self.norm, "weight", None),
                rmsnorm_eps=getattr(self.norm, "epsilon", 1e-5),
                outproj_weight=getattr(self.out_proj, "kernel", None),
                outproj_bias=getattr(self.out_proj, "bias", None),
                headdim=self.headdim,
                ngroups=self.ngroups,
                norm_before_gate=False,
                initial_states=initial_states,
                **dt_limit_kwargs,
            )
        else:
            z, xBC, dt = tf.split(
                zxbcdt, [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], axis=-1
            )
            dt = tf.nn.softplus(dt + self.dt_bias)  # (B, L, nheads)
            assert self.activation in ["silu", "swish"]

            # 1D Convolution
            if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
                xBC = self.act(
                    self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)
                )  # (B, L, self.d_inner + 2 * ngroups * d_state)
                xBC = xBC[:, :seqlen, :]
            else:
                xBC = causal_conv1d_fn(
                    x=xBC.transpose(1, 2),
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                ).transpose(1, 2)

            # Split into 3 main branches: X, B, C
            # These correspond to V, K, Q respectively in the SSM/attention duality
            x, B, C = tf.split(xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], axis=-1)
            y = mamba_chunk_scan_combined_fwd_tf(
                rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                dt,
                A,
                rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
                rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
                chunk_size=self.chunk_size,
                D=self.D,
                z=None,
                seq_idx=seq_idx,
                initial_states=initial_states,
                **dt_limit_kwargs,
            )
            y = rearrange(y, "b l h p -> b l (h p)")

            # Multiply "gate" branch and apply extra normalization layer
            y = self.norm(y, z)
            out = self.out_proj(y)
        return out
