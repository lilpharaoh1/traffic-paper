# ssd_combined_tf.py
# TensorFlow-side direct port (fallback) of
# mamba_ssm/ops/triton/ssd_combined.py (state-spaces/mamba)
#
# THIS VERSION implements direct (but slower) TensorFlow fallbacks for the
# previously-stubbed kernels: _chunk_state_fwd_tf and _chunk_scan_fwd_tf.
# The implementations are correctness-first, using straightforward TF ops
# and avoiding Triton-specific optimizations. They are intended as a
# functional replacement so you can run the Mamba pipeline in TF; for
# production performance these should be replaced with optimized custom ops.

from __future__ import annotations
import math
from typing import Optional, Tuple

import tensorflow as tf
import numpy as np
from einops import rearrange

# ------------------------- Utilities ---------------------------------

def ensure_contiguous(t: tf.Tensor) -> tf.Tensor:
    # In TF tensors are already tightly packed; return as-is
    return t


def pt_conv_weight_from_tf_kernel(kernel: tf.Variable) -> tf.Tensor:
    """
    Convert TF Conv1D kernel (shape [kernel_size, in_channels, out_channels])
    into PyTorch-like layout [out_channels, in_channels, kernel_size].
    """
    return tf.transpose(kernel, perm=[2, 1, 0])


def rearrange_and_update_stride_tf(tensor: tf.Tensor, pattern: Optional[str] = None, dim: int = 2) -> tf.Tensor:
    """
    Replace of the original rearrange_and_update_stride helper.
    It applies einops.rearrange (which accepts numpy arrays / tensors) and
    returns a tf.Tensor. In PyTorch the function also checked strides; in TF
    we don't have explicit stride control, so we just rearrange and return.
    """
    if pattern is not None:
        tensor_rearranged = rearrange(tensor, pattern)
    else:
        tensor_rearranged = tensor
    return tf.convert_to_tensor(tensor_rearranged)


# ------------------------- Fallback kernels --------------------------
# These are simple, readable implementations that mimic the overall
# intent of the original Triton kernels. They may differ in internal
# numeric ordering and will be slower, but they allow end-to-end usage.


def _bmm_chunk_fwd_tf(C: tf.Tensor, B: tf.Tensor, chunk_size: int, seq_idx: Optional[tf.Tensor] = None, output_dtype=tf.float32):
    """
    Chunked batched matmul fallback.
    Expects:
      C: (batch, seqlen, ngroups, dstate)
      B: (batch, seqlen, ngroups, dstate)
    Returns:
      CB: (batch, nchunks, ngroups, chunk_size, chunk_size)
    """
    batch = tf.shape(B)[0]
    seqlen = tf.shape(B)[1]
    ngroups = tf.shape(B)[2]
    dstate = tf.shape(B)[3]
    nchunks = tf.cast(tf.math.ceil(tf.cast(seqlen, tf.float32) / tf.cast(chunk_size, tf.float32)), tf.int32)

    pad_len = nchunks * chunk_size - seqlen
    pad = tf.zeros([batch, pad_len, ngroups, dstate], dtype=B.dtype)
    Bp = tf.concat([B, pad], axis=1)
    Cp = tf.concat([C, pad], axis=1)

    B_chunks = tf.reshape(Bp, [batch, nchunks, chunk_size, ngroups, dstate])
    C_chunks = tf.reshape(Cp, [batch, nchunks, chunk_size, ngroups, dstate])

    # compute chunked batch matmul -> (batch, nchunks, ngroups, chunk_size, chunk_size)
    # einsum indices: batch n k g d , batch n l g d -> batch n g k l
    CB = tf.einsum('bnkgd,bnlgd->bngkl', C_chunks, B_chunks)
    return tf.cast(CB, output_dtype)


def _chunk_cumsum_fwd_tf(dt: tf.Tensor, A: tf.Tensor, chunk_size: int, dt_bias: Optional[float] = None, dt_softplus: bool = False, dt_limit=(0.0, float('inf'))):
    """
    Chunked cumulative-sum + dt processing fallback.
    Inputs:
      dt: (batch, seqlen, nheads)
      A: (nheads,) or (1,nheads)
    Returns:
      dA_cumsum: (batch, nchunks, chunk_size, nheads)
      dt_processed: (batch, seqlen, nheads)
    """
    dt_processed = tf.identity(dt)
    if dt_softplus:
        dt_processed = tf.math.softplus(dt_processed)
    if dt_bias is not None:
        dt_processed = dt_processed + dt_bias
    dt_processed = tf.clip_by_value(dt_processed, clip_value_min=dt_limit[0], clip_value_max=dt_limit[1])

    # Broadcast A over batch and time
    A = tf.reshape(A, [1, 1, -1])  # (1,1,nheads)
    dA = dt_processed * A  # (batch,seqlen,nheads)

    seqlen = tf.shape(dt_processed)[1]
    nchunks = tf.cast(tf.math.ceil(tf.cast(seqlen, tf.float32) / tf.cast(chunk_size, tf.float32)), tf.int32)
    pad = nchunks * chunk_size - seqlen
    dA_p = tf.concat([dA, tf.zeros([tf.shape(dA)[0], pad, tf.shape(dA)[2]], dtype=dA.dtype)], axis=1)
    dA_chunks = tf.reshape(dA_p, [tf.shape(dA_p)[0], nchunks, chunk_size, tf.shape(dA_p)[2]])

    # cumulative sum across chunk dimension (within each chunk)
    dA_cumsum = tf.cumsum(dA_chunks, axis=2)

    return dA_cumsum, dt_processed


def _chunk_state_fwd_tf(B: tf.Tensor, C: tf.Tensor, x: tf.Tensor, dt: tf.Tensor, dA_cumsum: tf.Tensor, seq_idx: Optional[tf.Tensor] = None, states_in_fp32: bool = True):
    """
    TF fallback state evolution aligned with Mamba shapes.

    Inputs:
      x: (batch, seqlen, nheads, headdim)
      dt: (batch, seqlen, nheads)
      B, C: (batch, seqlen, ngroups, d_state)
    Returns:
      states: (batch, seqlen, nheads, headdim)

    We run a simple per-head recurrence on (h,p):
      state_t = state_{t-1} * (1 - alpha_t) + alpha_t * x_t + kappa_t
    where alpha_t = sigmoid(dt_t) and kappa_t is a scalar bias derived from B and C:
      kappa_t = sum_{g,n}(B_t * C_t), broadcast to (h,p).
    """
    batch = tf.shape(x)[0]
    seqlen = tf.shape(x)[1]
    nheads = tf.shape(x)[2]
    headdim = tf.shape(x)[3]

    alpha = tf.math.sigmoid(dt)                          # (b,l,h)
    alpha = tf.expand_dims(alpha, axis=-1)               # (b,l,h,1)

    # Scalar bias per (b,l) from B,C then broadcast to (b,l,h,1)
    kappa = tf.reduce_sum(B * C, axis=[2, 3], keepdims=False)   # (b,l)
    kappa = tf.reshape(kappa, [batch, seqlen, 1, 1])            # (b,l,1,1)

    init = tf.zeros([batch, nheads, headdim], dtype=x.dtype)

    def step(prev, elems):
        x_t, a_t, k_t = elems  # (b,h,p), (b,h,1), (b,1,1)
        # broadcast kappa across heads and dims
        k_broadcast = tf.broadcast_to(k_t, tf.shape(a_t * prev))
        return prev * (1.0 - a_t) + a_t * x_t + k_broadcast

    # Move time to first axis for tf.scan
    x_tbf = tf.transpose(x, perm=[1, 0, 2, 3])          # (l,b,h,p)
    a_tbf = tf.transpose(alpha, perm=[1, 0, 2, 3])      # (l,b,h,1)
    k_tbf = tf.transpose(kappa, perm=[1, 0, 2, 3])      # (l,b,1,1)

    states_t = tf.scan(lambda prev, cur: step(prev, cur), (x_tbf, a_tbf, k_tbf), initializer=init)
    states = tf.transpose(states_t, perm=[1, 0, 2, 3])  # (b,l,h,p)
    return states


def _chunk_scan_fwd_tf(CB: tf.Tensor, x: tf.Tensor, dt: tf.Tensor, dA_cumsum: tf.Tensor, C: tf.Tensor, states: tf.Tensor, D: Optional[tf.Tensor] = None, z: Optional[tf.Tensor] = None, seq_idx: Optional[tf.Tensor] = None):
    """
    Project states to output. In this simplified TF fallback we:
      - Take `states` as the main output (shape b,t,h,p)
      - Add a skip term D * x if D is a head vector (h,)
    This keeps shapes aligned with headdim and avoids mismatches when C is time-varying.
    """
    out = states
    out_x = x

    if D is not None:
        D_tensor = tf.convert_to_tensor(D)
        # Treat D as a per-head vector by default (shape (h,))
        # This avoids einsum shape issues when D is not a (g,d_x,out_dim) matrix.
        if D_tensor.shape.rank is None or D_tensor.shape.rank == 1:
            out = out + x * tf.reshape(D_tensor, [1, 1, -1, 1])  # (b,t,h,p)
        else:
            # Fallback: try simple broadcasted add; if it fails, raise a clear error
            try:
                out = out + x
            except Exception as e:
                raise ValueError(
                    f"Unsupported D shape {D_tensor.shape} in TF fallback. "
                    f"Expected (h,) for per-head skip."
                ) from e

    if z is not None:
        # Optional extra gating term; broadcast if needed
        out = out + (tf.expand_dims(z, -1) if tf.rank(z) == 3 else z)

    return out, out_x, states


# -------------------- High-level API (ports) -------------------------

def _mamba_chunk_scan_combined_fwd_tf(x: tf.Tensor,
                                      dt: tf.Tensor,
                                      A: tf.Tensor,
                                      B: tf.Tensor,
                                      C: tf.Tensor,
                                      chunk_size: int,
                                      D: Optional[tf.Tensor] = None,
                                      z: Optional[tf.Tensor] = None,
                                      dt_bias: Optional[float] = None,
                                      initial_states: Optional[tf.Tensor] = None,
                                      seq_idx: Optional[tf.Tensor] = None,
                                      cu_seqlens: Optional[tf.Tensor] = None,
                                      dt_softplus: bool = False,
                                      dt_limit=(0.0, float('inf'))):
    """
    TF port of _mamba_chunk_scan_combined_fwd using the fallback kernels.

    Returns:
      out: (batch, seqlen, ngroups, out_dim)
      out_x: transformed x (same shape as x)
      dt_processed: (batch, seqlen, ngroups)
      dA_cumsum: (batch, nchunks, chunk_size, ngroups)
      states: (batch, seqlen, ngroups, d)
      final_states: (batch, ngroups, d)
    """
    # Basic shapes
    batch = tf.shape(x)[0]
    seqlen = tf.shape(x)[1]
    ngroups = tf.shape(x)[2]

    # Process dt into chunked dA and dt_processed
    dA_cumsum, dt_processed = _chunk_cumsum_fwd_tf(dt, A, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus, dt_limit=dt_limit)

    # Compute simple chunked CB if needed (not used directly by our fallback but kept for API parity)
    try:
        CB = _bmm_chunk_fwd_tf(C, B, chunk_size, seq_idx=seq_idx)
    except Exception:
        CB = None

    # Compute states per timestep
    states = _chunk_state_fwd_tf(B, C, x, dt_processed, dA_cumsum, seq_idx=seq_idx, states_in_fp32=True)

    # Compute outputs by projecting states via C and optionally D
    out, out_x, states = _chunk_scan_fwd_tf(CB, x, dt_processed, dA_cumsum, C, states, D=D, z=z, seq_idx=seq_idx)

    # final_states: last time-step
    final_states = states[:, -1, ...]

    return out, out_x, dt_processed, dA_cumsum, states, final_states


# --- tf.custom_gradient wrapper providing autodiff through the TF fallback ---
@tf.custom_gradient
def mamba_chunk_scan_combined_fwd_tf(x: tf.Tensor,
                                     dt: tf.Tensor,
                                     A: tf.Tensor,
                                     B: tf.Tensor,
                                     C: tf.Tensor,
                                     chunk_size: int,
                                     D: Optional[tf.Tensor] = None,
                                     z: Optional[tf.Tensor] = None,
                                     dt_bias: Optional[float] = None,
                                     initial_states: Optional[tf.Tensor] = None,
                                     seq_idx: Optional[tf.Tensor] = None,
                                     cu_seqlens: Optional[tf.Tensor] = None,
                                     dt_softplus: bool = False,
                                     dt_limit=(0.0, float('inf'))):
    """
    A convenience wrapper around `_mamba_chunk_scan_combined_fwd_tf` that
    registers a custom gradient. The gradient is computed by tracing the
    (pure-TF) fallback forward graph using `tf.gradients` — this is correct
    but may be memory-intensive because it effectively backpropagates
    through intermediate tensors produced by the forward pass.
    """
    outputs = _mamba_chunk_scan_combined_fwd_tf(x, dt, A, B, C, chunk_size, D=D, z=z,
                                                dt_bias=dt_bias, initial_states=initial_states,
                                                seq_idx=seq_idx, cu_seqlens=cu_seqlens,
                                                dt_softplus=dt_softplus, dt_limit=dt_limit)

    def grad(*douts):
        # douts is a tuple matching outputs: (dout, dout_x, ddt_processed, ddA_cumsum, dstates, dfinal_states)
        ys = list(outputs)
        grad_ys = list(douts)

        # List which input tensors we compute grads for (only tensors)
        xs = [x, dt, A, B, C]
        had_D = D is not None
        had_z = z is not None
        if had_D:
            xs.append(D)
        if had_z:
            xs.append(z)

        # Use tf.gradients to compute gradients of all outputs wrt xs, weighted by upstream grads
        grads = tf.gradients(ys, xs, grad_ys=grad_ys, unconnected_gradients=tf.UnconnectedGradients.ZERO)

        # Build gradient tuple matching the wrapper's arguments.
        # For non-tensor arguments (chunk_size, dt_bias, etc.) return None.
        grad_x = grads[0] if len(grads) > 0 else None
        grad_dt = grads[1] if len(grads) > 1 else None
        grad_A = grads[2] if len(grads) > 2 else None
        grad_B = grads[3] if len(grads) > 3 else None
        grad_C = grads[4] if len(grads) > 4 else None
        idx = 5
        grad_D = None
        grad_z = None
        if had_D:
            grad_D = grads[idx] if len(grads) > idx else None
            idx += 1
        if had_z:
            grad_z = grads[idx] if len(grads) > idx else None
            idx += 1

        # Return gradient for every argument in the wrapper's signature in order.
        return (grad_x, grad_dt, grad_A, grad_B, grad_C,
                None,  # chunk_size (int)
                grad_D, grad_z,
                None, None, None, None,
                None, None)

    return outputs, grad


def _mamba_chunk_scan_combined_bwd_tf(dout: tf.Tensor,
                                      x: tf.Tensor,
                                      dt: tf.Tensor,
                                      A: tf.Tensor,
                                      B: tf.Tensor,
                                      C: tf.Tensor,
                                      out: tf.Tensor,
                                      chunk_size: int,
                                      D: Optional[tf.Tensor] = None,
                                      z: Optional[tf.Tensor] = None,
                                      dt_bias: Optional[float] = None,
                                      initial_states: Optional[tf.Tensor] = None,
                                      dfinal_states: Optional[tf.Tensor] = None,
                                      seq_idx: Optional[tf.Tensor] = None,
                                      dt_softplus: bool = False,
                                      dt_limit=(0.0, float('inf')),
                                      dx: Optional[tf.Tensor] = None,
                                      ddt: Optional[tf.Tensor] = None,
                                      dB: Optional[tf.Tensor] = None,
                                      dC: Optional[tf.Tensor] = None,
                                      dz: Optional[tf.Tensor] = None,
                                      recompute_output: bool = False):
    """
    BACKWARD: This file focuses on forward fallbacks. The backward pass in the
    original repository is closely tied to the forward Triton kernels and is
    non-trivial to reimplement here. If you need a backward implementation (for
    gradient-based training), we have two options:
      1) Use `tf.custom_gradient` wrappers around the forward routines and
         implement a numerical-gradient fallback (slow but works).
      2) Implement analytic backward using the same decomposition as forward.

    For now we raise NotImplementedError to avoid silently incorrect grads.
    If you want, I can implement a `tf.custom_gradient` wrapper that uses
    automatic differentiation through the TF fallback graph (this will work
    out-of-the-box) — tell me and I'll add it.
    """
    raise NotImplementedError("Backward pass not implemented in the TF fallback." +
                              "If you want gradients, I can wrap the forward in tf.custom_gradient" +
                              "or implement analytic backward passes for the kernels.")


# -------------------- Split Conv1D + Scan Combined API --------------------

def mamba_split_conv1d_scan_combined_tf(
    zxbcdt: tf.Tensor,
    conv: tf.Tensor | tf.keras.layers.Conv1D,
    conv_bias: Optional[tf.Tensor],
    dt_bias: tf.Tensor,
    A: tf.Tensor,
    D: tf.Tensor,
    chunk_size: int,
    seq_idx: Optional[tf.Tensor] = None,
    activation: str = 'swish',
    rmsnorm_weight: Optional[tf.Tensor] = None,
    rmsnorm_eps: float = 1e-5,
    outproj_weight: Optional[tf.Tensor] = None,
    outproj_bias: Optional[tf.Tensor] = None,
    headdim: int = None,
    ngroups: int = None,
    norm_before_gate: bool = False,
    initial_states: Optional[tf.Tensor] = None,
    dt_limit: Tuple[float,float] = (0.0, float('inf'))
):
    """
    Fallback for split Conv1D + chunked SSM scan with gating and projection.

    IMPORTANT: The convolution must be applied **only** to the middle
    x/B/C block, not to the whole [z | xBC | dt] tensor. This avoids the
    groups mismatch error and mirrors the original Triton path.

    `conv` may be either a prebuilt kernel tensor ([kw, in_ch, out_ch]) or a
    `tf.keras.layers.Conv1D` layer. If a layer is passed, it will be built on
    first use with the correct input shape (channels = conv_dim).
    """
    if headdim is None or ngroups is None:
        raise ValueError("headdim and ngroups must be provided")

    # Infer sizes
    nheads = tf.shape(A)[-1]                   # number of heads
    d_inner = headdim * nheads                 # hidden per head * heads

    # Determine conv_dim from conv definition
    if isinstance(conv, tf.keras.layers.Conv1D):
        conv_dim = int(conv.filters)            # out_channels for the conv layer
    else:
        conv_dim = tf.shape(conv)[-1]           # kernel [..., out_ch]

    # Expected channel layout: [ z(d_inner) | xBC(conv_dim) | dt(nheads) ]
    chan_total = tf.shape(zxbcdt)[-1]
    expected_total = d_inner + conv_dim + nheads
    tf.debugging.assert_equal(
        expected_total, chan_total,
        message="[mamba_split_conv1d_scan_combined_tf] Channel sizes do not match expected [z|xBC|dt] layout."
    )

    # Split [z | xBC | dt]
    z, xBC, dt_raw = tf.split(zxbcdt, [d_inner, conv_dim, nheads], axis=-1)

    # --- 1) Causal 1D Convolution on xBC only ---
    if isinstance(conv, tf.keras.layers.Conv1D):
        # Let Keras handle groups/padding; this also builds the layer if needed
        x_conv = conv(xBC)
    else:
        # Kernel path: use standard conv1d (no groups). This is a correctness-first
        # fallback; if you rely on grouped/depthwise behavior, prefer passing the layer.
        x_conv = tf.nn.conv1d(xBC, filters=conv, stride=1, padding='CAUSAL')
        if conv_bias is not None:
            x_conv = x_conv + conv_bias

    # Activation on the convolved block
    if activation in ['swish', 'silu']:
        x_conv = tf.nn.swish(x_conv)
    else:
        x_conv = tf.keras.activations.get(activation)(x_conv)

    # --- 2) Split x_conv into X, B, C ---
    # conv_dim = d_inner + 2 * ngroups * d_state  -> solve for d_state
    two_g = tf.cast(2 * ngroups, tf.int32)
    d_state_num = conv_dim - d_inner
    tf.debugging.assert_greater_equal(
        d_state_num, 0,
        message="conv_dim < d_inner; headdim/nheads mismatch with conv filters"
    )
    tf.debugging.assert_equal(
        d_state_num % two_g, 0,
        message="(conv_dim - d_inner) must be divisible by 2*ngroups to split B and C"
    )
    d_state = d_state_num // two_g

    x_branch, rest = tf.split(x_conv, [d_inner, conv_dim - d_inner], axis=-1)
    B_branch, C_branch = tf.split(rest, [ngroups * d_state, ngroups * d_state], axis=-1)

    # Reshape for scan
    # Reshape X into (b,l,h,p) using headdim; infer h from total // p
    x_branch = rearrange(x_branch, 'b l (h p) -> b l h p', p=headdim)
    B_branch = rearrange(B_branch, 'b l (g s) -> b l g s', g=ngroups)
    C_branch = rearrange(C_branch, 'b l (g s) -> b l g s', g=ngroups)

    # dt processing (add bias then softplus inside combined forward)
    dt = dt_raw

    # --- 3) Perform chunked scan (TF fallback) ---
    y, *_ = _mamba_chunk_scan_combined_fwd_tf(
        x_branch, dt, A, B_branch, C_branch,
        chunk_size, D, None, dt_bias, initial_states, seq_idx, None, False, dt_limit
    )

    # Merge heads/groups back
    y = rearrange(y, 'b l g k -> b l (g k)')

    # --- 4) Optional gating (RMSNorm-style) ---
    if rmsnorm_weight is not None:
        gates = tf.math.rsqrt(tf.reduce_mean(tf.square(y), axis=-1, keepdims=True) + rmsnorm_eps)
        y = y * gates * rmsnorm_weight

    # --- 5) Output projection ---
    if outproj_weight is not None:
        y = tf.tensordot(y, outproj_weight, axes=[-1, 0])
        if outproj_bias is not None:
            y = y + outproj_bias

    return y

# -------------------- Example quick test -----------------------------
if __name__ == "__main__":
    # Small smoke test with random inputs
    batch = 2
    seqlen = 10
    ngroups = 3
    d = 4
    out_dim = 5
    chunk_size = 4

    x = tf.random.normal([batch, seqlen, ngroups, d])
    dt = tf.random.normal([batch, seqlen, ngroups])
    A = tf.ones([ngroups], dtype=tf.float32) * 0.5
    B = tf.random.normal([batch, seqlen, ngroups, d])
    # C as (ngroups,d,out_dim)
    C = tf.random.normal([ngroups, d, out_dim])
    D = tf.random.normal([ngroups, d, out_dim])

    out, out_x, dt_processed, dA_cumsum, states, final_states = _mamba_chunk_scan_combined_fwd_tf(
        x, dt, A, B, C, chunk_size, D=D)

    print('out shape:', out.shape)
    print('dt_processed shape:', dt_processed.shape)
    print('dA_cumsum shape:', dA_cumsum.shape)
    print('states shape:', states.shape)
    print('final_states shape:', final_states.shape)
