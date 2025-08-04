# Copyright (c) 2024, Albert Gu and Tri Dao.
"""Minimal implementation of SSD.

This is the same as Listing 1 from the paper.
"""

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from ray.rllib.utils.framework import try_import_tf
_, tf, _ = try_import_tf()

from policies.drama.mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined


def segsum_unstable(x):
    """Naive segment sum calculation."""
    T = x.size(-1)
    x_cumsum = tf.math.cumsum(x, axis=-1)
    x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
    # mask = torch.tril(tf.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    mask = tril_mask = tf.linalg.band_part(tf.ones((T, T), dtype=tf.bool), -1, 0)
    x_segsum = x_segsum.masked_fill(~mask, -tf.constant(float('inf')))
    return x_segsum

def segsum(x):
    """More stable segment sum calculation."""
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    # mask = torch.tril(tf.ones(T, T, device=x.device, dtype=bool), diagonal=-1)
    mask = tril_mask = tf.linalg.band_part(tf.ones((T, T), dtype=tf.bool), -1, -1) # EMRAN check if diagnol correctly converted
    x = x.masked_fill(~mask, 0)
    x_segsum = tf.math.cumsum(x, axis=-2)
    # mask = torch.tril(tf.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    mask = tril_mask = tf.linalg.band_part(tf.ones((T, T), dtype=tf.bool), -1, 0)
    x_segsum = x_segsum.masked_fill(~mask, -tf.constant(float('inf')))
    return x_segsum

def ssd_minimal_discrete(X, A, B, C, block_len, initial_states=None):
    """
    Arguments:
        X: (batch, length, n_heads, d_head)
        A: (batch, length, n_heads)
        B: (batch, length, n_heads, d_state)
        C: (batch, length, n_heads, d_state)
    Return:
        Y: (batch, length, n_heads, d_head)
    """
    assert X.dtype == A.dtype == B.dtype == C.dtype
    assert X.shape[1] % block_len == 0

    # Rearrange into blocks/chunks
    X, A, B, C = [rearrange(x, "b (c l) ... -> b c l ...", l=block_len) for x in (X, A, B, C)]

    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = tf.math.cumsum(A, axis=-1)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    L = tf.exp(segsum(A))
    Y_diag  = tf.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    decay_states = tf.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
    states = tf.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    if initial_states is None:
        initial_states = tf.zeros_like(states[:, :1])
    states = tf.concat([initial_states, states], axis=1)
    decay_chunk = tf.exp(segsum(last_elem_padded = tf.pad(A_cumsum[:, :, :, -1], paddings=[[0, 0], [0, 0], [0, 0], [1, 0]])))
    new_states = tf.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    state_decay_out = tf.exp(A_cumsum)
    Y_off = tf.einsum('bclhn,bchpn,bhcl->bclhp', C, states, state_decay_out)

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    Y = rearrange(Y_diag+Y_off, "b c l h p -> b (c l) h p")
    return Y, final_state


# Simple test
def test_correctness():
    tf.random.set_seed(42)

    ## Dimensions
    # Denoted (B, T, Q, D, P) in the paper
    batch, seqlen, chunk_size, dim, headdim = 1, 2048, 64, 2048, 64
    nheads = dim // headdim  # (H) in the paper
    ngroups = 1 # (G) in the paper
    dstate = 64  # (N) in the paper
    dtype = tf.float32
    device = "cuda"

    x = tf.random.normal(shape=(batch, seqlen, nheads, headdim), dtype=dtype)
    dt = tf.nn.softplus(tf.random.uniform(shape=(batch, seqlen, nheads), dtype=tf.float32) - 4).requires_grad_()
    A = (-tf.exp(tf.random.uniform(nheads, dtype=tf.float32, device=device))).requires_grad_()
    B = tf.random.normal(shape=(batch, seqlen, ngroups, headdim), dtype=dtype)
    C = tf.random.normal(shape=(batch, seqlen, ngroups, headdim), dtype=dtype)
    D = tf.random.normal(shape=(nheads,), dtype=dtype)

    # Comparing fused version and minimal version
    y = mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size, D=None)
    y_min, _ = ssd_minimal_discrete(x*dt.unsqueeze(-1), A*dt, B, C, chunk_size)
