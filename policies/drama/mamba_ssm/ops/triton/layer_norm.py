# Copyright (c) 2024, Tri Dao.
# Implement dropout + residual + layer_norm / rms_norm.

# Based on the Triton LayerNorm tutorial: https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
# For the backward pass, we keep weight_grad and bias_grad in registers and accumulate.
# This is faster for dimensions up to 8k, but after that it's much slower due to register spilling.
# The models we train have hidden dim up to 8k anyway (e.g. Llama 70B), so this is fine.

import math
import warnings

import torch
import torch.nn.functional as F
from policies.drama.mamba_ssm.utils.torch import custom_bwd, custom_fwd

from ray.rllib.utils.framework import try_import_tf
_, tf, _ = try_import_tf()

import triton
import triton.language as tl


def layer_norm_ref(
    x,
    weight,
    bias,
    residual=None,
    x1=None,
    weight1=None,
    bias1=None,
    epsilon=1e-6,
    dropout_p=0.0,
    rowscale=None,
    prenorm=False,
    dropout_mask=None,
    dropout_mask1=None,
    upcast=False,
):
    dtype = x.dtype
    if upcast:
        x = x.float()
        weight = weight.float()
        bias = bias.float() if bias is not None else None
        residual = residual.float() if residual is not None else residual
        x1 = x1.float() if x1 is not None else None
        weight1 = weight1.float() if weight1 is not None else None
        bias1 = bias1.float() if bias1 is not None else None
    if x1 is not None:
        assert rowscale is None, "rowscale is not supported with parallel LayerNorm"
    if rowscale is not None:
        x = x * rowscale[..., None]
    if dropout_p > 0.0:
        if dropout_mask is not None:
            x = x.masked_fill(~dropout_mask, 0.0) / (1.0 - dropout_p)
        else:
            x = F.dropout(x, p=dropout_p)
        if x1 is not None:
            if dropout_mask1 is not None:
                x1 = x1.masked_fill(~dropout_mask1, 0.0) / (1.0 - dropout_p)
            else:
                x1 = F.dropout(x1, p=dropout_p)
    if x1 is not None:
        x = x + x1
    if residual is not None:
        x = (x + residual).to(x.dtype)
    out = F.layer_norm(x.to(weight.dtype), x.shape[-1:], weight=weight, bias=bias, epsilon=epsilon).to(
        dtype
    )
    if weight1 is None:
        return out if not prenorm else (out, x)
    else:
        out1 = F.layer_norm(
            x.to(weight1.dtype), x.shape[-1:], weight=weight1, bias=bias1, epsilon=epsilon
        ).to(dtype)
        return (out, out1) if not prenorm else (out, out1, x)


def rms_norm_ref(
    x,
    weight,
    bias,
    residual=None,
    x1=None,
    weight1=None,
    bias1=None,
    epsilon=1e-6,
    dropout_p=0.0,
    rowscale=None,
    prenorm=False,
    dropout_mask=None,
    dropout_mask1=None,
    upcast=False,
):
    dtype = x.dtype
    if upcast:
        x = x.float()
        weight = weight.float()
        bias = bias.float() if bias is not None else None
        residual = residual.float() if residual is not None else residual
        x1 = x1.float() if x1 is not None else None
        weight1 = weight1.float() if weight1 is not None else None
        bias1 = bias1.float() if bias1 is not None else None
    if x1 is not None:
        assert rowscale is None, "rowscale is not supported with parallel LayerNorm"
    if rowscale is not None:
        x = x * rowscale[..., None]
    if dropout_p > 0.0:
        if dropout_mask is not None:
            x = x.masked_fill(~dropout_mask, 0.0) / (1.0 - dropout_p)
        else:
            x = F.dropout(x, p=dropout_p)
        if x1 is not None:
            if dropout_mask1 is not None:
                x1 = x1.masked_fill(~dropout_mask1, 0.0) / (1.0 - dropout_p)
            else:
                x1 = F.dropout(x1, p=dropout_p)
    if x1 is not None:
        x = x + x1
    if residual is not None:
        x = (x + residual).to(x.dtype)
    rstd = 1 / tf.math.sqrt((x.square()).mean(dim=-1, keepdim=True) + epsilon)
    out = ((x * rstd * weight) + bias if bias is not None else (x * rstd * weight)).to(dtype)
    if weight1 is None:
        return out if not prenorm else (out, x)
    else:
        out1 = ((x * rstd * weight1) + bias1 if bias1 is not None else (x * rstd * weight1)).to(
            dtype
        )
        return (out, out1) if not prenorm else (out, out1, x)

# EMRAN commenting out because I don't want to deal with cuda unless I have to
def config_prune(configs):

    if torch.version.hip:
        try:
            # set warp size based on gcn architecure 
            gcn_arch_name = torch.cuda.get_device_properties(0).gcnArchName
            if "gfx10" in gcn_arch_name or "gfx11" in gcn_arch_name:
                # radeon
                warp_size = 32
            else:
                # instinct
                warp_size = 64
        except AttributeError as e:
            # fall back to crude method to set warp size
            device_name = torch.cuda.get_device_properties(0).name
            if 'instinct' in device_name.lower():
                warp_size = 64
            else:
                warp_size = 32
            warnings.warn(f"{e}, warp size set to {warp_size} based on device name: {device_name}", UserWarning)

    else:
        # cuda 
        warp_size = 32    

    max_block_sz = 1024
    max_num_warps = max_block_sz // warp_size
    pruned_configs = [config for config in configs if config.num_warps <= max_num_warps]
    return pruned_configs

configs_autotune = [
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
        ]

# EMRAN commenting out because I don't want to deal with cuda unless I have to
pruned_configs_autotune = config_prune(configs_autotune)

@triton.autotune(
    configs = pruned_configs_autotune,
    key=["N", "HAS_RESIDUAL", "STORE_RESIDUAL_OUT", "IS_RMS_NORM", "HAS_BIAS"],
)
# @triton.heuristics({"HAS_BIAS": lambda args: args["B"] is not None})
# @triton.heuristics({"HAS_RESIDUAL": lambda args: args["RESIDUAL"] is not None})
@triton.heuristics({"HAS_X1": lambda args: args["X1"] is not None})
@triton.heuristics({"HAS_W1": lambda args: args["W1"] is not None})
@triton.heuristics({"HAS_B1": lambda args: args["B1"] is not None})
@triton.jit
def _layer_norm_fwd_1pass_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    RESIDUAL,  # pointer to the residual
    X1,
    W1,
    B1,
    Y1,
    RESIDUAL_OUT,  # pointer to the residual
    ROWSCALE,
    SEEDS,  # Dropout seeds for each row
    DROPOUT_MASK,
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_y_row,
    stride_res_row,
    stride_res_out_row,
    stride_x1_row,
    stride_y1_row,
    M,  # number of rows in X
    N,  # number of columns in X
    epsilon,  # epsilon to avoid division by zero
    dropout_p,  # Dropout probability
    IS_RMS_NORM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    STORE_RESIDUAL_OUT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_DROPOUT: tl.constexpr,
    STORE_DROPOUT_MASK: tl.constexpr,
    HAS_ROWSCALE: tl.constexpr,
    HAS_X1: tl.constexpr,
    HAS_W1: tl.constexpr,
    HAS_B1: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    X += row * stride_x_row
    Y += row * stride_y_row
    if HAS_RESIDUAL:
        RESIDUAL += row * stride_res_row
    if STORE_RESIDUAL_OUT:
        RESIDUAL_OUT += row * stride_res_out_row
    if HAS_X1:
        X1 += row * stride_x1_row
    if HAS_W1:
        Y1 += row * stride_y1_row
    # Compute mean and variance
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    if HAS_ROWSCALE:
        rowscale = tl.load(ROWSCALE + row).to(tl.float32)
        x *= rowscale
    if HAS_DROPOUT:
        # Compute dropout mask
        # 7 rounds is good enough, and reduces register pressure
        keep_mask = tl.rand(tl.load(SEEDS + row).to(tl.uint32), cols, n_rounds=7) > dropout_p
        x = tl.where(keep_mask, x / (1.0 - dropout_p), 0.0)
        if STORE_DROPOUT_MASK:
            tl.store(DROPOUT_MASK + row * N + cols, keep_mask, mask=cols < N)
    if HAS_X1:
        x1 = tl.load(X1 + cols, mask=cols < N, other=0.0).to(tl.float32)
        if HAS_ROWSCALE:
            rowscale = tl.load(ROWSCALE + M + row).to(tl.float32)
            x1 *= rowscale
        if HAS_DROPOUT:
            # Compute dropout mask
            # 7 rounds is good enough, and reduces register pressure
            keep_mask = (
                tl.rand(tl.load(SEEDS + M + row).to(tl.uint32), cols, n_rounds=7) > dropout_p
            )
            x1 = tl.where(keep_mask, x1 / (1.0 - dropout_p), 0.0)
            if STORE_DROPOUT_MASK:
                tl.store(DROPOUT_MASK + (M + row) * N + cols, keep_mask, mask=cols < N)
        x += x1
    if HAS_RESIDUAL:
        residual = tl.load(RESIDUAL + cols, mask=cols < N, other=0.0).to(tl.float32)
        x += residual
    if STORE_RESIDUAL_OUT:
        tl.store(RESIDUAL_OUT + cols, x, mask=cols < N)
    if not IS_RMS_NORM:
        mean = tl.sum(x, axis=0) / N
        tl.store(Mean + row, mean)
        xbar = tl.where(cols < N, x - mean, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
    else:
        xbar = tl.where(cols < N, x, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(var + epsilon)
    tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    mask = cols < N
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    if HAS_BIAS:
        b = tl.load(B + cols, mask=mask).to(tl.float32)
    x_hat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
    y = x_hat * w + b if HAS_BIAS else x_hat * w
    # Write output
    tl.store(Y + cols, y, mask=mask)
    if HAS_W1:
        w1 = tl.load(W1 + cols, mask=mask).to(tl.float32)
        if HAS_B1:
            b1 = tl.load(B1 + cols, mask=mask).to(tl.float32)
        y1 = x_hat * w1 + b1 if HAS_B1 else x_hat * w1
        tl.store(Y1 + cols, y1, mask=mask)


# EMRAN commenting out because I don't want to deal with cuda unless I have to
def _layer_norm_fwd(
    x,
    weight,
    bias,
    epsilon,
    residual=None,
    x1=None,
    weight1=None,
    bias1=None,
    dropout_p=0.0,
    rowscale=None,
    out_dtype=None,
    residual_dtype=None,
    is_rms_norm=False,
    return_dropout_mask=False,
):
    if residual is not None:
        residual_dtype = residual.dtype
    M, N = x.shape
    assert x.shape[-1] == 1
    if residual is not None:
        assert residual.shape[-1] == 1
        assert residual.shape == (M, N)
    assert weight.shape == (N,)
    assert weight.shape[-1] == 1
    if bias is not None:
        assert bias.shape[-1] == 1
        assert bias.shape == (N,)
    if x1 is not None:
        assert x1.shape == x.shape
        assert rowscale is None
        assert x1.shape[-1] == 1
    if weight1 is not None:
        assert weight1.shape == (N,)
        assert weight1.shape[-1] == 1
    if bias1 is not None:
        assert bias1.shape == (N,)
        assert bias1.shape[-1] == 1
    if rowscale is not None:
        assert rowscale.is_contiguous()
        assert rowscale.shape == (M,)
    # allocate output
    y = tf.zeros_like(x, dtype=x.dtype if out_dtype is None else out_dtype)
    assert y.shape[-1] == 1
    if weight1 is not None:
        y1 = tf.zeros_like(y)
        assert y1.shape[-1] == 1
    else:
        y1 = None
    if (
        residual is not None
        or (residual_dtype is not None and residual_dtype != x.dtype)
        or dropout_p > 0.0
        or rowscale is not None
        or x1 is not None
    ):
        residual_out = tf.Variable(
            tf.zeros((M, N), dtype=residual_dtype if residual_dtype is not None else x.dtype)
            )
        assert residual_out.shape[-1] == 1
    else:
        residual_out = None
    mean = tf.Variable(tf.random.normal((M,), dtype=tf.float32)) if not is_rms_norm else None
    rstd = tf.Variable(tf.random.normal((M,), dtype=tf.float32))
    if dropout_p > 0.0:
        seeds = tf.random.uniform(shape=(M,) if x1 is None else (2 * M,), minval=0, maxval=2**32, dtype=tf.int64)

    else:
        seeds = None
    if return_dropout_mask and dropout_p > 0.0:
        dropout_mask = tf.Variable(tf.zeros((M if x1 is None else 2 * M, N), dtype=tf.bool)) 
    else:
        dropout_mask = None
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    with torch.cuda.device(x.device.index):
        _layer_norm_fwd_1pass_kernel[(M,)](
            x,
            y,
            weight,
            bias,
            residual,
            x1,
            weight1,
            bias1,
            y1,
            residual_out,
            rowscale,
            seeds,
            dropout_mask,
            mean,
            rstd,
            x.stride(0),
            y.stride(0),
            residual.stride(0) if residual is not None else 0,
            residual_out.stride(0) if residual_out is not None else 0,
            x1.stride(0) if x1 is not None else 0,
            y1.stride(0) if y1 is not None else 0,
            M,
            N,
            epsilon,
            dropout_p,
            is_rms_norm,
            BLOCK_N,
            residual is not None,
            residual_out is not None,
            bias is not None,
            dropout_p > 0.0,
            dropout_mask is not None,
            rowscale is not None,
        )
    # residual_out is None if residual is None and residual_dtype == input_dtype and dropout_p == 0.0
    if dropout_mask is not None and x1 is not None:
        dropout_mask, dropout_mask1 = dropout_mask.tensor_split(2, dim=0)
    else:
        dropout_mask1 = None
    return (
        y,
        y1,
        mean,
        rstd,
        residual_out if residual_out is not None else x,
        seeds,
        dropout_mask,
        dropout_mask1,
    )

# EMRAN commenting out because I don't want to deal with cuda unless I have to
@triton.autotune(
    configs=pruned_configs_autotune,
    key=["N", "HAS_DRESIDUAL", "STORE_DRESIDUAL", "IS_RMS_NORM", "HAS_BIAS", "HAS_DROPOUT"],
)
# @triton.heuristics({"HAS_BIAS": lambda args: args["B"] is not None})
# @triton.heuristics({"HAS_DRESIDUAL": lambda args: args["DRESIDUAL"] is not None})
# @triton.heuristics({"STORE_DRESIDUAL": lambda args: args["DRESIDUAL_IN"] is not None})
@triton.heuristics({"HAS_ROWSCALE": lambda args: args["ROWSCALE"] is not None})
@triton.heuristics({"HAS_DY1": lambda args: args["DY1"] is not None})
@triton.heuristics({"HAS_DX1": lambda args: args["DX1"] is not None})
@triton.heuristics({"HAS_B1": lambda args: args["DB1"] is not None})
@triton.heuristics({"RECOMPUTE_OUTPUT": lambda args: args["Y"] is not None})
@triton.jit
def _layer_norm_bwd_kernel(
    X,  # pointer to the input
    W,  # pointer to the weights
    B,  # pointer to the biases
    Y,  # pointer to the output to be recomputed
    DY,  # pointer to the output gradient
    DX,  # pointer to the input gradient
    DW,  # pointer to the partial sum of weights gradient
    DB,  # pointer to the partial sum of biases gradient
    DRESIDUAL,
    W1,
    DY1,
    DX1,
    DW1,
    DB1,
    DRESIDUAL_IN,
    ROWSCALE,
    SEEDS,
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_y_row,
    stride_dy_row,
    stride_dx_row,
    stride_dres_row,
    stride_dy1_row,
    stride_dx1_row,
    stride_dres_in_row,
    M,  # number of rows in X
    N,  # number of columns in X
    epsilon,  # epsilon to avoid division by zero
    dropout_p,
    rows_per_program,
    IS_RMS_NORM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_DRESIDUAL: tl.constexpr,
    STORE_DRESIDUAL: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_DROPOUT: tl.constexpr,
    HAS_ROWSCALE: tl.constexpr,
    HAS_DY1: tl.constexpr,
    HAS_DX1: tl.constexpr,
    HAS_B1: tl.constexpr,
    RECOMPUTE_OUTPUT: tl.constexpr,
):
    # Map the program id to the elements of X, DX, and DY it should compute.
    row_block_id = tl.program_id(0)
    row_start = row_block_id * rows_per_program
    # Do not early exit if row_start >= M, because we need to write DW and DB
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    X += row_start * stride_x_row
    if HAS_DRESIDUAL:
        DRESIDUAL += row_start * stride_dres_row
    if STORE_DRESIDUAL:
        DRESIDUAL_IN += row_start * stride_dres_in_row
    DY += row_start * stride_dy_row
    DX += row_start * stride_dx_row
    if HAS_DY1:
        DY1 += row_start * stride_dy1_row
    if HAS_DX1:
        DX1 += row_start * stride_dx1_row
    if RECOMPUTE_OUTPUT:
        Y += row_start * stride_y_row
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    if RECOMPUTE_OUTPUT and HAS_BIAS:
        b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_DY1:
        w1 = tl.load(W1 + cols, mask=mask).to(tl.float32)
    dw = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if HAS_BIAS:
        db = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if HAS_DY1:
        dw1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
        if HAS_B1:
            db1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    row_end = min((row_block_id + 1) * rows_per_program, M)
    for row in range(row_start, row_end):
        # Load data to SRAM
        x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
        dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
        if HAS_DY1:
            dy1 = tl.load(DY1 + cols, mask=mask, other=0).to(tl.float32)
        if not IS_RMS_NORM:
            mean = tl.load(Mean + row)
        rstd = tl.load(Rstd + row)
        # Compute dx
        xhat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
        xhat = tl.where(mask, xhat, 0.0)
        if RECOMPUTE_OUTPUT:
            y = xhat * w + b if HAS_BIAS else xhat * w
            tl.store(Y + cols, y, mask=mask)
        wdy = w * dy
        dw += dy * xhat
        if HAS_BIAS:
            db += dy
        if HAS_DY1:
            wdy += w1 * dy1
            dw1 += dy1 * xhat
            if HAS_B1:
                db1 += dy1
        if not IS_RMS_NORM:
            c1 = tl.sum(xhat * wdy, axis=0) / N
            c2 = tl.sum(wdy, axis=0) / N
            dx = (wdy - (xhat * c1 + c2)) * rstd
        else:
            c1 = tl.sum(xhat * wdy, axis=0) / N
            dx = (wdy - xhat * c1) * rstd
        if HAS_DRESIDUAL:
            dres = tl.load(DRESIDUAL + cols, mask=mask, other=0).to(tl.float32)
            dx += dres
        # Write dx
        if STORE_DRESIDUAL:
            tl.store(DRESIDUAL_IN + cols, dx, mask=mask)
        if HAS_DX1:
            if HAS_DROPOUT:
                keep_mask = (
                    tl.rand(tl.load(SEEDS + M + row).to(tl.uint32), cols, n_rounds=7) > dropout_p
                )
                dx1 = tl.where(keep_mask, dx / (1.0 - dropout_p), 0.0)
            else:
                dx1 = dx
            tl.store(DX1 + cols, dx1, mask=mask)
        if HAS_DROPOUT:
            keep_mask = tl.rand(tl.load(SEEDS + row).to(tl.uint32), cols, n_rounds=7) > dropout_p
            dx = tl.where(keep_mask, dx / (1.0 - dropout_p), 0.0)
        if HAS_ROWSCALE:
            rowscale = tl.load(ROWSCALE + row).to(tl.float32)
            dx *= rowscale
        tl.store(DX + cols, dx, mask=mask)

        X += stride_x_row
        if HAS_DRESIDUAL:
            DRESIDUAL += stride_dres_row
        if STORE_DRESIDUAL:
            DRESIDUAL_IN += stride_dres_in_row
        if RECOMPUTE_OUTPUT:
            Y += stride_y_row
        DY += stride_dy_row
        DX += stride_dx_row
        if HAS_DY1:
            DY1 += stride_dy1_row
        if HAS_DX1:
            DX1 += stride_dx1_row
    tl.store(DW + row_block_id * N + cols, dw, mask=mask)
    if HAS_BIAS:
        tl.store(DB + row_block_id * N + cols, db, mask=mask)
    if HAS_DY1:
        tl.store(DW1 + row_block_id * N + cols, dw1, mask=mask)
        if HAS_B1:
            tl.store(DB1 + row_block_id * N + cols, db1, mask=mask)


# EMRAN commenting out because I don't want to deal with cuda unless I have to
def _layer_norm_bwd(
    dy,
    x,
    weight,
    bias,
    epsilon,
    mean,
    rstd,
    dresidual=None,
    dy1=None,
    weight1=None,
    bias1=None,
    seeds=None,
    dropout_p=0.0,
    rowscale=None,
    has_residual=False,
    has_x1=False,
    is_rms_norm=False,
    x_dtype=None,
    recompute_output=False,
):
    M, N = x.shape
    assert x.stride(-1) == 1
    assert dy.stride(-1) == 1
    assert dy.shape == (M, N)
    if dresidual is not None:
        assert dresidual.stride(-1) == 1
        assert dresidual.shape == (M, N)
    assert weight.shape == (N,)
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N,)
    if dy1 is not None:
        assert weight1 is not None
        assert dy1.shape == dy.shape
        assert dy1.stride(-1) == 1
    if weight1 is not None:
        assert weight1.shape == (N,)
        assert weight1.stride(-1) == 1
    if bias1 is not None:
        assert bias1.shape == (N,)
        assert bias1.stride(-1) == 1
    if seeds is not None:
        assert seeds.is_contiguous()
        assert seeds.shape == (M if not has_x1 else M * 2,)
    if rowscale is not None:
        assert rowscale.is_contiguous()
        assert rowscale.shape == (M,)
    # allocate output
    dx = (
        tf.zeros_like(x)
        if x_dtype is None
        else tf.Variable(tf.zeros((M, N), dtype=x_dtype))
    )
    dresidual_in = (
        tf.zeros_like(x)
        if has_residual
        and (dx.dtype != x.dtype or dropout_p > 0.0 or rowscale is not None or has_x1)
        else None
    )
    dx1 = tf.zeros_like(dx) if (has_x1 and dropout_p > 0.0) else None
    y = tf.Variable(tf.zeros((M, N), dtype=dy.dtype)) if recompute_output else None
    if recompute_output:
        assert weight1 is None, "recompute_output is not supported with parallel LayerNorm"

    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
    _dw = tf.zeros((sm_count, N), dtype=tf.float32)
    _db = (
        tf.zeros((sm_count, N), dtype=tf.float32)
        if bias is not None
        else None
    )
    _dw1 = tf.zeros_like(_dw) if weight1 is not None else None
    _db1 = tf.zeros_like(_db) if bias1 is not None else None
    rows_per_program = math.ceil(M / sm_count)
    grid = (sm_count,)
    with torch.cuda.device(x.device.index):
        _layer_norm_bwd_kernel[grid](
            x,
            weight,
            bias,
            y,
            dy,
            dx,
            _dw,
            _db,
            dresidual,
            weight1,
            dy1,
            dx1,
            _dw1,
            _db1,
            dresidual_in,
            rowscale,
            seeds,
            mean,
            rstd,
            x.stride(0),
            0 if not recompute_output else y.stride(0),
            dy.stride(0),
            dx.stride(0),
            dresidual.stride(0) if dresidual is not None else 0,
            dy1.stride(0) if dy1 is not None else 0,
            dx1.stride(0) if dx1 is not None else 0,
            dresidual_in.stride(0) if dresidual_in is not None else 0,
            M,
            N,
            epsilon,
            dropout_p,
            rows_per_program,
            is_rms_norm,
            BLOCK_N,
            dresidual is not None,
            dresidual_in is not None,
            bias is not None,
            dropout_p > 0.0,
        )
    dw = _dw.sum(0).to(weight.dtype)
    db = _db.sum(0).to(bias.dtype) if bias is not None else None
    dw1 = _dw1.sum(0).to(weight1.dtype) if weight1 is not None else None
    db1 = _db1.sum(0).to(bias1.dtype) if bias1 is not None else None
    # Don't need to compute dresidual_in separately in this case
    if has_residual and dx.dtype == x.dtype and dropout_p == 0.0 and rowscale is None:
        dresidual_in = dx
    if has_x1 and dropout_p == 0.0:
        dx1 = dx
    return (
        (dx, dw, db, dresidual_in, dx1, dw1, db1)
        if not recompute_output
        else (dx, dw, db, dresidual_in, dx1, dw1, db1, y)
    )

# EMRAN used to be class LayerNormFn(torch.autograd.Function)
# @tf.custom_gradient
# def layer_norm_custom(x, weight, bias, epsilon=1e-6, dropout_p=0.0, training=True):
#     x_shape_og = tf.shape(x)
#     x_flat = tf.reshape(x, [-1, x_shape_og[-1]])

#     # Normalize
#     mean = tf.reduce_mean(x_flat, axis=-1, keepdims=True)
#     var = tf.reduce_mean(tf.square(x_flat - mean), axis=-1, keepdims=True)
#     rstd = tf.math.rsqrt(var + epsilon)
#     norm = (x_flat - mean) * rstd

#     # Affine transformation
#     y = norm * weight + bias if bias is not None else norm * weight

#     # Apply dropout if training
#     if training and dropout_p > 0.0:
#         y = tf.nn.dropout(y, rate=dropout_p)

#     y = tf.reshape(y, x_shape_og)

#     def grad(dy):
#         dy = tf.reshape(dy, [-1, x_shape_og[-1]])
#         N = tf.cast(tf.shape(dy)[-1], dy.dtype)

#         # Gradient of LayerNorm (simplified version)
#         dx = (dy - tf.reduce_mean(dy, axis=-1, keepdims=True)
#               - norm * tf.reduce_mean(dy * norm, axis=-1, keepdims=True)) * rstd

#         dw = tf.reduce_sum(dy * norm, axis=0)
#         db = tf.reduce_sum(dy, axis=0) if bias is not None else None

#         return tf.reshape(dx, x_shape_og), dw, db, None, None, None

#     return y, grad


# def layer_norm_custom(
#     x,
#     weight,
#     bias=None,
#     residual=None,
#     x1=None,
#     weight1=None,
#     bias1=None,
#     epsilon=1e-6,
#     dropout_p=0.0,
#     rowscale=None,
#     prenorm=False,
#     residual_in_fp32=False,
#     is_rms_norm=False,
#     return_dropout_mask=False,
#     training=False,
# ):
#     x_shape_og = tf.shape(x)
#     last_dim = x.shape[-1]
#     x = tf.reshape(x, [-1, last_dim])
#     if residual is not None:
#         residual = tf.reshape(residual, [-1, last_dim])
#     if x1 is not None:
#         x1 = tf.reshape(x1, [-1, last_dim])
#     if rowscale is not None:
#         rowscale = tf.reshape(rowscale, [-1])

#     # Normalization
#     if is_rms_norm:
#         mean_sq = tf.reduce_mean(tf.square(x), axis=-1, keepdims=True)
#         normed = x / tf.sqrt(mean_sq + epsilon)
#     else:
#         mean = tf.reduce_mean(x, axis=-1, keepdims=True)
#         var = tf.reduce_mean(tf.square(x - mean), axis=-1, keepdims=True)
#         normed = (x - mean) / tf.sqrt(var + epsilon)

#     output = normed * weight + (bias if bias is not None else 0.0)

#     # Dropout
#     if dropout_p > 0.0 and training:
#         output = tf.nn.dropout(output, rate=dropout_p)

#     # Residual connection
#     if residual is not None:
#         output += tf.cast(residual, tf.float32) if residual_in_fp32 else residual

#     output = tf.reshape(output, x_shape_og)
#     return output

# class CustomLayerNorm(tf.keras.layers.Layer):
#     def call(
#         self,
#         x,
#         weight,
#         bias,
#         residual=None,
#         x1=None,
#         weight1=None,
#         bias1=None,
#         epsilon=1e-6,
#         dropout_p=0.0,
#         rowscale=None,
#         prenorm=False,
#         residual_in_fp32=False,
#         is_rms_norm=False,
#         return_dropout_mask=False,
#         training=None,
#     ):
#     return layer_norm_custom(
#         x,
#         weight,
#         bias,
#         residual,
#         x1,
#         weight1,
#         bias1,
#         epsilon,
#         dropout_p,
#         rowscale,
#         prenorm,
#         residual_in_fp32,
#         is_rms_norm,
#         return_dropout_mask,
#         training,
#     )

class LayerNormFn(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(
        self,
        x,
        weight,
        bias,
        residual=None,
        x1=None,
        weight1=None,
        bias1=None,
        epsilon=1e-6,
        dropout_p=0.0,
        rowscale=None,
        prenorm=False,
        residual_in_fp32=False,
        is_rms_norm=False,
        return_dropout_mask=False,
    ):
        return lnf(
            x,
            weight,
            bias,
            residual,
            x1,
            weight1,
            bias1,
            epsilon,
            dropout_p,
            rowscale,
            prenorm,
            residual_in_fp32,
            is_rms_norm,
            return_dropout_mask,
        ) 

def lnf(
    x,
    weight,
    bias,
    residual=None,
    x1=None,
    weight1=None,
    bias1=None,
    epsilon=1e-6,
    dropout_p=0.0,
    rowscale=None,
    prenorm=False,
    residual_in_fp32=False,
    is_rms_norm=False,
    return_dropout_mask=False,
):

    SENTINEL = tf.constant([-999], dtype=x.dtype)

    def is_sentinel(t):
        # returns a boolean scalar tensor: True if t is sentinel, False otherwise
        return tf.reduce_all(tf.equal(t, SENTINEL))

    @tf.custom_gradient
    def inside_function(
        x,
        weight,
        bias,
        residual,
        x1,
        weight1,
        bias1,
        rowscale,
    ):
        SENTINEL = tf.constant([-999.], dtype=x.dtype)

        def is_sentinel(t):
            return tf.reduce_all(tf.equal(t, SENTINEL))

        use_bias = tf.logical_not(is_sentinel(bias))
        use_residual = tf.logical_not(is_sentinel(residual))
        use_x1 = tf.logical_not(is_sentinel(x1))
        use_weight1 = tf.logical_not(is_sentinel(weight1))
        use_bias1 = tf.logical_not(is_sentinel(bias1))
        use_rowscale = tf.logical_not(is_sentinel(rowscale))

        x = tf.reshape(x, (-1, x.shape[-1]))

        residual = tf.cond(use_residual,
            lambda: tf.reshape(residual, (-1, tf.shape(residual)[-1])),
            lambda: SENTINEL,
        )

        x1 = tf.cond(use_x1,
            lambda: tf.reshape(x1, (-1, tf.shape(x1)[-1])),
            lambda: SENTINEL,
        )

        rowscale = tf.cond(use_rowscale,
            lambda: tf.reshape(rowscale, [-1]),
            lambda: SENTINEL,
        )

        residual = None if is_sentinel(residual) else residual
        x1 = None if is_sentinel(x1) else x1
        rowscale = None if is_sentinel(rowscale) else rowscale
        bias = None if not use_bias else bias
        weight1 = None if not use_weight1 else weight1
        bias1 = None if not use_bias1 else bias1

        residual_dtype = tf.float32 if (residual_in_fp32 and residual is None) else residual.dtype

        y, y1, mean, rstd, residual_out, seeds, dropout_mask, dropout_mask1 = _layer_norm_fwd(
            x,
            weight,
            bias,
            epsilon,
            residual,
            x1,
            weight1,
            bias1,
            dropout_p=dropout_p,
            rowscale=rowscale,
            residual_dtype=residual_dtype,
            is_rms_norm=is_rms_norm,
            return_dropout_mask=return_dropout_mask,
        )

        # original_shape = tf.shape(x)
        # x_reshaped = tf.reshape(x, [-1, tf.shape(x)[-1]])

        # residual_reshaped = tf.cond(
        #     use_residual,
        #     lambda: tf.reshape(residual, [-1, tf.shape(residual)[-1]]),
        #     lambda: SENTINEL,
        # )
        # x1_reshaped = tf.cond(
        #     use_x1,
        #     lambda: tf.reshape(x1, [-1, tf.shape(x1)[-1]]),
        #     lambda: SENTINEL,
        # )
        # rowscale_reshaped = tf.cond(
        #     use_rowscale,
        #     lambda: tf.reshape(rowscale, [-1]),
        #     lambda: SENTINEL,
        # )

        # def _norm(input_x, gamma, beta):
        #     mean = tf.reduce_mean(input_x, axis=-1, keepdims=True)
        #     variance = tf.reduce_mean(tf.square(input_x - mean), axis=-1, keepdims=True)
        #     rstd = tf.math.rsqrt(variance + epsilon)
        #     normalized = (input_x - mean) * rstd
        #     print("normalized, gamma, beta:", normalized, gamma, beta)
        #     return normalized * gamma + beta, mean, rstd

        # y, mean, rstd = _norm(x_reshaped, weight, tf.where(use_bias, bias, tf.zeros_like(bias)))

        # y1 = tf.cond(
        #     tf.logical_and(use_x1, tf.logical_and(use_weight1, use_bias1)),
        #     lambda: _norm(x1_reshaped, weight1, bias1)[0],
        #     lambda: SENTINEL,
        # )

        # dropout_mask = None
        # if dropout_p > 0.0:
        #     dropout_mask = tf.cast(tf.random.uniform(tf.shape(y)) >= dropout_p, y.dtype)
        #     y = y * dropout_mask / (1.0 - dropout_p)

        # y = tf.reshape(y, original_shape)

        # y1 = tf.cond(
        #     use_x1 & use_weight1 & use_bias1,
        #     lambda: tf.reshape(y1, original_shape),
        #     lambda: SENTINEL,
        # )

        # residual_out = tf.cond(
        #     use_residual,
        #     lambda: tf.reshape(residual_reshaped, original_shape),
        #     lambda: SENTINEL,
        # )

        # def grad(dy, *grad_args):
        #     dy_reshaped = tf.reshape(dy, [-1, tf.shape(dy)[-1]])

        #     def _grad_norm(dy_, x_, gamma, mean_, rstd_):
        #         N = tf.cast(tf.shape(x_)[-1], dy_.dtype)
        #         x_mu = x_ - mean_
        #         dx_hat = dy_ * gamma
        #         dvar = tf.reduce_sum(dx_hat * x_mu, axis=-1, keepdims=True) * -0.5 * tf.pow(rstd_, 3)
        #         dmean = tf.reduce_sum(dx_hat * -rstd_, axis=-1, keepdims=True) + dvar * tf.reduce_mean(-2.0 * x_mu, axis=-1, keepdims=True)
        #         dx = dx_hat * rstd_ + dvar * 2 * x_mu / N + dmean / N
        #         dgamma = tf.reduce_sum(dy_ * (x_ - mean_) * rstd_, axis=0)
        #         dbeta = tf.reduce_sum(dy_, axis=0)
        #         return dx, dgamma, dbeta

        #     dx, dgamma, dbeta = _grad_norm(dy_reshaped, x_reshaped, weight, mean, rstd)

        #     if dropout_p > 0.0:
        #         dx = dx * dropout_mask / (1.0 - dropout_p)

        #     dx = tf.reshape(dx, original_shape)

            # return (
            #     dx,
            #     dgamma,
            #     dbeta,
            #     tf.zeros_like(residual) if use_residual else SENTINEL,
            #     tf.zeros_like(x1) if use_x1 else SENTINEL,
            #     tf.zeros_like(weight1) if use_weight1 else SENTINEL,
            #     tf.zeros_like(bias1) if use_bias1 else SENTINEL,
            #     tf.zeros_like(rowscale) if use_rowscale else SENTINEL,
            # )

        # Create zero tensors with the right shape for missing outputs
        zero_y1 = tf.zeros_like(y)

        if not return_dropout_mask:
            if not use_weight1:
                if not prenorm:
                    outputs = (y, zero_y1)  # include y1 placeholder
                else:
                    outputs = (y, zero_y1, residual_out)
            else:
                if not prenorm:
                    outputs = (y, y1)
                else:
                    outputs = (y, y1, residual_out)
        else:
            zero_dropout_mask = tf.zeros_like(y, dtype=tf.bool)  # or tf.zeros_like(y), adjust dtype as needed
            none_placeholder = tf.constant(0)  # or tf.zeros([]), a scalar placeholder

            if not use_weight1:
                if not prenorm:
                    outputs = (y, zero_dropout_mask, zero_y1, none_placeholder)
                else:
                    outputs = (y, zero_y1, residual_out, zero_dropout_mask, none_placeholder)
            else:
                if not prenorm:
                    outputs = (y, y1, dropout_mask, none_placeholder)
                else:
                    outputs = (y, y1, residual_out, dropout_mask, none_placeholder)

        print("RMS output:", outputs)

        return outputs, grad

    # Replace all None inputs with SENTINEL tensor
    return inside_function(
        x,
        weight,
        bias if bias is not None else SENTINEL,
        residual if residual is not None else SENTINEL,
        x1 if x1 is not None else SENTINEL,
        weight1 if weight1 is not None else SENTINEL,
        bias1 if bias1 is not None else SENTINEL,
        rowscale if rowscale is not None else SENTINEL,
    )



def layer_norm_fn():
    return LayerNormFn()


def rms_norm_fn(
    x,
    weight,
    bias,
    residual=None,
    x1=None,
    weight1=None,
    bias1=None,
    epsilon=1e-6,
    dropout_p=0.0,
    rowscale=None,
    prenorm=False,
    residual_in_fp32=False,
    return_dropout_mask=False,
):  
    return lnf(
        x,
        weight,
        bias,
        residual,
        x1,
        weight1,
        bias1,
        epsilon,
        dropout_p,
        rowscale,
        prenorm,
        residual_in_fp32,
        True,
        return_dropout_mask,
    )


class RMSNorm(tf.keras.layers.Layer):

    def __init__(self, hidden_size, epsilon=1e-5, dropout_p=0.0, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.epsilon = epsilon
        if dropout_p > 0.0:
            self.drop = tf.keras.layers.Dropout(dropout_p)
        else:
            self.drop = None
        self.weight = tf.Variable(tf.random.uniform([hidden_size]), trainable=True) 
        self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.assign(tf.ones_like(self.weight))

    def call(self, x, residual=None, prenorm=False, residual_in_fp32=False):
        return rms_norm_fn(
            x,
            self.weight,
            self.bias,
            residual=residual,
            epsilon=self.epsilon,
            dropout_p=self.drop.p if self.drop is not None and self.training else 0.0,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
        )


# EMRAN used to be class LayerNormLinearFn(torch.autograd.Function)
def layer_norm_linear_fn(
    x,
    norm_weight,
    norm_bias,
    linear_weight,
    linear_bias,
    residual=None,
    epsilon=1e-6,
    prenorm=False,
    residual_in_fp32=False,
    is_rms_norm=False,
):
    x_shape_og = tf.shape(x)
    x_dtype = x.dtype
    x_2d = tf.reshape(x, [-1, x.shape[-1]])

    if residual is not None:
        residual_2d = tf.reshape(residual, [-1, residual.shape[-1]])
        residual_dtype = tf.float32 if residual_in_fp32 else residual.dtype
    else:
        residual_2d = None
        residual_dtype = tf.float32 if residual_in_fp32 else None

    # Forward logic
    @tf.custom_gradient # EMRAN 
    def _layer_norm_linear(x_2d):

        # Assumed to be implemented similarly to PyTorch
        y, mean, rstd, residual_out, y_pre = _layer_norm_fwd_tf(
            x_2d,
            norm_weight,
            norm_bias,
            epsilon,
            residual=residual_2d,
            out_dtype=None,  # TensorFlow handles mixed precision natively
            residual_dtype=residual_dtype,
            is_rms_norm=is_rms_norm,
        )

        y = tf.reshape(y, x_shape_og)

        out = tf.linalg.matmul(y, tf.transpose(linear_weight))
        if linear_bias is not None:
            out += linear_bias

        if prenorm:
            out_return = (out, tf.reshape(residual_out, x_shape_og))
        else:
            out_return = out

        def grad(dout, *grad_args):
            dout_2d = tf.reshape(dout, [-1, dout.shape[-1]])
            dy = tf.linalg.matmul(dout_2d, linear_weight)

            if prenorm:
                dresidual = tf.reshape(grad_args[0], [-1, grad_args[0].shape[-1]])
            else:
                dresidual = None

            dx, dnorm_weight, dnorm_bias, dresidual_out, _, _, _, y_pre = _layer_norm_bwd_tf(
                dy,
                x_2d,
                norm_weight,
                norm_bias,
                epsilon,
                mean,
                rstd,
                dresidual=dresidual,
                has_residual=residual is not None,
                is_rms_norm=is_rms_norm,
                x_dtype=x_dtype,
                recompute_output=True,
            )

            dlinear_weight = tf.einsum("bo,bi->oi", dout_2d, y_pre)
            dlinear_bias = None if linear_bias is None else tf.reduce_sum(dout_2d, axis=0)

            dx = tf.reshape(dx, x_shape_og)
            dresidual_out = tf.reshape(dresidual_out, x_shape_og) if residual is not None else None

            return (
                dx,
                dnorm_weight,
                dnorm_bias,
                dlinear_weight,
                dlinear_bias,
                dresidual_out,
                None,
                None,
                None,
                None,
            )

        return out_return, grad

    return _layer_norm_linear(x_2d)



def layer_norm_linear_fn(
    x,
    norm_weight,
    norm_bias,
    linear_weight,
    linear_bias,
    residual=None,
    epsilon=1e-6,
    prenorm=False,
    residual_in_fp32=False,
    is_rms_norm=False,
):
    return layer_norm_linear_fn(
        x,
        norm_weight,
        norm_bias,
        linear_weight,
        linear_bias,
        residual,
        epsilon,
        prenorm,
        residual_in_fp32,
        is_rms_norm,
    )
