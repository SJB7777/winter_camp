from __future__ import annotations

from math import pi, sqrt, log

import torch
from torch import Tensor
import torch.nn.functional as F

from src.abeles import abeles


def abeles_constant_smearing(
    q: Tensor,
    thickness: Tensor,
    roughness: Tensor,
    sld: Tensor,
    sample_length: Tensor,
    beam_width: Tensor,
    dq: Tensor = None,
    gauss_num: int = 51,
    xrr_dq: bool = True,
    abeles_func=None,
) -> Tensor:
    """
    Abeles reflectivity with instrumental broadening (resolution smearing).
    """

    assert dq is not None

    dq = torch.atleast_2d(dq)

    batch_size = thickness.shape[0]

    if dq.shape[0] != batch_size:
        dq = dq.expand(batch_size, dq.shape[-1])

    abeles_func = abeles_func or abeles

    # q_lin: (batch, q_lin_num)
    q_lin = _get_q_axes(q, dq, gauss_num, xrr_dq=xrr_dq)
    
    # kernels: (batch, kernel_size)
    kernels = _get_t_gauss_kernels(dq, gauss_num)

    # Calculate Reflectivity on linear Q grid
    # curves: (batch, q_lin_num)
    curves = abeles_func(q_lin, thickness, roughness, sld, sample_length, beam_width)

    # Convolve with resolution kernel
    # Use torch.conv1d with groups=batch_size to apply different kernels per sample.
    # To do this, we treat the batch dimension as channels.
    # Input shape required: (1, batch_size, length)
    # Weight shape required: (batch_size, 1, kernel_size)
    
    padding = (kernels.shape[-1] - 1) // 2
    
    # curves: (batch, q_lin_num)
    # reshape to (1, batch, q_lin_num)
    inp = curves.unsqueeze(0) 
    inp_padded = F.pad(inp, (padding, padding), mode="reflect")
    
    # Weights: (batch, 1, kernel_size)
    w = kernels.unsqueeze(1)
    
    smeared_curves = F.conv1d(
        inp_padded,
        w,
        groups=batch_size,
    ) # Output: (1, batch, q_lin_num)
    
    smeared_curves = smeared_curves.squeeze(0) # (batch, q_lin_num)

    # Interpolate back to original Q
    # q might be (batch, num_q)
    if q.shape[0] != smeared_curves.shape[0]:
        q = q.expand(smeared_curves.shape[0], *q.shape[1:])

    smeared_curves = _batch_linear_interp1d(q_lin, smeared_curves, q)

    return smeared_curves


_FWHM = 2 * sqrt(2 * log(2.0))
_2PI_SQRT = 1.0 / sqrt(2 * pi)


def _batch_linspace(start: Tensor, end: Tensor, num: int):
    # Differentiable linspace
    # start, end: (batch, 1) or (batch, )
    steps = torch.linspace(0, 1, int(num), device=end.device, dtype=end.dtype)
    return start + (end - start) * steps.unsqueeze(0)


def _torch_gauss(x, s):
    return _2PI_SQRT / s * torch.exp(-0.5 * x**2 / s / s)


def _get_t_gauss_kernels(resolutions: Tensor, gaussnum: int = 51):
    # resolution: (batch, 1) usually
    
    # We need a fixed grid for convolution kernel relative to its width?
    # No, the kernel width in Q varies if dq varies?
    # Actually here we are generating kernels for the linear Q grid.
    
    # gauss_x range: [-1.7*dq, 1.7*dq]
    resolutions = resolutions.view(-1, 1)
    
    gauss_x = _batch_linspace(-1.7 * resolutions, 1.7 * resolutions, gaussnum)
    
    # spacing dx
    dx = (gauss_x[:, 1] - gauss_x[:, 0])[:, None]
    
    gauss_y = (
        _torch_gauss(gauss_x, resolutions / _FWHM)
        * dx
    )
    return gauss_y


def _get_q_axes(
    q: Tensor, resolutions: Tensor, gaussnum: int = 51, xrr_dq: bool = True
):
    if xrr_dq:
        return _get_q_axes_for_constant_dq(q, resolutions, gaussnum)
    else:
        return _get_q_axes_for_linear_dq(q, resolutions, gaussnum)


def _get_q_axes_for_linear_dq(q: Tensor, resolutions: Tensor, gaussnum: int = 51):
    gaussgpoint = (gaussnum - 1) / 2

    # min/max per batch
    # q: (batch, num_q)
    lowq = torch.clamp(q.min(1).values, min=1e-6)[..., None]
    highq = q.max(1).values[..., None]

    # Log space expansion
    start = torch.log10(lowq) - 6 * resolutions / _FWHM
    end = torch.log10(highq * (1 + 6 * resolutions / _FWHM))

    # Determine number of points needed
    # This part involves .to(int) which breaks differentiability w.r.t q bounds if used for indexing
    # But num_points defines the tensor size, so it must be int / non-differentiable.
    # The values in the grid can be differentiable.
    
    # We take the max required points across the batch to have a rectangular tensor
    num_pts_needed = (
        torch.abs((end - start) / (1.7 * resolutions / _FWHM / gaussgpoint))
        .max()
        .ceil()
        .int()
        .item()
    )
    
    # Ensure minimum points
    num_pts_needed = max(int(num_pts_needed), q.shape[1] + 10)

    q_lin = 10 ** _batch_linspace(start, end, num_pts_needed)

    return q_lin


def _get_q_axes_for_constant_dq(
    q: Tensor, resolutions: Tensor, gaussnum: int = 51
) -> Tensor:
    gaussgpoint = (gaussnum - 1) / 2

    start = q.min(1).values[:, None] - resolutions * 1.7
    end = q.max(1).values[:, None] + resolutions * 1.7
    
    # Calculate step size based on resolution
    step = (1.7 * resolutions / gaussgpoint).min()
    
    num_pts_needed = ((end.max() - start.min()) / step).ceil().int().item()
    num_pts_needed = max(num_pts_needed, q.shape[1] + 10)

    # We use a shared grid size but per-batch start/end?
    # Original code used _batch_linspace_with_padding to handle different ranges
    # Let's simplify to simple linear interp if possible, but keeping original logic structure
    
    q_lin = _batch_linspace(start, end, num_pts_needed)
    q_lin = torch.clamp(q_lin, min=1e-6)

    return q_lin


def _batch_linear_interp1d(x: Tensor, y: Tensor, x_new: Tensor) -> Tensor:
    """
    Differentiable linear interpolation.
    """
    # x, y: (batch, len)
    # x_new: (batch, len_new)
    
    eps = torch.finfo(y.dtype).eps
    
    # Indices for interpolation
    # searchsorted is not differentiable w.r.t x or x_new values directly (returns int indices)
    # But we use the indices to gather values, and then interpolate.
    
    ind = torch.searchsorted(x.contiguous(), x_new.contiguous())
    ind = torch.clamp(ind - 1, 0, x.shape[-1] - 2)
    
    # Gather pairs
    # Helper to gather along dim 1
    def gather_values(src, idx):
        return torch.gather(src, 1, idx)
        
    x_lo = gather_values(x, ind)
    x_hi = gather_values(x, ind + 1)
    y_lo = gather_values(y, ind)
    y_hi = gather_values(y, ind + 1)
    
    # Slopes
    slope = (y_hi - y_lo) / (x_hi - x_lo + eps)
    
    y_new = y_lo + slope * (x_new - x_lo)
    
    return y_new
