"""
Abeles Matrix Method (Final Optimization: Scalar to Tensor)
===========================================================
"""
from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

def abeles_core(q: Tensor, thickness: Tensor, roughness: Tensor, sld: Tensor) -> Tensor:
    # Pure PyTorch Eager Execution
    # ... (기존 로직 동일) ...
    k_n = torch.sqrt((q / 2.0)**2 - 4.0 * math.pi * sld)
    k_curr, k_next = k_n[..., :-1], k_n[..., 1:] 

    beta = 1j * thickness * k_curr
    exp_beta = torch.exp(beta)
    exp_m_beta = torch.exp(-beta)

    sig2 = roughness ** 2
    denom = k_curr + k_next
    epsilon = 1e-9
    
    # Pre-computation to avoid redundant divisions
    r_n = ((k_curr - k_next) / (denom + epsilon)) * torch.exp(-2.0 * k_curr * k_next * sig2)

    m_00, m_01 = exp_beta, r_n * exp_beta
    m_10, m_11 = r_n * exp_m_beta, exp_m_beta

    m_accum_00, m_accum_01 = m_00[..., 0], m_01[..., 0]
    m_accum_10, m_accum_11 = m_10[..., 0], m_11[..., 0]

    num_layers = thickness.shape[2] 
    for i in range(1, num_layers):
        a, b, c, d = m_accum_00, m_accum_01, m_accum_10, m_accum_11
        e, f, g, h = m_00[..., i], m_01[..., i], m_10[..., i], m_11[..., i]
        
        m_accum_00 = a * e + b * g
        m_accum_01 = a * f + b * h
        m_accum_10 = c * e + d * g
        m_accum_11 = c * f + d * h

    quotient = m_accum_10 / (m_accum_00 + epsilon)
    return quotient.real**2 + quotient.imag**2

class AbelesMatrix(nn.Module):
    def __init__(self, device: torch.device | None = None):
        super().__init__()
        self.device = device
        # [Optimization] 파장은 상수이므로 미리 텐서로 만들어둠 (하지만 config 의존적이므로 forward에서 처리)

    def forward(self, 
                q: Tensor, 
                thickness: Tensor, 
                roughness: Tensor, 
                sld: Tensor,
                sample_length: Optional[Tensor] = None,
                beam_width: Optional[Tensor] = None,
                wavelength: float = 1.5406) -> Tensor:
        
        if self.device is None: self.device = q.device
        
        # 1. Prepare Inputs
        q_f = q.float()
        
        # [FIX] wavelength를 스칼라가 아닌 0-d Tensor로 변환하여 Device에 올림
        # 이렇게 하면 아래 연산들이 모두 GPU 내부에서 처리됨 (CPU 개입 차단)
        wl_tensor = torch.tensor(wavelength, device=self.device, dtype=torch.float32)
        
        # SLD & Complex handling
        if torch.is_complex(sld):
            sld_c = sld.to(dtype=torch.complex64)
        else:
            sld_c = sld.float().to(dtype=torch.complex64)
            
        amb_sld = sld_c[:, 0:1]
        sld_comp = (sld_c - amb_sld) * 1e-6 + 1j * 1e-9
        
        # Expand dims
        q_in = q_f.unsqueeze(-1).to(dtype=torch.complex64)
        sld_in = sld_comp.unsqueeze(1)
        t_in = thickness.float().unsqueeze(1)
        r_in = roughness.float().unsqueeze(1)

        # 2. Core Calculation
        reflectivity = abeles_core(q_in, t_in, r_in, sld_in)
        
        # 3. Footprint Correction (GPU Optimized)
        if sample_length is not None and beam_width is not None:
            # 입력값들도 확실하게 GPU Float Tensor로 변환
            sl = sample_length.float()
            bw = beam_width.float()
            
            # [Optimization] wavelength 텐서 사용
            sin_alpha = torch.clamp(q_f * wl_tensor / (4.0 * math.pi), -1.0, 1.0)
            
            sigma_b = bw / 2.3548
            # Epsilon 추가
            arg = (sl * sin_alpha) / (2.8284 * sigma_b + 1e-9)
            reflectivity = reflectivity * torch.erf(arg)

        return reflectivity
    
def abeles(*args, **kwargs):
    model = AbelesMatrix(device=args[0].device if args else None)
    return model(*args, **kwargs)