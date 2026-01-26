import torch
import numpy as np
from src.config import CONFIG

def simulate_reflectivity(param_tensors, q_tensor, physics_engine, device):
    """
    Refine Optimizer용 시뮬레이션 함수.
    [핵심 수정] 0.2도(Anchor)에서 시뮬레이션 값도 강제로 1.0으로 맞춤 (Scale Locking)
    """
    layer_names = [l.name for l in CONFIG.sample.layers] 

    # 1. Thickness Assembly
    t_list = [torch.zeros(1, device=device)] # Ambient
    for name in layer_names:
        key = f"{name}.thickness"
        val = param_tensors.get(key, torch.tensor([10.0], device=device))
        t_list.append(val.view(-1))
    thickness = torch.stack(t_list, dim=1)

    # 2. Roughness Assembly
    r_list = []
    for name in layer_names:
        key = f"{name}.roughness"
        val = param_tensors.get(key, torch.tensor([0.0], device=device))
        r_list.append(val.view(-1))
    r_sub = param_tensors.get("Substrate.roughness", torch.tensor([0.0], device=device))
    r_list.append(r_sub.view(-1))
    roughness = torch.stack(r_list, dim=1)

    # 3. SLD Assembly (Complex)
    sld_list = [torch.zeros(1, device=device, dtype=torch.complex64)] # Ambient
    for name in layer_names:
        real = param_tensors.get(f"{name}.sld", torch.tensor([0.0], device=device))
        imag = param_tensors.get(f"{name}.sld_imag", torch.tensor([0.0], device=device))
        c_val = torch.complex(real.double(), imag.double())
        sld_list.append(c_val.view(-1))
        
    sub_real = param_tensors.get("Substrate.sld", torch.tensor([20.0], device=device))
    sub_imag = param_tensors.get("Substrate.sld_imag", torch.tensor([0.0], device=device))
    c_sub = torch.complex(sub_real.double(), sub_imag.double())
    sld_list.append(c_sub.view(-1))
    sld = torch.stack(sld_list, dim=1)

    # 4. Pure Physics Simulation
    q_in = q_tensor.view(-1)
    beam_w = param_tensors.get("beam_width", torch.tensor([CONFIG.instrument.beam_width], device=device))
    sample_L = param_tensors.get("L", torch.tensor([10.0], device=device))
    
    r_sim_pure = physics_engine(
        q=q_in, thickness=thickness, roughness=roughness, sld=sld,
        sample_length=sample_L, beam_width=beam_w,
        wavelength=CONFIG.instrument.wavelength
    )
    
    # --------------------------------------------------------------------------
    # [CORE FIX] Anchor Normalization on Simulation
    # 시뮬레이션 결과도 0.2도(Anchor) 지점에서 1.0이 되도록 자동 스케일링
    # 이렇게 하면 i0 파라미터는 사실상 무시되거나 미세 조정용으로만 쓰임
    # --------------------------------------------------------------------------
    
    # 1. Find Anchor Index (0.2 deg)
    # 매번 계산하면 느리니 q_tensor가 고정이라면 캐싱할 수도 있지만, 여기선 안전하게 계산
    wavelength = CONFIG.instrument.wavelength
    anchor_tth = 0.2 # Config에서 가져와도 됨
    q_anchor = 4 * np.pi * np.sin(np.radians(anchor_tth / 2)) / wavelength
    
    # q_in에서 가장 가까운 인덱스 찾기
    idx_anchor = torch.argmin(torch.abs(q_in - q_anchor))
    
    # 2. Get Simulation Value at Anchor
    val_at_anchor = r_sim_pure[..., idx_anchor]
    
    # 3. Calculate Correction Factor (Target 1.0 / Current)
    # 0으로 나누기 방지
    correction_factor = 1.0 / (val_at_anchor + 1e-12)
    
    # 4. Apply Correction (스케일 맞춤)
    r_sim_norm = r_sim_pure * correction_factor
    
    # 5. Apply Background (bkg는 정규화된 스케일 위에서의 백그라운드)
    bkg = param_tensors.get("bkg", torch.tensor([-7.0], device=device))
    
    # i0는 이제 1.0 근처의 미세 조정 값(Fine-tuning)으로 사용됨
    i0_fine = param_tensors.get("i0", torch.tensor([1.0], device=device))
    
    return (r_sim_norm * i0_fine) + torch.pow(10.0, bkg)