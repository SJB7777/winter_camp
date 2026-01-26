import numpy as np
from src.physics_utils import tth2q

def apply_anchor_normalization(q_vals, y_raw, wavelength, anchor_tth=0.2):
    """
    데이터의 스케일을 특정 2theta 각도(anchor)에서 1.0이 되도록 강제 조정합니다.
    """
    # 2Theta -> Q for anchor
    q_anchor = tth2q(anchor_tth, wavelength)
    
    # Find closest index
    idx = np.argmin(np.abs(q_vals - q_anchor))
    
    # Scale Factor Calculation
    val_at_anchor = y_raw[idx]
    
    # 데이터가 0이거나 음수면 정규화 포기 (Raw 반환)
    if val_at_anchor <= 0: 
        return y_raw, 1.0 
    
    scale_factor = 1.0 / val_at_anchor
    y_norm = y_raw * scale_factor
    
    return y_norm, scale_factor

def normalize(y_vals):
    """
    데이터의 최대값이 1.0이 되도록 정규화합니다.
    """
    max_val = np.max(y_vals)
    if max_val == 0:
        return y_vals
    return y_vals / max_val