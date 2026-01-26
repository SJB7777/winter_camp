import matplotlib.pyplot as plt
import numpy as np
import io
import base64

def plot_fit_result(q_raw, y_raw, r_sim, title="Fit Result", save_path=None, show_mask_limit=-14.0):
    """
    Matplotlib 그래프를 생성하여 저장하거나 객체를 반환합니다.
    """
    plt.figure(figsize=(10, 6))
    
    # Log Scale Masking
    log_y = np.log10(np.clip(y_raw, 1e-15, None))
    valid_mask = log_y > show_mask_limit
    
    plt.semilogy(q_raw[valid_mask], y_raw[valid_mask], 'ko', alpha=0.3, label='Experiment')
    plt.semilogy(q_raw, r_sim, 'r-', linewidth=2, label='Simulation')
    
    plt.title(title)
    plt.xlabel("Q [$\\AA^{-1}$]")
    plt.ylabel("Reflectivity (Log)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Mode 1: Save to file (Debug)
    if save_path:
        plt.savefig(save_path)
        plt.close()
        return save_path
    
    # Mode 2: Return Base64 (Server)
    else:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        return img_str