from flask import Flask, request, jsonify
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import numpy as np
import pandas as pd
import io
import base64
import json

from src.constants import LOG_VALID_THRESHOLD
from src.refine import refine_with_gradient
from src.abeles import AbelesMatrix
from src.physics_utils import tth2q
from src.config import CONFIG
from src.nn_glue import load_model_from_checkpoint, predict_initial_params


CHECKPOINT_PATH = "checkpoints/model.pt"
AI_MODEL = None
AI_CONFIG = None


AI_MODEL, AI_CONFIG = load_model_from_checkpoint(CHECKPOINT_PATH, torch.device('cpu'))
print("AI Model Loaded Successfully!")


app = Flask(__name__)

# ==============================================================================
# 1. Simulator Bridge (The most important part)
# ==============================================================================
# 전역 시뮬레이터 인스턴스 (비용 절감)
# device는 요청 들어올 때마다 확인하지만, 초기화는 CPU로 해둠 (Railway 환경)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PHYSICS_ENGINE = AbelesMatrix(device=DEVICE)

def generate_plot(q_vals, y_obs, y_sim_dict, title_prefix="Result"):
    """
    Args:
        q_vals (np.array): Q축 데이터
        y_obs (np.array): 실험 데이터 (Linear Scale)
        y_sim_dict (dict): {'Label': y_sim_data} 형태의 시뮬레이션 결과들
    """
    plt.figure(figsize=(7, 5), dpi=100)
    
    # 1. Masking (Valid Range Only)
    # 실험 데이터의 Log값이 Threshold보다 큰 구간만 살림
    # y_obs가 0 이하일 수 있으므로 안전하게 log 변환
    log_y_obs = np.log10(np.clip(y_obs, 1e-15, None))
    mask = log_y_obs > LOG_VALID_THRESHOLD
    
    # 마스크된 데이터 (Plotting용)
    q_plot = q_vals[mask]
    y_obs_plot = y_obs[mask]
    
    # 2. Plot Real Data
    plt.semilogy(q_plot, y_obs_plot, 'ko', markersize=3, alpha=0.4, label='Experiment')
    
    loss_texts = []
    
    # 3. Plot Simulations & Calc Loss
    colors = ['b', 'r', 'g']
    for idx, (label, y_sim) in enumerate(y_sim_dict.items()):
        # 마스크 적용
        y_sim_plot = y_sim[mask]
        
        # Plot
        style = '--' if 'Initial' in label else '-'
        color = colors[idx % len(colors)]
        plt.semilogy(q_plot, y_sim_plot, color=color, linestyle=style, linewidth=1.5, label=label)
        
        # Calc Metrics (Log MSE)
        # 1e-15 클리핑으로 -inf 방지
        log_sim = np.log10(np.clip(y_sim_plot, 1e-15, None))
        log_obs = np.log10(np.clip(y_obs_plot, 1e-15, None))
        mse = np.mean((log_sim - log_obs)**2)
        loss_texts.append(f"{label} Loss: {mse:.4f}")

    # 4. Decoration
    plt.xlabel("Q [$\AA^{-1}$]")
    plt.ylabel("Reflectivity (Log)")
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    
    # Title with Loss info
    loss_str = " | ".join(loss_texts)
    plt.title(f"{title_prefix}\n[{loss_str}]", fontsize=10)
    plt.tight_layout()
    
    # 5. Export
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close('all')
    return img_str


def simulator_wrapper(param_tensors: dict[str, torch.Tensor], q_tensor: torch.Tensor) -> torch.Tensor:
    """
    refine.py의 Optimizer가 넘겨주는 '평평한 파라미터 딕셔너리'를
    AbelesMatrix가 이해하는 '구조적 텐서(thickness, roughness, sld)'로 변환하여 시뮬레이션 수행.
    
    [핵심] 이곳의 모든 연산은 PyTorch Tensor 연산이어야 하며, Gradient가 끊기면 안 됨.
    """
    
    # 1. 딕셔너리에서 값 추출 (없으면 기본값 0.0 처리)
    # 키 형식: "{LayerName}.{Property}" 예: "Target_Film.thickness"
    
    # 레이어 순서 정의 (config.yaml이나 기본 CONFIG 순서 따름)
    # 실제로는 동적으로 파싱하거나, 요청에서 layer_names를 받아야 더 정확하지만,
    # 여기서는 CONFIG의 기본 레이어 순서를 따른다고 가정합니다.
    layer_names = [l.name for l in CONFIG.sample.layers] # ['Top_Layer', 'Target_Film']
    
    # 배치 사이즈 (보통 1)
    batch_size = q_tensor.shape[0] if q_tensor.dim() > 1 else 1

    # --------------------------------------------------------------------------
    # A. Thickness Assembly (Ambient + Layers)
    # --------------------------------------------------------------------------
    # Ambient(Air) Thickness = 0
    t_list = [torch.zeros(1, device=DEVICE)] 
    
    for name in layer_names:
        key = f"{name}.thickness"
        if key in param_tensors:
            t_list.append(param_tensors[key].view(-1))
        else:
            # 파라미터 딕셔너리에 없으면(최적화 대상 아니면), 고정된 상수값이라고 가정하거나 에러
            # 해커톤용: 안전하게 10.0 (dummy) 혹은 에러 처리
            # 여기서는 Gradient가 필요 없는 상수로 처리
            t_list.append(torch.tensor([10.0], device=DEVICE)) 
            
    thickness = torch.stack(t_list, dim=1) # (Batch, N_layers)

    # --------------------------------------------------------------------------
    # B. Roughness Assembly (Layers + Substrate)
    # --------------------------------------------------------------------------
    r_list = []
    
    # Layers
    for name in layer_names:
        key = f"{name}.roughness"
        val = param_tensors.get(key, torch.tensor([0.0], device=DEVICE))
        r_list.append(val.view(-1))
        
    # Substrate
    sub_key = "Substrate.roughness"
    r_sub = param_tensors.get(sub_key, torch.tensor([0.0], device=DEVICE))
    r_list.append(r_sub.view(-1))
    
    roughness = torch.stack(r_list, dim=1)

    # --------------------------------------------------------------------------
    # C. SLD Assembly (Ambient + Layers + Substrate) -> Complex Support
    # --------------------------------------------------------------------------
    # Ambient (0+0j)
    sld_list = [torch.zeros(1, device=DEVICE, dtype=torch.complex64)]
    
    # Layers
    for name in layer_names:
        real_key = f"{name}.sld"
        imag_key = f"{name}.sld_imag"
        
        rho = param_tensors.get(real_key, torch.tensor([0.0], device=DEVICE))
        rho_imag = param_tensors.get(imag_key, torch.tensor([0.0], device=DEVICE))
        
        c_val = rho.complex() + 1j * rho_imag.complex()
        sld_list.append(c_val.view(-1))
        
    # Substrate
    sub_real = param_tensors.get("Substrate.sld", torch.tensor([20.0], device=DEVICE))
    sub_imag = param_tensors.get("Substrate.sld_imag", torch.tensor([0.0], device=DEVICE))
    c_sub = sub_real.complex() + 1j * sub_imag.complex()
    sld_list.append(c_sub.view(-1))
    
    sld = torch.stack(sld_list, dim=1) # (Batch, N+2)

    # --------------------------------------------------------------------------
    # D. Simulate
    # --------------------------------------------------------------------------
    # Instrument params
    # 파라미터에 있으면 쓰고, 없으면 CONFIG 기본값 사용
    curr_beam_width = param_tensors.get("beam_width", torch.tensor([CONFIG.instrument.beam_width], device=DEVICE))
    curr_sample_len = param_tensors.get("L", torch.tensor([10.0], device=DEVICE)) # Default 10mm
    
    # Call Physics Engine
    # q_tensor가 1D면 (Q_len,), 2D면 (Batch, Q_len)
    if q_tensor.dim() == 1:
        q_in = q_tensor
    else:
        q_in = q_tensor.view(-1) # 단순화

    r_sim = PHYSICS_ENGINE(
        q=q_in,
        thickness=thickness,
        roughness=roughness,
        sld=sld,
        sample_length=curr_sample_len,
        beam_width=curr_beam_width,
        wavelength=CONFIG.instrument.wavelength
    )
    
    # Scale & Background (I = I0 * R + Bkg)
    i0 = param_tensors.get("i0", torch.tensor([1.0], device=DEVICE))
    bkg = param_tensors.get("bkg", torch.tensor([-7.0], device=DEVICE)) # Log10 scale
    
    r_final = (i0 * r_sim) + torch.pow(10.0, bkg)
    
    return r_final


# ==============================================================================
# 2. Flask Routes
# ==============================================================================

@app.route('/predict_initial', methods=['POST'])
def api_predict_initial():
    """
    [Step 1] AI Guess + 초기 시각화
    """
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file"}), 400
        
        file = request.files['file']
        
        # Data Load
        df = pd.read_csv(file, sep=r'\s+', comment='#', header=None, dtype=str)
        df = df.replace({r'[()]': ''}, regex=True).astype(float).dropna()
        x_raw, y_raw = df.iloc[:, 0].values, df.iloc[:, 1].values
        
        # Physics Convert
        q_vals = tth2q(x_raw, wavelen=CONFIG.instrument.wavelength)
        q_tensor = torch.from_numpy(q_vals).float().to(DEVICE)
        log_r_obs = torch.log10(torch.clamp(torch.from_numpy(y_raw).float().to(DEVICE), min=1e-12))
        
        # 1. AI Prediction
        initial_params = {}
        if AI_MODEL:
            try:
                initial_params = predict_initial_params(AI_MODEL, q_tensor, log_r_obs, DEVICE)
            except:
                pass # Fallback
                
        if not initial_params:
            # Fallback Defaults
            initial_params = {"Target_Film.thickness": 300.0, "i0": 1.0, "bkg": -6.0} 
            # (필요한 모든 키 추가)

        # 2. Simulate Initial Guess (for visualization)
        # wrapper에 넣기 위해 딕셔너리 값을 텐서로 변환
        with torch.no_grad():
            init_tensors = {k: torch.tensor([v], device=DEVICE) for k, v in initial_params.items()}
            # simulator_wrapper 구현 필요 (이전 답변 참조)
            # 여기선 에러 방지를 위해 try 감쌈
            try:
                r_sim = simulator_wrapper(init_tensors, q_tensor)
                r_sim_np = r_sim.cpu().numpy().flatten()
                
                # 3. Generate Plot (Raw vs Initial)
                plot_base64 = generate_plot(
                    q_vals, y_raw, 
                    {'AI Initial': r_sim_np}, 
                    title_prefix="Step 1: AI Initial Guess"
                )
            except:
                plot_base64 = "" # 시뮬레이터 에러 시 그림 생략

        return jsonify({
            "status": "success",
            "initial_params": initial_params,
            "valid_keys": list(initial_params.keys()),
            "plot_base64": plot_base64  # <--- n8n에서 바로 확인 가능!
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/refine_gradient', methods=['POST'])
def api_refine():
    """
    [Step 2] Refinement + 결과 비교 시각화
    """
    try:
        # Input Parsing (File, Params, Spec)
        file = request.files['file']
        current_params = json.loads(request.form.get('current_params', '{}'))
        optimize_spec = json.loads(request.form.get('optimize_spec', '{}'))
        
        # Data Load
        df = pd.read_csv(file, sep=r'\s+', comment='#', header=None, dtype=str)
        df = df.replace({r'[()]': ''}, regex=True).astype(float).dropna()
        x_raw, y_raw = df.iloc[:, 0].values, df.iloc[:, 1].values
        
        q_vals = tth2q(x_raw, wavelen=CONFIG.instrument.wavelength)
        q_tensor = torch.from_numpy(q_vals).float().to(DEVICE)
        y_tensor = torch.from_numpy(y_raw).float().to(DEVICE)
        log_r_obs = torch.log10(torch.clamp(y_tensor, min=1e-12))
        
        data_payload = {'q': q_tensor, 'log_r_obs': log_r_obs}
        
        # 1. Run Refinement
        refined_params, final_loss = refine_with_gradient(
            current_params, data_payload, optimize_spec, simulator_wrapper, DEVICE
        )
        
        # 2. Simulate for Plot (Initial vs Refined)
        with torch.no_grad():
            # Initial
            init_tensors = {k: torch.tensor([v], device=DEVICE) for k, v in current_params.items()}
            r_init = simulator_wrapper(init_tensors, q_tensor).cpu().numpy().flatten()
            
            # Refined
            ref_tensors = {k: torch.tensor([v], device=DEVICE) for k, v in refined_params.items()}
            r_final = simulator_wrapper(ref_tensors, q_tensor).cpu().numpy().flatten()
            
        # 3. Generate Comparison Plot (Masked)
        plot_base64 = generate_plot(
            q_vals, y_raw,
            {
                'Initial': r_init,
                'Refined': r_final
            },
            title_prefix=f"Step 2: Refinement ({optimize_spec.get('loss_type', 'LogMSE')})"
        )
        
        return jsonify({
            "status": "success",
            "updated_params": refined_params,
            "final_loss": final_loss,
            "plot_base64": plot_base64
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)