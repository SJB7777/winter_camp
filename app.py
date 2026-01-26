from flask import Flask, request, jsonify
import matplotlib
matplotlib.use('Agg') # [중요] GUI 없는 환경에서 Plotting 충돌 방지
import torch
import numpy as np
import pandas as pd
import json
import os
import traceback

# --- Modules ---
# (src 폴더가 있는 경로에서 실행해야 합니다)
from src.refine import refine_with_gradient
from src.abeles import AbelesMatrix
from src.physics_utils import tth2q
from src.config import CONFIG, XRefineConfig
from src.nn_glue import load_model_from_checkpoint, predict_initial_params
from src.data_processing import apply_anchor_normalization
from src.simulation import simulate_reflectivity
from src.visualization import plot_fit_result
from src.losses import compute_standard_loss

app = Flask(__name__)

# ==============================================================================
# 0. Global Setup & Config
# ==============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PHYSICS_ENGINE = AbelesMatrix(device=DEVICE)

# Config Loading
config_path = "config.yaml"
if os.path.exists(config_path):
    real_config = XRefineConfig.load_yaml(config_path)
    CONFIG.sample = real_config.sample
    CONFIG.instrument = real_config.instrument

# AI Model Loading
ckpt_path = os.environ.get("MODEL_PATH", "checkpoints/model.pt")
AI_MODEL = None
if os.path.exists(ckpt_path):
    AI_MODEL, _ = load_model_from_checkpoint(ckpt_path, DEVICE)
    print(f"✅ AI Model Loaded from {ckpt_path}")
else:
    print("⚠️ AI Model not found. Using Fallback heuristics.")

# Simulator Wrapper
def simulator_wrapper(param_tensors, q_tensor):
    return simulate_reflectivity(param_tensors, q_tensor, PHYSICS_ENGINE, DEVICE)

# ==============================================================================
# 1. Helper Functions (Serialization & Type Safety)
# ==============================================================================
def safe_serialize(obj):
    """
    JSON으로 변환 불가능한 타입(Bytes, Numpy, Tensor 등)을 안전하게 변환
    - Bytes (이미지) -> String (UTF-8)
    - Numpy/Tensor -> List
    """
    if isinstance(obj, bytes):
        return obj.decode('utf-8')  # [핵심 Fix] base64 bytes를 string으로 변환
    if isinstance(obj, (np.ndarray, np.generic)):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.cpu().detach().numpy().tolist()
    return obj

def sanitize_params(params):
    """
    파라미터 딕셔너리 내부의 Numpy/Tensor 값을 Python native float으로 변환
    (JSON 직렬화 에러 방지용)
    """
    clean_params = {}
    for k, v in params.items():
        if isinstance(v, (torch.Tensor, np.generic)):
            clean_params[k] = float(v)
        elif isinstance(v, list): # 혹시 리스트로 감싸진 경우
            clean_params[k] = float(v[0]) if len(v) > 0 else 0.0
        else:
            clean_params[k] = v
    return clean_params

def json_to_tensors(json_data):
    """
    Client(n8n)에서 받은 JSON 데이터 패키지를 PyTorch Tensor로 복원
    """
    if not json_data or 'q' not in json_data or 'y_norm' not in json_data:
        raise ValueError("Invalid Data Package: Missing 'q' or 'y_norm'.")
    
    q_vals = np.array(json_data['q'], dtype=float)
    y_norm = np.array(json_data['y_norm'], dtype=float)
    
    # Tensor 변환 (GPU/CPU)
    q_tensor = torch.from_numpy(q_vals).float().to(DEVICE)
    # 로그 변환 및 0 방지 (Clamp)
    log_r_obs = torch.log10(torch.clamp(torch.from_numpy(y_norm).float().to(DEVICE), min=1e-12))
    
    return {
        'q_tensor': q_tensor,
        'log_r_obs': log_r_obs,
        'q_vals': q_vals,
        'y_norm': y_norm
    }

# ==============================================================================
# 2. API Routes
# ==============================================================================

@app.route('/load_data', methods=['POST'])
def api_load_data():
    """
    [Step 0] 파일 업로드 -> 데이터 정제 -> JSON 패키지 반환
    - 이 단계에서만 'multipart/form-data' 사용
    """
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part in request"}), 400
        
        file = request.files['file']
        
        # 1. Read File
        df = pd.read_csv(file, sep=r'\s+', comment='#', header=None, dtype=str)
        df = df.replace({r'[()]': ''}, regex=True).astype(float).dropna()
        x_raw, y_raw = df.iloc[:, 0].values, df.iloc[:, 1].values
        
        # 2. Physics Logic (2theta -> q)
        q_vals = tth2q(x_raw, wavelen=CONFIG.instrument.wavelength)
        
        # 3. Normalize (Anchor at 0.2 deg)
        y_norm, scale_factor = apply_anchor_normalization(
            q_vals, y_raw, CONFIG.instrument.wavelength
        )
        
        # 4. Return Pure Data Package (No Image)
        return jsonify({
            "status": "success",
            "scale_factor": float(scale_factor),
            "data_package": {
                "q": q_vals.tolist(),
                "y_norm": y_norm.tolist()
            }
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/predict_initial', methods=['POST'])
def api_predict_initial():
    """
    [Step 1] JSON 데이터 -> AI 추론 -> 결과 & 그래프 반환
    - JSON Body 사용
    """
    try:
        # 1. Parse Input
        req_json = request.json
        data_package = req_json.get('data_package')
        
        # 2. Convert to Tensors
        tensors = json_to_tensors(data_package)
        
        # 3. AI Prediction
        initial_params = {}
        if AI_MODEL:
            try:
                initial_params = predict_initial_params(
                    AI_MODEL, tensors['q_tensor'], tensors['log_r_obs'], DEVICE
                )
            except Exception as e:
                print(f"AI Error: {e}")
        
        # Fallback if AI fails or not loaded
        if not initial_params:
            initial_params = {"i0": 1.0, "bkg": -6.0, "Target_Film.thickness": 300.0}

        initial_params['i0'] = 1.0 # Anchor Lock
        
        # 4. Simulation & Plotting
        with torch.no_grad():
            # Clean params for simulation
            clean_init_params = sanitize_params(initial_params)
            p_tensors = {k: torch.tensor([v], device=DEVICE) for k, v in clean_init_params.items()}
            
            r_sim = simulator_wrapper(p_tensors, tensors['q_tensor'])
            initial_loss = compute_standard_loss(r_sim, tensors['log_r_obs'])
            
            # Create Plot (Returns bytes)
            plot_base64_bytes = plot_fit_result(
                tensors['q_vals'], tensors['y_norm'], r_sim.cpu().numpy().flatten(), 
                title=f"AI Guess | Loss: {initial_loss:.4f}"
            )
            # Safe Convert (bytes -> string)
            plot_base64_str = safe_serialize(plot_base64_bytes)

        return jsonify({
            "status": "success",
            "initial_params": clean_init_params,
            "initial_loss": float(initial_loss),
            "plot_base64": plot_base64_str # String으로 전송
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/refine_gradient', methods=['POST'])
def api_refine():
    """
    [Step 2] JSON 데이터 + 파라미터 + 전략 -> 최적화 -> 결과 반환
    - JSON Body 사용
    """
    try:
        # 1. Parse Input
        req_json = request.json
        current_params = req_json.get('current_params', {})
        optimize_spec = req_json.get('optimize_spec', {})
        data_package = req_json.get('data_package')
        
        # 2. Convert to Tensors
        tensors = json_to_tensors(data_package)
        data_payload = {'q': tensors['q_tensor'], 'log_r_obs': tensors['log_r_obs']}
        
        # Guard: i0 range
        if 'target_params' in optimize_spec and 'i0' in optimize_spec['target_params']:
            optimize_spec['target_params']['i0'] = [0.8, 1.2]

        # 3. Refine (Gradient Descent)
        refined_params, opt_loss = refine_with_gradient(
            current_params, data_payload, optimize_spec, simulator_wrapper, DEVICE
        )
        
        # 4. Final Simulation & Plotting
        with torch.no_grad():
            clean_refined_params = sanitize_params(refined_params)
            p_ref = {k: torch.tensor([v], device=DEVICE) for k, v in clean_refined_params.items()}
            
            r_final = simulator_wrapper(p_ref, tensors['q_tensor'])
            standard_loss = compute_standard_loss(r_final, tensors['log_r_obs'])
            
            loss_name = optimize_spec.get('loss_type', 'LogMSE')
            
            plot_base64_bytes = plot_fit_result(
                tensors['q_vals'], tensors['y_norm'], r_final.cpu().numpy().flatten(), 
                title=f"Refined ({loss_name}) | StdLoss: {standard_loss:.4f}"
            )
            plot_base64_str = safe_serialize(plot_base64_bytes)

        return jsonify({
            "status": "success",
            "updated_params": clean_refined_params,
            "final_loss": float(standard_loss),
            "opt_loss": float(opt_loss),
            "plot_base64": plot_base64_str
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # n8n 및 외부 접속 허용 (0.0.0.0)
    app.run(host='0.0.0.0', port=5000, debug=True)