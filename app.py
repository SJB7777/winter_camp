from flask import Flask, request, jsonify
import matplotlib
matplotlib.use('Agg')
import torch
import numpy as np
import pandas as pd
import os

# --- Modules ---
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
# 0. Global Setup
# ==============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PHYSICS_ENGINE = AbelesMatrix(device=DEVICE)

config_path = "config.yaml"
if os.path.exists(config_path):
    real_config = XRefineConfig.load_yaml(config_path)
    CONFIG.sample = real_config.sample
    CONFIG.instrument = real_config.instrument

ckpt_path = os.environ.get("MODEL_PATH", "checkpoints/model.pt")
AI_MODEL = None
if os.path.exists(ckpt_path):
    AI_MODEL, _ = load_model_from_checkpoint(ckpt_path, DEVICE)
    print("✅ AI Model Ready.")

# Wrapper
def simulator_wrapper(param_tensors, q_tensor):
    return simulate_reflectivity(param_tensors, q_tensor, PHYSICS_ENGINE, DEVICE)

# ==============================================================================
# 1. Helper: JSON Arrays -> PyTorch Tensors
# ==============================================================================
def json_to_tensors(json_data):
    """
    클라이언트가 보낸 JSON 리스트(q, y_norm)를 받아서
    계산 가능한 Tensor로 변환
    """
    if 'q' not in json_data or 'y_norm' not in json_data:
        raise ValueError("Missing 'q' or 'y_norm' list in JSON body.")
    
    # 1. List -> Numpy
    q_vals = np.array(json_data['q'], dtype=float)
    y_norm = np.array(json_data['y_norm'], dtype=float)
    
    # 2. Numpy -> Tensor (GPU/CPU)
    q_tensor = torch.from_numpy(q_vals).float().to(DEVICE)
    # 로그 변환 및 안전장치
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
    [Step 0] 파일 업로드 -> 데이터 정규화 -> JSON 배열 반환
    (이후 단계에서는 이 JSON 데이터만 핑퐁함)
    """
    try:
        if 'file' not in request.files: return jsonify({"error": "No file"}), 400
        file = request.files['file']
        
        # 1. Parsing
        df = pd.read_csv(file, sep=r'\s+', comment='#', header=None, dtype=str)
        df = df.replace({r'[()]': ''}, regex=True).astype(float).dropna()
        x_raw, y_raw = df.iloc[:, 0].values, df.iloc[:, 1].values
        
        # 2. Physics & Normalization
        q_vals = tth2q(x_raw, wavelen=CONFIG.instrument.wavelength)
        y_norm, scale_factor = apply_anchor_normalization(
            q_vals, y_raw, CONFIG.instrument.wavelength
        )
        
        # 3. Return Pure Data (No Image yet)
        return jsonify({
            "status": "success",
            "scale_factor": float(scale_factor),
            "data_package": {
                "q": q_vals.tolist(),      # 리스트로 변환
                "y_norm": y_norm.tolist()  # 리스트로 변환
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/predict_initial', methods=['POST'])
def api_predict_initial():
    """
    [Step 1] JSON 데이터 수신 -> AI 추론 -> 결과 & 그래프 반환
    (파일 업로드 X, JSON Body O)
    """
    try:
        # Request Body에서 데이터 패키지 추출
        req_json = request.json
        data_package = req_json.get('data_package')
        
        # Tensor 변환
        tensors = json_to_tensors(data_package)
        
        # AI Prediction
        initial_params = {}
        if AI_MODEL:
            try:
                initial_params = predict_initial_params(
                    AI_MODEL, tensors['q_tensor'], tensors['log_r_obs'], DEVICE
                )
            except: pass
        
        if not initial_params:
            initial_params = {"i0": 1.0, "bkg": -6.0, "Target_Film.thickness": 300.0}

        initial_params['i0'] = 1.0 # Anchor Lock

        # Simulate
        with torch.no_grad():
            p_tensors = {k: torch.tensor([v], device=DEVICE) for k, v in initial_params.items()}
            r_sim = simulator_wrapper(p_tensors, tensors['q_tensor'])
            initial_loss = compute_standard_loss(r_sim, tensors['log_r_obs'])
            
            plot_base64 = plot_fit_result(
                tensors['q_vals'], tensors['y_norm'], r_sim.cpu().numpy().flatten(), 
                title=f"AI Guess | Loss: {initial_loss:.4f}"
            )

        return jsonify({
            "status": "success",
            "initial_params": initial_params,
            "initial_loss": initial_loss,
            "plot_base64": plot_base64
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/refine_gradient', methods=['POST'])
def api_refine():
    """
    [Step 2] JSON 데이터 + 파라미터 수신 -> 최적화 -> 결과 반환
    (파일 업로드 X, JSON Body O)
    """
    try:
        # JSON Body 파싱
        req_json = request.json
        
        current_params = req_json.get('current_params', {})
        optimize_spec = req_json.get('optimize_spec', {})
        data_package = req_json.get('data_package') # [중요] 데이터도 JSON으로 받음
        
        # Tensor 변환
        tensors = json_to_tensors(data_package)
        data_payload = {'q': tensors['q_tensor'], 'log_r_obs': tensors['log_r_obs']}
        
        if 'target_params' in optimize_spec and 'i0' in optimize_spec['target_params']:
            optimize_spec['target_params']['i0'] = [0.8, 1.2]

        # Refine
        refined_params, opt_loss = refine_with_gradient(
            current_params, data_payload, optimize_spec, simulator_wrapper, DEVICE
        )
        
        # Plot
        with torch.no_grad():
            p_ref = {k: torch.tensor([v], device=DEVICE) for k, v in refined_params.items()}
            r_final = simulator_wrapper(p_ref, tensors['q_tensor'])
            standard_loss = compute_standard_loss(r_final, tensors['log_r_obs'])
            
            loss_name = optimize_spec.get('loss_type', 'LogMSE')
            plot_base64 = plot_fit_result(
                tensors['q_vals'], tensors['y_norm'], r_final.cpu().numpy().flatten(), 
                title=f"Refined ({loss_name}) | StdLoss: {standard_loss:.4f}"
            )
            
        return jsonify({
            "status": "success",
            "updated_params": refined_params,
            "final_loss": standard_loss,
            "plot_base64": plot_base64
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)