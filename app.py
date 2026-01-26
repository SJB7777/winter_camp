from flask import Flask, request, jsonify
import matplotlib
matplotlib.use('Agg')
import torch
import numpy as np
import pandas as pd
import json
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
# [NEW] 표준 로스 함수 import
from src.losses import compute_standard_loss

app = Flask(__name__)

# ... (Global Setup 등은 그대로 유지) ...
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


@app.route('/predict_initial', methods=['POST'])
def api_predict_initial():
    try:
        if 'file' not in request.files: return jsonify({"error": "No file"}), 400
        file = request.files['file']
        
        # 1. Load & Norm
        df = pd.read_csv(file, sep=r'\s+', comment='#', header=None, dtype=str)
        df = df.replace({r'[()]': ''}, regex=True).astype(float).dropna()
        x_raw, y_raw = df.iloc[:, 0].values, df.iloc[:, 1].values
        
        q_vals = tth2q(x_raw, wavelen=CONFIG.instrument.wavelength)
        y_norm, scale_factor = apply_anchor_normalization(q_vals, y_raw, CONFIG.instrument.wavelength)
        
        q_tensor = torch.from_numpy(q_vals).float().to(DEVICE)
        log_r_obs = torch.log10(torch.clamp(torch.from_numpy(y_norm).float().to(DEVICE), min=1e-12))
        
        # 2. AI Guess
        initial_params = {}
        if AI_MODEL:
            try: initial_params = predict_initial_params(AI_MODEL, q_tensor, log_r_obs, DEVICE)
            except: pass
        
        if not initial_params:
            initial_params = {"i0": 1.0, "bkg": -6.0, "Target_Film.thickness": 300.0}

        initial_params['i0'] = 1.0 # Anchor Lock

        # 3. Simulate & Standard Loss Check
        with torch.no_grad():
            p_tensors = {k: torch.tensor([v], device=DEVICE) for k, v in initial_params.items()}
            r_sim_tensor = simulator_wrapper(p_tensors, q_tensor)
            
            # [Standardized Metric] 여기서 표준 함수 사용!
            initial_loss = compute_standard_loss(r_sim_tensor, log_r_obs)
            
            r_sim_np = r_sim_tensor.cpu().numpy().flatten()
            plot_base64 = plot_fit_result(q_vals, y_norm, r_sim_np, title=f"AI Guess | Loss: {initial_loss:.4f}")

        return jsonify({
            "status": "success",
            "initial_params": initial_params,
            "initial_loss": initial_loss, # 표준화된 점수 리턴
            "valid_keys": list(initial_params.keys()),
            "scale_factor": float(scale_factor),
            "plot_base64": plot_base64
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/refine_gradient', methods=['POST'])
def api_refine():
    try:
        file = request.files['file']
        current_params = json.loads(request.form.get('current_params', '{}'))
        optimize_spec = json.loads(request.form.get('optimize_spec', '{}'))
        
        # 1. Load & Norm
        df = pd.read_csv(file, sep=r'\s+', comment='#', header=None, dtype=str)
        df = df.replace({r'[()]': ''}, regex=True).astype(float).dropna()
        x_raw, y_raw = df.iloc[:, 0].values, df.iloc[:, 1].values
        
        q_vals = tth2q(x_raw, wavelen=CONFIG.instrument.wavelength)
        y_norm, _ = apply_anchor_normalization(q_vals, y_raw, CONFIG.instrument.wavelength)
        
        q_tensor = torch.from_numpy(q_vals).float().to(DEVICE)
        log_r_obs = torch.log10(torch.clamp(torch.from_numpy(y_norm).float().to(DEVICE), min=1e-12))
        data_payload = {'q': q_tensor, 'log_r_obs': log_r_obs}
        
        # Guard
        if 'target_params' in optimize_spec and 'i0' in optimize_spec['target_params']:
            optimize_spec['target_params']['i0'] = [0.8, 1.2]

        # 2. Refine
        # 여기서 반환되는 opt_loss는 Optimizer가 쓴 로스(예: Correlation)임
        refined_params, opt_loss = refine_with_gradient(
            current_params, data_payload, optimize_spec, simulator_wrapper, DEVICE
        )
        
        # 3. Final Standard Check
        # Optimizer가 무슨 로스를 썼든, 보고는 Standard LogMSE로 통일
        with torch.no_grad():
            p_ref = {k: torch.tensor([v], device=DEVICE) for k, v in refined_params.items()}
            r_final_tensor = simulator_wrapper(p_ref, q_tensor)
            
            # [Standardized Metric] 비교를 위한 표준 점수 계산
            standard_loss = compute_standard_loss(r_final_tensor, log_r_obs)
            
            r_final_np = r_final_tensor.cpu().numpy().flatten()
            loss_name = optimize_spec.get('loss_type', 'LogMSE')
            
            # 그래프 제목에는 두 로스 모두 표시 (참고용)
            title = f"Refined ({loss_name}) | StdLoss: {standard_loss:.4f}"
            plot_base64 = plot_fit_result(q_vals, y_norm, r_final_np, title=title)
            
        return jsonify({
            "status": "success",
            "updated_params": refined_params,
            "final_loss": standard_loss, # [중요] 표준화된 점수로 덮어씀 (일관성 유지)
            "opt_loss": opt_loss,       # 참고용: 실제로 최적화한 로스 값
            "plot_base64": plot_base64
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)