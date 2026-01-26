from flask import Flask, request, jsonify
import matplotlib
matplotlib.use('Agg') # Server-side rendering
import torch
import numpy as np
import pandas as pd
import json
import os

# --- Core Modules (Shared with Debug Pipeline) ---
from src.refine import refine_with_gradient
from src.abeles import AbelesMatrix
from src.physics_utils import tth2q
from src.config import CONFIG, XRefineConfig
from src.nn_glue import load_model_from_checkpoint, predict_initial_params
from src.data_processing import apply_anchor_normalization
from src.simulation import simulate_reflectivity
from src.visualization import plot_fit_result

app = Flask(__name__)

# ==============================================================================
# 0. Global Setup
# ==============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PHYSICS_ENGINE = AbelesMatrix(device=DEVICE)

# Config Load
config_path = "config.yaml"
if os.path.exists(config_path):
    print(f"📂 Server Config Loaded: {config_path}")
    real_config = XRefineConfig.load_yaml(config_path)
    CONFIG.sample = real_config.sample
    CONFIG.instrument = real_config.instrument

# AI Model Load
ckpt_path = os.environ.get("MODEL_PATH", "checkpoints/model.pt")
AI_MODEL = None
if os.path.exists(ckpt_path):
    AI_MODEL, _ = load_model_from_checkpoint(ckpt_path, DEVICE)
    print("✅ AI Model Ready.")
else:
    print(f"⚠️ Model not found at {ckpt_path}")

# ==============================================================================
# 1. Simulator Interface
# ==============================================================================
def simulator_wrapper(param_tensors: dict[str, torch.Tensor], q_tensor: torch.Tensor) -> torch.Tensor:
    """
    Refine Engine이 호출하는 표준 인터페이스.
    src.simulation.simulate_reflectivity로 위임하여 테스트 코드와 100% 동일 로직 보장.
    """
    return simulate_reflectivity(param_tensors, q_tensor, PHYSICS_ENGINE, DEVICE)

# ==============================================================================
# 2. API Routes
# ==============================================================================

@app.route('/predict_initial', methods=['POST'])
def api_predict_initial():
    """
    [Step 1] 데이터 로드 -> 정규화 -> AI 초안 -> 그래프(Base64)
    """
    try:
        if 'file' not in request.files: 
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        
        # 1. Load Data
        df = pd.read_csv(file, sep=r'\s+', comment='#', header=None, dtype=str)
        df = df.replace({r'[()]': ''}, regex=True).astype(float).dropna()
        x_raw, y_raw = df.iloc[:, 0].values, df.iloc[:, 1].values
        
        # 2. Apply Anchor Normalization (Core Logic)
        q_vals = tth2q(x_raw, wavelen=CONFIG.instrument.wavelength)
        y_norm, scale_factor = apply_anchor_normalization(
            q_vals, y_raw, CONFIG.instrument.wavelength
        )
        
        # Prepare Tensors
        q_tensor = torch.from_numpy(q_vals).float().to(DEVICE)
        # AI 입력도 정규화된 데이터를 사용 (log10)
        log_r_obs = torch.log10(torch.clamp(torch.from_numpy(y_norm).float().to(DEVICE), min=1e-12))
        
        # 3. AI Prediction
        initial_params = {}
        if AI_MODEL:
            try:
                initial_params = predict_initial_params(AI_MODEL, q_tensor, log_r_obs, DEVICE)
            except Exception as e:
                print(f"AI Prediction Failed: {e}")
        
        # Fallback Defaults
        if not initial_params:
            initial_params = {"i0": 1.0, "bkg": -6.0}
            for l in CONFIG.sample.layers:
                initial_params[f"{l.name}.thickness"] = 300.0
                initial_params[f"{l.name}.roughness"] = 3.0
                initial_params[f"{l.name}.sld"] = 50.0
                initial_params[f"{l.name}.sld_imag"] = 1.0
            initial_params["Substrate.roughness"] = 2.0
            initial_params["Substrate.sld"] = 20.0
            initial_params["Substrate.sld_imag"] = 0.0

        # [SCALE LOCK] 정규화되었으므로 i0는 1.0 강제
        initial_params['i0'] = 1.0

        # 4. Simulate & Plot (Normalized View)
        with torch.no_grad():
            p_tensors = {k: torch.tensor([v], device=DEVICE) for k, v in initial_params.items()}
            r_sim = simulator_wrapper(p_tensors, q_tensor).cpu().numpy().flatten()
            
            # src.visualization 모듈을 사용하여 이미지 생성 (Base64 리턴)
            plot_base64 = plot_fit_result(
                q_vals, y_norm, r_sim, 
                title="Step 1: AI Initial Guess (Normalized)",
                save_path=None # None이면 Base64 문자열 반환
            )

        return jsonify({
            "status": "success",
            "initial_params": initial_params,
            "valid_keys": list(initial_params.keys()),
            "scale_factor": float(scale_factor), # 클라이언트에 스케일 정보 제공
            "plot_base64": plot_base64
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/refine_gradient', methods=['POST'])
def api_refine():
    """
    [Step 2] 현재 파라미터 + 옵션 -> Optimizer 실행 -> 결과 & 그래프
    """
    try:
        # Input Parsing
        file = request.files['file']
        current_params = json.loads(request.form.get('current_params', '{}'))
        optimize_spec = json.loads(request.form.get('optimize_spec', '{}'))
        
        # 1. Load & Normalize (Same as Step 1)
        df = pd.read_csv(file, sep=r'\s+', comment='#', header=None, dtype=str)
        df = df.replace({r'[()]': ''}, regex=True).astype(float).dropna()
        x_raw, y_raw = df.iloc[:, 0].values, df.iloc[:, 1].values
        
        q_vals = tth2q(x_raw, wavelen=CONFIG.instrument.wavelength)
        y_norm, _ = apply_anchor_normalization(
            q_vals, y_raw, CONFIG.instrument.wavelength
        )
        
        q_tensor = torch.from_numpy(q_vals).float().to(DEVICE)
        y_tensor = torch.from_numpy(y_norm).float().to(DEVICE)
        log_r_obs = torch.log10(torch.clamp(y_tensor, min=1e-12))
        
        data_payload = {'q': q_tensor, 'log_r_obs': log_r_obs}
        
        # [Safety Guard] i0 Bounds Check
        # 클라이언트가 i0 범위를 너무 넓게 잡았을 경우 서버에서 안전하게 클램핑
        if 'target_params' in optimize_spec and 'i0' in optimize_spec['target_params']:
            bounds = optimize_spec['target_params']['i0']
            # i0는 1.0 근처여야 함 (0.8 ~ 1.2)
            safe_min = max(bounds[0], 0.8)
            safe_max = min(bounds[1], 1.2)
            optimize_spec['target_params']['i0'] = [safe_min, safe_max]
            print(f"🔒 Server enforced i0 bounds: [{safe_min}, {safe_max}]")

        # 2. Run Refinement
        print(f"🔥 Starting Refinement ({optimize_spec.get('loss_type', 'LogMSE')})...")
        refined_params, final_loss = refine_with_gradient(
            current_params, data_payload, optimize_spec, simulator_wrapper, DEVICE
        )
        print(f"✅ Refinement Done. Loss: {final_loss:.6f}")
        
        # 3. Plot Result (Normalized View)
        with torch.no_grad():
            # Initial Curve (Comparison)
            p_init = {k: torch.tensor([v], device=DEVICE) for k, v in current_params.items()}
            # Refined Curve
            p_ref = {k: torch.tensor([v], device=DEVICE) for k, v in refined_params.items()}
            
            r_final = simulator_wrapper(p_ref, q_tensor).cpu().numpy().flatten()
            
            # 그래프 생성
            loss_name = optimize_spec.get('loss_type', 'LogMSE')
            plot_base64 = plot_fit_result(
                q_vals, y_norm, r_final, 
                title=f"Refined Result ({loss_name}) | Loss: {final_loss:.4f}",
                save_path=None
            )
            
        return jsonify({
            "status": "success",
            "updated_params": refined_params,
            "final_loss": final_loss,
            "plot_base64": plot_base64
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Production Level: 0.0.0.0 for external access
    app.run(host='0.0.0.0', port=5000, debug=False)