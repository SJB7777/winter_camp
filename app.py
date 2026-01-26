from flask import Flask, request, jsonify
import matplotlib
matplotlib.use('Agg')

import os
import traceback
import base64

import torch
import numpy as np
import pandas as pd


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
# 0. Global Setup & Config
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
    print(f"✅ AI Model Loaded from {ckpt_path}")
else:
    print("⚠️ AI Model not found. Using Fallback heuristics.")


def simulator_wrapper(param_tensors, q_tensor):
    return simulate_reflectivity(param_tensors, q_tensor, PHYSICS_ENGINE, DEVICE)


# ==============================================================================
# 1. Helper Functions (Serialization & Type Safety)
# ==============================================================================
def safe_serialize(obj):
    """
    JSON으로 변환 불가능한 타입(Bytes, Numpy, Tensor 등)을 안전하게 변환
    """
    if isinstance(obj, bytes):
        # 여기서는 '일반 bytes'일 수도 있으니 무조건 decode만 하지 않음.
        try:
            return obj.decode('utf-8')
        except UnicodeDecodeError:
            return base64.b64encode(obj).decode('utf-8')

    if isinstance(obj, (np.ndarray, np.generic)):
        return obj.tolist()

    if isinstance(obj, torch.Tensor):
        return obj.cpu().detach().numpy().tolist()

    return obj


def sanitize_params(params):
    """
    파라미터 딕셔너리 내부의 Numpy/Tensor 값을 Python native float으로 변환
    """
    clean_params = {}
    for k, v in (params or {}).items():
        if isinstance(v, (torch.Tensor, np.generic)):
            clean_params[k] = float(v)
        elif isinstance(v, list):
            clean_params[k] = float(v[0]) if len(v) > 0 else 0.0
        else:
            clean_params[k] = v
    return clean_params


def ensure_base64_str(x):
    """
    plot_fit_result()가 bytes 또는 str을 줄 수 있으니,
    n8n Convert to File이 먹을 수 있게 '순수 base64 문자열(str)'로 통일.
    """
    if x is None:
        return None
    if isinstance(x, bytes):
        try:
            return x.decode("utf-8")
        except UnicodeDecodeError:
            return base64.b64encode(x).decode("utf-8")
    if isinstance(x, str):
        return x
    return safe_serialize(x)


def json_to_tensors(data_package):
    """
    Client(n8n)에서 받은 data_package(q, y_norm)를 PyTorch Tensor로 복원
    """
    if not data_package or 'q' not in data_package or 'y_norm' not in data_package:
        raise ValueError("Invalid Data Package: Missing 'q' or 'y_norm'.")

    q_vals = np.array(data_package['q'], dtype=float)
    y_norm = np.array(data_package['y_norm'], dtype=float)

    q_tensor = torch.from_numpy(q_vals).float().to(DEVICE)
    log_r_obs = torch.log10(torch.clamp(torch.from_numpy(y_norm).float().to(DEVICE), min=1e-12))

    return {
        'q_tensor': q_tensor,
        'log_r_obs': log_r_obs,
        'q_vals': q_vals,
        'y_norm': y_norm
    }


# ==============================================================================
# 2. Unified API Envelope
# ==============================================================================
def api_ok(stage, *,
           data_package=None,
           scale_factor=None,
           params=None,
           loss=None,
           plot_base64=None,
           opt_loss=None,
           meta=None,
           http_code=200):
    payload = {
        "status": "success",
        "stage": stage,

        # data / preprocessing
        "data_package": data_package,  # dict or None
        "scale_factor": float(scale_factor) if scale_factor is not None else None,

        # fitting state
        "params": sanitize_params(params or {}),
        "loss": float(loss) if loss is not None else None,
        "plot_base64": ensure_base64_str(plot_base64),
        "opt_loss": float(opt_loss) if opt_loss is not None else None,

        # misc
        "meta": meta or {},
        "error": None,
    }
    return jsonify(payload), http_code


def api_error(stage, message, *, http_code=500, meta=None):
    payload = {
        "status": "error",
        "stage": stage,

        "data_package": None,
        "scale_factor": None,

        "params": {},
        "loss": None,
        "plot_base64": None,
        "opt_loss": None,

        "meta": meta or {},
        "error": {"message": str(message)},
    }
    return jsonify(payload), http_code


# ==============================================================================
# 3. API Routes
# ==============================================================================

@app.route('/load_data', methods=['POST'])
def api_load_data():
    """
    [Step 0] 파일 업로드 -> 데이터 정제 -> data_package 반환
    - multipart/form-data, request.files['file']
    """
    stage = "load_data"
    try:
        if 'file' not in request.files:
            return api_error(stage, "No file part in request", http_code=400)

        file = request.files['file']

        # 1) Read File
        df = pd.read_csv(file, sep=r'\s+', comment='#', header=None, dtype=str)
        df = df.replace({r'[()]': ''}, regex=True).astype(float).dropna()
        x_raw, y_raw = df.iloc[:, 0].values, df.iloc[:, 1].values

        # 2) 2theta -> q
        q_vals = tth2q(x_raw, wavelen=CONFIG.instrument.wavelength)

        # 3) Normalize (Anchor at 0.2 deg)
        y_norm, scale_factor = apply_anchor_normalization(
            q_vals, y_raw, CONFIG.instrument.wavelength
        )

        data_package = {
            "q": q_vals.tolist(),
            "y_norm": y_norm.tolist()
        }

        return api_ok(
            stage,
            data_package=data_package,
            scale_factor=scale_factor,
            params={},            # 아직 파라미터 없음
            loss=None,
            plot_base64=None,
            opt_loss=None,
            meta={"note": "data_package generated"},
            http_code=200
        )

    except Exception as e:
        traceback.print_exc()
        return api_error(stage, e, http_code=500)


@app.route('/predict_initial', methods=['POST'])
def api_predict_initial():
    """
    [Step 1] JSON(data_package) -> AI 추론 -> params/loss/plot 반환
    """
    stage = "initial"
    try:
        req_json = request.json or {}
        data_package = req_json.get('data_package')
        tensors = json_to_tensors(data_package)

        initial_params = {}
        if AI_MODEL:
            try:
                initial_params = predict_initial_params(
                    AI_MODEL, tensors['q_tensor'], tensors['log_r_obs'], DEVICE
                )
            except Exception as e:
                print(f"AI Error: {e}")

        if not initial_params:
            initial_params = {"i0": 1.0, "bkg": -6.0, "Target_Film.thickness": 300.0}

        # Anchor lock
        initial_params['i0'] = 1.0

        with torch.no_grad():
            clean_params = sanitize_params(initial_params)
            p_tensors = {k: torch.tensor([v], device=DEVICE) for k, v in clean_params.items()}
            r_sim = simulator_wrapper(p_tensors, tensors['q_tensor'])
            std_loss = compute_standard_loss(r_sim, tensors['log_r_obs'])

            plot_b64 = plot_fit_result(
                tensors['q_vals'], tensors['y_norm'], r_sim.cpu().numpy().flatten(),
                title=f"AI Guess | Loss: {std_loss:.4f}"
            )

        return api_ok(
            stage,
            data_package=data_package,   # echo back for convenience
            scale_factor=req_json.get("scale_factor"),
            params=clean_params,
            loss=std_loss,
            plot_base64=plot_b64,
            opt_loss=None,
            meta={"i0_locked": True},
            http_code=200
        )

    except Exception as e:
        traceback.print_exc()
        return api_error(stage, e, http_code=500)


@app.route('/refine_gradient', methods=['POST'])
def api_refine_gradient():
    """
    [Step 2] JSON(data_package + current_params + optimize_spec) -> refine -> params/loss/plot 반환
    """
    stage = "refine"
    try:
        req_json = request.json or {}

        data_package = req_json.get('data_package')
        current_params = req_json.get('current_params', {})
        optimize_spec = req_json.get('optimize_spec', {})

        tensors = json_to_tensors(data_package)
        data_payload = {'q': tensors['q_tensor'], 'log_r_obs': tensors['log_r_obs']}

        # Guard: i0 range
        if 'target_params' in optimize_spec and 'i0' in optimize_spec['target_params']:
            optimize_spec['target_params']['i0'] = [0.8, 1.2]

        refined_params, opt_loss = refine_with_gradient(
            current_params, data_payload, optimize_spec, simulator_wrapper, DEVICE
        )

        with torch.no_grad():
            clean_params = sanitize_params(refined_params)
            p_ref = {k: torch.tensor([v], device=DEVICE) for k, v in clean_params.items()}
            r_final = simulator_wrapper(p_ref, tensors['q_tensor'])
            std_loss = compute_standard_loss(r_final, tensors['log_r_obs'])

            loss_name = optimize_spec.get('loss_type', 'LogMSE')
            plot_b64 = plot_fit_result(
                tensors['q_vals'], tensors['y_norm'], r_final.cpu().numpy().flatten(),
                title=f"Refined ({loss_name}) | StdLoss: {std_loss:.4f}"
            )

        return api_ok(
            stage,
            data_package=data_package,  # echo back for convenience
            scale_factor=req_json.get("scale_factor"),
            params=clean_params,
            loss=std_loss,
            plot_base64=plot_b64,
            opt_loss=opt_loss,
            meta={"loss_name": loss_name},
            http_code=200
        )

    except Exception as e:
        traceback.print_exc()
        return api_error(stage, e, http_code=500)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
