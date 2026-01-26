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
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"MODEL_PATH not found: {ckpt_path}")

AI_MODEL, _ = load_model_from_checkpoint(ckpt_path, DEVICE)
print(f"✅ AI Model Loaded from {ckpt_path}")


def simulator_wrapper(param_tensors, q_tensor):
    return simulate_reflectivity(param_tensors, q_tensor, PHYSICS_ENGINE, DEVICE)


# ==============================================================================
# 1. Helper Functions
# ==============================================================================
def safe_serialize(obj):
    if isinstance(obj, bytes):
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
    if params is None:
        raise ValueError("params is required (got None)")
    if not isinstance(params, dict):
        raise ValueError(f"params must be dict (got {type(params)})")

    clean_params = {}
    for k, v in params.items():
        if isinstance(v, (torch.Tensor, np.generic)):
            clean_params[k] = float(v)
        elif isinstance(v, list):
            if len(v) != 1:
                raise ValueError(f"param '{k}' list must have length 1 (got {len(v)})")
            clean_params[k] = float(v[0])
        elif isinstance(v, (int, float, str)):
            clean_params[k] = v
        else:
            # 여기서 애매한 타입은 바로 죽게 해서 버그 노출
            raise ValueError(f"param '{k}' has unsupported type: {type(v)}")
    return clean_params


def ensure_base64_str(x):
    if x is None:
        raise ValueError("plot_base64 is required but None returned from plot_fit_result()")

    if isinstance(x, bytes):
        try:
            return x.decode("utf-8")
        except UnicodeDecodeError:
            return base64.b64encode(x).decode("utf-8")

    if isinstance(x, str):
        if len(x) == 0:
            raise ValueError("plot_base64 is empty string")
        return x

    raise ValueError(f"plot_base64 must be bytes or str (got {type(x)})")


def require_json():
    req_json = request.json
    if req_json is None:
        raise ValueError("Request body must be JSON (Content-Type: application/json)")
    if not isinstance(req_json, dict):
        raise ValueError("Request JSON must be an object")
    return req_json


def require_data_package(req_json):
    data_package = req_json.get('data_package')
    if data_package is None:
        raise ValueError("Missing required field: data_package")
    if not isinstance(data_package, dict):
        raise ValueError("data_package must be an object")
    return data_package


def json_to_tensors(data_package):
    if not data_package or 'q' not in data_package or 'y_norm' not in data_package:
        raise ValueError("Invalid data_package: missing 'q' or 'y_norm'")

    q = data_package['q']
    y_norm = data_package['y_norm']
    if not isinstance(q, list) or not isinstance(y_norm, list):
        raise ValueError("'q' and 'y_norm' must be lists")

    q_vals = np.array(q, dtype=float)
    y_norm_vals = np.array(y_norm, dtype=float)

    q_tensor = torch.from_numpy(q_vals).float().to(DEVICE)
    log_r_obs = torch.log10(torch.clamp(torch.from_numpy(y_norm_vals).float().to(DEVICE), min=1e-12))

    return {
        'q_tensor': q_tensor,
        'log_r_obs': log_r_obs,
        'q_vals': q_vals,
        'y_norm': y_norm_vals
    }


# ==============================================================================
# 2. Unified API Envelope (Strict)
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

        "data_package": data_package,  # dict or None
        "scale_factor": float(scale_factor) if scale_factor is not None else None,

        "params": sanitize_params(params) if params is not None else {},
        "loss": float(loss) if loss is not None else None,
        "plot_base64": ensure_base64_str(plot_base64) if plot_base64 is not None else None,
        "opt_loss": float(opt_loss) if opt_loss is not None else None,

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
    stage = "load_data"
    try:
        if 'file' not in request.files:
            return api_error(stage, "No file part in request", http_code=400)

        file = request.files['file']

        df = pd.read_csv(file, sep=r'\s+', comment='#', header=None, dtype=str)
        df = df.replace({r'[()]': ''}, regex=True).astype(float).dropna()
        x_raw, y_raw = df.iloc[:, 0].values, df.iloc[:, 1].values

        q_vals = tth2q(x_raw, wavelen=CONFIG.instrument.wavelength)
        y_norm, scale_factor = apply_anchor_normalization(q_vals, y_raw, CONFIG.instrument.wavelength)

        data_package = {
            "q": q_vals.tolist(),
            "y_norm": y_norm.tolist()
        }

        return api_ok(
            stage,
            data_package=data_package,
            scale_factor=scale_factor,
            params={},          # load 단계는 params 없음이 정상
            loss=None,
            plot_base64=None,
            opt_loss=None,
            meta={"note": "data_package generated"},
            http_code=200
        )

    except ValueError as e:
        return api_error(stage, e, http_code=400)
    except Exception as e:
        traceback.print_exc()
        return api_error(stage, e, http_code=500)


@app.route('/predict_initial', methods=['POST'])
def api_predict_initial():
    stage = "initial"
    try:
        req_json = require_json()
        data_package = require_data_package(req_json)
        tensors = json_to_tensors(data_package)

        # strict: 모델이 없으면 진행 금지 (이미 전역에서 로드 실패 시 raise)
        initial_params = predict_initial_params(
            AI_MODEL, tensors['q_tensor'], tensors['log_r_obs'], DEVICE
        )
        if not isinstance(initial_params, dict) or len(initial_params) == 0:
            raise ValueError("AI returned empty initial_params (strict mode)")

        # strict: i0 강제는 유지(원하면 이 줄도 제거 가능)
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
            data_package=data_package,
            scale_factor=req_json.get("scale_factor"),
            params=clean_params,
            loss=std_loss,
            plot_base64=plot_b64,
            opt_loss=None,
            meta={"i0_locked": True},
            http_code=200
        )

    except ValueError as e:
        return api_error(stage, e, http_code=400)
    except Exception as e:
        traceback.print_exc()
        return api_error(stage, e, http_code=500)


@app.route('/refine_gradient', methods=['POST'])
def api_refine_gradient():
    stage = "refine"
    try:
        req_json = require_json()

        data_package = require_data_package(req_json)
        current_params = req_json.get('current_params', None)
        optimize_spec = req_json.get('optimize_spec', None)

        if current_params is None:
            raise ValueError("Missing required field: current_params")
        if optimize_spec is None:
            raise ValueError("Missing required field: optimize_spec")
        if not isinstance(optimize_spec, dict):
            raise ValueError("optimize_spec must be an object")

        tensors = json_to_tensors(data_package)
        data_payload = {'q': tensors['q_tensor'], 'log_r_obs': tensors['log_r_obs']}

        # strict: current_params는 dict이고 비어있으면 에러
        clean_current = sanitize_params(current_params)
        if len(clean_current) == 0:
            raise ValueError("current_params is empty (strict mode)")

        # Guard: i0 range (원하면 이 guard도 제거 가능)
        if 'target_params' in optimize_spec and 'i0' in optimize_spec['target_params']:
            optimize_spec['target_params']['i0'] = [0.8, 1.2]

        refined_params, opt_loss = refine_with_gradient(
            clean_current, data_payload, optimize_spec, simulator_wrapper, DEVICE
        )
        clean_refined = sanitize_params(refined_params)
        if len(clean_refined) == 0:
            raise ValueError("refine returned empty params (strict mode)")

        with torch.no_grad():
            p_ref = {k: torch.tensor([v], device=DEVICE) for k, v in clean_refined.items()}
            r_final = simulator_wrapper(p_ref, tensors['q_tensor'])
            std_loss = compute_standard_loss(r_final, tensors['log_r_obs'])

            loss_name = optimize_spec.get('loss_type', None)
            if loss_name is None:
                raise ValueError("optimize_spec.loss_type is required (strict mode)")

            plot_b64 = plot_fit_result(
                tensors['q_vals'], tensors['y_norm'], r_final.cpu().numpy().flatten(),
                title=f"Refined ({loss_name}) | StdLoss: {std_loss:.4f}"
            )

        return api_ok(
            stage,
            data_package=data_package,
            scale_factor=req_json.get("scale_factor"),
            params=clean_refined,
            loss=std_loss,
            plot_base64=plot_b64,
            opt_loss=opt_loss,
            meta={"loss_name": loss_name},
            http_code=200
        )

    except ValueError as e:
        return api_error(stage, e, http_code=400)
    except Exception as e:
        traceback.print_exc()
        return api_error(stage, e, http_code=500)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
