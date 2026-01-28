from flask import Flask, request, jsonify
import torch

from src.nn_glue import load_model_from_checkpoint, predict_initial_params
from src.simulation import simulate_reflectivity
from src.abeles import AbelesMatrix
from src.losses import compute_standard_loss
from src.visualization import plot_fit_result

ckpt_path = "checkpoints/model.pt"
DEVICE = torch.device("cpu")
AI_MODEL, _ = load_model_from_checkpoint(ckpt_path, DEVICE)
PHYSICS_ENGINE = AbelesMatrix(device=DEVICE)
HTTP_CODE = 200

app = Flask(__name__)

@app.route('/nn_predict', methods=['POST'])
def api_nn_predict():
    data = request.get_json()
    q = torch.Tensor(data["q"])
    refl = torch.Tensor(data["refl"])

    initial_params = predict_initial_params(
        AI_MODEL, q, torch.log(refl), DEVICE
    )

    with torch.no_grad():
        r_sim = simulate_reflectivity(initial_params, q, PHYSICS_ENGINE, DEVICE)
        std_loss = compute_standard_loss(r_sim, refl)
    
    return jsonify({
        "q": data["q"],
        "raw_refl": data["refl"],
        "refl_sim": r_sim.cpu().numpy().tolist(),
        "std_loss": std_loss,
        "params": initial_params
        }), HTTP_CODE
