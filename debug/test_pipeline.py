import sys
import os
import torch
import numpy as np
import json

# --- Modules ---
from src.load_data import load_dat_file
from src.nn_glue import load_model_from_checkpoint, predict_initial_params
from src.refine import refine_with_gradient
from src.abeles import AbelesMatrix
from src.physics_utils import tth2q
from src.config import CONFIG, XRefineConfig
from src.data_processing import apply_anchor_normalization
from src.simulation import simulate_reflectivity
from src.visualization import plot_fit_result

# ==============================================================================
# 0. Setup
# ==============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PHYSICS_ENGINE = AbelesMatrix(device=DEVICE)

# [User Request] 하드코딩된 경로 유지
PLOT_DIR = "debug/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# Config Load
config_path = "config.yaml"
if os.path.exists(config_path):
    print(f"📂 Loading Config: {config_path}")
    real_config = XRefineConfig.load_yaml(config_path)
    CONFIG.sample = real_config.sample
    CONFIG.instrument = real_config.instrument
else:
    print("⚠️ config.yaml not found! Using default dummy config.")

print(f"🚀 Pipeline Test Started on {DEVICE}")
print(f"📂 Plots will be saved to: {PLOT_DIR}")

# Wrapper
def simulator_wrapper(param_tensors, q_tensor):
    return simulate_reflectivity(param_tensors, q_tensor, PHYSICS_ENGINE, DEVICE)

# ==============================================================================
# UX Helper Functions
# ==============================================================================
def simplify_key(key):
    return key.replace("Target_Film", "Target").replace("Top_Layer", "Top").replace("Substrate", "Sub").replace("thickness", "thick").replace("roughness", "rough")

def display_dashboard(params):
    keys = list(params.keys())
    print("\n" + "="*80)
    print(f"📊 CURRENT PARAMETERS (SLD controls Real + Imag)")
    print("-" * 80)
    
    visible_keys = [k for k in keys if "sld_imag" not in k]
    
    col_width = 38
    for i in range(0, len(visible_keys), 2): 
        row_str = ""
        for j in range(2):
            if i + j < len(visible_keys):
                raw_key = visible_keys[i + j]
                simple_name = simplify_key(raw_key)
                val = params[raw_key]
                val_str = f"{val:.4f}"
                
                if "sld" in raw_key and not "imag" in raw_key:
                    imag_key = raw_key + "_imag"
                    if imag_key in params:
                        val_str += f" (+{params[imag_key]:.2f}j)"
                
                item = f"• {simple_name}: {val_str}"
                row_str += f"{item:<{col_width}}"
        print(row_str)
    print("="*80)
    return keys

def parse_targets(cmd, all_keys):
    # [CORE FEATURE] 'all' 입력 시 모든 파라미터 선택
    if cmd.lower() == 'all': 
        return all_keys

    target_keys = []
    keywords = cmd.split()
    for token in keywords:
        sub_tokens = token.split('.') 
        for key in all_keys:
            if all(st.lower() in key.lower() for st in sub_tokens):
                if key not in target_keys:
                    target_keys.append(key)
                # sld 선택 시 imag 자동 포함
                if "sld" in key and "imag" not in key:
                    imag_key = key + "_imag"
                    if imag_key in all_keys and imag_key not in target_keys:
                        target_keys.append(imag_key)
    return target_keys

# ==============================================================================
# Main Simulation
# ==============================================================================
def run_agent_simulation(data_path, ckpt_path):
    print("\n🤖 [Agent Mode] Initializing...")
    
    # 1. Load Data & TRANSFORM GRAPH
    try:
        df = load_dat_file(data_path).copy()
        x_raw, y_raw = df.iloc[:, 0].values.copy(), df.iloc[:, 1].values.copy()
        q_vals = tth2q(x_raw, wavelen=CONFIG.instrument.wavelength)
        
        y_norm, scale_factor = apply_anchor_normalization(q_vals, y_raw, CONFIG.instrument.wavelength)
        print(f"⚓ Graph Scaled by x{scale_factor:.4e} (Normalized to ~1.0)")
        
        q_tensor = torch.from_numpy(q_vals).float().to(DEVICE)
        y_tensor = torch.from_numpy(y_norm).float().to(DEVICE)
        log_r_obs = torch.log10(torch.clamp(y_tensor, min=1e-12))
        data_payload = {'q': q_tensor, 'log_r_obs': log_r_obs}
        
    except Exception as e:
        print(f"❌ Data Load Error: {e}"); return

    # 2. NN Guess
    current_params = {}
    if os.path.exists(ckpt_path):
        try:
            print("🧠 Neural Network Guessing...")
            model, _ = load_model_from_checkpoint(ckpt_path, DEVICE)
            current_params = predict_initial_params(model, q_tensor, log_r_obs, DEVICE)
        except Exception as e: print(f"⚠️ NN Failed: {e}")
    
    if not current_params:
        current_params = {"i0": 1.0, "bkg": -6.0}
        for l in CONFIG.sample.layers:
            current_params[f"{l.name}.thickness"] = 300.0
            current_params[f"{l.name}.roughness"] = 3.0
            current_params[f"{l.name}.sld"] = 50.0
            current_params[f"{l.name}.sld_imag"] = 1.0
        current_params["Substrate.roughness"] = 2.0
        current_params["Substrate.sld"] = 20.0
        current_params["Substrate.sld_imag"] = 0.0

    current_params['i0'] = 1.0 
    
    # Save Initial
    with torch.no_grad():
        p_tensors = {k: torch.tensor([v], device=DEVICE) for k, v in current_params.items()}
        r_sim = simulator_wrapper(p_tensors, q_tensor).cpu().numpy().flatten()
        save_path = os.path.join(PLOT_DIR, "00_Initial_State.png")
        plot_fit_result(q_vals, y_norm, r_sim, "Step 1: AI Guess (Normalized)", save_path=save_path)
        print(f"🖼️  Initial plot saved: {save_path}")

    # 3. Interactive Loop
    step_count = 1
    while True:
        all_keys = display_dashboard(current_params)
        
        print("\n🔧 ENTER TARGETS (Smart Match)")
        print("   Ex: 'Target.thick' -> Matches Target_Film.thickness")
        print("   Ex: 'all'          -> Optimizes EVERYTHING (Fine Tuning)")  # [UI Update]
        
        cmd = input(f"Targets (Brain) > ").strip()
        
        if cmd.lower() == 'q': break
        if cmd.lower() == 'r': print("Resetting..."); continue
        
        target_keys = parse_targets(cmd, all_keys)
        
        if not target_keys:
            print("⚠️  No matching parameters found! Try again.")
            continue
            
        simple_selected = [simplify_key(k) for k in target_keys]
        if len(simple_selected) > 5:
            print(f"🎯 Selected ({len(target_keys)}): {simple_selected[:5]} ...")
        else:
            print(f"🎯 Selected: {simple_selected}")

        print("\n📉 Loss Function? (1:Log, 2:Linear, 3:Corr, 4:Grad, 5:Hybrid)")
        loss_in = input("Select Loss (default 1) > ").strip()
        loss_map = {'1': 'log_mse', '2': 'linear_mse', '3': 'correlation', '4': 'gradient', '5': 'hybrid'}
        loss_type = loss_map.get(loss_in, 'log_mse')
        
        optimize_spec = {
            "target_params": {}, "loss_type": loss_type,
            "method": "lbfgs", "lr": 1.0, "max_iter": 40
        }
        
        # Auto Bounds
        for k in target_keys:
            val = current_params[k]
            min_v, max_v = val * 0.5, val * 1.5
            if "roughness" in k: min_v = 0.0
            if "thickness" in k: min_v, max_v = 10.0, 3000.0
            
            # [SCALE LOCK] i0 restricted
            if "i0" in k: 
                min_v, max_v = 0.8, 1.2
                
            optimize_spec["target_params"][k] = [min_v, max_v]

        print(f"\n🚀 Running Optimizer ({loss_type})...")
        try:
            refined_params, final_loss = refine_with_gradient(
                current_params, data_payload, optimize_spec, simulator_wrapper, DEVICE
            )
            print(f"✅ Done! Loss: {final_loss:.6f}")
            current_params = refined_params
            
            step_name = f"{step_count:02d}_{cmd.replace(' ', '_').replace('.','')}_{loss_type}"
            save_path = os.path.join(PLOT_DIR, f"{step_name}.png")
            with torch.no_grad():
                p_tensors = {k: torch.tensor([v], device=DEVICE) for k, v in current_params.items()}
                r_sim = simulator_wrapper(p_tensors, q_tensor).cpu().numpy().flatten()
                plot_fit_result(q_vals, y_norm, r_sim, f"Loss: {final_loss:.4f}", save_path=save_path)
            
            print(f"🖼️  Result saved: {save_path}")
            step_count += 1
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    # [User Request] 하드코딩된 경로 유지
    data_path = r"C:\Warehouse\data\dat_files\jinhuan\#1_xrr.dat"
    ckpt_path = r"checkpoints\model.pt"
    
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
        
    run_agent_simulation(data_path, ckpt_path)