import torch
import torch.optim as optim
import logging
from typing import Dict, Any, Tuple, Callable

# Loss import
from src.losses import (
    log_mse_loss, linear_mse_loss
)

logger = logging.getLogger(__name__)

LOSS_MAP = {
    "log_mse": log_mse_loss,
    "linear_mse": linear_mse_loss,
    # "correlation": correlation_loss,
    # "gradient": log_gradient_loss, # [NEW]
    # "hybrid": hybrid_loss          # [NEW]
}

def refine_with_gradient(
    current_params: Dict[str, float],
    data: Dict[str, torch.Tensor],
    optimize_spec: Dict[str, Any],
    simulator_fn: Callable[[Dict[str, torch.Tensor], torch.Tensor], torch.Tensor],
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> Tuple[Dict[str, float], float]:
    """
    [ROBUST OPTIMIZER] Parameter Normalization 적용 버전
    모든 파라미터를 0~1 범위(Normalized Space)로 변환하여 최적화함.
    """
    
    # 1. Data Setup
    q_grid = data['q'].to(device)
    log_r_obs = data['log_r_obs'].to(device)
    
    q_min = optimize_spec.get('q_min', 0.0)
    q_max = optimize_spec.get('q_max', q_grid.max().item())
    mask = (log_r_obs > -10.0) & (q_grid >= q_min) & (q_grid <= q_max)

    # 2. Parameter Normalization Setup
    target_keys = list(optimize_spec.get('target_params', {}).keys())
    
    # 학습할 텐서 리스트 (0~1 범위로 정규화된 값들)
    trainable_tensors = []
    param_info = [] # 메타데이터 (key, min, max)
    
    # 고정 파라미터 (값 그대로)
    fixed_tensors = {}

    for key, val in current_params.items():
        if key in target_keys:
            # 범위 가져오기
            min_v, max_v = optimize_spec['target_params'][key]
            
            # 현재 값을 0~1로 정규화
            # norm = (val - min) / (max - min)
            current_val = max(min_v, min(val, max_v)) # Clamp first
            norm_val = (current_val - min_v) / (max_v - min_v + 1e-9)
            
            # 텐서 생성 (Requires Grad)
            t = torch.tensor([norm_val], dtype=torch.float32, device=device, requires_grad=True)
            trainable_tensors.append(t)
            
            param_info.append({
                'key': key,
                'min': min_v,
                'max': max_v,
                'tensor': t
            })
        else:
            fixed_tensors[key] = torch.tensor([val], dtype=torch.float32, device=device)

    if not trainable_tensors:
        return current_params, 0.0

    # 3. Optimizer (LBFGS for robust physics fitting)
    method = optimize_spec.get('method', 'lbfgs').lower()
    lr = optimize_spec.get('lr', 1.0)
    max_iter = optimize_spec.get('max_iter', 20)

    if method == 'adam':
        optimizer = optim.Adam(trainable_tensors, lr=0.05) # Normalized space라 LR 작게
    else:
        # LBFGS is king for XRR
        optimizer = optim.LBFGS(trainable_tensors, lr=1.0, max_iter=20, history_size=10, line_search_fn="strong_wolfe")

    loss_name = optimize_spec.get('loss_type', 'log_mse')
    loss_fn = LOSS_MAP.get(loss_name, log_mse_loss)

    # 4. Helper: Denormalize (0~1 -> Real Value)
    def get_real_params():
        real_params = fixed_tensors.copy()
        for info in param_info:
            # 0~1 클리핑 (학습 중 범위 이탈 방지)
            norm_v = torch.clamp(info['tensor'], 0.0, 1.0)
            # 복원: val = norm * (max - min) + min
            real_v = norm_v * (info['max'] - info['min']) + info['min']
            real_params[info['key']] = real_v
        return real_params

    # 5. Optimization Loop
    def closure():
        optimizer.zero_grad()
        
        try:
            # A. Get Physical Values
            p_tensors = get_real_params()
            
            # B. Simulate
            r_sim = simulator_fn(p_tensors, q_grid)
            
            # C. Loss
            loss = loss_fn(r_sim, log_r_obs, mask)
            
            # D. Backward
            if loss.requires_grad:
                loss.backward()
            return loss
            
        except Exception:
            return torch.tensor(1e9, device=device, requires_grad=True)

    try:
        optimizer.step(closure)
    except Exception as e:
        logger.error(f"Optimization failed: {e}")

    # 6. Result Extraction
    updated_params = current_params.copy()
    final_loss = closure().item()
    
    with torch.no_grad():
        final_tensors = get_real_params()
        for k, v in final_tensors.items():
            updated_params[k] = v.item()

    return updated_params, final_loss