import torch
import torch.optim as optim
import logging
from typing import Dict, Any, Tuple, Callable, List

# -----------------------------------------------------------------------------
# 1. Loss Functions Import
# (losses.py가 같은 폴더나 src 패키지 안에 있다고 가정)
# -----------------------------------------------------------------------------

from src.losses import (
    log_mse_loss, 
    linear_mse_loss, 
    correlation_loss, 
    gradient_loss,
    make_fringe_weighted_loss
)

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 2. Loss Map: AI가 문자열로 선택할 수 있는 무기들
# -----------------------------------------------------------------------------
LOSS_MAP = {
    # [기본] 전체 형상 맞춤
    "log_mse": log_mse_loss,
    
    # [진폭/임계각] 진폭이 안 맞거나 Critical Angle이 틀렸을 때
    "linear_mse": linear_mse_loss,
    
    # [패턴 매칭] 두께(주기)가 아예 틀렸을 때 강제 정렬
    "correlation": correlation_loss,
    
    # [미세 구조] 기울기(Gradient)를 강조하여 프린지 디테일 보존
    "gradient": lambda s, o, m: log_mse_loss(s, o, m) + 10.0 * gradient_loss(s, o, m),
    
    # [프린지 집중] 고각도 노이즈 무시하고 프린지 영역만 집중
    "fringe_log": make_fringe_weighted_loss(log_mse_loss),
    "fringe_linear": make_fringe_weighted_loss(linear_mse_loss)
}

# -----------------------------------------------------------------------------
# 3. Core Refinement Logic (Gradient Descent)
# -----------------------------------------------------------------------------
def refine_with_gradient(
    current_params: Dict[str, float],
    data: Dict[str, torch.Tensor],
    optimize_spec: Dict[str, Any],
    simulator_fn: Callable[[Dict[str, torch.Tensor], torch.Tensor], torch.Tensor],
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> Tuple[Dict[str, float], float]:
    """
    Autograd를 이용한 미분 기반 파라미터 최적화 (L-BFGS / Adam).
    
    Args:
        current_params: 현재 파라미터 값 (JSON 호환 Dict)
        data: {'q': Tensor, 'log_r_obs': Tensor}
        optimize_spec: AI가 생성한 전략 JSON
            {
                "target_params": {"Target_Film.thickness": [200, 400], ...},
                "loss_type": "log_mse",
                "lr": 1.0,
                "max_iter": 20
            }
        simulator_fn: (params_tensor_dict, q_tensor) -> r_sim_tensor
                      반드시 PyTorch Differentiable 연산으로만 구성되어야 함.
    
    Returns:
        (updated_params_dict, final_loss_float)
    """
    
    # 1. Prepare Data
    q_grid = data['q'].to(device)
    log_r_obs = data['log_r_obs'].to(device)
    
    # AI가 Cutoff를 지정했으면 적용, 없으면 전체 범위
    q_min = optimize_spec.get('q_min', 0.0)
    q_max = optimize_spec.get('q_max', q_grid.max().item())
    
    # 유효 데이터 마스크 (Log값이 너무 작은 노이즈 제외)
    mask = (log_r_obs > -10.0) & (q_grid >= q_min) & (q_grid <= q_max)

    # 2. Prepare Parameters (Variable Wrapping)
    target_keys = list(optimize_spec.get('target_params', {}).keys())
    param_tensors = {} # 시뮬레이터에 들어갈 모든 파라미터
    trainable_tensors = [] # 최적화할(미분할) 파라미터 리스트

    for key, val in current_params.items():
        # 스칼라 값을 텐서로 변환
        t_val = torch.tensor([val], dtype=torch.float32, device=device)
        
        if key in target_keys:
            t_val.requires_grad_(True) # Gradient 추적 켜기!
            trainable_tensors.append(t_val)
        
        param_tensors[key] = t_val

    if not trainable_tensors:
        logger.warning("⚠️ No trainable parameters specified in optimize_spec.")
        return current_params, 0.0

    # 3. Optimizer Setup
    # LBFGS: XRR처럼 매끄러운 물리 모델에서 수렴 속도가 압도적으로 빠름 (추천)
    # Adam: 노이즈가 많거나 불안정한 경우 사용
    method = optimize_spec.get('method', 'lbfgs').lower()
    lr = optimize_spec.get('lr', 1.0) # LBFGS는 보통 lr=1.0이 default
    max_iter = optimize_spec.get('max_iter', 20)

    if method == 'adam':
        optimizer = optim.Adam(trainable_tensors, lr=0.1)
    else:
        optimizer = optim.LBFGS(
            trainable_tensors, 
            lr=lr, 
            max_iter=max_iter, 
            history_size=10, 
            line_search_fn="strong_wolfe" # 물리 피팅 안정성 확보
        )

    # 4. Loss Function Selection
    loss_name = optimize_spec.get('loss_type', 'log_mse')
    loss_fn = LOSS_MAP.get(loss_name, log_mse_loss)
    
    # 5. Optimization Loop (Closure)
    # LBFGS는 한 Step 안에서 loss를 여러 번 평가하므로 closure 함수가 필수입니다.
    def closure():
        optimizer.zero_grad()
        
        try:
            # A. Simulation (Forward)
            # param_tensors는 {str: Tensor} 형태이며, trainable 텐서들과 연결되어 있음
            r_sim = simulator_fn(param_tensors, q_grid)
            
            # B. Loss Calculation
            base_loss = loss_fn(r_sim, log_r_obs, mask)
            
            # C. Soft Constraints (Penalty Method)
            # 하드 바운드(Bounds) 대신 범위를 벗어나면 로스를 급격히 키워서 밀어넣음
            penalty = 0.0
            penalty_weight = 1000.0 # 페널티 강도
            
            for i, key in enumerate(target_keys):
                val_tensor = trainable_tensors[i]
                min_v, max_v = optimize_spec['target_params'][key]
                
                # Lower bound violation
                penalty += torch.relu(min_v - val_tensor) * penalty_weight
                # Upper bound violation
                penalty += torch.relu(val_tensor - max_v) * penalty_weight
            
            total_loss = base_loss + penalty.sum()
            
            # D. Backward (Gradient Calculation)
            if total_loss.requires_grad:
                total_loss.backward()
                
            return total_loss

        except Exception as e:
            # 시뮬레이션 폭발 방지 (NaN 등)
            logger.error(f"Simulation failed during optimization: {e}")
            return torch.tensor(1e9, device=device, requires_grad=True)

    # 6. Run Optimization
    # LBFGS는 step() 한 번 호출에 내부적으로 max_iter만큼 돕니다.
    try:
        optimizer.step(closure)
    except Exception as e:
        logger.error(f"Optimization step failed: {e}")

    # 7. Finalize & Extract Results
    updated_params = current_params.copy()
    
    # 최종 로스 계산 (No grad)
    with torch.no_grad():
        final_loss = closure().item()
        
        for i, key in enumerate(target_keys):
            # 텐서 값을 꺼내서 float로 변환
            val = trainable_tensors[i].item()
            
            # 안전장치: 혹시라도 범위 밖으로 튀어나갔으면 강제 클리핑
            min_v, max_v = optimize_spec['target_params'][key]
            val = max(min_v, min(val, max_v))
            
            updated_params[key] = val

    return updated_params, final_loss