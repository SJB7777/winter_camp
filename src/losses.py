import torch
from torch import Tensor
from collections.abc import Callable

LossFn = Callable[[Tensor, Tensor, Tensor], Tensor]

# --- Existing Loss Functions ---
def log_mse_loss(r_sim: Tensor, log_r_obs: Tensor, mask: Tensor) -> Tensor:
    log_r_sim = torch.log10(torch.clamp(r_sim, min=1e-12))
    return torch.mean(((log_r_sim - log_r_obs) ** 2) * mask)

def linear_mse_loss(r_sim: Tensor, log_r_obs: Tensor, mask: Tensor) -> Tensor:
    r_obs = 10 ** log_r_obs
    return torch.mean(((r_sim - r_obs) ** 2) * mask)

def correlation_loss(r_sim: Tensor, log_r_obs: Tensor, mask: Tensor) -> Tensor:
    log_r_sim = torch.log10(torch.clamp(r_sim, min=1e-12))
    sim_masked = log_r_sim[mask]
    obs_masked = log_r_obs[mask]
    vx = sim_masked - sim_masked.mean()
    vy = obs_masked - obs_masked.mean()
    corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-9)
    return 1.0 - corr

def log_gradient_loss(r_sim: Tensor, log_r_obs: Tensor, mask: Tensor) -> Tensor:
    log_r_sim = torch.log10(torch.clamp(r_sim, min=1e-12))
    diff_sim = log_r_sim[..., 1:] - log_r_sim[..., :-1]
    diff_obs = log_r_obs[..., 1:] - log_r_obs[..., :-1]
    mask_diff = mask[..., 1:] & mask[..., :-1]
    val_loss = torch.mean(((log_r_sim - log_r_obs) ** 2) * mask)
    grad_loss = torch.mean(((diff_sim - diff_obs) ** 2) * mask_diff)
    return val_loss + 10.0 * grad_loss

def hybrid_loss(r_sim: Tensor, log_r_obs: Tensor, mask: Tensor) -> Tensor:
    loss_lin = linear_mse_loss(r_sim, log_r_obs, mask)
    loss_log = log_mse_loss(r_sim, log_r_obs, mask)
    return 20.0 * loss_lin + loss_log

# ==============================================================================
# [NEW] Standard Metric Calculator (통일된 채점 기준)
# ==============================================================================
def compute_standard_loss(r_sim: Tensor, log_r_obs: Tensor) -> float:
    """
    [Standard Reporting Metric]
    Optimizer가 무슨 로스를 쓰든(Correlation, Gradient 등),
    사용자(LLM)에게 보고할 때는 항상 'Log MSE' 기준으로 점수를 매깁니다.
    그래야 Step 1 -> Step 2로 갈 때 좋아졌는지 객관적 비교가 가능합니다.
    """
    # 표준 마스크 (-14.0 이하 노이즈 제외)
    mask = log_r_obs > -14.0
    
    # 평가용이므로 no_grad() 처리는 호출 측에서 하거나, 여기서 item()만 뽑음
    loss_val = log_mse_loss(r_sim, log_r_obs, mask).item()
    return loss_val