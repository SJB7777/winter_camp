import torch
from torch import Tensor
from collections.abc import Callable

LossFn = Callable[[Tensor, Tensor, Tensor], Tensor]

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

# [NEW] Gradient Loss (형상/기울기 집중)
def log_gradient_loss(r_sim: Tensor, log_r_obs: Tensor, mask: Tensor) -> Tensor:
    log_r_sim = torch.log10(torch.clamp(r_sim, min=1e-12))
    
    # 1차 미분 (인접 포인트 차이)
    diff_sim = log_r_sim[..., 1:] - log_r_sim[..., :-1]
    diff_obs = log_r_obs[..., 1:] - log_r_obs[..., :-1]
    
    # 마스크도 길이 맞춤
    mask_diff = mask[..., 1:] & mask[..., :-1]
    
    val_loss = torch.mean(((log_r_sim - log_r_obs) ** 2) * mask)
    grad_loss = torch.mean(((diff_sim - diff_obs) ** 2) * mask_diff)
    
    # 값 차이 + 기울기 차이 (가중치 10배)
    return val_loss + 10.0 * grad_loss

# [NEW] Hybrid Loss (Log + Linear)
def hybrid_loss(r_sim: Tensor, log_r_obs: Tensor, mask: Tensor) -> Tensor:
    # Linear (진폭/스케일) + Log (전체 형상)
    loss_lin = linear_mse_loss(r_sim, log_r_obs, mask)
    loss_log = log_mse_loss(r_sim, log_r_obs, mask)
    return 20.0 * loss_lin + loss_log