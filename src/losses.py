"""
Loss Functions for XRR Refinement
=================================
1급 객체(함수)로 정의된 Loss Functions

Usage:
    from xrefine.refine.losses import log_mse_loss, linear_mse_loss, correlation_loss
    
    loss_fn = log_mse_loss
    loss = loss_fn(r_sim, r_obs, mask)
"""
import torch
from torch import Tensor
from collections.abc import Callable

# Type alias for loss function signature
# (r_sim, log_r_obs, mask) -> scalar loss
LossFn = Callable[[Tensor, Tensor, Tensor], Tensor]


def log_mse_loss(r_sim: Tensor, log_r_obs: Tensor, mask: Tensor) -> Tensor:
    """
    Log10 공간에서의 MSE Loss
    
    Args:
        r_sim: Simulated reflectivity (linear scale)
        log_r_obs: Observed reflectivity (log10 scale)
        mask: Valid data mask
    
    Returns:
        Scalar loss tensor
    """
    log_r_sim = torch.log10(torch.clamp(r_sim, min=1e-12))
    return torch.mean(((log_r_sim - log_r_obs) ** 2) * mask)


def linear_mse_loss(r_sim: Tensor, log_r_obs: Tensor, mask: Tensor) -> Tensor:
    """
    Linear R 공간에서의 MSE Loss (프린지 진폭에 민감)
    
    Args:
        r_sim: Simulated reflectivity (linear scale)
        log_r_obs: Observed reflectivity (log10 scale)
        mask: Valid data mask
    
    Returns:
        Scalar loss tensor
    """
    r_obs = 10 ** log_r_obs
    return torch.mean(((r_sim - r_obs) ** 2) * mask)


def correlation_loss(r_sim: Tensor, log_r_obs: Tensor, mask: Tensor) -> Tensor:
    """
    Pearson Correlation Loss (1 - corr)
    두께 검색에 적합 (패턴 매칭)
    
    Args:
        r_sim: Simulated reflectivity (linear scale)
        log_r_obs: Observed reflectivity (log10 scale)
        mask: Valid data mask
    
    Returns:
        1 - correlation (0 = perfect match)
    """
    log_r_sim = torch.log10(torch.clamp(r_sim, min=1e-12))
    
    sim_masked = log_r_sim[mask]
    obs_masked = log_r_obs[mask]
    
    vx = sim_masked - sim_masked.mean()
    vy = obs_masked - obs_masked.mean()
    
    corr = torch.sum(vx * vy) / (
        torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-9
    )
    
    return 1.0 - corr


def gradient_loss(r_sim: Tensor, log_r_obs: Tensor, mask: Tensor, weight: float = 5.0) -> Tensor:
    """
    Gradient Loss (프린지 형상 보존)
    
    Args:
        r_sim: Simulated reflectivity (linear scale)
        log_r_obs: Observed reflectivity (log10 scale)
        mask: Valid data mask
        weight: Gradient loss weight
    
    Returns:
        Weighted gradient loss
    """
    log_r_sim = torch.log10(torch.clamp(r_sim, min=1e-12))
    
    diff_sim = torch.diff(log_r_sim, dim=-1)
    diff_obs = torch.diff(log_r_obs, dim=-1)
    diff_mask = mask[:, 1:] if mask.dim() > 1 else mask[1:]
    
    return weight * torch.mean(((diff_sim - diff_obs) ** 2) * diff_mask)


def combined_loss(
    r_sim: Tensor, 
    log_r_obs: Tensor, 
    mask: Tensor,
    base_loss: LossFn = log_mse_loss,
    add_gradient: bool = True,
    gradient_weight: float = 5.0
) -> Tensor:
    """
    Combined Loss (Base + Gradient)
    
    Args:
        r_sim: Simulated reflectivity
        log_r_obs: Observed reflectivity (log10)
        mask: Valid data mask
        base_loss: Base loss function
        add_gradient: Whether to add gradient loss
        gradient_weight: Weight for gradient loss
    
    Returns:
        Combined loss
    """
    loss = base_loss(r_sim, log_r_obs, mask)
    
    if add_gradient:
        loss = loss + gradient_loss(r_sim, log_r_obs, mask, gradient_weight)
    
    return loss


# Factory function to create custom combined losses
def make_combined_loss(
    base: LossFn = log_mse_loss,
    gradient: bool = False,
    gradient_weight: float = 5.0
) -> LossFn:
    """
    커스텀 Combined Loss 생성 팩토리
    
    Usage:
        my_loss = make_combined_loss(linear_mse_loss, gradient=True, gradient_weight=3.0)
    """
    def loss_fn(r_sim: Tensor, log_r_obs: Tensor, mask: Tensor) -> Tensor:
        return combined_loss(r_sim, log_r_obs, mask, base, gradient, gradient_weight)
    
    return loss_fn


# Preset losses for convenience
LOG_MSE = log_mse_loss
LINEAR_MSE = linear_mse_loss
CORRELATION = correlation_loss
LOG_MSE_WITH_GRAD = make_combined_loss(log_mse_loss, gradient=True)
LINEAR_MSE_WITH_GRAD = make_combined_loss(linear_mse_loss, gradient=True)


def make_fringe_weighted_loss(
    base_loss: LossFn = log_mse_loss,
    fringe_q_range: tuple = (0.05, 0.25),
    fringe_weight: float = 3.0,
    high_q_damping: float = 0.3
) -> LossFn:
    """
    프린지 영역 가중치 Loss 생성
    
    중간각도(프린지 선명) 영역에 높은 가중치, 고각도(노이즈 플로어) 낮은 가중치
    
    Args:
        base_loss: 기본 loss 함수
        fringe_q_range: 프린지 영역 Q 범위 (min, max)
        fringe_weight: 프린지 영역 가중치 (>1 = 강조)
        high_q_damping: 고각도 가중치 감소율 (<1)
    
    Returns:
        Q-weighted loss function
    """
    def weighted_loss(r_sim: Tensor, log_r_obs: Tensor, mask: Tensor) -> Tensor:
        # 기본 loss 계산을 위한 요소별 오차
        log_r_sim = torch.log10(torch.clamp(r_sim, min=1e-12))
        
        # Pointwise error
        if base_loss == linear_mse_loss:
            r_obs = 10 ** log_r_obs
            error = (r_sim - r_obs) ** 2
        else:
            error = (log_r_sim - log_r_obs) ** 2
        
        # Q grid 추정 (mask 길이에서)
        n_points = mask.shape[-1]
        q_estimated = torch.linspace(0.0, 0.5, n_points, device=mask.device)
        
        # Q-based weights
        q_min, q_max = fringe_q_range
        weights = torch.ones_like(q_estimated)
        
        # 프린지 영역 강조 (Q_min ~ Q_max)
        fringe_mask = (q_estimated >= q_min) & (q_estimated <= q_max)
        weights[fringe_mask] = fringe_weight
        
        # 고각도 감쇠 (Q > Q_max)
        high_q_mask = q_estimated > q_max
        weights[high_q_mask] = high_q_damping
        
        # Weighted loss
        weighted_error = error * weights * mask
        return torch.sum(weighted_error) / (torch.sum(mask * weights) + 1e-9)
    
    return weighted_loss


# Fringe-weighted presets
FRINGE_WEIGHTED_LOG = make_fringe_weighted_loss(log_mse_loss)
FRINGE_WEIGHTED_LINEAR = make_fringe_weighted_loss(linear_mse_loss)

