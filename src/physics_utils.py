from typing import Final, overload

from torch import Tensor
import torch
import numpy as np
from numpy.typing import NDArray

r_e: Final[float] = 2.8179403227e-5  # AA

@overload
def tth2q(tth: float, wavelen: float = 1.54) -> float: ...

@overload
def tth2q(tth: NDArray[np.float64], wavelen: float = 1.54) -> NDArray[np.float64]: ...

def tth2q(tth, wavelen: float = 1.5406):
    """
    Convert 2θ (in degrees) and wavelength (in Å) to scattering vector q (in 1/Å).
    Formula: q = (4 * pi / lambda) * sin(theta)
    """
    # tth is 2*theta, so divide by 2 to get theta
    th_rad = np.deg2rad(0.5 * tth)
    result = (4 * np.pi / wavelen) * np.sin(th_rad)

    if isinstance(tth, (int, float)):
        return float(result)
    return result


def soft_clamp(
    x: Tensor, 
    min_val: float, 
    max_val: float, 
    sharpness: float = 10.0
) -> Tensor:
    """
    Soft clamping that preserves gradients at boundaries.
    
    Unlike torch.clamp which sets gradient to 0 at boundaries,
    this uses sigmoid to provide smooth transitions with weak gradients.
    
    Args:
        x: Input tensor
        min_val: Minimum value
        max_val: Maximum value
        sharpness: Controls transition steepness (higher = closer to hard clamp)
                   Default 10.0 is a good balance for physics constraints
    
    Returns:
        Soft-clamped tensor with gradients preserved
        
    Example:
        >>> sin_alpha = soft_clamp(q * wavelength / (4 * pi), -1.0, 1.0)
        >>> # Near boundaries, still has small gradient unlike torch.clamp
    """
    # Normalize to [0, 1]
    x_normalized = (x - min_val) / (max_val - min_val + 1e-9)
    
    # Sigmoid with adjustable sharpness
    # When sharpness high, sigmoid(10 * (0.5 - 0.5)) = 0.5
    # sigmoid(10 * (x - 0.5)) transitions from ~0 to ~1 in [0, 1]
    x_sigmoid = torch.sigmoid(sharpness * (x_normalized - 0.5))
    
    # Map back to [min_val, max_val]
    return min_val + (max_val - min_val) * x_sigmoid


def soft_clamp_min(x: Tensor, min_val: float, sharpness: float = 10.0) -> Tensor:
    """
    Soft lower bound only.
    
    Useful for reflectivity: R >= 1e-15
    
    Args:
        x: Input tensor
        min_val: Minimum value
        sharpness: Transition steepness
        
    Returns:
        Tensor with soft lower bound
    """
    # Use softplus: log(1 + exp(sharpness * (x - min_val))) / sharpness + min_val
    # More numerically stable than exp
    return min_val + torch.nn.functional.softplus(x - min_val, beta=sharpness) / sharpness


def curriculum_weight(
    epoch: int, 
    max_epochs: int, 
    warmup_fraction: float = 0.3,
    start_weight: float = 0.1,
    end_weight: float = 1.0
) -> float:
    """
    Calculate curriculum learning weight.
    
    Starts low (easy learning) and increases to full weight (strict enforcement).
    
    Args:
        epoch: Current epoch (0-indexed)
        max_epochs: Total training epochs
        warmup_fraction: Fraction of training for warmup (default 30%)
        start_weight: Initial weight
        end_weight: Final weight
        
    Returns:
        Current weight for this epoch
        
    Example:
        >>> # In training loop
        >>> phys_weight = curriculum_weight(epoch, 1000)
        >>> penalty = physics_penalty * phys_weight
    """
    warmup_epochs = int(max_epochs * warmup_fraction)
    
    if epoch < warmup_epochs:
        # Linear warmup
        progress = epoch / warmup_epochs
        return start_weight + (end_weight - start_weight) * progress
    else:
        return end_weight


def robust_log10(x: Tensor, min_val: float = 1e-15) -> Tensor:
    """
    Numerically stable log10.
    
    Instead of clamp then log (which loses gradient),
    this uses log(x + epsilon) which preserves gradient.
    
    Args:
        x: Input tensor (e.g., reflectivity)
        min_val: Small constant for stability
        
    Returns:
        log10(x + min_val)
    """
    return torch.log10(x + min_val)


def physics_constraint_penalty(
    thickness: Tensor,
    roughness: Tensor,
    constraint_ratio: float = 0.5,
    penalty_type: str = 'relu'
) -> Tensor:
    """
    Physical constraint: roughness < ratio * thickness
    
    Args:
        thickness: Layer thickness (B, L)
        roughness: Interface roughness (B, L)
        constraint_ratio: Maximum roughness as fraction of thickness
        penalty_type: 'relu' or 'quadratic'
        
    Returns:
        Scalar penalty term
    """
    violation = roughness - constraint_ratio * thickness
    
    if penalty_type == 'relu':
        # ReLU: only penalize violations
        return torch.mean(torch.relu(violation))
    elif penalty_type == 'quadratic':
        # Quadratic: stronger penalty for larger violations
        return torch.mean(torch.relu(violation) ** 2)
    else:
        raise ValueError(f"Unknown penalty_type: {penalty_type}")


def adaptive_epsilon(x: Tensor, base_eps: float = 1e-9) -> Tensor:
    """
    Adaptive epsilon based on tensor magnitude.
    
    Useful for division to prevent numerical issues.
    
    Args:
        x: Input tensor
        base_eps: Base epsilon value
        
    Returns:
        Adaptive epsilon: max(base_eps, |x| * 1e-7)
    """
    magnitude_eps = torch.abs(x) * 1e-7
    return torch.maximum(
        torch.tensor(base_eps, device=x.device, dtype=x.dtype),
        magnitude_eps
    )
