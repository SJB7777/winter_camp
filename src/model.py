"""
Common Components for PINN Models (Enhanced)
============================================
Enhanced with curriculum learning and soft constraints.

Key improvements:
- Curriculum learning for progressive constraint enforcement
- Soft clamping in simulate() for better gradient flow
- Enhanced physics validation with dynamic weights
"""
import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor

from src.abeles import AbelesMatrix
from src.param import ParamSet
from src.config import XRefineConfig
from src.smearing import abeles_constant_smearing
from src.physics_utils import (
    soft_clamp_min, 
    curriculum_weight, 
    physics_constraint_penalty
)
from src.constants import LOG_MASK_VALUE


# ==============================================================================
# 1. Feature Encoding Modules (Unchanged from original)
# ==============================================================================

class FourierFeatureMapping(nn.Module):
    """
    Random Fourier Feature Mapping (NeRF-style).
    """
    def __init__(
        self, 
        mapping_size: int = 64, 
        scale: float = 40.0,
        output_format: str = 'cnn'
    ):
        super().__init__()
        self.mapping_size = mapping_size
        self.output_format = output_format
        random_matrix = torch.randn(1, mapping_size) * scale
        self.register_buffer('B', random_matrix)

    def forward(self, x: Tensor) -> Tensor:
        if self.output_format == 'cnn':
            if x.dim() == 2: x = x.unsqueeze(1)
            x_proj = x.transpose(1, 2) @ self.B
            x_proj = x_proj * 2 * math.pi
            out = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
            return out.transpose(1, 2)
        else:
            if x.dim() == 2: x = x.unsqueeze(-1)
            x_proj = x @ self.B
            x_proj = x_proj * 2 * math.pi
            return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)



class BasePINNModel(nn.Module, ABC):
    """
    Enhanced PINN model with curriculum learning and soft constraints.
    
    Improvements:
    - Progressive physics constraint enforcement via curriculum learning
    - Soft clamping for better gradient flow
    - Dynamic epoch tracking for weight scheduling
    """
    def __init__(self, config: XRefineConfig):
        super().__init__()
        self.config = config
        
        # [ENHANCED] Use enhanced Abeles implementation
        self.simulator = AbelesMatrix(device=config.device)
        self.register_buffer('q_grid', config.q_grid)
        
        self.param_map = self._build_param_map(config)
        self.output_dim = len(self.param_map)
        
        # [NEW] Curriculum learning state
        self.current_epoch = 0
        self.max_epochs = getattr(config.train, 'epochs', 1000) if hasattr(config, 'train') else 1000

    def set_epoch(self, epoch: int):
        """Update current epoch for curriculum learning"""
        self.current_epoch = epoch

    def _build_param_map(self, config: XRefineConfig) -> list[tuple[str, list[float]]]:
        """
        Builds a flattened list of (param_key, range) for the neural network output head.
        Returns: [('Film.thickness', [100, 500]), ('Film.roughness', [1, 5]), ...]
        """
        mapping = []

        # 1. Layers
        for layer in config.sample.layers:
            mapping.append((f"{layer.name}.thickness", layer.thickness))
            mapping.append((f"{layer.name}.roughness", layer.roughness))
            mapping.append((f"{layer.name}.sld", layer.sld))
            
            if getattr(layer, 'sld_imag', None) is not None:
                mapping.append((f"{layer.name}.sld_imag", layer.sld_imag))

        # 2. Substrate
        mapping.append(("Substrate.roughness", config.sample.substrate.roughness))
        mapping.append(("Substrate.sld", config.sample.substrate.sld))
        
        if getattr(config.sample.substrate, 'sld_imag', None) is not None:
             mapping.append(("Substrate.sld_imag", config.sample.substrate.sld_imag))

        # 3. Instrument (Global)
        for key in ParamSet.INSTRUMENT_KEYS:
            if hasattr(config.sample, key):
                rng = getattr(config.sample, key)
                if rng is not None:
                    mapping.append((key, rng))

        return mapping

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def unnormalize(self, raw_output: Tensor, fixed_values: ParamSet | None = None) -> ParamSet:
        """
        Convert NN output (0~1) to physical ranges.
        Constructs a ParamSet with optional fixed value constraints.
        """
        batch_size = raw_output.size(0)
        vals = torch.split(raw_output, 1, dim=1)
        params_dict = {}

        # 1. Base: Convert Network Predictions to Physical Values
        for (key, (p_min, p_max)), val in zip(self.param_map, vals):
            physical_val = val * (p_max - p_min) + p_min
            params_dict[key] = physical_val

        # 2. Override: Apply Fixed Value Constraints
        if fixed_values is not None:
            for key, target_val in fixed_values._params.items():
                if key in params_dict:
                    if target_val.size(0) == 1 and batch_size > 1:
                        target_val = target_val.expand(batch_size, -1)
                    params_dict[key] = target_val

        layer_names = [l.name for l in self.config.sample.layers]
        return ParamSet(params_dict, layer_names, device=self.device)

    def validate_physics(self, params: ParamSet) -> Tensor:
        """
        Enhanced physics constraints with curriculum learning.
        
        Early training: weak constraints (exploration)
        Late training: strong constraints (refinement)
        """
        penalty = torch.tensor(0.0, device=self.device)
        
        # [ENHANCED] Curriculum weight: 0.1 (early) -> 1.0 (late)
        weight = curriculum_weight(
            self.current_epoch, 
            self.max_epochs,
            warmup_fraction=0.3,
            start_weight=0.1,
            end_weight=1.0
        )
        
        for name in params.layer_names:
            d = params[f"{name}.thickness"]
            sigma = params[f"{name}.roughness"]
            
            # [ENHANCED] Use utility function
            curr_penalty = physics_constraint_penalty(
                thickness=d,
                roughness=sigma,
                constraint_ratio=0.5,
                penalty_type='relu'
            )
            penalty = penalty + curr_penalty * weight
            
        return penalty

    def simulate(self, params: ParamSet) -> Tensor:
        """
        Enhanced simulation with soft clamping.
        
        ParamSet -> Reflectivity with improved gradient flow
        """
        thickness, roughness, sld_full = params.assemble_structure()
        batch_size = thickness.size(0)
        
        q_batch = self.q_grid.unsqueeze(0).expand(batch_size, -1).to(self.device)
        
        if 'tth_offset' in params._params:
            q_batch = q_batch + params.tth_offset * 0.01
        
        L = params.L
        beam_width = params.beam_width
        dq = params.dq if 'dq' in params._params else None
        
        if dq is not None and dq.mean() > 1e-5:
             r_sim = abeles_constant_smearing(
                q=q_batch,
                thickness=thickness,
                roughness=roughness,
                sld=sld_full,
                sample_length=L,
                beam_width=beam_width,
                dq=dq,
                abeles_func=self.simulator
            )
        else:
            r_sim = self.simulator(
                q=q_batch,
                thickness=thickness,
                roughness=roughness,
                sld=sld_full,
                sample_length=L,
                beam_width=beam_width,
                wavelength=self.config.instrument.wavelength
            )
        
        i0 = params.i0
        bkg = params.bkg
        r_final = (i0 * r_sim) + torch.pow(10.0, bkg)
        
        # [ENHANCED] Soft clamping instead of hard clamp
        return soft_clamp_min(r_final, min_val=1e-15, sharpness=10.0)

    def forward_with_params(self, params: ParamSet) -> tuple[ParamSet, Tensor, Tensor]:
        params = params.to(self.device)
        r_final = self.simulate(params)
        penalty = self.validate_physics(params)
        return params, r_final, penalty

    @abstractmethod
    def encode(self, x: Tensor) -> Tensor:
        """Subclasses must implement the encoder network"""
        pass

    def forward(self, r_obs_log: Tensor, fixed_values: ParamSet | None = None) -> tuple[ParamSet, Tensor, Tensor]:
        if torch.isnan(r_obs_log).any():
            r_obs_log = torch.nan_to_num(r_obs_log, nan=-10.0)
        
        raw_output = self.encode(r_obs_log)
        params = self.unnormalize(torch.sigmoid(raw_output), fixed_values)
        
        r_final = self.simulate(params)
        penalty = self.validate_physics(params)
        
        return params, r_final, penalty



class FourierConvEncoder(nn.Module):
    """
    Ported XRRPhysicsModel Encoder logic.
    Structure: [Conv1d(7) -> BN -> Leaky -> Drop -> MaxPool(2)] x Depth
    """
    def __init__(self, config: XRefineConfig, output_dim: int):
        super().__init__()

        # Hyperparameter Mapping
        q_len = config.q_len
        n_channels = config.model.hidden_dim        
        depth = config.model.encoder_depth          
        mlp_hidden = config.model.dim_feedforward   
        dropout = config.model.dropout

        # Fourier Settings
        mapping_size = config.model.mapping_size
        use_fourier = mapping_size > 0
        fourier_scale = config.model.fourier_scale

        self.use_fourier = use_fourier

        # 1. Fourier Feature Mapping
        if use_fourier:
            self.fourier = FourierFeatureMapping(
                mapping_size=mapping_size,
                scale=fourier_scale,
                output_format='cnn' # Returns (B, 2M, L)
            )
            # Legacy: Input(1) + Fourier(2M)
            in_channels = 1 + (mapping_size * 2)
        else:
            in_channels = 1

        # 2. CNN Encoder
        layers = []
        curr_dim = in_channels

        for i in range(depth):
            # Channel scaling strategy
            out_dim = n_channels * (2 ** min(i, 3))

            layers.append(nn.Sequential(
                nn.Conv1d(curr_dim, out_dim, kernel_size=7, padding=3, bias=False),
                nn.BatchNorm1d(out_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(dropout),
                nn.MaxPool1d(kernel_size=2)
            ))
            curr_dim = out_dim

        # [FIX 1] 변수명 통일 (self.encoder -> self.backbone)
        self.backbone = nn.Sequential(*layers)

        # 3. Global Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # 4. Regressor (MLP)
        # [FIX 2] 입력 차원 수정: Global Stats(3)가 Concat되므로 +3 추가
        self.regressor = nn.Sequential(
            nn.Linear(curr_dim + 3, mlp_hidden), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(mlp_hidden // 2, output_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, Q_Len) or (B, 1, Q_Len)
        Returns:
            (B, Output_Dim)
        """
        # (B, L) -> (B, 1, L)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # [FIX 2 Related] Global Hint Extraction
        # 1. Max Intensity
        stat_max = x.max(dim=-1)[0] # (B, 1)
        
        # 2. Low-Q Mean
        stat_low_q = x[..., :50].mean(dim=-1) # (B, 1)
        
        # 3. Std Dev
        stat_std = x.std(dim=-1) # (B, 1)
        
        # 합치기: (B, 3)
        global_stats = torch.cat([stat_max, stat_low_q, stat_std], dim=1)

        # Intensity Normalization (-1 ~ 1)
        x_norm = (torch.clamp(x, min=LOG_MASK_VALUE, max=0.0) - LOG_MASK_VALUE/2) / (-LOG_MASK_VALUE/2)

        if self.use_fourier:
            fourier_feat = self.fourier(x_norm)
            x_in = torch.cat([x_norm, fourier_feat], dim=1)
        else:
            x_in = x_norm
            
        # CNN Backbone
        # [FIX 1 Related] 이제 self.backbone이 정의되어 있으므로 에러 없음
        feat = self.backbone(x_in) # (B, C, L_pooled)
        feat = self.global_pool(feat).flatten(1) # (B, C) -> (B, Hidden_Dim)

        # [FIX 3] Feature Fusion (Conv Feature + Global Hint)
        # (B, Hidden) + (B, 3) -> (B, Hidden + 3)
        feat_enriched = torch.cat([feat, global_stats], dim=1)

        return self.regressor(feat_enriched)


class FourierConvPINN(BasePINNModel):
    """
    Legacy CNN Implementation (Reference: Exp07)
    """
    def __init__(self, config: XRefineConfig):
        super().__init__(config)
        self.encoder_net = FourierConvEncoder(config, self.output_dim)

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder_net(x)