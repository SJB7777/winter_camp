"""
Physics informed CNN Model (Updated Normalization)
==========================================
CNN with Fourier Features
"""
import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor

from src.abeles import AbelesMatrix
from src.smearing import abeles_constant_smearing
from src.param import ParamSet
from src.constants import LOG_MASK_VALUE
from src.config import XRefineConfig


class BasePINNModel(nn.Module, ABC):
    """
    PINN 모델의 공통 인터페이스 (Dynamic Refactor).
    """
    def __init__(self, config: XRefineConfig):
        super().__init__()
        self.config = config
        self.simulator = AbelesMatrix(device=config.device)
        self.register_buffer('q_grid', config.q_grid)
        
        # [REFAC] Calculate output dimension based on Dynamic Layer Config
        self.param_map = self._build_param_map(config)
        self.output_dim = len(self.param_map)

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
            
            # [FIX] sld_imag (Absorption) 추가 확인
            if getattr(layer, 'sld_imag', None) is not None:
                mapping.append((f"{layer.name}.sld_imag", layer.sld_imag))

        # 2. Substrate
        mapping.append(("Substrate.roughness", config.sample.substrate.roughness))
        mapping.append(("Substrate.sld", config.sample.substrate.sld))
        
        # [FIX] Substrate sld_imag 추가 확인
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
        Constructs a DynamicParamSet.

        [Feature] Constraint Learning:
        If 'fixed_values' is provided, parameters present in it will OVERWRITE 
        the network predictions. This cuts the gradient for those parameters, 
        treating them as constants (Ground Truth).
        """
        batch_size = raw_output.size(0)

        # Split output into chunks of 1 (since all scalars)
        vals = torch.split(raw_output, 1, dim=1)

        params_dict = {}

        # 1. Base: Convert Network Predictions to Physical Values
        for (key, (p_min, p_max)), val in zip(self.param_map, vals):
            # val: (B, 1) Normalized 0~1
            physical_val = val * (p_max - p_min) + p_min
            params_dict[key] = physical_val

        # 2. Override: Apply Fixed Value Constraints
        if fixed_values is not None:
            # DynamicParamSet 내부 딕셔너리 순회
            for key, target_val in fixed_values._params.items():
                if key in params_dict:
                    # 배치 사이즈 브로드캐스팅 (1 -> B)
                    if target_val.size(0) == 1 and batch_size > 1:
                        target_val = target_val.expand(batch_size, -1)

                    # [Core Logic] 예측값 덮어쓰기
                    # 이 시점에서 해당 파라미터는 신경망의 Computational Graph에서 분리됩니다.
                    params_dict[key] = target_val

        # Order of layer names for assembly
        layer_names = [l.name for l in self.config.sample.layers]

        return ParamSet(params_dict, layer_names, device=self.device)

    def validate_physics(self, params: ParamSet) -> Tensor:
        """
        Dynamic physics constraints.
        Iterate over all layers and check roughness < 0.5 * thickness.
        """
        penalty = torch.tensor(0.0, device=self.device)
        
        for name in params.layer_names:
            d = params[f"{name}.thickness"]
            sigma = params[f"{name}.roughness"]
            
            # ReLU(sigma - 0.5*d)
            curr_penalty = torch.mean(torch.relu(sigma - 0.5 * d))
            penalty = penalty + curr_penalty
            
        return penalty

    def simulate(self, params: ParamSet) -> Tensor:
        """
        ParamSet -> Reflectivity
        """
        # Assemble big tensors
        thickness, roughness, sld_full = params.assemble_structure()
        batch_size = thickness.size(0)
        
        # Q Grid
        q_batch = self.q_grid.unsqueeze(0).expand(batch_size, -1).to(self.device)
        
        # Tth Offset
        if 'tth_offset' in params._params: # Direct check
            q_batch = q_batch + params.tth_offset * 0.01
        
        # Instrument Params
        L = params.L
        beam_width = params.beam_width
        
        # Resolution Smearing Check
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
        
        # I0 / Bkg
        i0 = params.i0
        bkg = params.bkg
        
        r_final = (i0 * r_sim) + torch.pow(10.0, bkg)
        
        return torch.clamp(r_final, min=1e-15, max=2.0)

    def forward_with_params(self, params: ParamSet) -> tuple[ParamSet, Tensor, Tensor]:
        params = params.to(self.device)
        r_final = self.simulate(params)
        penalty = self.validate_physics(params)
        return params, r_final, penalty

    @abstractmethod
    def encode(self, x: Tensor) -> Tensor:
        pass

    def forward(self, r_obs_log: Tensor, fixed_values: ParamSet | None = None) -> tuple[ParamSet, Tensor, any]:
        if torch.isnan(r_obs_log).any():
            r_obs_log = torch.nan_to_num(r_obs_log, nan=-10.0)
        
        raw_output = self.encode(r_obs_log)
        
        # Pass fixed_values if needed (currently implementation ignores it but keeps signature)
        params = self.unnormalize(torch.sigmoid(raw_output), fixed_values)
        
        r_final = self.simulate(params)
        penalty = self.validate_physics(params)
        
        return params, r_final, penalty

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
            if x.dim() == 2:
                x = x.unsqueeze(1)
            x_proj = x.transpose(1, 2) @ self.B
            x_proj = x_proj * 2 * math.pi
            out = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
            return out.transpose(1, 2)
        else:
            if x.dim() == 2:
                x = x.unsqueeze(-1)
            x_proj = x @ self.B
            x_proj = x_proj * 2 * math.pi
            return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class FourierConvEncoder(nn.Module):
    """
    Ported XRRPhysicsModel Encoder logic.
    Structure: [Conv1d(7) -> BN -> Leaky -> Drop -> MaxPool(2)] x Depth
    """
    def __init__(self, config: XRefineConfig, output_dim: int):
        super().__init__()

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

        self.encoder = nn.Sequential(*layers)

        # 3. Global Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # 4. Regressor (MLP)
        self.regressor = nn.Sequential(
            nn.Linear(curr_dim, mlp_hidden),
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
            x: (B, Q_Len)
        Returns:
            (B, Output_Dim)
        """
        # (B, L) -> (B, 1, L)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # [FIX] Intensity Normalization (Sync with ViT logic)
        # LOG_MASK_VALUE(-15.0) ~ 0.0 범위를 -1.0 ~ 1.0 (또는 0~1)로 정규화
        # 여기서는 -1 ~ 1 범위로 맞춥니다.
        x_norm = (torch.clamp(x, min=LOG_MASK_VALUE, max=0.0) - LOG_MASK_VALUE/2) / (-LOG_MASK_VALUE/2)

        if self.use_fourier:
            # Fourier returns (B, 2M, L)
            fourier_feat = self.fourier(x_norm)
            # Concatenate along channel dim: (B, 1+2M, L)
            x_in = torch.cat([x_norm, fourier_feat], dim=1)
        else:
            x_in = x_norm

        feat = self.encoder(x_in) # (B, C_last, L_pooled)
        feat = self.global_pool(feat).squeeze(-1) # (B, C_last)

        return self.regressor(feat)


class FourierConvPINN(BasePINNModel):
    """
    Legacy CNN Implementation (Reference: Exp07)
    """
    def __init__(self, config: XRefineConfig):
        super().__init__(config)
        self.encoder_net = FourierConvEncoder(config, self.output_dim)

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder_net(x)