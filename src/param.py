from __future__ import annotations
from typing import Self

from torch import Tensor
import torch

from src.config import SampleConfig

class ParamSet:
    """
    Dynamic container for learnable physical parameters.
    Replaces the hardcoded 'ParamSet' with a dictionary-based approach
    that adapts to any layer structure defined in SampleConfig.
    
    Structure:
    - layers: list of layer names (ordered)
    - params: dict[str, Tensor] -> 'Film.thickness', 'Film.roughness', etc.
    - instrument: dict[str, Tensor] -> 'i0', 'bkg', etc.
    """
    
    # Standard keys for instrument parameters
    INSTRUMENT_KEYS = {'L', 'i0', 'bkg', 'beam_width', 'dq', 'tth_offset'}

    def __init__(self, 
                 params: dict[str, Tensor], 
                 layer_names: list[str],
                 device: torch.device | None = None):
        self._params = params
        self.layer_names = layer_names # Order matters for assembly!
        self.device = device or (next(iter(params.values())).device if params else torch.device('cpu'))

    @classmethod
    def from_config(cls, config: SampleConfig, batch_size: int = 1, device: torch.device | None = None) -> Self:
        """
        Initialize parameters (randomly or centrally) based on ranges in SampleConfig.
        This creates the 'Parameter State' for a batch.
        """
        device = device or torch.device('cpu')
        params = {}
        layer_names = [l.name for l in config.layers]

        def sample_val(rng):
            # Uniform sampling: min + rand * (max - min)
            val = rng[0] + torch.rand(batch_size, 1, device=device) * (rng[1] - rng[0])
            return val

        # 1. Layer Parameters
        for layer in config.layers:
            params[f"{layer.name}.thickness"] = sample_val(layer.thickness)
            params[f"{layer.name}.roughness"] = sample_val(layer.roughness)
            params[f"{layer.name}.sld"] = sample_val(layer.sld)
            
            # [FIX] Handle sld_imag (Absorption)
            if hasattr(layer, 'sld_imag') and layer.sld_imag is not None:
                params[f"{layer.name}.sld_imag"] = sample_val(layer.sld_imag)

        # 2. Substrate
        params["Substrate.roughness"] = sample_val(config.substrate.roughness)
        params["Substrate.sld"] = sample_val(config.substrate.sld)
        
        # [FIX] Handle Substrate sld_imag
        if hasattr(config.substrate, 'sld_imag') and config.substrate.sld_imag is not None:
            params["Substrate.sld_imag"] = sample_val(config.substrate.sld_imag)

        # 3. Ambient (Usually fixed, but ready for extension)
        
        # 4. Instrument/Global Parameters
        for key in cls.INSTRUMENT_KEYS:
            if hasattr(config, key):
                rng = getattr(config, key)
                # Pydantic model might return None if optional, check it
                if rng is not None:
                    params[key] = sample_val(rng)

        return cls(params, layer_names, device)

    def __getattr__(self, name: str) -> Tensor:
        """Allow dot access: p.i0, p['Film.thickness']"""
        if name in self._params:
            return self._params[name]
        # Fallback for old code: d_f -> Film.thickness? 
        # For now, strict access only.
        raise AttributeError(f"'DynamicParamSet' has no attribute '{name}'")
    
    def __getitem__(self, key: str) -> Tensor:
        return self._params[key]

    def __setitem__(self, key: str, value: Tensor):
        self._params[key] = value

    @classmethod
    def cat(cls, ps_list: list[ParamSet]) -> ParamSet:
        """Concatenate a list of ParamSets along the batch dimension."""
        if not ps_list:
            raise ValueError("Empty ParamSet list")
        
        layer_names = ps_list[0].layer_names
        device = ps_list[0].device
        
        merged_params = {}
        keys = ps_list[0]._params.keys()
        
        for key in keys:
            tensors = [ps[key] for ps in ps_list]
            merged_params[key] = torch.cat(tensors, dim=0)
            
        return cls(merged_params, layer_names, device)

    def to(self, device: torch.device) -> Self:
        """Move all parameter tensors to the specified device."""
        self.device = device
        self._params = {k: v.to(device) for k, v in self._params.items()}
        return self

    def detach(self) -> Self:
        return self.__class__(
            {k: v.detach() for k, v in self._params.items()},
            self.layer_names,
            self.device
        )
    
    def clone(self) -> Self:
        return self.__class__(
            {k: v.clone() for k, v in self._params.items()},
            self.layer_names,
            self.device
        )
        
    def requires_grad_(self, mode: bool = True) -> Self:
        for v in self._params.values():
            v.requires_grad_(mode)
        return self

    def assemble_structure(self) -> tuple[Tensor, Tensor, Tensor]:
        """
        Construct the monolithic tensors required by the Abeles engine.
        Now supports Complex SLD (Absorption).
        """
        # Batch size check
        first_key = next(iter(self._params))
        batch_size = self._params[first_key].shape[0]

        # 1. Thickness
        t_amb = torch.zeros((batch_size, 1), device=self.device)
        t_layers = [self._params[f"{name}.thickness"] for name in self.layer_names]
        thickness = torch.cat([t_amb] + t_layers, dim=1) 

        # 2. Roughness
        r_layers = [self._params[f"{name}.roughness"] for name in self.layer_names]
        r_sub = self._params["Substrate.roughness"]
        roughness = torch.cat(r_layers + [r_sub], dim=1) 

        # 3. SLD (Complex Support)
        # Ambient is 0 (Air), assumed real=0, imag=0
        # If we wanted complex ambient, we'd need config support, but Air is usually 0+0j.
        # We start with a complex tensor for ambient.
        sld_amb = torch.zeros((batch_size, 1), device=self.device, dtype=torch.complex128)
        
        sld_list = []
        all_layers = self.layer_names + ["Substrate"]
        
        for name in all_layers:
            # Real Part
            rho_real = self._params[f"{name}.sld"]
            
            # Imaginary Part (Absorption) Check
            imag_key = f"{name}.sld_imag"
            if imag_key in self._params:
                rho_imag = self._params[imag_key]
                # Combine: Real + 1j * Imag
                # Ensure double precision for compatibility with new AbelesMatrix
                sld_val = rho_real.double() + 1j * rho_imag.double()
            else:
                # If no imag part, treat as real (0j)
                sld_val = rho_real.double() + 0j
            
            sld_list.append(sld_val)

        # Concatenate: Ambient + Layers + Substrate
        # Result shape: (B, N+2), dtype=complex128
        sld = torch.cat([sld_amb] + sld_list, dim=1)

        return thickness, roughness, sld