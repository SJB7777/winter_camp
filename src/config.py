import torch
import yaml
from pathlib import Path
from typing import Literal, Self, Tuple, Optional, List, Annotated
from pydantic import BaseModel, Field, ConfigDict, field_validator, field_serializer, PrivateAttr

from src.physics_utils import tth2q

# Helper Type for Ranges [min, max]
Range = Annotated[list[float], Field(min_length=2, max_length=2)]

# ==============================================================================
# 1. Physical Structure Configuration (Dynamic Layers)
# ==============================================================================

class MaterialConfig(BaseModel):
    """Configuration for a single material property (SLD, Roughness)."""
    sld: Range
    # [FIX] Mandatory: Must be explicitly defined in YAML
    sld_imag: Range 
    roughness: Range

class LayerConfig(BaseModel):
    """Configuration for a single layer in the stack."""
    name: str
    thickness: Range
    roughness: Range
    sld: Range
    # [FIX] Mandatory: Optional제거 -> YAML에 없으면 에러 발생 (안전장치)
    sld_imag: Range 
    repeats: int = Field(default=1, ge=1, description="Number of times this layer repeats")

class SubstrateConfig(BaseModel):
    name: str = "Substrate"
    roughness: Range
    sld: Range
    # [FIX] Mandatory for strict complex simulation
    sld_imag: Range 

class AmbientConfig(BaseModel):
    name: str = "Ambient"
    sld: float = 0.0 # Usually fixed 0 for air
    # Ambient is usually transparent (0), but keeping structure consistent
    sld_imag: Range = Field(default=[0.0, 0.0]) 

class SampleConfig(BaseModel):
    """
    Complete sample definition replacing the old flat PhysicalRangeConfig.
    """
    ambient: AmbientConfig = Field(default_factory=AmbientConfig)
    substrate: SubstrateConfig
    layers: list[LayerConfig]
    
    # Global/Instrumental parameters that vary per sample
    L: Range = Field(default=[5.0, 25.0], description="Sample Length [mm]")
    beam_width: Range = Field(default=[0.01, 0.3], description="Beam Width [mm]")
    bkg: Range = Field(default=[-9.0, -5.0], description="Background Log10")
    i0: Range = Field(default=[0.8, 1.2], description="Intensity Scale")
    dq: Range = Field(default=[0.0005, 0.01], description="Resolution [A^-1]")
    tth_offset: Range = Field(default=[-0.05, 0.05], description="TwoTheta Offset [deg]")

# ==============================================================================
# 2. Existing Configs (Refined)
# ==============================================================================

class InstrumentConfig(BaseModel):
    wavelength: float = 1.540606
    beam_width: float = 0.1
    res: float = 0.002
    I0: float = 1.0
    Ibkg_init: float = 1e-7

class AugmentationConfig(BaseModel):
    intensity_noise_scale: float = 0.2
    bg_range: Range = [-9.0, -5.0]
    res_sigma_range: Range = [0.0001, 0.005]
    q_shift_sigma: float = 0.001
    beam_profile: Literal['gaussian', 'square'] = 'gaussian'
    measure_step_range: Range = [0.001, 0.1]

class ModelArchConfig(BaseModel):
    model_type: Literal['fourier_conv', 'fourier_vit', 'latent_trans'] = 'fourier_conv'
    hidden_dim: int = 128
    encoder_depth: int = 4
    
    # CNN specific
    kernel_size: int = 15
    dilation: int = 2
    fourier_scale: float = 40.0
    mapping_size: int = 0 
    
    # Transformer/ViT
    n_heads: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    activation: str = 'gelu'
    
    # ViT specific
    patch_size: int = 16

class WeightConfig(BaseModel):
    global_weights: dict[str, float] = Field(
        default={
            "L": 5.0, "i0": 1.0, "bkg": 1.0, "beam_width": 10.0,
            "dq": 1.0, "tth_offset": 5.0
        }
    )
    layer_weights: dict[str, float] = Field(default_factory=dict)
    specific_layer_weights: dict[str, dict[str, float]] = Field(default_factory=dict)

class TrainingConfig(BaseModel):
    batch_size: int = 64
    lr: float = 5e-4
    weight_decay: float = 1e-4
    epochs: int = 300
    steps_per_epoch: int = 100
    warmup_steps: int = 1000
    early_stop_patience: int = 50
    
    sup_w_start: float = 100.0
    sup_w_end: float = 2000.0
    phys_penalty_w: float = 10.0
    
    w_linear: float = 100.0
    w_gradient: float = 50.0
    w_log_base: float = 10.0
    
    weights: WeightConfig = Field(default_factory=WeightConfig)

# ==============================================================================
# 3. Root Configuration
# ==============================================================================

class XRefineConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='ignore')

    experiment_name: str = 'XRefine_NextGen'
    save_root: Path = Path("./data")
    log_dir: Path = Path("./logs")
    
    device: torch.device = Field(
        default_factory=lambda: torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )

    q_min_tth: float = 0.05
    q_max_tth: float = 12.0
    q_len: int = 2000

    instrument: InstrumentConfig = Field(default_factory=InstrumentConfig)
    sample: SampleConfig = Field(description="Dynamic Layer Configuration")
    augmentations: AugmentationConfig = Field(default_factory=AugmentationConfig)
    train: TrainingConfig = Field(default_factory=TrainingConfig)
    model: ModelArchConfig = Field(default_factory=ModelArchConfig)

    _q_grid_override: Optional[torch.Tensor] = PrivateAttr(default=None)

    @field_validator('device', mode='before')
    def parse_device(cls, v):
        if isinstance(v, str):
            return torch.device(v)
        return v

    @field_serializer('device')
    def serialize_device(self, device: torch.device, _info):
        return str(device)

    @property
    def q_grid(self) -> torch.Tensor:
        if self._q_grid_override is not None:
            return self._q_grid_override
        q_start = tth2q(self.q_min_tth, self.instrument.wavelength)
        q_end = tth2q(self.q_max_tth, self.instrument.wavelength)
        return torch.linspace(q_start, q_end, self.q_len)

    @q_grid.setter
    def q_grid(self, value: torch.Tensor):
        self._q_grid_override = value

    def save_yaml(self, path: str | Path):
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        data = self.model_dump(mode='json')
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, sort_keys=False)

    @classmethod
    def load_yaml(cls, path: str | Path) -> Self:
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls(**data)

# Dummy Default for strict testing
DEFAULT_SAMPLE = SampleConfig(
    substrate=SubstrateConfig(
        roughness=[0.5, 5.0], 
        sld=[19.0, 21.0], 
        sld_imag=[0.0, 0.5] # Explicitly added
    ),
    layers=[
        LayerConfig(
            name="Film", 
            thickness=[100.0, 500.0], 
            roughness=[1.0, 5.0], 
            sld=[20.0, 60.0],
            sld_imag=[0.0, 2.0] # Explicitly added
        )
    ]
)

CONFIG = XRefineConfig(sample=DEFAULT_SAMPLE)