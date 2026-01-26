import torch
import numpy as np
import logging
from src.config import XRefineConfig
from src.model import FourierConvPINN

logger = logging.getLogger(__name__)

def load_model_from_checkpoint(ckpt_path: str, device: torch.device):
    """
    .pt 체크포인트에서 모델과 학습 당시의 설정을 복원합니다.
    """
    if not torch.cuda.is_available():
        map_loc = 'cpu'
    else:
        map_loc = device

    logger.info(f"📂 Loading checkpoint from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=map_loc)

    # 1. Config 복원 (학습 당시의 설정이 모델 구조를 결정함)
    config_data = checkpoint.get('config')
    if isinstance(config_data, dict):
        config = XRefineConfig(**config_data)
    else:
        config = config_data # 객체째로 저장된 경우
    
    # 강제로 현재 디바이스 설정
    config.device = device 

    # 2. 모델 초기화
    model = FourierConvPINN(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, config

def predict_initial_params(
    model: FourierConvPINN, 
    q_input: torch.Tensor, 
    log_r_input: torch.Tensor,
    device: torch.device
) -> dict[str, float]:
    """
    관측 데이터(Q, R)를 모델의 입력 크기(Q_len)에 맞춰 인터폴레이션하고,
    NN을 통과시켜 초기 파라미터 딕셔너리를 반환합니다.
    """
    # 1. 모델이 학습된 Q Grid 가져오기
    # (모델은 고정된 입력 사이즈(예: 2000)를 가짐)
    model_q_grid = model.q_grid.to(device)
    
    # 2. 데이터 인터폴레이션 (관측 데이터 -> 모델 입력 포맷)
    # q_input, log_r_input은 (N,) 형태
    # np.interp를 위해 CPU로 내림
    q_np = q_input.cpu().numpy()
    r_np = log_r_input.cpu().numpy()
    target_q = model_q_grid.cpu().numpy()
    
    # 선형 보간
    interp_r = np.interp(target_q, q_np, r_np)
    
    # 3. 텐서 변환 및 정규화
    # (모델 학습 시 -15.0 ~ 0.0 범위를 주로 썼다고 가정)
    input_tensor = torch.from_numpy(interp_r).float().to(device)
    input_tensor = input_tensor.view(1, -1) # (1, Q_Len)

    # 4. 추론 (Inference)
    with torch.no_grad():
        # 모델 내부에서 unnormalize를 수행하여 ParamSet 객체 반환
        # (src/model.py의 forward 로직 활용)
        # forward 결과: (params, r_sim, penalty)
        predicted_params_set, _, _ = model(input_tensor)

    # 5. Dict 변환
    # ParamSet._params는 {key: Tensor(batch, 1)} 형태임
    result_dict = {}
    for key, tensor_val in predicted_params_set._params.items():
        result_dict[key] = float(tensor_val.item())

    return result_dict