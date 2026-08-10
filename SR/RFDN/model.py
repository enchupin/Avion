import sys
import os
import torch
# RFDN 폴더 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'RFDN'))
from RFDN import RFDN

def load_model(weights_path, device='cpu'):
    # RFDN 모델 초기화
    model = RFDN(in_nc=3, nf=50, num_modules=4, out_nc=3, upscale=4).to(device)
    
    # 가중치 로드
    if weights_path:
        model.load_state_dict(torch.load(weights_path, map_location=device))
    
    model.eval()
    return model