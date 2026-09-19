import torch
import cv2
import numpy as np
import os
import glob
import argparse
from model import load_model

def get_first_image_in_dir(directory):
    # 지원하는 이미지 확장자 목록
    exts = ('*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG', '*.bmp')
    candidates = []
    for ext in exts:
        candidates.extend(glob.glob(os.path.join(directory, ext)))
    
    # 생성된 결과물(output.png 등)은 입력 후보에서 제외
    candidates = [p for p in candidates if not os.path.basename(p).startswith("output")]
    
    if not candidates:
        raise FileNotFoundError(f" '{directory}' 폴더 내에 처리할 이미지 파일(.png, .jpg 등)이 없습니다.")
    
    # 첫 번째 이미지 파일 정렬 후 반환
    return sorted(candidates)[0]

def run_image_upscale(weights_path, input_image_path=None, output_image_path=None, scale_factor=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. 가중치 파일 검증 (best_model.pth 고정)
    if not os.path.isabs(weights_path):
        weights_path = os.path.join(base_dir, weights_path)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"가중치 파일을 찾을 수 없습니다: {weights_path}")

    # 2. 이미지 파일 자동 감지 (파일명과 상관없이 첫 번째 이미지 선택)
    if input_image_path is None or not os.path.exists(input_image_path):
        input_image_path = get_first_image_in_dir(base_dir)
      

    # 3. 출력 경로 설정 (기본: output.png)
    if output_image_path is None:
        output_image_path = os.path.join(base_dir, "output.png")

    # 4. 이미지 로드
    img_bgr = cv2.imread(input_image_path)
    if img_bgr is None:
        raise ValueError(f" 입력 이미지를 읽을 수 없습니다: {input_image_path}")

    # 5. 모델 로드
    model = load_model(weights_path, scale_factor=scale_factor, device=device)

    # 6. 전처리 (BGR -> RGB 변환 및 정규화)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    input_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
    
   

    # 7. 모델 추론
    with torch.no_grad():
        sr_tensor = model(input_tensor)

    # 8. 후처리 및 저장
    sr_rgb = sr_tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.0
    sr_bgr = cv2.cvtColor(np.round(sr_rgb).astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(output_image_path, sr_bgr)
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FSRCNN Auto-detect Image Upscaler")
    parser.add_argument('--weights', type=str, default='best_model.pth', help='가중치 파일 경로 (기본: best_model.pth)')
    parser.add_argument('--input', type=str, default=None, help='입력 이미지 경로 (미입력 시 폴더 내 이미지 자동 인식)')
    parser.add_argument('--output', type=str, default=None, help='출력 이미지 경로 (기본: output.png)')
    parser.add_argument('--scale', type=int, default=4, help='업스케일 배율 (기본: 4)')

    args = parser.parse_args()
    run_image_upscale(args.weights, args.input, args.output, scale_factor=args.scale)