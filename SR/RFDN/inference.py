import torch
import cv2
import numpy as np
import os
import glob
import argparse
from model import load_model

def get_first_image_in_dir(directory):
    exts = ('*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG', '*.bmp')
    candidates = []
    for ext in exts:
        candidates.extend(glob.glob(os.path.join(directory, ext)))
    # 결과 이미지 제외
    candidates = [p for p in candidates if not os.path.basename(p).startswith("output")]
    if not candidates:
        raise FileNotFoundError(f" '{directory}' 폴더 내에 처리할 이미지 파일이 없습니다.")
    return sorted(candidates)[0]

def run_image_upscale(weights_path, input_image_path=None, output_image_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. 가중치 경로 결정
    if not os.path.isabs(weights_path):
        weights_path = os.path.join(base_dir, weights_path)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"가중치 파일을 찾을 수 없습니다: {weights_path}")

    # 2. 이미지 결정 
    if input_image_path is None or not os.path.exists(input_image_path):
        input_image_path = get_first_image_in_dir(base_dir)
       

    if output_image_path is None:
        output_image_path = os.path.join(base_dir, "output.png")

    # 3. 모델 로드
    model = load_model(weights_path, device=device)

    # 4. 이미지 로드 및 전처리
    lr_bgr = cv2.imread(input_image_path)
    if lr_bgr is None:
        raise ValueError(f"❌ 이미지를 열 수 없습니다: {input_image_path}")

    lr_rgb = cv2.cvtColor(lr_bgr, cv2.COLOR_BGR2RGB)
    h, w = lr_rgb.shape[:2]

    lr_tensor = torch.from_numpy(lr_rgb).permute(2, 0, 1).float() / 255.0
    lr_tensor = lr_tensor.unsqueeze(0).to(device)

   
    # 5. 모델 추론
    with torch.no_grad():
        sr_tensor = model(lr_tensor)

    # 6. 후처리
    sr_rgb = sr_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    sr_rgb = np.clip(sr_rgb, 0.0, 1.0)
    sr_bgr = cv2.cvtColor((sr_rgb * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)

    cv2.imwrite(output_image_path, sr_bgr)
    print(f"업스케일 완료! 결과 저장: {output_image_path} (확대 해상도: {sr_bgr.shape[1]}x{sr_bgr.shape[0]})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RFDN Auto-detect Image Upscaler")
    parser.add_argument('--weights', type=str, default='best_model.pth', help='가중치 파일 경로')
    parser.add_argument('--input', type=str, default=None, help='입력 이미지 경로')
    parser.add_argument('--output', type=str, default=None, help='출력 이미지 경로')

    args = parser.parse_args()
    run_image_upscale(args.weights, args.input, args.output)