import torch
import cv2
import numpy as np
from model import load_model

def run_upscale(weights_path, input_video_path, output_video_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 모델 로드
    model = load_model(weights_path, device)
    
    # 영상 로드
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 저장 설정 (4배 업스케일)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_sr = cv2.VideoWriter(output_video_path, fourcc, fps, (in_w * 4, in_h * 4))
    
    print("Model inference started")
    
    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # 전처리
            input_tensor = torch.from_numpy(frame).permute(2,0,1).float().unsqueeze(0).to(device) / 255.0
            
            # 추론
            sr_tensor = model(input_tensor).squeeze(0).permute(1,2,0).clamp(0, 1).cpu().numpy() * 255.0
            
            # 저장
            out_sr.write(sr_tensor.astype(np.uint8))
            
    cap.release()
    out_sr.release()
    print(f"Upscaling completed. Saved to: {output_video_path}")

if __name__ == "__main__":
    # 경로 설정
    WEIGHTS = 'best_model.pth' # best_model.pth 위치
    INPUT = 'input.mp4'       # 입력 영상 경로
    OUTPUT = 'output.mp4'     # 출력 영상 경로
    
    run_upscale(WEIGHTS, INPUT, OUTPUT)