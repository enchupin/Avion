# video_sr.py
import cv2
import torch
import numpy as np
import time
from model import architecture
import utils
import os

# ========== 설정 ==========
INPUT_VIDEO = 'downscale_result.mp4'
OUTPUT_VIDEO = 'output_x2_bech32_epoch50.mp4'
CHECKPOINT = 'checkpoint_x2_b32/epoch_50.pth'
SCALE = 2
DATASET = 'Apex 직접 수집 (50장)'
EPOCHS = 50
# ==========================

# 모델 로드
model = architecture.IMDN(upscale=SCALE)
model.load_state_dict(utils.load_state_dict(CHECKPOINT))
model.eval().cuda()

# 영상 읽기
cap = cv2.VideoCapture(INPUT_VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 출력 영상 설정
out = cv2.VideoWriter(OUTPUT_VIDEO,
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps, (w*SCALE, h*SCALE))

# 시간 측정 리스트
frame_times = []
total_start = time.time()

current = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 전처리
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).cuda()

    # 프레임당 처리 시간 측정
    frame_start = time.time()
    with torch.no_grad():
        sr = model(img)
    torch.cuda.synchronize()
    frame_end = time.time()
    frame_times.append((frame_end - frame_start) * 1000)

    # 후처리
    sr_img = utils.tensor2np(sr.detach()[0])
    sr_img = cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR)

    out.write(sr_img)
    current += 1
    print(f"\r진행률: {current}/{frame_count} 프레임 | 현재 프레임 처리 시간: {frame_times[-1]:.2f}ms", end="")

total_end = time.time()
total_time = total_end - total_start

cap.release()
out.release()

# 결과 출력
print("\n")
print("=" * 50)
print(f"모델:               IMDN")
print(f"데이터셋:           {DATASET}")
print(f"파인튜닝 에폭:      {EPOCHS}")
print(f"배치 사이즈:        8")
print(f"학습률:             2e-4")
print(f"손실 함수:          L1 Loss")
print(f"옵티마이저:         Adam")
print(f"스케일:             x{SCALE}")
print(f"원본 해상도:        {w}x{h}")
print(f"결과 해상도:        {w*SCALE}x{h*SCALE}")
print(f"총 프레임 수:       {frame_count}")
print(f"프레임당 처리 시간: {np.mean(frame_times):.2f}ms (평균)")
print(f"전체 처리 시간:     {total_time:.2f}초 ({total_time/60:.2f}분)")
print(f"결과 영상:          {OUTPUT_VIDEO}")
print("=" * 50)