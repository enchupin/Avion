import cv2
import torch
import numpy as np
import time
from model import architecture
import utils
import os
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# ========== 설정 ==========
LR_FOLDER     = "D:/joljack_backup/data/DOOM/LR"
HR_FOLDER     = "D:/joljack_backup/data/DOOM/HR"
OUTPUT_FOLDER = "D:/joljack_backup/data/DOOM/SR/x2_b32_e30"
CHECKPOINT    = "checkpoint_x2_b32/epoch_30.pth"
SCALE         = 2
DATASET       = "DOOM 1542장"
EPOCHS        = 30
NUM_IMAGES    = 10   # 처리할 이미지 수
# ==========================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Y채널 변환 (IMDN 논문 방식)
def rgb2y(img):
    img = img.astype(np.float64)
    y = 16.0 + (65.481*img[:,:,0] + 128.553*img[:,:,1] + 24.966*img[:,:,2]) / 255.0
    return y

# 모델 로드
model = architecture.IMDN(upscale=SCALE)
model.load_state_dict(utils.load_state_dict(CHECKPOINT))
model.eval().cuda()

# 이미지 목록 (5장만)
files = sorted([f for f in os.listdir(LR_FOLDER) if f.endswith(('.png', '.jpg'))])[:NUM_IMAGES]

frame_times = []
psnr_y_list = []
ssim_list   = []
mse_list    = []
mae_list    = []

print(f"{'파일명':<25} {'PSNR_Y':>8} {'SSIM':>8} {'MSE':>10} {'MAE':>8} {'시간(ms)':>10}")
print("-" * 75)

total_start = time.time()

for filename in files:
    lr_path = os.path.join(LR_FOLDER, filename)
    hr_path = os.path.join(HR_FOLDER, filename)

    # LR 읽기 및 전처리
    img = cv2.imread(lr_path, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_f = img_rgb.astype(np.float32) / 255.0
    img_t = torch.from_numpy(np.transpose(img_f, (2, 0, 1))).unsqueeze(0).cuda()

    # 추론
    frame_start = time.time()
    with torch.no_grad():
        sr = model(img_t)
    torch.cuda.synchronize()
    elapsed = (time.time() - frame_start) * 1000
    frame_times.append(elapsed)

    # 후처리
    sr_img = utils.tensor2np(sr.detach()[0])  # RGB uint8
    sr_img_bgr = cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, filename), sr_img_bgr)

    # HR 읽기 및 지표 계산
    if os.path.exists(hr_path):
        hr_img = cv2.imread(hr_path, cv2.IMREAD_COLOR)
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
        hr_img = cv2.resize(hr_img, (w*SCALE, h*SCALE))

        # ① PSNR Y채널 (IMDN 논문 방식)
        hr_y = rgb2y(hr_img)
        sr_y = rgb2y(sr_img)
        mse_y = np.mean((hr_y - sr_y) ** 2)
        p_y = 10 * np.log10(255.0 ** 2 / mse_y)

        # ② SSIM
        s = structural_similarity(hr_img, sr_img, channel_axis=-1, data_range=255)

        # ③ MSE
        mse_v = np.mean((hr_img.astype(np.float64) - sr_img.astype(np.float64)) ** 2)

        # ④ MAE (L1 Loss와 동일)
        mae_v = np.mean(np.abs(hr_img.astype(np.float64) - sr_img.astype(np.float64)))

        psnr_y_list.append(p_y)
        ssim_list.append(s)
        mse_list.append(mse_v)
        mae_list.append(mae_v)

        print(f"{filename:<25} {p_y:>8.4f} {s:>8.4f} {mse_v:>10.4f} {mae_v:>8.4f} {elapsed:>10.2f}")

total_time = time.time() - total_start

print("-" * 75)
print(f"{'평균':<25} {np.mean(psnr_y_list):>8.4f} {np.mean(ssim_list):>8.4f} {np.mean(mse_list):>10.4f} {np.mean(mae_list):>8.4f} {np.mean(frame_times):>10.2f}")

print("\n")
print("=" * 55)
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
print(f"처리 이미지 수:     {NUM_IMAGES}장")
print(f"이미지당 처리 시간: {np.mean(frame_times):.2f}ms (평균)")
print(f"전체 처리 시간:     {total_time:.2f}초")
print(f"Mean PSNR (Y채널):  {np.mean(psnr_y_list):.4f} dB")
print(f"Mean SSIM:          {np.mean(ssim_list):.4f}")
print(f"Mean MSE:           {np.mean(mse_list):.4f}")
print(f"Mean MAE (L1):      {np.mean(mae_list):.4f}")
print(f"결과 저장 위치:     {OUTPUT_FOLDER}")
print("=" * 55)