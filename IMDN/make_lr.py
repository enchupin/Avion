# make_lr.py
from PIL import Image
import os

hr_folder = "dataset/HR"   # 고화질 이미지 폴더
lr_folder = "dataset/LR"   # LR 저장 폴더
scale = 2

os.makedirs(lr_folder, exist_ok=True)

for filename in os.listdir(hr_folder):
    if filename.endswith(('.png', '.jpg')):
        hr = Image.open(os.path.join(hr_folder, filename))
        w, h = hr.size
        lr = hr.resize((w//scale, h//scale), Image.BICUBIC)
        lr.save(os.path.join(lr_folder, filename))
        print(f"생성 완료: {filename} ({w}x{h} → {w//scale}x{h//scale})")

print("전체 완료!")