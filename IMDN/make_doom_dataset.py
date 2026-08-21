# make_doom_dataset.py
from PIL import Image
import os

# 설정
src_folder = "D:/졸작 백업/data/DOOM/raw"
output_folders = {
    "D:/졸작 백업/data/DOOM/doom_320": (320, 180),    # LR (원본 유사)
    "D:/졸작 백업/data/DOOM/doom_640": (640, 360),    # HR x2
    "D:/졸작 백업/data/DOOM/doom_960": (960, 540),    # HR x3
}

# 출력 폴더 생성
for folder in output_folders:
    os.makedirs(folder, exist_ok=True)

# 변환
files = [f for f in os.listdir(src_folder) if f.endswith(('.png', '.jpg'))]
total = len(files)

for i, filename in enumerate(files, 1):
    src_path = os.path.join(src_folder, filename)
    img = Image.open(src_path)

    for folder, (w, h) in output_folders.items():
        resized = img.resize((w, h), Image.BICUBIC)
        resized.save(os.path.join(folder, filename))

    print(f"\r진행률: {i}/{total}", end="")

print("\n완료!")