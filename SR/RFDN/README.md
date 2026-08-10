# RFDN Video Upscaler

RFDN 기반 비디오 업스케일링 파이프라인.

## Setup

1. RFDN 원본 저장소 클론:
   ```bash
   git clone https://github.com/njulj/RFDN.git

2. 종속성 설치:

   ```bash
   pip install -r requirements.txt
   

3. 실행:
   best_model.pth 파일 다운로드 (one drive/RFDN/가중치 폴더) 후 프로젝트 폴더 내 위치

   input.mp4 파일 준비

   추론 실행:

   ```bash
   python inference.py
