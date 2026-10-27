# Avion

Electron control surface with a native C++ Windows Graphics Capture preview host.

## Requirements

- Windows 10 1903 or newer
- Visual Studio 2022 Build Tools
- Desktop development with C++ workload
- Windows 10/11 SDK

## Run

```powershell
cd electron
npm install
npm run build:native
npm start
```

Press the ON/OFF button in Electron to start or stop the native capture preview window. The native host captures the primary display through Windows Graphics Capture and renders frames through Direct3D 11.

By default, each capture writes every captured frame as lossless raw BGRA data:

- `avion-capture-YYYYMMDD-HHMMSS-001.bgra`: uncompressed BGRA frame data
- `avion-capture-YYYYMMDD-HHMMSS-001.csv`: per-frame timestamp and instantaneous FPS
- `avion-capture-YYYYMMDD-HHMMSS-001.json`: width, height, pixel format, frame count, and duration metadata

Uncompressed capture files are large. A 1920x1080 capture at 60 FPS writes about 475 MiB per second before filesystem overhead.

Enable `1 FPS PNG` before starting capture to save one lossless PNG per second instead. That mode creates a new folder for each ON/OFF capture session, such as `avion-capture-YYYYMMDD-HHMMSS-001-1fps`, with PNG frames, `frames.csv`, and `capture.json`.

## Extract Recorded Frames

Install `ffmpeg`, start the Electron app, then use `Frame Splitter` -> `Select Source`.

For normal video files, Avion analyzes the source FPS and exports PNG frames at that rate.

For raw Avion `.bgra` captures, select the `.bgra`, `.csv`, or `.json` sidecar file. Avion uses the `.json` metadata for frame size and the `.csv` timing file for the per-frame timestamps, then exports one PNG per captured BGRA frame. The output folder includes a new `frames.csv` that maps each PNG back to its original capture timestamp.

## SR 성능 평가

앱 하단 **SR 성능 평가 → 성능 측정 시작**을 누르면 저장소의 Set5 이미지와 IMDN 가중치로 실제 평가를 실행합니다. 게임 프레임 폴더 또는 기존 Avion BGRA 캡처도 선택할 수 있습니다. 같은 프레임을 Nearest, Bilinear, Bicubic, IMDN FP32로 처리하며, CUDA에서는 IMDN FP16 비교를 추가할 수 있습니다. 기본 가중치는 ×2/×3/×4 배율에 맞춰 선택합니다. 사용자 가중치는 해당 배율의 IMDN 구조와 일치해야 합니다.

Python 실행 환경에 평가 의존성을 설치하세요. CUDA 버전의 PyTorch가 필요한 환경은 [PyTorch 설치 안내](https://pytorch.org/get-started/locally/)를 따릅니다.

```powershell
python -m pip install -r SR/requirements-evaluation.txt
# Python 경로를 직접 지정해야 하는 경우, 같은 터미널에서 설정 후 앱 실행
$env:AVION_SR_PYTHON = 'C:\path\to\python.exe'
cd electron
npm start
```

화면에서 화질, 평균/P95 지연, 처리 FPS, GPU 전체 사용률을 확인하고 비교 이미지를 볼 수 있습니다. **발표용 보고서**는 표·그래프·이미지를 포함한 HTML을 엽니다. **결과 CSV**, **결과 폴더**로 원자료를 확인할 수 있습니다. 취소한 평가를 완료 결과로 저장하지 않으며, 앱을 다시 실행하면 최근 완료 결과를 불러옵니다.

개발 환경의 결과는 `output/benchmarks/bench-*/`에 저장합니다. 패키지 앱은 문서 폴더의 `Avion Benchmarks`에 저장합니다. `AVION_BENCHMARK_OUTPUT`으로 결과 위치를 변경할 수 있습니다.

- `report.html`: 이미지가 내장된 발표용 보고서
- `report.json`: 조건, 환경, 가중치/기준 프레임 해시, 프레임별 화질과 요약
- `summary.csv`: 방식별 성능 표
- `frames.csv`: 모든 반복의 프레임 처리 지연
- `gpu-samples.csv`: GPU 계측 원자료
- `preview-*.png`: 동일 프레임의 기준 영상과 처리 결과

PSNR·SSIM을 계산하기 위해 선택한 영상을 기준으로 삼고 bicubic 축소한 LR을 복원합니다. 기본 설정은 기준 영상 최대 변 512 px, 5프레임, ×2, 프레임별 워밍업 3회, 최소 10회 반복 및 방식별 최소 1초 측정입니다. RGB 0–255에서 배율만큼 테두리를 제외한 후 프레임별 지표를 평균합니다. 이 재구성 평가는 원래 저해상도 게임에 존재하지 않는 고해상도 정보를 복원했다는 증거가 아닙니다. Set5 샘플 결과와 게임 캡처 결과는 구분해 발표하세요.

지연은 메모리의 RGB 입력부터 전처리·추론·후처리를 거쳐 CPU RGB 출력까지이며, GPU 완료를 동기화합니다. 파일 I/O, 지표 계산, 모델 로딩, 캡처·IPC·화면 표시는 제외합니다. 처리 FPS는 `1000 / 평균 지연(ms)`으로 계산하므로 실제 게임 FPS나 전체 캡처·출력 지연으로 해석하면 안 됩니다. 보간 방식은 CPU, IMDN은 선택한 장치에서 실행하며 장치를 표에 표시합니다.

GPU 값은 `nvidia-smi`의 카드 전체 사용률과 메모리를 약 100 ms 간격으로 수집합니다. 다른 앱과 드라이버의 표본 구간 영향을 받습니다. CUDA 외의 실행이나 계측 미지원 시 0 대신 N/A를 표시합니다. 모델/가중치 오류가 나면 실패를 표시하며 보간으로 조용히 대체하지 않습니다.

IMDN FP16 비교는 합성곱을 FP16으로 실행하고 채널 평균·분산의 누적 합은 FP32로 계산해 오버플로를 방지합니다. 워밍업과 화질 계산에서는 출력의 NaN/Infinity를 검사하며, 비정상 출력이 있으면 보고서를 완료하지 않습니다.

### 같은 DOOM 장면으로 전체 SR / ROI SR 비교

1. **SR 성능 평가 → DOOM 장면 수집**을 누릅니다. 기존 ROI 지원 `Doom/build-local/src/Release/chocolate-doom.exe`와 `Doom/iwads/freedoom-0.13.0/freedoom-0.13.0/freedoom2.wad`를 사용합니다. 실행 파일이 없으면 [DOOM 빌드 안내](Doom/README.md)에 따라 ROI 변경이 포함된 현재 소스를 빌드하세요.
2. 열린 DOOM에서 몬스터·아이템이 보이는 곳으로 이동합니다. 게임 엔진의 깨끗한 RGB 입력과 그 프레임의 ROI 좌표를 0.5초마다 함께 저장합니다. 최대 60장 또는 2분 후 자동 종료됩니다.
3. **수집 종료·선택**을 누릅니다. Avion이 시작한 DOOM과 SR worker를 종료하고 해당 데이터의 **ROI SR 비교**를 자동 선택합니다. 게임을 직접 닫아도 저장한 데이터를 선택합니다.
4. **축소·복원 (화질 + 속도)**, ×2, CUDA를 선택하고 **성능 측정 시작**을 누릅니다. 같은 입력과 가중치로 Bicubic / 전체 SR FP32 / ROI SR FP32를 비교합니다. FP16을 켜면 전체/ROI FP16도 함께 측정합니다.
5. 실제 게임 입력 크기의 속도는 **원본 입력 (속도 · 화질 N/A)** 모드로 다시 측정하세요. 동일한 데이터가 유지됩니다. 예를 들어 원본 DOOM 320×200일 때 축소·복원은 160×100→320×200, 원본 입력은 320×200→640×400입니다. 원본 입력 모드는 최대 변 설정을 무시하며, 고해상도 정답이 없으므로 PSNR·SSIM을 계산하지 않습니다.

수집 데이터는 `output/benchmarks/doom-scenes-*/roi-manifest.json`과 무손실 PNG로 남습니다. **기존 DOOM ROI 데이터**에서 `roi-manifest.json`을 다시 선택할 수 있습니다. 일반 화면 캡처에는 엔진 ROI가 없으므로 ROI 비교를 활성화하지 않습니다. 게임 수집과 성능 측정은 동시에 실행할 수 없습니다.

ROI 비교는 실시간 worker의 `selective_upscale` 함수를 그대로 호출합니다. 입력에서 오브젝트 영역을 12 px 확장하고 겹치는 영역을 합칩니다. 배경은 Bicubic, 오브젝트는 IMDN, 외곽 4×배율 px는 혼합합니다. 합친 영역이 8개 초과이거나 프레임 면적의 60% 이상이면 전체 SR로 전환합니다. **ROI / 전체 전환 / 배경만**의 장면 수를 표시해, 전환된 프레임을 ROI 처리 성능으로 오해하지 않도록 합니다.

전체 화면 PSNR·SSIM과 별도로 **ROI PSNR·SSIM**을 모든 방식에 대해 계산합니다. ROI는 몬스터·아이템 사각형의 합집합이며 배경 일부를 포함할 수 있습니다. 패딩은 화질 마스크에 포함하지 않고 중복 픽셀은 한 번만 셉니다. ROI SSIM은 11×11 창이 영역 안에 완전히 들어가는 픽셀만 사용합니다. 빈 ROI나 작은 ROI의 측정 불가능한 지표는 N/A입니다. 유효 프레임 수, 입력 해시·좌표, 실제 처리 경로, 측정 시간 구간은 JSON/CSV에 기록됩니다.

앱 밖에서 기존 ROI 실행기로 수집하려면, 아직 존재하지 않는 새 폴더를 지정합니다:

```powershell
$env:AVION_ROI_RECORD_DIR = "$PWD\output\benchmarks\doom-scenes-manual"
$env:AVION_ROI_RECORD_MAX_FRAMES = '60'
.\Doom\run-freedoom2-roi.bat
# 게임 종료 후 다음 실행에 기록 설정이 남지 않도록 해제
Remove-Item Env:AVION_ROI_RECORD_DIR
```

CLI와 검증:

```powershell
python -B SR/benchmark.py --output output/benchmarks/bench-my-run --device cuda --methods nearest bilinear bicubic imdn imdn-fp16
python -B SR/benchmark.py --source 'C:\frames' --output output/benchmarks/bench-game --scale 2 --max-frames 20
python -B SR/benchmark.py --source output/benchmarks/doom-scenes-manual/roi-manifest.json --output output/benchmarks/bench-roi --methods bicubic imdn imdn-roi imdn-fp16 imdn-roi-fp16 --device cuda --max-frames 20
# 실제 입력 크기 측정: 위 명령에 --mode native 추가 (출력 폴더는 새 이름 사용)
python -B -m unittest discover -s SR -p 'test_*.py'
cd electron
npm test
```

구현 기준: [scikit-image PSNR·SSIM](https://scikit-image.org/docs/stable/api/skimage.metrics.html), [PyTorch CUDA 동기화](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.synchronize.html), [NVIDIA 계측 문서](https://docs.nvidia.com/deploy/nvidia-smi/index.html).
