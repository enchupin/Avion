# Avion

Electron에서 SR 실행 환경을 설정하고, 선택한 SR 방식으로 DOOM을 실행하는 앱입니다.

## Requirements

- Windows 10 1903 or newer
- 소스 실행: Node.js / npm
- DOOM 소스 재빌드 시에만 Visual Studio 2022 C++ Build Tools와 Windows SDK가 필요합니다.

## Run

Windows 배포본은 [GitHub Releases](https://github.com/enchupin/Avion/releases)에서 `Avion-v1.0.2-win-x64.zip`을 내려받아 압축을 푼 뒤 `Avion-win32-x64/Avion.exe`를 더블클릭합니다. 실행 파일 옆의 DLL과 `resources` 폴더를 함께 유지하세요. Node.js나 Visual Studio 설치 없이 실행할 수 있으며, SR은 앱의 **설치 여부 확인 → 미설치 항목 설치**로 준비합니다. 자동 PyTorch 설치는 CPU용입니다.

소스에서 실행하는 경우:

```powershell
cd electron
npm install
npm start
```

앱에서 **SR 실행 환경 → 설치 여부 확인**을 누른 뒤, **DOOM 실행**에서 SR 방식을 선택하세요.

## SR 실행 환경

**SR 실행 환경 → 설치 여부 확인**을 먼저 누르세요. Python, NumPy, OpenCV, scikit-image, PyTorch의 설치 상태와 버전을 표시합니다. 미설치 또는 실행 불가 항목에만 **설치** 버튼이 나타납니다. Python이 없다면 Python부터 설치하고 나머지 패키지를 설치하세요. 설치 중에는 진행 상태·로그·취소 버튼이 표시되며, 완료 후 자동으로 재검사합니다. IMDN ×2/×3/×4와 ROI 추론까지 통과하면 DOOM 실행 버튼이 활성화됩니다.

Windows x64에서는 기존 Python이 없어도 공식 CPython NuGet 패키지를 다운로드해 SHA-256을 검증하고 앱 전용 환경을 만듭니다. 개발 실행은 저장소의 `.avion-runtime/`, 배포 앱은 사용자 데이터 폴더의 `sr-runtime/`에 설치하므로 앱 설치 폴더의 쓰기 권한이나 시스템 Python 설정 변경이 필요하지 않습니다. 기존 프로젝트 `.venv`도 인식합니다. 설치된 환경은 다음 실행에도 사용하며, 앱을 다시 열면 **설치 여부 확인**으로 점검하세요. 다른 PC에는 가상환경 폴더를 복사하지 말고 같은 버튼으로 설치하세요. 최초 설치에는 인터넷 연결과 패키지 저장 공간이 필요합니다. 네트워크 오류는 로그를 확인한 뒤 해당 설치 버튼으로 재시도할 수 있습니다.

**GPU SR**은 CUDA 장치 인식뿐 아니라 GPU 합성곱과 실제 IMDN 추론도 확인합니다. CPU용 PyTorch, GPU/드라이버 인식 실패, 추론 오류를 구분해 표시합니다. GPU 검증이 안 된 상태에서는 CPU로 실행하며 CUDA/FP16 선택을 비활성화합니다. 자동 설치 버튼은 CPU용 PyTorch를 설치합니다. CUDA용 PyTorch 설치는 [PyTorch 설치 안내](https://pytorch.org/get-started/locally/)에서 GPU·드라이버에 맞는 명령을 선택해 진단에 표시된 Python 경로로 실행한 뒤 다시 확인하세요.

터미널에서 직접 설정하는 경우:

```powershell
python -m pip install -r SR/requirements-evaluation.txt
# Python 경로를 직접 지정해야 하는 경우, 같은 터미널에서 설정 후 앱 실행
$env:AVION_SR_PYTHON = 'C:\path\to\python.exe'
cd electron
npm start
```


## DOOM 실행

게임을 바로 실행하려면 Electron의 **SR 실행 환경 → 설치 여부 확인** 후 **DOOM 실행 → 실시간 SR 설정**에서 방식을 선택하고 **선택한 방식으로 DOOM 실행**을 누르세요. **SR 입력 320×200**과 **SR 결과**를 별도의 게임 창에서 비교합니다. 입력 창은 worker에 전달하는 RGB 픽셀을 확대·보간·4:3 비율 보정 없이 1:1로 표시합니다. 입력 창을 늘려도 영상은 320×200을 유지하고 여백만 늘어납니다. 이 입력은 고해상도 GT가 아닙니다. SR 결과는 ×2에서 640×400, ×3에서 960×600, ×4에서 1280×800이며, 화면 공간이 부족하면 결과 창의 표시 크기만 맞춥니다. ROI 경계는 SR 결과 창에만 표시하므로 입력 영상은 그대로 유지됩니다.

- **ROI SR FP32 / FP16**: 몬스터·아이템은 IMDN, 배경은 바이큐빅으로 처리합니다. ROI가 넓거나 많으면 전체 SR로 전환합니다. 녹색 ROI 경계 표시를 켜거나 끌 수 있습니다.
- **전체 SR FP32 / FP16**: 전체 프레임에 IMDN을 적용합니다.
- **바이큐빅 / 바이리니어 / 최근접 보간**: IMDN을 로딩하지 않고 CPU 보간으로 확대합니다.

×2/×3/×4 배율과 자동/CPU/CUDA 장치를 선택할 수 있습니다. FP16은 CUDA GPU 확인 후 자동 또는 CUDA 장치에서만 사용할 수 있습니다. 게임 중에는 설정 변경·중복 실행·환경 설치를 막습니다. **DOOM 종료** 또는 게임 창에서 종료한 뒤 다른 방식으로 다시 실행하세요. Electron을 닫으면 이 버튼으로 시작한 게임과 worker도 종료됩니다. worker의 첫 프레임 처리를 확인한 뒤 선택 방식의 적용 완료를 표시하며, SR 오류가 나면 게임을 종료하고 오류를 표시합니다.

기존 `Doom/run-freedoom2-roi.bat`도 같은 입력/SR 두 창으로 ROI FP32를 실행합니다. Electron은 공통 `run-freedoom2.bat`에 선택한 ROI 옵션과 worker 환경 설정을 전달합니다. 소스를 직접 빌드했다면 입력 표시 변경이 반영되도록 해당 DOOM 실행 파일도 다시 빌드하세요.


## 개발용 CLI 도구

Electron에는 환경 설정과 DOOM 실행만 표시합니다. 데이터 수집·성능 평가는 저장소의 독립 CLI 도구로 실행할 수 있습니다.

저장소의 CLI 도구로 장면을 수집하려면, 아직 존재하지 않는 새 폴더를 지정합니다:

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
# 실제 Electron → DOOM → SR worker 실행·종료 검증 (숨김 SDL 렌더러)
npm run test:doom
```

구현 기준: [scikit-image PSNR·SSIM](https://scikit-image.org/docs/stable/api/skimage.metrics.html), [PyTorch CUDA 동기화](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.synchronize.html), [NVIDIA 계측 문서](https://docs.nvidia.com/deploy/nvidia-smi/index.html).
