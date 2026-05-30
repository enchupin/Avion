# TSD-SR 벤치마크 테스트 코드 분석 및 결과 보고서

## 1. `run_benchmark_scenarios.py` 코드 동작 원리 분석

이 스크립트는 단순히 모델을 한 번 실행하는 것이 아니라, 코드 내부적으로 정해진 여러 설정값(시나리오)과 이미지 상태(가혹 조건)를 조합하여 **반복적으로 추론(Inference) 및 평가(Metrics)**를 수행하는 자동화 파이썬 스크립트입니다. 

코드의 핵심 동작 순서는 다음과 같습니다.

### 단계 1: 설정값 및 시나리오 로드 (`parse_args`, `load_scenarios`)
* 코드가 실행되면 가장 먼저 커맨드라인으로 입력받은 경로(모델, 로라(LoRA), 데이터셋 경로 등)를 파싱합니다.
* `SCENARIOS_QUICK` 또는 `SCENARIOS_EXTENDED` 리스트를 불러옵니다. 이 리스트에는 모델을 실행할 때 줄 하이퍼파라미터들(예: `align_method`: wavelet/adain, `process_size`: 512/768, 타일링 사용 여부 등)이 딕셔너리 형태로 정의되어 있습니다.

### 단계 2: (옵션) 가혹 조건(Perturbation) 생성 (`build_perturbation_dirs`)
* `--enable_perturbations` 옵션이 켜져 있을 경우 실행됩니다.
* OpenCV(`cv2`)를 사용하여 원본 입력 이미지(LR)에 인위적인 손상을 가합니다.
  * **Gaussian Noise**: 정규 분포 난수를 더해 노이즈 추가
  * **JPEG Compression**: JPEG 인코딩/디코딩 과정을 거쳐 압축 손실 발생
  * **Blur**: 가우시안 블러(GaussianBlur) 적용
  * **Low Light**: 픽셀 값에 0.6을 곱해 어둡게 만듦
* 변형된 이미지들은 `outputs/benchmarks_tmp` 폴더 내부에 각각의 조건 이름으로 임시 저장됩니다.

### 단계 3: 추론(Inference) 스크립트 자동 실행 (`test/test_tsdsr.py`)
* 3중 반복문(데이터셋 $\rightarrow$ 가혹 조건 $\rightarrow$ 시나리오)을 돌면서 모든 조합에 대해 테스트를 진행합니다.
* 파이썬의 `subprocess.run()` 모듈을 이용해 백그라운드에서 실제 화질 개선 스크립트인 `test_tsdsr.py`를 실행하는 터미널 명령어를 조립하고 실행시킵니다.
* 이때 앞서 로드한 시나리오의 파라미터들(`--align_method`, `--process_size` 등)이 명령어 인자로 자동으로 들어갑니다.
* 처리된 결과물은 `outputs/benchmarks/[런아이디]` 폴더에 저장됩니다.

### 단계 4: 평가지표(Metrics) 계산 자동 실행 (`test/test_metrics.py`)
* 정답 이미지(GT) 경로가 제공된 경우, 결과물이 나온 직후 곧바로 `test_metrics.py` 스크립트를 `subprocess`로 실행합니다.
* 이 스크립트는 추론 결과 이미지와 원본 정답 이미지를 비교하여 다양한 평가지표(PSNR, SSIM, LPIPS, FID 등)를 계산하고 로그 파일(`.log`)로 저장합니다.

### 단계 5: 요약 파일(Summary) 파싱 및 생성 (`parse_metrics_from_log`)
* 모든 반복이 끝나거나(또는 지정된 `--max_runs` 도달 시), 정규화 표현식(`re` 모듈)을 이용해 생성된 텍스트 로그 파일에서 평가지표 숫자들만 추출합니다.
* 추출한 데이터를 모아 최종적으로 모든 시나리오에 대한 결과가 담긴 `benchmark_summary.json`과 `benchmark_summary.csv` 파일을 `logs/benchmarks` 폴더에 생성합니다.

---

## 2. 테스트 환경 요약 및 사용 모델 설명

* **사용된 모델 (Pretrained Model & LoRA) 두 가지 비교:**
  본 테스트에서는 서로 다른 목적을 가진 두 가지 TSD-SR 모델 가중치를 교차 검증하였습니다.
  - **1) 기본 모델 (TSD-SR)** (`checkpoint/tsdsr`): 일반적인 화질 복원에 널리 사용되도록 최적화된 범용 가중치.
  - **2) 평가 전용 모델 (TSD-SR-MSE)** (`checkpoint/tsdsr-mse`): 벤치마크 테스트와 평가지표(Metrics) 상에서 조금 더 수학적으로 일관된 결과를 얻도록(Loss를 줄이도록) 조정된 가중치.
  - 프롬프트 엠베딩(Embedding) 경로: `dataset/default`

* **사용된 데이터셋 경로:** `imgs/test` (GT(정답) 이미지가 없는 실제(Real) 저해상도 이미지 데이터셋을 사용)
* **Perturbation(가혹 조건) 적용 여부:** 사용 안 함 (`clean` 상태 원본 이미지로만 테스트)

### 💡 각 시나리오(설정)별 차이점 설명
본 벤치마크 테스트에 사용된 총 6가지 시나리오(Quick 세트)는 이미지 처리 해상도 크기와 `align_method`(디테일 및 색감 보정 방식)에 따라 다음과 같은 차이가 있습니다.
1. **`baseline_wavelet_512` / `wavelet_384` / `wavelet_768`**: 화질 복원 시 **Wavelet 변환**을 사용하여 원본의 색상과 질감 디테일을 정교하게 보정(Align)합니다. 뒤에 붙은 숫자는 모델이 한 번에 처리하는 기본 타겟 해상도 크기(384, 512, 768)를 의미합니다.
2. **`adain_512`**: Wavelet 기술 대신 **AdaIN (Adaptive Instance Normalization)** 방식을 사용하여 결과 이미지의 전체적인 스타일(색감 등)을 원본에 맞게 조정합니다.
3. **`nofix_512`**: 별도의 디테일 및 색감 보정 과정 없이 순수하게 화질 확대(업스케일링) 알고리즘만 수행합니다.
4. **`tile_wavelet_512`**: 큰 이미지를 여러 개의 작은 타일(조각)로 쪼개어 처리한 뒤 이어 붙이는 타일링(Tiling) 방식입니다. 메모리(VRAM)가 부족한 그래픽카드 환경에서 오류 없이 거대한 초고해상도 이미지를 만들어낼 때 주로 사용됩니다.

---

## 3. 테스트 결과 요약 (기본 모델 vs MSE 모델 비교)

*참고: 입력된 데이터셋에 품질 비교를 위한 원본 고화질 정답(GT: Ground Truth) 파일이 없기 때문에 PSNR, SSIM, LPIPS 등의 화질 관련 평가 점수는 산출할 수 없었습니다. 따라서 동작 성공 여부와 전체 처리 소요 시간 차이를 기록하였습니다.*

| 시나리오 이름 (Scenario) | Align 방식 | 처리 해상도 | [기본 모델] 결과 | [기본 모델] 실행 시간 | [MSE 모델] 결과 | [MSE 모델] 실행 시간 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`baseline_wavelet_512`** | Wavelet | 512 | `inference_ok` | **215.377 초** | `inference_ok` | 285.123 초 |
| **`adain_512`** | AdaIN | 512 | `inference_ok` | **214.296 초** | `inference_ok` | 265.798 초 |
| **`nofix_512`** | 안 함 (nofix) | 512 | `inference_ok` | **215.823 초** | `inference_ok` | 293.828 초 |
| **`tile_wavelet_512`** | Wavelet (타일 분할) | 512 | `inference_ok` | **226.726 초** | `inference_ok` | 251.676 초 |
| **`wavelet_384`** | Wavelet | 384 | `inference_ok` | **216.141 초** | `inference_ok` | 287.693 초 |
| **`wavelet_768`** | Wavelet | 768 | `inference_ok` | **209.876 초** | `inference_ok` | 278.306 초 |

---

## 4. 최종 분석 및 결론

* **전체 테스트 안정성:** 총 5장의 원본 이미지에 대해 기본 모델과 MSE 모델 양쪽 다 메모리 오류나 비정상 종료 없이 **모든 테스트가 100% 성공적으로 동작(`inference_ok`)** 하였습니다. MSE 모델의 결과물은 `outputs/benchmarks_mse/` 폴더 내부에 별도로 저장되었습니다.
* **소요 시간 분석 (모델 간 차이):** 
  * 기본 모델(TSD-SR)은 전체 5장을 처리하는데 시나리오당 평균 **210~226초**가 소요된 것에 반해, MSE 특화 모델(TSD-SR-MSE)은 시나리오 당 **251~293초**로 평균적으로 시간이 더 오래 걸리는 양상을 보였습니다. (MSE 가이드가 포함된 평가 특화 방식의 연산이 조금 더 무거운 것으로 추측됩니다.)
  * 흥미롭게도 MSE 모델에서는 이미지를 쪼개서 작업하는 **타일링 방식(`tile_wavelet_512`)**이 251.6초로 **가장 빠른 연산 속도**를 보였습니다. 이는 일반 모델(타일링 연산이 더 오래 걸린 모델) 결과와 상반된 경향성입니다.
* **향후 권장 사항:** 
  * 속도와 기본 효율성에 중점을 두신다면 가벼운 **기본 TSD-SR 모델**이 적합하며, 시간 지연을 감수하더라도 엄격한 평가 테스트 검증 목적의 결과가 필요하시다면 **TSD-SR-MSE 모델**을 권장합니다. 
  * 두 모델 간의 색감/디테일 복원 느낌(정성적 차이)을 직접 확인하시고 싶다면 `outputs/benchmarks` 폴더와 `outputs/benchmarks_mse` 폴더 안의 결과 사진들을 나란히 띄워두고 비교하시면 각 모델의 특징을 정확하게 파악하실 수 있습니다.
