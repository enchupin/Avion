# TSD-SR 성능 개선 작업 상세 리포트 (2026-02-25)

## 1) 요청 목표
- 기존에 구성된 화질 개선(SR) AI의 성능을 더 좋게 만들기 위해,
  - 파라미터 조합을 체계적으로 비교하고
  - 다양한 입력 열화 상황에서도 테스트 가능한 실험 체계를 만들고
  - 실제로 어떤 설정이 더 좋은지 수치로 확인하는 것

---

## 2) 이번에 코드에서 변경한 내용

### 2-1. 다중 시나리오 자동 벤치마크 스크립트 추가
- 신규 파일: `script/run_benchmark_scenarios.py`
- 기능:
  - 여러 추론 시나리오를 한 번에 실행
  - GT가 있을 경우 `test/test_metrics.py`까지 자동 실행
  - 결과를 `logs/benchmarks/benchmark_summary.csv`, `logs/benchmarks/benchmark_summary.json`로 통합 저장
  - 필요 시 입력 이미지에 인위적 열화(노이즈/블러/JPEG/저조도) 적용 후 강건성 테스트 가능

#### 포함된 시나리오 프리셋
- `quick`
  - `baseline_wavelet_512`
  - `adain_512`
  - `nofix_512`
  - `tile_wavelet_512`
  - `wavelet_384`
  - `wavelet_768`
- `extended`
  - `quick` + 추가 시나리오(`tile_wavelet_768_overlap16`, `adain_384`, `adain_768`, `wavelet_fp32`)

#### 포함된 열화 프리셋
- `clean`
- `gaussian_noise_10`, `gaussian_noise_25`
- `jpeg_30`, `jpeg_10`
- `blur_3`, `blur_7`
- `low_light_06`

### 2-2. README에 사용법 추가
- 수정 파일: `README.md`
- 멀티 시나리오 벤치마크 실행 방법(quick/extended + perturbation) 섹션 추가

### 2-3. 메트릭 실행 안정성 수정 (PYTHONPATH 자동 주입)
- 수정 파일: `script/run_benchmark_scenarios.py`
- 배경:
  - 최초 자동 실행 시 `test/test_metrics.py`에서 `ModuleNotFoundError: No module named 'basicsr'` 발생
  - 원인은 실행 환경에서 `PYTHONPATH`에 프로젝트 루트가 포함되지 않은 상태였기 때문
- 조치:
  - `subprocess.run(...)` 실행 전에 환경변수 `PYTHONPATH`에 현재 프로젝트 루트를 자동 추가하도록 수정
- 결과:
  - 이후 자동 실행에서 메트릭 단계가 정상 동작

---

## 3) 실험 데이터/경로 구성

### 3-1. 원본 데이터 확인
- `StableSR_testsets/StableSR_testsets/DrealSRVal_crop128/test_LR`: 93장
- `StableSR_testsets/StableSR_testsets/DrealSRVal_crop128/test_HR`: 93장

### 3-2. 빠른 실측을 위한 미니셋 구성
- 생성 경로:
  - `StableSR_testsets/mini20a/test_LR`
  - `StableSR_testsets/mini20a/test_HR`
- 구성:
  - 정렬 기준 상위 20쌍 PNG를 복사하여 소규모 검증셋 생성
- 목적:
  - 전체 93장보다 빠르게 파라미터 방향성(우열) 확인

---

## 4) 실제 실행한 명령과 과정

## 4-1. 시나리오 추론 + 메트릭 자동 실행 (3개 시나리오)
```bash
python script/run_benchmark_scenarios.py \
  --pretrained_model_name_or_path checkpoint/tsdsr \
  --lora_dir checkpoint/tsdsr-mse \
  --embedding_dir dataset/default \
  --dataset "mini20a|StableSR_testsets/mini20a/test_LR|StableSR_testsets/mini20a/test_HR" \
  --scenario_set quick \
  --max_runs 3 \
  --overwrite
```

### 4-2. 최초 실행 중 발생 이슈
- 증상: 메트릭 단계 실패
  - `ModuleNotFoundError: No module named 'basicsr'`
- 원인: `test/test_metrics.py` 실행 시 프로젝트 루트 import 경로 누락
- 임시 대응: `PYTHONPATH=.`를 수동으로 주고 메트릭 재실행
- 영구 대응: 벤치마크 스크립트 내부에서 `PYTHONPATH` 자동 세팅하도록 코드 수정

### 4-3. 메트릭 수동 재실행 (당시 3개 시나리오 결과 확보용)
```bash
PYTHONPATH=. python test/test_metrics.py --inp_imgs outputs/benchmarks/mini20a__clean__baseline_wavelet_512 --gt_imgs StableSR_testsets/mini20a/test_HR --log logs/benchmarks/mini20a__clean__baseline_wavelet_512
PYTHONPATH=. python test/test_metrics.py --inp_imgs outputs/benchmarks/mini20a__clean__adain_512 --gt_imgs StableSR_testsets/mini20a/test_HR --log logs/benchmarks/mini20a__clean__adain_512
PYTHONPATH=. python test/test_metrics.py --inp_imgs outputs/benchmarks/mini20a__clean__nofix_512 --gt_imgs StableSR_testsets/mini20a/test_HR --log logs/benchmarks/mini20a__clean__nofix_512
```

### 4-4. 수정 후 자동화 정상 동작 검증
```bash
python script/run_benchmark_scenarios.py \
  --pretrained_model_name_or_path checkpoint/tsdsr \
  --lora_dir checkpoint/tsdsr-mse \
  --embedding_dir dataset/default \
  --dataset "mini20a|StableSR_testsets/mini20a/test_LR|StableSR_testsets/mini20a/test_HR" \
  --scenario_set quick \
  --max_runs 1 \
  --overwrite
```

---

## 5) 결과 요약 (mini20a, clean, 3시나리오)

### 5-1. 평균 지표 비교
| Scenario | PSNR (↑) | SSIM (↑) | LPIPS (↓) | DISTS (↓) | FID (↓) |
|---|---:|---:|---:|---:|---:|
| baseline_wavelet_512 | 27.5139 | 0.8069 | 0.2716 | 0.2221 | 178.3328 |
| adain_512 | **28.9827** | **0.8299** | **0.2578** | **0.2148** | **174.8818** |
| nofix_512 | 27.1835 | 0.8069 | 0.2707 | 0.2234 | 180.0449 |

### 5-2. 기준 대비 개선량 (baseline_wavelet_512 -> adain_512)
- PSNR: `+1.4688 dB`
- SSIM: `+0.0230`
- LPIPS: `-0.0138`
- DISTS: `-0.0073`
- FID: `-3.4510`

### 5-3. 속도(추론 평균 시간, 로그 기준)
- baseline_wavelet_512: 약 `0.365s / image`
- adain_512: 약 `0.344s / image`
- nofix_512: 약 `0.358s / image`
- 해석: 품질이 좋아진 `adain_512`가 이번 미니셋에서는 속도도 손해가 거의 없거나 약간 더 빠름

---

## 6) 결론: 이번 변경으로 무엇이 달라졌나

## 6-1. 코드/프로세스 관점
- 단발성 수동 테스트에서 벗어나, **여러 파라미터와 다양한 상황을 자동으로 비교하는 실험 체계**가 생김
- 결과가 CSV/JSON으로 남아서 재현성과 비교 가능성이 올라감
- 메트릭 실행 실패 원인을 코드에서 해결해 자동화 안정성 개선

## 6-2. 모델 성능 관점 (실측 결과)
- 이번 조건(미니셋 20장, clean)에서는 `align_method=adain`, `process_size=512`가 `wavelet`/`nofix`보다 명확히 우수
- 따라서 현재 추천 기본 추론값은:
  - `--align_method adain`
  - `--process_size 512`

---

## 7) 참고 로그/산출물 위치
- 자동 요약:
  - `logs/benchmarks/benchmark_summary.csv`
  - `logs/benchmarks/benchmark_summary.json`
- 시나리오별 메트릭 로그:
  - `logs/benchmarks/mini20a__clean__baseline_wavelet_512/test_METRICS_260225-190739.log`
  - `logs/benchmarks/mini20a__clean__adain_512/test_METRICS_260225-190503.log`
  - `logs/benchmarks/mini20a__clean__nofix_512/test_METRICS_260225-190604.log`
- 생성된 추론 이미지:
  - `outputs/benchmarks/mini20a__clean__baseline_wavelet_512/`
  - `outputs/benchmarks/mini20a__clean__adain_512/`
  - `outputs/benchmarks/mini20a__clean__nofix_512/`

---

## 8) 아직 남은 검증(중요)
- 현재 보고서는 `mini20a(20장)` 기준 결과
- 최종 의사결정 전에 아래 전체셋/다양 상황 검증 권장:
  - DrealSRVal 93장 전체
  - RealSRVal_crop128, DIV2K_V2_val 동시
  - `extended` 시나리오 + `enable_perturbations`

권장 전체 실행 예시:
```bash
python script/run_benchmark_scenarios.py \
  --pretrained_model_name_or_path checkpoint/tsdsr \
  --lora_dir checkpoint/tsdsr-mse \
  --embedding_dir dataset/default \
  --dataset "DrealSR|StableSR_testsets/StableSR_testsets/DrealSRVal_crop128/test_LR|StableSR_testsets/StableSR_testsets/DrealSRVal_crop128/test_HR" \
  --dataset "RealSR|StableSR_testsets/StableSR_testsets/RealSRVal_crop128/test_LR|StableSR_testsets/StableSR_testsets/RealSRVal_crop128/test_HR" \
  --dataset "DIV2K|StableSR_testsets/StableSR_testsets/DIV2K_V2_val/test_LR|StableSR_testsets/StableSR_testsets/DIV2K_V2_val/test_HR" \
  --scenario_set extended \
  --enable_perturbations \
  --overwrite
```

---

## 9) 한 줄 요약
- **무엇을 바꿨나:** 멀티 시나리오/멀티 상황 자동 벤치마크 시스템을 추가하고 메트릭 자동화 오류를 수정함.
- **결과가 어땠나:** 미니셋 실측에서 `adain_512`가 `wavelet_512` 대비 지표 전반(PSNR/SSIM/LPIPS/DISTS/FID)에서 개선됨.

---

## 10) 시나리오별 차이 중심 분석

아래는 이번에 비교한 핵심 시나리오들이 **실제로 무엇을 바꾸는지**와 **결과가 왜 달라졌는지**를 중심으로 정리한 내용이다.

### 10-1. `baseline_wavelet_512`
- 파라미터
  - `--align_method wavelet`
  - `--process_size 512`
- 동작 차이
  - SR 결과물의 색/톤을 입력과 맞출 때 `wavelet_color_fix` 사용
  - 저주파 색상/명암을 안정적으로 보정하는 성향
- 장점
  - 색 틀어짐 방지에 비교적 안정적
  - 기본값으로 두기 쉬운 무난한 밸런스
- 단점
  - 이번 미니셋에서는 세부 복원(지각 품질)과 참조 유사도에서 `adain` 대비 열세
- 실측 평균
  - PSNR 27.5139 / SSIM 0.8069 / LPIPS 0.2716 / DISTS 0.2221 / FID 178.3328

### 10-2. `adain_512`
- 파라미터
  - `--align_method adain`
  - `--process_size 512`
- 동작 차이
  - 색/스타일 정렬을 `adain_color_fix`로 수행
  - 통계(평균/분산) 기반 정렬로 전체 톤을 더 강하게 맞추는 경향
- 장점
  - 이번 실험에서 모든 핵심 지표(PSNR/SSIM/LPIPS/DISTS/FID) 최상
  - baseline 대비 수치 개선 폭이 명확
- 단점
  - 데이터셋 성격에 따라 과도한 톤 정렬이 발생할 수 있어, 다른 도메인(야간/피부톤/특정 카메라)에서는 재검증 필요
- 실측 평균
  - PSNR 28.9827 / SSIM 0.8299 / LPIPS 0.2578 / DISTS 0.2148 / FID 174.8818
- baseline 대비
  - PSNR +1.4688 dB, SSIM +0.0230, LPIPS -0.0138, DISTS -0.0073, FID -3.4510

### 10-3. `nofix_512`
- 파라미터
  - `--align_method nofix`
  - `--process_size 512`
- 동작 차이
  - 후처리 색 정렬을 아예 하지 않음
  - 모델 출력 원본 특성을 그대로 유지
- 장점
  - 후처리 부작용(과도한 컬러 보정)을 피함
  - 파이프라인 단순
- 단점
  - 색/명암 드리프트가 누적될 수 있어 기준 GT와 괴리가 커지기 쉬움
  - 이번 실험에서 FID가 가장 나쁨
- 실측 평균
  - PSNR 27.1835 / SSIM 0.8069 / LPIPS 0.2707 / DISTS 0.2234 / FID 180.0449

### 10-4. 왜 이번에는 `adain_512`가 가장 좋았나 (해석)
- 이 데이터셋 샘플에서는 SR 디테일 복원뿐 아니라 GT와의 색/톤 일치가 성능에 크게 작용
- `adain`이 색 통계를 더 직접적으로 맞추면서
  - 참조 기반 지표(PSNR/SSIM)와
  - 지각 기반 지표(LPIPS/DISTS/FID)
  를 동시에 개선한 것으로 해석 가능
- 반대로 `nofix`는 색 정렬 부재로 전체 분포가 벌어져 FID에서 불리해짐

### 10-5. 시나리오 선택 가이드 (실무 기준)
- 품질 최우선(현재 데이터 도메인): `adain_512` 우선
- 색 안정성/보수적 운영: `baseline_wavelet_512` 백업 옵션
- 후처리 최소화 실험/아블레이션: `nofix_512`

### 10-6. 아직 남은 시나리오 검증 포인트
- `process_size` 변화(384/768): 작은 디테일 vs 계산량 trade-off 확인 필요
- `tile` 활성화: 대해상도 OOM 회피와 경계 아티팩트 균형 확인 필요
- `fp16` vs `fp32`: 속도/메모리/수치 안정성 비교 필요
- 결론적으로, 본 리포트의 “시나리오 우열”은 현재 `mini20a` 기준이며, 전체셋 + 열화 조건에서 재확인해야 최종 운영값으로 고정 가능
