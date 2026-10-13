# 프로젝트 4. DefectVision AI — 정밀 부품 외관 이상탐지 검사 시스템

> **상태 (2026-10-13):** ML 코어·FastAPI 백엔드 완료, MVTec AD 확보·검증 완료.
> 실데이터 측정은 **사전 등록 ablation 64회 중 1회(B0 · metal_nut · seed 0)만** 실행했습니다. 나머지(CPU 약 19시간)는 보류 중이며 `NOT_RUN`입니다.
> 용어가 낯설면 [용어 설명](docs/GLOSSARY.md)을 먼저 보세요.

## 요약
정밀 제조 부품 이미지에서 정상/이상을 판정하고, 결함 위치를 픽셀 단위 heatmap(이상 정도를 색으로 칠한 그림)으로 추정하는 검사 시스템입니다.
**정상 이미지만으로 학습**하는 PatchCore(정상 패치 특징을 저장해 두고 얼마나 다른지로 점수를 매기는 이상탐지 방법)가 주력 모델이고,
결과를 REST API와 웹 대시보드로 제공하는 것이 목표입니다.

포트폴리오 공통 주제인 **"AI의 판단을 믿어도 되는가"** 를 이 프로젝트에서는 이렇게 구체화합니다.
- 임계값을 테스트셋에 맞추지 않았을 때, 실제 라인에서 기대할 수 있는 오경보율·미검출률은 얼마인가?
- heatmap이 가리키는 위치는 실제 결함과 얼마나 겹치는가? (pixel AUROC, AUPRO, IoU)
- 설계 선택(백본·레이어·메모리 뱅크 크기·입력 크기)은 품질과 비용을 어떻게 맞바꾸는가? → [사전 등록 ablation](docs/ABLATION_STUDY.md)

공개 민간 산업 데이터(MVTec AD)만 사용합니다. 특정 기업의 내부 시스템이나 군 장비 검사를 재현한다고 주장하지 않습니다.

## 진행 현황
| Phase | 내용 | 상태 |
|---|---|---|
| 0 | 저장소·환경 점검, 설계 | 완료 ([ARCHITECTURE](docs/ARCHITECTURE.md)) |
| 1 | 데이터 검증, PatchCore 학습/추론, heatmap, 평가, 스모크 테스트 | 완료. MVTec AD 4개 카테고리 확보·검증 PASS ([DATASET](docs/DATASET.md)) |
| 1+ | 사전 등록 ablation (13개 설정 × 4개 카테고리) | 실행기 완료, **1/64회 실측**, 나머지 보류 ([ABLATION_STUDY](docs/ABLATION_STUDY.md)) |
| 2 | FastAPI 백엔드 (검사 API, 모델 로딩, 이력 저장, readiness) | **완료** — 실서버(uvicorn) 확인, 테스트 30개 |
| 3 | React 대시보드 | 대기 — Node.js 미설치 |
| 4 | Docker Compose, PostgreSQL, CI | CI 워크플로 작성(push 전 미실행). Compose는 Docker 미설치로 대기 |
| 5 | 비교 모델(CNN), 임계값 검증, 강건성, 벤치마크 보고서 | 예정 |

## 지금까지의 실측 결과 (1회 실행, 일반화 금지)
B0 설정(wide_resnet50_2 torchvision 가중치, layer2+3, k-center coreset 10%, 256 px) · metal_nut · seed 0.
임계값은 정상 검증 이미지 44장으로만 정했고, test는 결과 보고에만 썼습니다.

| 지표 | 값 | 의미 |
|---|---|---|
| test image AUROC | 0.9985 | 결함/정상 이미지 순위 매기기. 1 = 완벽 |
| test pixel AUROC | 0.989 | heatmap이 결함 픽셀을 정상 픽셀보다 뜨겁게 칠하는 정도 |
| test AUPRO (FPR ≤ 0.3) | 0.942 | 결함 영역 하나하나를 얼마나 덮는지 (작은 결함도 같은 비중) |
| 검증 임계값 적용 시 정상 오경보 / 결함 검출률 | 1/22 (4.5%) / 96.8% | 목표 오경보율은 약 1%였음 → 검증셋이 작아 임계값이 낮게 잡힌 영향으로 추정 |
| batch-1 지연 평균 / P95 | 744 ms / 838 ms | CPU 4스레드. P95 = 100건 중 95번째로 느린 요청 |
| 서비스 최대 메모리 / 모델 로드 | 1.18 GB / 8.6 s | 컨테이너 메모리 설정 근거 |
| 학습 시간 | 829 s (그중 coreset 선택 693 s) | coreset = 메모리 뱅크를 대표 패치로 줄이는 단계 |

원본: `results/ablation/runs.csv`, `results/ablation/tables.md` (재현: `python -m ml.ablation.run_ablation --categories metal_nut --experiments B0 --max-runs 1`).
seed 1개·카테고리 1개 결과라, 설정 간 비교나 일반화 근거로 쓰지 않습니다.

이 1회 실행에서 발견한 것:
- **검증용 합성 결함의 AUROC가 0.998로 거의 상한**입니다. 설정을 고르는 기준으로 쓰기에는 너무 쉬울 수 있어, 본 실행 결과와 함께 보고할 예정입니다.
- 전체 ablation은 CPU로 **약 19시간**이 걸릴 것으로 추정됩니다 (coreset 선택 시간이 패치 수의 제곱에 비례).

## 실행 방법
작업 폴더: `project4_defectvision/`. 가상환경은 저장소 공용 `../.venv`를 씁니다.

```powershell
# 0) 의존성
..\.venv\Scripts\python.exe -m pip install -r requirements.txt -r backend\requirements.txt

# 1) 린트 + 테스트 (오프라인, 약 1분. 랜덤 백본 + 합성 데이터, 87개)
..\.venv\Scripts\ruff.exe check .
..\.venv\Scripts\python.exe -m pytest -q

# 2) 데이터셋 검증 (MVTec AD를 data\mvtec_ad\ 에 풀어 둔 뒤)
..\.venv\Scripts\python.exe scripts\inspect_dataset.py --data-root data\mvtec_ad --categories metal_nut screw capsule transistor

# 3) 학습 (train/good만 사용, 그중 20%는 임계값 보정용) → 4) 평가 (공식 test, 임계값은 아티팩트 값 그대로)
..\.venv\Scripts\python.exe -m ml.training.train_patchcore --data-root data\mvtec_ad --category metal_nut --version 1.0.0
..\.venv\Scripts\python.exe -m ml.evaluation.evaluate --artifact artifacts\models\metal_nut\patchcore\1.0.0 --data-root data\mvtec_ad

# 5) API 서버 (모델 없이 띄우면 Demo Mode: 검사 요청을 503으로 거부)
$env:DV_MODEL_ARTIFACT_DIR = "artifacts\models\metal_nut\patchcore\1.0.0"
..\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --port 8000   # 문서: http://localhost:8000/docs

# 6) ablation (재실행 시 끝난 회차는 건너뜀, 기본 오프라인)
..\.venv\Scripts\python.exe -m ml.ablation.run_ablation --dry-run
```
`pre_trained=true`(기본값)는 첫 실행 때 timm 백본 가중치를 Hugging Face Hub에서 내려받습니다. 받은 뒤에는 `HF_HUB_OFFLINE=1`로 실행할 수 있습니다.

## API 요약
| 경로 | 설명 |
|---|---|
| `GET /api/v1/health` · `/health/ready` | 생존 확인 · 모델 준비 여부 (준비 전이면 503) |
| `GET /api/v1/models` | 활성 모델 버전, 데이터셋 지문, 임계값, 체크섬 |
| `POST /api/v1/inspections` | 이미지 검사(multipart) → 점수·임계값·판정·heatmap URL, DB 저장 |
| `GET /api/v1/inspections[/{id}[/heatmap\|overlay\|mask\|original]]` | 이력(필터·페이지), 상세, 결과 이미지 |
| `GET /api/v1/metrics/summary` | 건수, 이상 비율, 지연 P50/P95, 최근 24시간 시간대별 이상 비율 |

점수는 확률이 아닌 거리 값이라 응답에 항상 임계값을 함께 넣습니다. 설정은 `DV_*` 환경변수로 바꿉니다 ([ARCHITECTURE §5.1](docs/ARCHITECTURE.md)).

## 폴더 구조
| 경로 | 내용 |
|---|---|
| `ml/configs/` | YAML 설정 (이미지 크기, 백본, coreset 비율, 샘플링 방식, 임계값 규칙, 디바이스) |
| `ml/datasets/` | MVTec 구조 검증·분할·전처리, 합성 스모크 데이터, 검증용 합성 결함 |
| `ml/inference/patchcore.py` | anomalib `PatchcoreModel` 래퍼: fit / calibrate / predict / save / load (safetensors + 해시 검증) |
| `ml/training/`, `ml/evaluation/` | 학습 CLI, 지표(공식 코드와 일치 확인한 AUPRO 포함)·평가 CLI·메모리 측정 |
| `ml/ablation/` | 사전 등록 계획(`ablation_v1.yaml`), 실행기, 요약기 |
| `ml/visualization/` | heatmap, overlay, 이진 마스크 패널 |
| `backend/` | FastAPI 서비스와 테스트 |
| `scripts/` | 데이터셋 점검, 합성 데이터 생성, AUPRO 교차검증 |
| `tests/`, `backend/tests/` | pytest 87개 (ML 57, 백엔드 30) |
| `results/` | 커밋하는 결과 CSV·표 (`aupro_crosscheck.csv`, `ablation/`) |
| `data/`, `artifacts/`, `var/` | git 제외 (데이터·메모리 뱅크·API 저장소, 재생성 가능) |

## 면접에서 설명할 포인트
- **테스트셋을 보지 않는 설정 선택:** 공식 분할에서 결함은 test에만 있어서, 검증 이미지에 결함을 합성해 선택 기준으로 쓰고 그 기준이 test 순위와 맞는지까지 재도록 설계했습니다.
- **지표 구현을 공식 코드로 검증:** MVTec 공식 평가 코드와 비교해 영역 연결 방식(4-연결 vs 8-연결) 버그를 찾아 고쳤고, 4개 사례에서 소수점 5자리까지 일치시켰습니다.
- **재현성:** anomalib coreset의 숨은 난수(torch·numpy 전역 RNG)를 찾아 seed를 고정했고, 백본 가중치 해시로 다른 가중치가 섞이면 로드를 거부합니다.
- **서비스 안전장치:** 모델이 없으면 가짜 점수 대신 503(Demo Mode), 위장 파일·압축 폭탄·경로 조작 차단, DB 실패 시 저장 파일까지 되돌림.

## 다음 단계
1. 보류 중인 ablation 본 실행 (약 19시간, 여러 번에 나눠 실행 가능)
2. Node.js·Docker 설치 후 React 대시보드(Phase 3), Docker Compose + PostgreSQL(Phase 4)
3. 임계값 안정성(검증셋 크기에 따른 흔들림), 강건성 시험(Phase 5)

## 문서
[GLOSSARY](docs/GLOSSARY.md) · [ARCHITECTURE](docs/ARCHITECTURE.md) · [DATASET](docs/DATASET.md) · [MODEL_DESIGN](docs/MODEL_DESIGN.md) · [EVALUATION](docs/EVALUATION.md) · [ABLATION_STUDY](docs/ABLATION_STUDY.md) · [TESTING](docs/TESTING.md) · [LIMITATIONS](docs/LIMITATIONS.md)
DEPLOYMENT, PORTFOLIO 문서는 해당 Phase에서 실제 결과와 함께 작성합니다.
