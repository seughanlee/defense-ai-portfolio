# 프로젝트 4. DefectVision AI — 정밀 부품 외관 이상탐지 검사 시스템

> **상태: Phase 1 (ML 코어) 구현 완료, 실제 데이터 성능은 아직 측정하지 않음 (NOT_RUN).**
> 아래 문서의 모든 성능 칸은 MVTec AD로 실험을 돌린 뒤에 채웁니다. 합성 데이터 결과는 성능 근거로 쓰지 않습니다.

## 요약
정밀 제조 부품 이미지에서 정상/이상을 판정하고, 결함 위치를 픽셀 단위 heatmap으로 추정하는 검사 시스템입니다.
**정상 이미지만으로 학습**하는 PatchCore(비지도 이상탐지)가 주력 모델이고, 결과를 REST API와 웹 대시보드로 제공하는 것이 최종 목표입니다.

포트폴리오 공통 주제인 **"AI의 판단을 믿어도 되는가"** 를 이 프로젝트에서는 이렇게 구체화합니다.
- 임계값을 테스트셋에 맞추지 않았을 때, 실제 라인에서 기대할 수 있는 오경보율·미검출률은 얼마인가?
- heatmap이 가리키는 위치는 실제 결함과 얼마나 겹치는가? (pixel AUROC, AUPRO, IoU)
- 조명·노이즈·블러가 바뀌면 판정이 어떻게 흔들리는가?

공개 민간 산업 데이터(MVTec AD)만 사용합니다. 특정 기업의 내부 시스템이나 군 장비 검사를 재현한다고 주장하지 않습니다.

## 진행 현황
| Phase | 내용 | 상태 |
|---|---|---|
| 0 | 저장소·환경 점검, 설계 | 완료 ([ARCHITECTURE.md](docs/ARCHITECTURE.md)) |
| 1 | 데이터 검증, PatchCore 학습/추론, heatmap, 평가 코드, 스모크 테스트 | **코드 완료, MVTec 실험 NOT_RUN** (데이터 미확보) |
| 2 | FastAPI 백엔드 (검사 API, 모델 로딩, 이력 저장) | 예정 |
| 3 | React 대시보드 | 예정 — Node.js 미설치 |
| 4 | Docker Compose, PostgreSQL, CI | 예정 — Docker 미설치 |
| 5 | 비교 모델(CNN), 임계값 검증, 강건성, 벤치마크 보고서 | 예정 |

## 실행 방법 (Phase 1)
작업 폴더: `project4_defectvision/`. 가상환경은 저장소 공용 `../.venv`를 씁니다.

```powershell
# 0) 의존성 (저장소 공용 venv에 추가 설치)
..\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 1) 테스트 (오프라인, 약 20초. 랜덤 백본 + 합성 데이터)
..\.venv\Scripts\python.exe -m pytest -q

# 2) 데이터셋 검증 (MVTec AD를 data/mvtec_ad/ 에 풀어 둔 뒤)
..\.venv\Scripts\python.exe scripts\inspect_dataset.py --data-root data\mvtec_ad --categories metal_nut screw capsule transistor

# 3) 학습: train/good만 사용, 그중 20%는 임계값 보정용으로 분리
..\.venv\Scripts\python.exe -m ml.training.train_patchcore --data-root data\mvtec_ad --category metal_nut --version 1.0.0

# 4) 평가: 공식 test split, 임계값은 아티팩트에 저장된 값 그대로
..\.venv\Scripts\python.exe -m ml.evaluation.evaluate --artifact artifacts\models\metal_nut\patchcore\1.0.0 --data-root data\mvtec_ad
```
데이터셋 없이 파이프라인만 확인하려면:
```powershell
..\.venv\Scripts\python.exe scripts\make_synthetic.py --out data\synthetic
..\.venv\Scripts\python.exe -m ml.training.train_patchcore --data-root data\synthetic --category synthetic_part --set model.pre_trained=false preprocessing.image_size=64
```
`pre_trained=true`(기본값)는 첫 실행 때 timm 백본 가중치를 Hugging Face Hub에서 내려받습니다.

## 폴더 구조
| 경로 | 내용 |
|---|---|
| `ml/configs/` | YAML 설정 (이미지 크기, 백본, coreset 비율, 임계값 규칙, 디바이스) |
| `ml/datasets/` | MVTec 구조 검증, 분할, 전처리, 합성 스모크 데이터 |
| `ml/inference/patchcore.py` | anomalib `PatchcoreModel` 래퍼: fit / calibrate / predict / save / load |
| `ml/training/` | 학습 CLI |
| `ml/evaluation/` | 지표(AUROC, AP, P/R/F1, pixel AUROC, AUPRO, IoU, 지연시간)와 평가 CLI |
| `ml/visualization/` | heatmap, overlay, 이진 마스크 패널 |
| `scripts/` | 데이터셋 점검, 합성 데이터 생성 |
| `tests/` | pytest (56개) |
| `docs/` | 설계·데이터·모델·평가·테스트 문서 |
| `data/`, `artifacts/` | git 제외 (재생성 가능) |

## 문서
[ARCHITECTURE](docs/ARCHITECTURE.md) · [DATASET](docs/DATASET.md) · [MODEL_DESIGN](docs/MODEL_DESIGN.md) · [EVALUATION](docs/EVALUATION.md) · [TESTING](docs/TESTING.md)
DEPLOYMENT, LIMITATIONS, PORTFOLIO 문서는 해당 Phase에서 실제 결과와 함께 작성합니다.
