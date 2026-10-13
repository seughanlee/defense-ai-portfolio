# 실행·배포

| 환경 | 상태 |
|---|---|
| Windows 10 + Python 3.14 (venv) | **검증함** (2026-10-13: 테스트 87개, uvicorn 실서버 확인) |
| Ubuntu / WSL2 + Python venv | 절차 작성, **미검증** (이 PC에 Linux 배포판 없음) |
| Docker Compose (ai-api + web-dashboard + postgres) | **미구현** — Docker 미설치 (Phase 4) |

## 1. 공통 준비물
- Python 3.12 이상 (개발은 3.14.7), CPU만으로 동작. RAM 4 GB 이상 권장 — 실측 서비스 최대 메모리 1.18 GB, 학습 1.8 GB (metal_nut, B0).
- 디스크: MVTec AD 압축본 5.3 GB + 4개 카테고리 풀면 1.1 GB, 백본 가중치 캐시 약 0.7 GB.
- 학습된 아티팩트(`metadata.json` + `memory_bank.safetensors`). 저장소에는 없으니 README의 학습 명령으로 만듭니다.

## 2. Windows (검증함)
```powershell
cd defense-ai-portfolio
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
cd project4_defectvision
..\.venv\Scripts\python.exe -m pip install -r requirements.txt -r backend\requirements.txt
..\.venv\Scripts\python.exe -m pytest -q
$env:DV_MODEL_ARTIFACT_DIR = "artifacts\models\metal_nut\patchcore\1.0.0"
$env:HF_HUB_OFFLINE = "1"   # 가중치를 한 번 받은 뒤에는 오프라인 실행
..\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

## 3. Ubuntu / WSL2 (미검증)
```bash
sudo apt-get update && sudo apt-get install -y python3-venv python3-pip libgl1 libglib2.0-0   # OpenCV 런타임
git clone https://github.com/seughanlee/defense-ai-portfolio.git && cd defense-ai-portfolio
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
cd project4_defectvision && pip install -r requirements.txt -r backend/requirements.txt
python -m pytest -q
export DV_MODEL_ARTIFACT_DIR=artifacts/models/metal_nut/patchcore/1.0.0 HF_HUB_OFFLINE=1
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```
WSL2에서 Windows 쪽 폴더(`/mnt/c/...`)를 쓰면 파일 입출력이 느립니다. 저장소와 데이터는 WSL 안쪽 경로(`~/...`)에 두는 것을 권장합니다.

## 4. 환경변수 (`DV_` 접두사)
| 변수 | 기본값 | 설명 |
|---|---|---|
| `DV_ENV` | `dev` | `dev` / `test` / `prod`. `prod`는 아래 안전 조건이 없으면 기동을 거부 |
| `DV_MODEL_ARTIFACT_DIR` | 없음 | 서비스할 아티팩트 폴더. 없으면 Demo Mode(검사 요청 503) |
| `DV_MODEL_DEVICE` | 아티팩트 값 | 예: `cpu` |
| `DV_DATABASE_URL` | `sqlite:///.../var/defectvision.db` | 운영: `postgresql+psycopg://user:pass@host:5432/db` |
| `DV_STORAGE_DIR` | `var/inspections` | 원본·heatmap·overlay·mask PNG 저장 위치 |
| `DV_MAX_UPLOAD_MB` | 10 | 업로드 크기 상한 |
| `DV_MAX_IMAGE_PIXELS` | 40,000,000 | 픽셀 수 상한 (압축 폭탄 방어) |
| `DV_API_KEY` | 없음 | 설정하면 health를 뺀 모든 요청에 `X-API-Key` 필요. `prod`에서는 16자 이상 필수 |
| `DV_CORS_ORIGINS` | `["http://localhost:5173", ...]` | JSON 배열. `prod`에서 `*` 금지 |
| `DV_ENABLE_DOCS` | `true` | `/docs` (OpenAPI) 노출 여부. 외부 공개 시 `false` 권장 |
| `HF_HUB_OFFLINE` | (미설정) | `1`이면 백본 가중치를 캐시에서만 읽음 — 폐쇄망 권장 |

## 5. 운영 시 확인 사항
- **Readiness:** 오케스트레이터(로드밸런서, Compose healthcheck)는 `/api/v1/health/ready`를 봐야 합니다. `/health`는 모델이 없어도 200입니다.
- **워커 수:** 추론은 프로세스당 하나씩 처리합니다(잠금). 워커를 늘리면 처리량은 늘지만 워커마다 모델(약 1.2 GB)을 따로 올립니다.
- **지연:** CPU 4스레드 기준 이미지 1장 평균 744 ms, P95 838 ms (metal_nut, B0). 실시간 라인 속도가 필요하면 GPU나 더 가벼운 설정이 필요합니다 — [ABLATION_STUDY](ABLATION_STUDY.md)에서 측정 예정.
- **폐쇄망:** 백본 가중치(Hugging Face 캐시)를 미리 받아 함께 배포하고 `HF_HUB_OFFLINE=1`로 실행합니다. 아티팩트는 로드할 때 백본 가중치 해시를 검사하므로 다른 가중치가 섞이면 기동 시 `failed` 상태가 됩니다.
- **보관:** `DV_STORAGE_DIR`과 DB는 계속 늘어납니다. 보존 기간 정책은 아직 없습니다 ([LIMITATIONS](LIMITATIONS.md)).
- **스키마 변경:** 마이그레이션 도구가 없어서, 테이블 구조를 바꾸면 DB를 새로 만들어야 합니다.

## 6. Docker Compose (Phase 4, 미구현)
계획: `ai-api`(비루트 사용자, 가중치·아티팩트 볼륨, `/health/ready` healthcheck), `web-dashboard`(정적 빌드 + nginx),
`postgres`(볼륨, healthcheck), CPU 기본 구성. Docker 설치 후 작성하고 실제 `docker compose up`으로 검증한 뒤 이 절을 채웁니다.
