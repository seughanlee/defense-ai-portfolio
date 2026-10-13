# 아키텍처와 개발 계획

## 1. Phase 0 점검 결과 (2026-10-12)
| 항목 | 결과 | 영향 |
|---|---|---|
| OS | Windows 10 Pro 19045, Git Bash / PowerShell 5.1 | 개발은 Windows, 배포 문서는 Ubuntu/WSL2 기준으로 별도 작성 |
| Python | 3.14.7, 저장소 공용 `.venv` | anomalib 2.7.0 / torchvision 0.29.1 / pytest 9.1 추가 설치. 기존 패키지 버전 변경 없음 (dry-run으로 사전 확인) |
| PyTorch | 2.14.1+cpu | GPU 없음 → CPU 기본 구성 |
| CPU / RAM / 디스크 | i5-1135G7 4코어 8스레드 / 15.7 GB / 여유 57.9 GB | 학습 스레드 4개로 제한 (다른 학습 작업과 공유) |
| Node.js | **미설치** | Phase 3 (React) 차단 |
| Docker / Compose | **미설치** | Phase 4 차단 |
| WSL | 배포판 없음 | Linux 실행 검증 차단 |
| 데이터 | MVTec AD 없음, 백본 가중치 캐시 없음 | 실제 성능 실험 차단 → 사용자 동의 후 다운로드 |
| 저장소 | `defense-ai-portfolio` (프로젝트 1–3), `origin` GitHub 연결 | `project4_defectvision/` 폴더로 추가. 미커밋 파일 `project3_phm/monitor.py`는 건드리지 않음 |

## 2. 설계 원칙
- **수직 단위 먼저:** 데이터 → 모델 → 평가가 끝까지 도는 최소 경로를 먼저 만들고, API·UI는 그 위에 얹습니다.
- **학습과 추론 분리:** 학습 CLI는 아티팩트(메모리 뱅크 + 메타데이터)를 만들고, 추론 쪽(API)은 아티팩트만 읽습니다.
- **pickle 금지:** 메모리 뱅크는 safetensors, 나머지는 JSON. 로드 시 파일 해시와 백본 가중치 해시를 검증합니다.
- **테스트셋 격리:** 임계값과 하이퍼파라미터는 train/good에서 떼어 낸 정상 검증셋으로만 정합니다.
- **설정 파일 우선:** 이미지 크기, 백본, 레이어, coreset 비율, 임계값 규칙, 디바이스, 스레드 수는 YAML로 바꿉니다.

## 3. ML 파이프라인
```
이미지 (RGB uint8)
  └─ preprocess: 정사각 resize(INTER_AREA) → ImageNet 정규화          ml/datasets/transforms.py
      └─ 백본 wide_resnet50_2 layer2 + layer3 (timm, eval 고정)        anomalib TimmFeatureExtractor
          └─ 3×3 avg-pool → 업샘플·결합 → 패치 임베딩 (1536차원)
              ├─ [fit]  정상 이미지 패치 전체 → k-center greedy coreset → memory bank
              └─ [predict] 패치별 최근접 거리 → 이미지 점수(재가중) + anomaly map(업샘플, σ=4 블러)
                  └─ 임계값 비교 (정상 검증셋 분위수) → 정상/이상 판정, 이진 결함 마스크
```

## 4. 아티팩트 구조
```
artifacts/models/<category>/patchcore/<version>/
├── memory_bank.safetensors      # (N, 1536) float32
├── metadata.json                # 모델명·버전·설정·데이터셋 지문·분할 목록·임계값·체크섬·라이브러리 버전·평가 요약
└── evaluation/                  # evaluate 실행 시 생성: evaluation.json, per_image.csv, report.md, panels/
```
로드 시 거부 조건: 파일 누락, JSON 손상, 포맷 버전 불일치, 메모리 뱅크 SHA-256 불일치, 백본 가중치 해시 불일치.

## 5. 전체 시스템 (목표)
```
React 대시보드 ──HTTP──▶ FastAPI (ai-api) ──▶ PatchCoreDetector (프로세스당 1회 로드)
                              │
                              └──▶ PostgreSQL (검사 이력, 모델 버전)
                              └──▶ 결과 이미지 저장소 (볼륨, UUID 파일명만 사용)
```
- API는 기동 직후 `ready=false`로 시작해, 아티팩트 검증과 백본 로딩이 끝난 뒤에만 readiness를 통과시킵니다.
- 모델이 없으면 대시보드는 "Demo Mode"를 표시하고, 가짜 점수를 만들지 않습니다.

## 5.1 백엔드 (Phase 2, 구현 완료)
```
backend/app/
├── main.py              create_app(): 설정·DB·저장소·모델 서비스 조립, 요청 ID·크기 제한·오류 미들웨어
├── core/config.py       DV_* 환경변수 (dev/test/prod). prod는 API 키·PostgreSQL·명시적 CORS가 없으면 기동 거부
├── core/logging.py      JSON 한 줄 로그 + request_id. 파일 내용·사용자 파일명·비밀값은 기록하지 않음
├── core/errors.py       {"error": {code, message, request_id}} 형식 통일
├── db/                  SQLAlchemy 2: inspections 테이블, 요청 단위 세션 (끝에서 commit, 오류 시 rollback)
├── services/
│   ├── model_service.py 상태 not_configured / loading / ready / failed. 백그라운드 로딩, 추론 직렬화(잠금)
│   ├── image_validation.py  내용 기반 형식 판별, 크기·픽셀 수 제한, 전체 디코딩, 파일명 정리
│   └── storage.py       결과 PNG 저장. 경로는 서버가 만든 UUID + 고정 이름만 사용
├── schemas/             Pydantic 응답 모델
└── api/routes.py        엔드포인트
```

| 메서드 · 경로 | 설명 | 주요 상태 코드 |
|---|---|---|
| `GET /api/v1/health` | 생존 확인 + 모델 상태 (인증 없음) | 200 |
| `GET /api/v1/health/ready` | 검증된 모델이 로드된 경우만 200 (인증 없음) | 200 / 503 |
| `GET /api/v1/models` | 활성 모델 메타데이터 (버전, 데이터셋 지문, 임계값, 평가 요약, 체크섬) | 200 |
| `POST /api/v1/inspections` | multipart 이미지 검사 → 저장 | 201 / 400 / 413 / 415 / 422 / 503 |
| `GET /api/v1/inspections` | 이력 (page, page_size ≤ 100, prediction, model_version, since, until) | 200 / 422 |
| `GET /api/v1/inspections/{id}` | 상세 | 200 / 404 / 422 |
| `GET /api/v1/inspections/{id}/{original,heatmap,overlay,mask}` | 결과 PNG | 200 / 404 / 422 |
| `GET /api/v1/metrics/summary` | 건수, 이상 비율, 지연 p50/p95, 최근 24시간 시간대별 이상 비율, 모델 버전별 건수 | 200 |

검사 요청 처리 순서: 모델 준비 확인(503) → 본문 크기 제한(413) → 내용 기반 이미지 검증(400/413/415) → 추론(잠금)
→ 결과 PNG 저장 → DB flush. **flush가 실패하면 저장한 파일을 지우고** 트랜잭션을 되돌립니다. 처리되지 않은 예외는
미들웨어에서 request_id를 붙여 500으로 응답하며, 내부 메시지는 응답에 넣지 않고 로그에만 남깁니다.

## 6. Phase 계획
| Phase | 산출물 | 완료 기준 | 선행 조건 |
|---|---|---|---|
| 1 | ML 코어 | 합성 데이터로 전 경로 테스트 통과 ✅ / MVTec 확보·검증 ✅ / 실측 ablation ⏳ (CPU 수 시간, 보류 중) | — |
| 2 | `backend/` FastAPI | 6개 엔드포인트, 업로드 검증, readiness, pytest ✅ (PostgreSQL 실행 검증은 Phase 4) | Phase 1 |
| 3 | `frontend/` React+TS+Vite+Tailwind | 4개 화면, 오류 상태 테스트, 빌드 | Node.js LTS 설치 |
| 4 | Docker Compose (ai-api, web-dashboard, postgres), GitHub Actions | `docker compose up` 헬스체크 통과, CI green | Docker Desktop 또는 WSL2 (CI 워크플로는 작성 완료, push 전이라 미실행) |
| 5 | CNN 비교 모델, 임계값 검증, 강건성, 벤치마크 | 재현 가능한 보고서 | Phase 1 실측 |

## 7. 위험 요소
| 위험 | 대응 |
|---|---|
| CPU 전용이라 학습·추론이 느림 | coreset 비율·이미지 크기를 설정으로 조절, 지연시간을 실측해 문서화 |
| 정상 검증셋이 작아 임계값이 불안정 | 분위수 임계값의 한계를 보고서에 명시, Phase 5에서 부트스트랩으로 변동 폭 측정 |
| anomalib API 변경 | 버전 고정(2.7.0), 래퍼 계층에서만 사용 |
| timm 백본 가중치 버전 변경 | 백본 해시를 메타데이터에 저장, 불일치 시 로드 거부 |
| MVTec AD 비상업 라이선스 | 데이터·파생 가중치 미배포, 포트폴리오(비상업) 용도로만 사용 |
| Hugging Face Hub 접속 의존 | 가중치가 캐시에 있어도 로드 때마다 Hub에 요청을 보냄 → ablation 실행기는 기본 오프라인(`HF_HUB_OFFLINE=1`), 배포 이미지도 가중치 포함 + 오프라인 실행 예정 |
| 업로드 악용 (거대 파일, 압축 폭탄, 위장 파일, 경로 조작) | 크기·픽셀 수 제한, 내용 기반 판별, UUID 경로 — 각각 테스트로 고정 |
