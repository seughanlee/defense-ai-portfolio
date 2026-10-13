# 테스트

## 실행
```powershell
..\.venv\Scripts\python.exe -m pytest -q                 # 전체 (오프라인, 약 20초)
..\.venv\Scripts\python.exe -m pytest -q -m "not slow"   # 백본을 만들지 않는 빠른 테스트 (약 4초)
```
`slow` 테스트는 랜덤 가중치 `wide_resnet50_2`를 만들어 쓰므로 네트워크가 필요 없습니다. 실제 MVTec AD가 필요한 테스트(`dataset` 마커)는 아직 없습니다.

## 실행 기록
| 날짜 | 환경 | 결과 |
|---|---|---|
| 2026-10-12 | Windows 10, Python 3.14.7, torch 2.14.1+cpu, anomalib 2.7.0 | **43 passed** in 19.5 s |
| 2026-10-12 (ablation 추가 후) | 같은 환경 + psutil 7.2.2 | **56 passed** in 63 s |
| 2026-10-13 (백엔드 추가 후) | 같은 환경 + FastAPI 0.143, SQLAlchemy 2.1.4, `HF_HUB_OFFLINE=1` | **87 passed** in 51 s (ML 57 + 백엔드 30), ruff 통과 |

## 범위
| 파일 | 다루는 요구사항 |
|---|---|
| `test_config.py` | 설정 로딩, 오버라이드, 잘못된 값·오타 키·테스트셋 튜닝 방식 거부 |
| `test_dataset.py` | 데이터셋 없음, 카테고리 없음, 빈 학습 폴더, 구조 검증, 마스크 누락·크기 불일치, 손상 이미지, train–test 누수, 분할 결정성·배타성, 흑백→RGB |
| `test_metrics_and_transforms.py` | 전처리 출력 형태·정규화, 잘못된 입력 거부, 마스크 이진성, 이미지·픽셀 지표 정답값, 지연시간 요약 |
| `backend/tests/test_api.py` | health·readiness·Demo Mode(가짜 점수 없이 503), 모델 로딩 실패, 검사 왕복(점수·판정·결과 PNG 4종), 형식(PNG/JPEG/BMP/흑백), 페이지네이션·필터, 통계, 재기동 후 유지, 빈 파일·위장 파일·잘린 PNG·GIF·필드 누락·크기 초과·픽셀 수 초과(압축 폭탄), 없는/잘못된 ID, 경로 조작, 파일명 정리, API 키, **DB 실패 시 롤백 + 저장 파일 삭제 + 500에 request_id**, 운영 설정 가드, 로그에 파일명 미기록, OpenAPI 문서 |
| `backend/tests/test_api_real_model.py` | 실제 PatchCore 아티팩트(합성 데이터, 랜덤 백본)를 API로 서비스: readiness → 검사 → heatmap 원본 해상도 복원 → 모델 메타데이터 |
| `test_patchcore_pipeline.py` | 학습 전 추론 거부, 빈 입력, 메모리 뱅크 생성, 학습 중 백본 불변, 점수·heatmap 형태, 임계값 판정, 저장·로드 동일성, 덮어쓰기 거부, 체크포인트 누락, 메모리 뱅크 변조, 백본 불일치, 손상 메타데이터, 학습→평가 전 경로 |

추가된 테스트 (`test_ablation.py`, `test_metrics_and_transforms.py`): 샘플링 seed 재현성(k-center, random), 전체 뱅크, 합성 결함 결정성·마스크 밖 픽셀 불변·입력 불변, 전경 마스크, 메모리 측정, 계획 병합·실행 순서·NOT_RUN 제외, 합성 데이터 ablation 전 경로, AUPRO 8-연결.

## 테스트가 잡은 버그
- Windows에서 safetensors `load_file`이 파일을 메모리 매핑한 채로 잡아 두어 아티팩트 파일을 지울 수 없었습니다. 이제 파일을 한 번 읽고, 그 바이트로 해시를 검증한 뒤 같은 바이트에서 로드합니다 (검증-사용 사이 틈도 제거).
- AUPRO가 4-연결로 영역을 나눠 공식 정의(8-연결)와 달랐습니다. 공식 평가 코드와 비교하다 발견해 수정했습니다.
- 학습 CLI는 seed 42로, 로더는 seed 0으로 랜덤 백본을 만들어 로드가 실패했습니다. 백본 해시 검증이 이 불일치를 잡았고, 생성 seed를 메타데이터에 저장하도록 고쳤습니다.

- 처리되지 않은 예외가 요청 미들웨어 바깥에서 처리되어 500 응답·로그에 request_id가 빠졌습니다 (DB 실패 테스트에서 발견). 미들웨어 안에서 잡도록 고쳤습니다.

## 실서버 확인 (2026-10-13)
uvicorn으로 합성 데이터 아티팩트를 띄워 HTTP로 확인: 모델 로드 약 9 s 후 `/health/ready` 200, 검사 201·DB 저장, 텍스트 파일 업로드 400 `INVALID_IMAGE`, 통계 반영, 로그 12줄에 업로드 파일명 없음.

## 아직 없는 테스트
- PostgreSQL 실DB 테스트 (Docker 설치 후 Phase 4). 지금은 SQLite로만 검증했습니다.
- 프론트엔드, Docker, 동시 요청 부하 시험.
- CI 워크플로(`.github/workflows/project4-defectvision.yml`)는 작성했지만 push 전이라 GitHub에서 아직 실행하지 않았습니다.
