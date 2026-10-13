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

## 6. Phase 계획
| Phase | 산출물 | 완료 기준 | 선행 조건 |
|---|---|---|---|
| 1 | ML 코어 | 합성 데이터로 전 경로 테스트 통과 ✅ / MVTec 1개 카테고리 실측 ⏳ | MVTec AD, 백본 가중치 다운로드 동의 |
| 2 | `backend/` FastAPI | 6개 엔드포인트, 업로드 검증, readiness, pytest | Phase 1 |
| 3 | `frontend/` React+TS+Vite+Tailwind | 4개 화면, 오류 상태 테스트, 빌드 | Node.js LTS 설치 |
| 4 | Docker Compose (ai-api, web-dashboard, postgres), GitHub Actions | `docker compose up` 헬스체크 통과, CI green | Docker Desktop 또는 WSL2 |
| 5 | CNN 비교 모델, 임계값 검증, 강건성, 벤치마크 | 재현 가능한 보고서 | Phase 1 실측 |

## 7. 위험 요소
| 위험 | 대응 |
|---|---|
| CPU 전용이라 학습·추론이 느림 | coreset 비율·이미지 크기를 설정으로 조절, 지연시간을 실측해 문서화 |
| 정상 검증셋이 작아 임계값이 불안정 | 분위수 임계값의 한계를 보고서에 명시, Phase 5에서 부트스트랩으로 변동 폭 측정 |
| anomalib API 변경 | 버전 고정(2.7.0), 래퍼 계층에서만 사용 |
| timm 백본 가중치 버전 변경 | 백본 해시를 메타데이터에 저장, 불일치 시 로드 거부 |
| MVTec AD 비상업 라이선스 | 데이터·파생 가중치 미배포, 포트폴리오(비상업) 용도로만 사용 |
