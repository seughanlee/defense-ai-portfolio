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

## 범위
| 파일 | 다루는 요구사항 |
|---|---|
| `test_config.py` | 설정 로딩, 오버라이드, 잘못된 값·오타 키·테스트셋 튜닝 방식 거부 |
| `test_dataset.py` | 데이터셋 없음, 카테고리 없음, 빈 학습 폴더, 구조 검증, 마스크 누락·크기 불일치, 손상 이미지, train–test 누수, 분할 결정성·배타성, 흑백→RGB |
| `test_metrics_and_transforms.py` | 전처리 출력 형태·정규화, 잘못된 입력 거부, 마스크 이진성, 이미지·픽셀 지표 정답값, 지연시간 요약 |
| `test_patchcore_pipeline.py` | 학습 전 추론 거부, 빈 입력, 메모리 뱅크 생성, 학습 중 백본 불변, 점수·heatmap 형태, 임계값 판정, 저장·로드 동일성, 덮어쓰기 거부, 체크포인트 누락, 메모리 뱅크 변조, 백본 불일치, 손상 메타데이터, 학습→평가 전 경로 |

추가된 테스트 (`test_ablation.py`, `test_metrics_and_transforms.py`): 샘플링 seed 재현성(k-center, random), 전체 뱅크, 합성 결함 결정성·마스크 밖 픽셀 불변·입력 불변, 전경 마스크, 메모리 측정, 계획 병합·실행 순서·NOT_RUN 제외, 합성 데이터 ablation 전 경로, AUPRO 8-연결.

## 테스트가 잡은 버그
- Windows에서 safetensors `load_file`이 파일을 메모리 매핑한 채로 잡아 두어 아티팩트 파일을 지울 수 없었습니다. 이제 파일을 한 번 읽고, 그 바이트로 해시를 검증한 뒤 같은 바이트에서 로드합니다 (검증-사용 사이 틈도 제거).
- AUPRO가 4-연결로 영역을 나눠 공식 정의(8-연결)와 달랐습니다. 공식 평가 코드와 비교하다 발견해 수정했습니다.
- 학습 CLI는 seed 42로, 로더는 seed 0으로 랜덤 백본을 만들어 로드가 실패했습니다. 백본 해시 검증이 이 불일치를 잡았고, 생성 seed를 메타데이터에 저장하도록 고쳤습니다.

## 아직 없는 테스트
API, DB, 프론트엔드, Docker는 해당 Phase에서 추가합니다. Lint(ruff)는 미설치라 실행하지 않았습니다.
