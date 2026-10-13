# 평가

## 현재 상태
| 실험 | 상태 |
|---|---|
| MVTec AD 카테고리별 PatchCore 평가 | [ABLATION_STUDY.md](ABLATION_STUDY.md) 참고 |
| CNN 비교 모델 | NOT_RUN |
| 강건성 시험 | NOT_RUN |
| 최대 메모리 | ablation에서 측정 (프로세스 peak working set) · CPU/GPU 사용률은 NOT_RUN |
| 합성 데이터 스모크 테스트 | 실행함. **성능 근거 아님** (아래 참고) |

## 평가 규칙
1. 임계값과 하이퍼파라미터는 정상 검증셋(train/good의 20%)으로만 정합니다. 공식 test split의 결과를 보고 설정을 바꾸지 않습니다.
2. 임계값에 의존하는 지표(precision, recall, F1, confusion matrix, IoU)는 항상 사용한 임계값과 함께 기록합니다.
3. 픽셀 지표는 **모델 입력 해상도(기본 256 px)** 에서 계산합니다. 원본 해상도 평가와 다르므로 논문 수치와 직접 비교하지 않습니다.
4. IoU는 정답 마스크가 있는 결함 이미지에서만 평균합니다. 정상 이미지의 오검출은 이미지 단위 오경보율로 따로 봅니다.
5. 실행하지 않은 실험은 수치 대신 `NOT_RUN`으로 둡니다.

## 지표
| 수준 | 지표 | 구현 |
|---|---|---|
| 이미지 | AUROC, Average Precision | scikit-learn |
| 이미지 | precision / recall / F1 / 정상 오경보율 / confusion matrix @ 임계값 | `metrics.image_metrics` |
| 픽셀 | pixel AUROC | scikit-learn (전체 픽셀) |
| 영역 | AUPRO (FPR ≤ 0.3, 0~1 정규화) | `metrics.aupro` — 8-연결 영역 단위 겹침, 모든 점수를 임계값으로 쓰는 정확 계산. MVTec 공식 평가 코드와 일치 확인 (`scripts/crosscheck_aupro.py`, [결과](../results/aupro_crosscheck.csv)) |
| 픽셀 | mean IoU @ 픽셀 임계값 | `metrics.mean_iou` |
| 운영 | 모델 로드 시간, batch-1 지연 평균/p50/p95, 처리량 | `evaluate.py` (워밍업 2회 제외, 디코딩 후 시간) |

지표 함수 검증 (pytest): 완벽한 예측 → AUROC·AUPRO·IoU = 1, 무작위 map → AUPRO ≈ 0.15 (이론값 FPR 한계 0.3의 절반), 한 클래스만 있으면 `None`.

## 산출물
`artifacts/models/<category>/patchcore/<version>/evaluation/`
- `evaluation.json` — 전체 수치, 임계값 출처, 데이터셋 지문
- `per_image.csv` — 이미지별 점수와 판정 (실패 사례 분석용)
- `report.md` — 사람이 읽는 요약
- `panels/*.png` — 점수가 가장 높은 정상(오경보 후보)과 가장 낮은 결함(미검출 후보)

## 합성 스모크 테스트 기록 (성능 근거 아님)
2026-10-12, 랜덤 백본(`pre_trained=false`), 64 px, 합성 원판 20장 학습(fit 16 / val 4), test 12장.
- 파이프라인이 학습 → 보정 → 저장 → 로드(해시 검증) → 평가 → 패널 생성까지 끝까지 동작함을 확인했습니다.
- 관찰: 순위 지표는 높았지만 정상 6장 중 4장이 오경보였습니다. **검증 정상 이미지가 4장뿐이라 0.99 분위수 임계값이 낮게 잡힌 것**이 원인입니다.
  MVTec에서도 검증셋이 수십 장 수준이므로 같은 문제가 생길 수 있습니다. Phase 5에서 검증셋 크기에 따른 임계값 변동을 부트스트랩으로 측정합니다.

## 강건성 시험 계획 (Phase 5)
| 변형 | 가정 | 한계 |
|---|---|---|
| 밝기 ±, 감마 | 조명 변동은 결함이 아님 | `color` 결함(metal_nut 등)은 색·밝기 자체가 결함 신호라 해석에 주의 |
| 가우시안 노이즈, 블러 | 센서 잡음·초점 변화 | 강한 블러는 미세 긁힘을 실제로 지워 버림 → 미검출이 "모델 잘못"만은 아님 |
| JPEG 압축 | 전송·저장 손실 | 학습 데이터는 PNG |
| 회전 | 부품 자세 변화 | 방향이 결함 정의인 경우(예: 뒤집힘·잘못 놓임 유형)는 회전이 결함 의미를 바꿈 → 해당 카테고리는 제외하거나 따로 보고 |
