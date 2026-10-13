# 데이터셋

## MVTec AD
- 출처: MVTec Software GmbH, "MVTec Anomaly Detection Dataset" — https://www.mvtec.com/company/research/datasets/mvtec-ad
- 논문: Bergmann et al., *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*, CVPR 2019;
  Bergmann et al., *The MVTec Anomaly Detection Dataset*, IJCV 2021 (PRO 지표 정의)
- 라이선스: **CC BY-NC-SA 4.0** (비상업, 출처 표시, 동일 조건 변경 허락). 데이터와 이 데이터로 만든 메모리 뱅크는 저장소에 올리지 않습니다.
- 다운로드: 라이선스에 동의한 뒤, anomalib 2.7.0 소스(`anomalib/data/datamodules/image/mvtecad.py`의 `DOWNLOAD_INFO`)에 기록된
  MVTec 공식 링크(mydrive.ch)로 받았습니다. URL을 추측하지 않았고, anomalib에 기록된 해시로 검증했습니다.

| 파일 | 크기 | SHA-256 | 받은 날 |
|---|---|---|---|
| `mvtec_anomaly_detection.tar.xz` | 5,264,982,680 bytes | `cf4313b13603bec67abb49ca959488f7eedce2a9f7795ec54446c649ac98cd3d` (anomalib 기록값과 일치) | 2026-10-12 ~ 13 (서버가 연결을 끊어 이어받기로 완료) |

압축본 전체 중 아래 4개 카테고리와 `license.txt`, `readme.txt`만 `data/mvtec_ad/`에 풀었습니다 (1.1 GB).

### 디렉터리 구조 (코드가 기대하는 형식)
```
data/mvtec_ad/<category>/
├── train/good/*.png                      # 정상만 존재
├── test/good/*.png                       # 정상 (마스크 없음)
├── test/<defect_type>/*.png              # 결함 유형별 폴더
└── ground_truth/<defect_type>/<stem>_mask.png   # 결함 이미지 1장당 마스크 1장 (0/255)
```

### 사용 카테고리 (계획)
정밀 부품과 관련 있는 `metal_nut`, `screw`, `capsule`, `transistor`를 우선 사용하고, 비교용으로 `hazelnut`을 둘 수 있습니다.
**카테고리마다 모델을 따로 학습합니다** (카테고리 간 메모리 뱅크를 섞지 않음). 결과도 카테고리별로 따로 보고합니다.
카테고리별 이미지 수·해상도·채널은 데이터를 받은 뒤 `scripts/inspect_dataset.py` 출력으로 채웁니다. 문헌 값을 옮겨 적지 않습니다.

| category | train/good | test/good | test 결함 유형 (장수) | 해상도·모드 | 지문 (SHA-256 앞 12자) |
|---|---|---|---|---|---|
| metal_nut | 220 | 22 | bent 25, color 22, flip 23, scratch 23 | 700×700 RGB | `6f2234a047df` |
| screw | 320 | 41 | manipulated_front 24, scratch_head 24, scratch_neck 25, thread_side 23, thread_top 23 | 1024×1024 **흑백(L)** | `bf27cac48f8d` |
| capsule | 219 | 23 | crack 23, faulty_imprint 22, poke 21, scratch 23, squeeze 20 | 1000×1000 RGB | `8c624edd7a8c` |
| transistor | 213 | **60** | bent_lead 10, cut_lead 10, damaged_case 10, misplaced 10 | 1024×1024 RGB | `930c02c31a1d` |

검증 결과 (2026-10-13, `scripts/inspect_dataset.py`): 4개 카테고리 모두 **PASS** — 손상 파일 0, 마스크 누락·크기 불일치 0, 고아 마스크 0, train–test 동일 파일 0.

데이터에서 확인한 주의점:
- **transistor는 test에서 정상(60)이 결함(40)보다 많습니다.** 다른 카테고리와 양성 비율이 달라 AP를 카테고리끼리 직접 비교하면 안 됩니다.
- **metal_nut `flip`과 transistor `misplaced`는 방향·위치 자체가 결함입니다.** 회전·이동 강건성 시험은 이 유형의 의미를 바꾸므로 따로 다룹니다 ([EVALUATION.md](EVALUATION.md)).
- screw는 흑백이라 3채널로 복제해 넣습니다. 컬러 정보가 없는 입력에 ImageNet 컬러 백본을 쓰는 셈입니다.

## 분할 규칙
| 분할 | 출처 | 용도 |
|---|---|---|
| fit | 공식 `train/good`의 80% | 메모리 뱅크 구축 |
| val | 공식 `train/good`의 20% (seed 42, 파일 목록을 메타데이터에 기록) | 정상 분포 기반 임계값 보정 |
| test | 공식 `test/*` 전체 | 최종 평가 **전용**. 학습·임계값·하이퍼파라미터 결정에 사용하지 않음 |

- 이상 샘플로 임계값을 최적화하지 않습니다. 공식 분할에는 이상 샘플이 test에만 있기 때문입니다.
- `test/good`은 마스크 파일이 없습니다. 픽셀 지표를 계산할 때만 "결함 없음 = 전부 0인 마스크"로 취급하며, UI에는 정답 마스크가 있는 것처럼 표시하지 않습니다.
- 흑백 카테고리(예: screw)는 3채널로 복제해 같은 백본에 넣습니다.

## 검증 항목 (`validate_category`)
| 항목 | 판정 |
|---|---|
| 필수 폴더 존재, train/test 이미지 1장 이상 | 없으면 `DatasetError` (실행 중단) |
| 이미지 손상 (PIL verify + 전체 디코딩) | 오류 |
| 결함 이미지의 마스크 누락, 마스크 크기 불일치 | 오류 |
| 마스크 값이 0/255가 아님, 결함 이미지의 빈 마스크 | 경고 |
| 이미지 없는 고아 마스크 | 경고 |
| train–test 간 동일 파일 (SHA-256) | **오류 (데이터 누수)** |
| 같은 분할 안 중복 | 경고 |
| 해상도·모드 혼재 | 경고 |
| 데이터셋 지문 | 전체 파일 (경로, SHA-256) 목록의 해시. 아티팩트 메타데이터에 기록 |

근접 중복(재압축·리사이즈된 같은 사진)은 아직 검사하지 않습니다 (지각 해시는 Phase 5 후보).

## 전처리와 증강
- 전처리: 정사각 resize (`INTER_AREA`, 기본 256 px) → ImageNet 평균·표준편차 정규화. 마스크는 최근접 보간.
- 증강: PatchCore 메모리 뱅크에는 **증강을 쓰지 않습니다.** 증강된 "정상"이 메모리 뱅크에 들어가면 회전·색 변화 같은 실제 결함 신호까지 정상으로 흡수할 수 있기 때문입니다. 증강은 Phase 5의 CNN 비교 모델 학습에만 적용합니다.

## 합성 스모크 데이터
`ml/datasets/synthetic.py`가 MVTec과 같은 구조의 작은 데이터(회색 원판 + 긁힘/점 결함)를 만듭니다.
파이프라인 동작 확인용이며, 폴더에 `SYNTHETIC_DATA_NOT_FOR_PERFORMANCE_CLAIMS.txt`를 남기고 평가 보고서 상태를 `SYNTHETIC_SMOKE_TEST`로 표시합니다.
