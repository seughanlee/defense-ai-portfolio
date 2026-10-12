# 방산(대공방어 체계) AI 엔지니어 소형 프로젝트 포트폴리오

L-SAM 같은 다기능레이더(MFR) 기반 대공방어 체계 기업(LIG넥스원, 한화시스템, 한화에어로스페이스 등)에 지원하기 위한
소형 프로젝트 모음입니다. **공개 자료와 합성(시뮬레이션) 데이터만 사용**합니다. 실제 체계 데이터와 보안 사항은 다루지 않습니다.

## 방산 AI 직무에서 보는 역량

| 역량 | 일반 IT AI 직무와의 차이 |
|---|---|
| 신호처리·추적 기초 | 칼만필터, 레이더 방정식, 좌표 변환 등 **도메인 수학**이 기본 소양 |
| 신뢰성·검증(V&V) | 정확도보다 **일관성(consistency), 최악 사례, 재현성**이 중요 |
| 임베디드·실시간 | C/C++, 경량화 모델, 지연시간 상한 보장 |
| 폐쇄망 환경 | 외부 API 없이 동작하는 **온프레미스** 모델과 파이프라인 |
| 문서화 | 무기체계 SW 개발 절차에 맞춘 요구사항·시험 문서 작성 능력 |

## 프로젝트 로드맵 (각 2~4주)

| # | 프로젝트 | 체계 연관성 | 핵심 기술 | 상태 |
|---|---|---|---|---|
| 1 | [레이더 표적 추적 필터 비교 (KF / IMM / CT-EKF)](project1_radar_tracking/) | MFR 추적 처리 | 칼만필터, IMM, EKF, 몬테카를로, NEES 일관성, 강건성 시험 | **v2 완료** |
| 2 | [마이크로도플러 기반 표적 분류](project2_microdoppler/) | 표적 식별(드론, 조류, 회전익) | 물리 기반 시뮬레이터, CNN vs 특징+RF, 보정, 미지 표적 탐지, ONNX INT8 | **v1 완료** |
| 3 | [장비 고장 예측(PHM)](project3_phm/) | 체계 가동률·후속 군수지원 | NASA C-MAPSS, 잔여수명(RUL) 예측, conformal 예측 구간, 정비 비용 분석 | **v1 완료** |
| 4 | 폐쇄망 기술문서 RAG | 정비 교범·규격 검색 | 로컬 LLM, 하이브리드 검색, 평가셋 구축 | 예정 |

### 진행 순서를 이렇게 잡은 이유
1. **추적 필터(1)**: 레이더 체계 엔지니어와 대화할 공통 언어입니다. AI 모델 이전에 갖춰야 하는 기초입니다.
2. **표적 분류(2)**: 딥러닝 역량과 임베디드 배포(경량화)를 함께 보여줍니다.
3. **PHM(3)**: 방산 업계에서 AI 실적용 사례가 가장 많은 분야이고, 공개 데이터셋으로 바로 시작할 수 있습니다.
4. **RAG(4)**: 폐쇄망 LLM 수요가 커지고 있고, 일반 AI 직무와도 겹치는 기술입니다.

## 모든 프로젝트에 공통으로 적용할 기준
- 베이스라인 대비 정량 비교표
- 몬테카를로 반복 실험으로 통계적 신뢰성 확보
- 실패 사례와 한계를 README에 명시
- 시드 고정으로 `python run_*.py` 한 번에 결과 재현

## 저장소 구조
| 경로 | 내용 |
|---|---|
| `project1_radar_tracking/` | 칼만필터·IMM 추적 (numpy만 사용) |
| `project2_microdoppler/` | 마이크로도플러 시뮬레이터와 표적 분류 |
| `project3_phm/` | C-MAPSS 잔여수명 예측과 예측 구간 |
| `prompts/` | 프로젝트 실행용 프롬프트 |
| [CHANGELOG.md](CHANGELOG.md) | 버전별 검증 질문과 결론 |

데이터(`data/`)와 학습된 가중치(`checkpoints/`)는 저장소에 넣지 않습니다. 각 프로젝트 README의 실행 방법대로 다시 만들 수 있습니다.
- 프로젝트 2: `python dataset.py`로 시뮬레이션 데이터 생성 (약 3분)
- 프로젝트 3: NASA PCoE 저장소에서 C-MAPSS를 받아 `project3_phm/data/cmapss/`에 풂 (출처와 SHA-256은 [EVALUATION.md](project3_phm/EVALUATION.md))

## 환경 설정
```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r project1_radar_tracking/requirements.txt -r project2_microdoppler/requirements.txt -r project3_phm/requirements.txt
```
개발 환경: Windows 10, Python 3.14, CPU 전용 (8스레드).

## 버전 태그
완료된 버전마다 태그를 붙입니다: `p1-v2.1`, `p2-v1`, …  자세한 내용은 [CHANGELOG.md](CHANGELOG.md)에 있습니다.
