# Sprint 1 Backlog

## 목표: Calibration Set 구성 알고리즘 설계 및 구현, 대상 모델(Upstage Solar 우선) 스터디
**기간:** 3/17 ~ 3/23

---

## 백로그

| ID | 항목 | 사고 문서 | 상태 |
|----|------|-----------|------|
| S1-1 | 대상 모델 우선순위 변경 및 업스테이지(Solar)/Llama 아키텍처 검증 | [01_EXAONE_아키텍처_스터디.md](thoughts/01_EXAONE_아키텍처_스터디.md), [02_대상모델_변경_업스테이지_Llama.md](thoughts/02_대상모델_변경_업스테이지_Llama.md) | ✅ 완료 |
| S1-2 | 형태소 인식 캘리브레이션 셋 알고리즘 설계 | [03_캘리브레이션셋_알고리즘_설계.md](thoughts/03_캘리브레이션셋_알고리즘_설계.md), [04_나무위키_데이터셋_선정.md](thoughts/04_나무위키_데이터셋_선정.md) | ✅ 완료 |
| S1-3 | 데이터 토큰화 및 시각화 코드 구현 | `src/` 디렉토리 내 5개 모듈 | ✅ 완료 |
| S1-4 | 최적화된 Calibration Set 최종 추출 | `results/calibration_set_*.json` | ✅ 완료 |

---

## 핵심 추진 과제
- 양자화 라이브러리와 완벽히 호환되는 범용 아키텍처(Solar 10.7B 등) 적용을 통해, 파이프라인 버그 우회.
- 데이터 전처리 시 Kiwi를 사용하여 형태소를 분리한 뒤, 이를 LLM의 Tokenizer 경계에 어떻게 정렬(Align)시킬 것인가?

## S1-1 핵심 결론 요약 (3/17)
- **1순위:** Solar 10.7B (DUS 48-layer, Llama 2 기반, vocab 32K) — AutoGPTQ 완벽 호환 ✅
- **2순위:** Llama 3.1 Ko 8B (GQA 32-layer, vocab 128K) — AutoGPTQ 완벽 호환 ✅
- **3순위:** EXAONE 3.5 (커스텀 아키텍처) — AutoGPTQ 미지원, 추후 편입 🔄
- **핵심 인사이트:** 두 모델의 **Tokenizer vocab 차이(32K vs 128K)**가 한국어 형태소-토큰 정렬에 서로 다른 fragmentation 패턴을 유발 → S1-2 알고리즘 설계 시 반영 필요
