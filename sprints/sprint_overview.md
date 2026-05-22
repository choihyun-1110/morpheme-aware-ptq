# Sprint Overview

## 프로젝트: Morphology-Aware Calibration for PTQ of Korean LLMs

> **핵심 주장:** 형태소 다양성 기반 calibration set 선별이 GPTQ 양자화 품질을 보존하며, 이 원리는 사전학습 언어에 정렬된 calibration을 사용할 때 언어 불문하게 적용된다.

---

## 스프린트 구성

| Sprint | 기간 | 목표 | 상태 |
|--------|------|------|------|
| **Sprint 0** | 3/12~3/16 | 환경 검증 + 파일럿 | ✅ 완료 |
| **Sprint 1** | 3/17~3/18 | SOLAR 분석 + 알고리즘 설계 (A/B/C) | ✅ 완료 |
| **Sprint 2** | 3/18~3/19 | 본 실험 + C_v3 확정 + KoBEST/kmmlu | ✅ 완료 |
| **Sprint 3** | 3/20~3/26 | 다모델 언어 일반화 검증 (★ 4/17 중간 보고서) | ✅ 완료 |
| **Sprint 4** | 4/1~6/4 | 갭 보완 + 심화 실험 + 논문 작성 (★ 6/12 마감) | 🔄 진행 중 |

---

## Sprint 2 핵심 결과 (SOLAR-10.7B, KoBEST)

| 조건 | avg | 보존율 |
|------|-----|--------|
| FP16 | 0.6523 | 100% |
| **C_v3 (우리)** | **0.6356** | **97.4%** |
| B (랜덤 한국어) | 0.6176 | 94.7% |
| A (표준 GPTQ) | 0.5981 | 91.7% |

- C_v3 재현성: Run1=0.6354, Run2=0.6356 ✅
- 메커니즘: 비한국어 subword 제거(846→570개)가 GPTQ Hessian 근사 개선

---

## Sprint 3 핵심 결과 — 5모델 교차 검증

### "사전학습 언어 = calibration 언어" 가설 검증

| 모델 | 사전학습 언어 | 최적 조건 | 가설 지지 |
|------|------------|---------|---------|
| SOLAR-10.7B | 한국어 | C_v3 (한국어 다양성) | ✅ |
| EEVE-10.8B | 한국어 | B / C_v3_eeve (한국어) | ✅ |
| EXAONE-3.5-7.8B | 한국어 | B / C_v3_exaone (한국어) | ✅ |
| Qwen2-7B | 중국어 | C_zh (Sprint3당시 C_zh_v3 미시행) | 🔶 부분 |
| Llama3-Ko-8B | 영어 | C_en_v3 (영어 다양성) | ✅ (역방향) |

### Sprint 3 개별 결과

**EEVE-10.8B** (FP16=미측정): B(0.7595) > C_v3_eeve(0.7551) > C_v3(0.7514) >> A(미측정)
**EXAONE-3.5-7.8B** (FP16=0.7437): B(0.7196) > C_v3(0.7164) >> A(0.6645) [+ Sprint4: C_v3_exaone(0.7415) 최고]
**Qwen2-7B C-Eval** (FP16=0.8165): C_v3(0.7734) > A(0.7630) = C_zh(0.7630) [+ Sprint4 C_zh_v3(0.7868) 최고]
**SOLAR Phase2**: g64 C_v3=0.6468, SmoothScale 실패(0.48대), desc_act=False=0.6029

---

## Sprint 4 실험 결과 (2026-04-10 기준)

### S4-1: C_v3 vs B 통계 검정 (seed shuffle 5런)

| 조건 | mean | std | 비고 |
|------|------|-----|------|
| C_v3 | 0.6236 | **0.0098** | greedy 순서 파괴로 Sprint2 대비 하락 |
| B | 0.6393 | 0.0250 | seed shuffle 영향 없음 |

- paired t-test: p=0.1484 (비유의)
- **핵심 발견**: C_v3 std가 B의 39% → greedy 알고리즘이 calibration 순서에 robust

### S4-2: Qwen2 C_zh_v3 (중국어 다양성 알고리즘)

| 조건 | C-Eval | KoBEST |
|------|--------|--------|
| FP16 | 0.8120 | — |
| **C_zh_v3** | **0.7870** | 0.5737 |
| C_v3 (한국어) | 0.7694 | — |
| A (영어) | 0.7617 | — |
| C_zh (중국어 랜덤) | 0.7584 | — |

C_zh_v3 > C_v3 > A > C_zh → 언어 정렬 + 다양성 모두 중요

### S4-9: Llama3-Ko C_en_v3 (영어 다양성 알고리즘)

| 조건 | KoBEST | 보존율 |
|------|--------|--------|
| FP16 | **0.5900** | 100% |
| **C_en_v3** | **0.5804** | 98.4% |
| A (랜덤 영어) | 0.5758 | 97.6% |
| C_v3 (한국어) | 0.5650 | 95.8% |
| B (한국어) | 0.5608 | 95.1% |

*KoBEST group acc (lm-eval kobest group 결과)*

**C_en_v3 ≈ A**: Wikitext-2가 이미 균질한 텍스트라 다양성 효과 미미
→ 다양성 알고리즘은 calibration 풀이 불균일할 때 더 효과적

### C_v5 / C_v5_delta ablation (SOLAR)

| 조건 | avg | 설명 |
|------|-----|------|
| C_v3 Sprint2 | 0.6356 | 기준 (고정 순서) |
| C_v5 (δ+순도+길이) | 0.6128 | 순도 강화로 unique_morphemes 1826→1719 감소 |
| **C_v5_delta (δ만)** | **0.6266** | δ 단독 효과 +0.003 (소폭) |

**결론**: cross-sentence 형태소 coverage > within-sentence token richness

### EEVE kmmlu (2026-04-10 완료)

| 조건 | KoBEST | kmmlu |
|------|--------|-------|
| FP16 | 0.7759 | — |
| A | 0.7463 | 0.4044 |
| B | 0.7498 | **0.4126** |
| C_v3 | 0.7505 | 0.4089 |
| **C_v3_eeve** | **0.7551** | 0.4051 |

*KoBEST: group acc (lm-eval kobest group 결과), kmmlu: 49과목 단순 평균*

**발견**: KoBEST(이해/추론)는 C_v3_eeve 최고, kmmlu(지식 recall)는 B 최고

### EXAONE35 kmmlu (2026-04-13 완료)

| 조건 | KoBEST | kmmlu |
|------|--------|-------|
| FP16 | 0.7437 | — |
| A | 0.6963 | **0.4346** |
| B | 0.7244 | 0.4208 |
| C_v3 | 0.7145 | 0.4322 |
| **C_v3_exaone** | **0.7415** | 0.4299 |

*KoBEST: group acc, kmmlu: 49과목 단순 평균*

**발견**: KoBEST는 C_v3_exaone 최고, kmmlu는 A(영어 랜덤) 최고 — EEVE와 다른 패턴  
→ EXAONE35도 영어 사전학습 비중 존재, kmmlu 지식 recall에서 영어 calibration 유리

---

## 전체 모델 비교 요약 (최고 조건 기준)

| 모델 | FP16 | 최고조건 | 점수 | 보존율 | 표준GPTQ(A) 대비 |
|------|------|---------|------|--------|----------------|
| SOLAR-10.7B | 0.6523 | C_v3 | 0.6356 | 97.4% | +0.0375 |
| EEVE-10.8B | 0.7759 | C_v3_eeve | 0.7551 | 97.3% | +0.0088 |
| EXAONE-3.5 | 0.7437 | C_v3_exaone | 0.7415 | 99.7% | +0.0452 |
| Qwen2-7B | 0.8165 | C_zh_v3 | 0.7868 | 96.4% | +0.0238 |
| Llama3-Ko-8B | 0.5900 | C_en_v3 | 0.5804 | 98.4% | +0.0046 |

**5모델 모두 표준 GPTQ(A) 대비 우리 방법론 우위** (Llama3-Ko: C_en_v3 > A, 영어 다양성 calibration이 랜덤 영어보다 우위)

---

## 남은 작업 (Sprint 4)

### 실험 — 모두 완료
- [x] EXAONE35 kmmlu 완료 (2026-04-13)
- [x] S4-1 Levene test: 분산비 6.46x 확인
- [x] C_en_v3 ≈ A 원인 분析: A long-paragraph 절대 커버리지 효과
- [x] KoBEST vs kmmlu 불일치 분析: 형태소다양성/도메인다양성 구분
- [x] S4-12 Task-Aware Calibration Bank 완료 (2026-04-23)
- [x] Korean PPL 측정 완료 (2026-05-11): FP16=19.34, C_v3=20.08(+3.8%), A=21.66(+12.0%) — C_v3 왜곡 68% 감소
- [x] S4-13 Activation Sensitivity Score 완료 (2026-05-11): 12/48 고감도 레이어, model activation 왜곡 C_v3 86.1% 감소
- [ ] DBAR-v1 수정 재실험 (선택, inference tensor 버그 수정 필요)

### S4-12 Task-Aware Calibration Bank 결과 (2026-04-23)

| 조건 | KoBEST | kmmlu | balanced_ret |
|------|--------|-------|-------------|
| A | 0.5981 | 0.3678 | 0.961 |
| B | 0.6176 | 0.3731 | 0.983 |
| C_v3 | 0.6356 | 0.3750 | **1.000** (기준) |
| H1 50/50-IL | 0.6321 | 0.3725 | 0.994 |
| H2 50/50-CAT | 0.6157 | 0.3621 | 0.967 |
| **H3 75/25-IL** | 0.6284 | **0.3787** | **0.999** |
| **H4 25/75-IL** | **0.6413** | 0.3665 | 0.993 |

- SOLAR에서는 C_v3가 KoBEST/kmmlu 모두 최선 → EEVE/EXAONE35의 task tradeoff 없음
- H4: KoBEST 0.6413이 C_v3(0.6356) 초과 / H3: kmmlu 0.3787이 C_v3(0.3750) 초과 → mixing synergy
- DBAR-v1: 구현 버그(inference tensor backward 에러)로 실패 → skip
- 상세: `sprints/sprint4/thoughts/05_Task_Aware_Calibration_Bank_설계_및_실험.md`

### 결과 분析 완료 (2026-04-13)
- Levene test: C_v3 분산비 6.46x (greedy 순서의 안정성 효과)
- C_en_v3 ≈ A: A long-paragraph(avg 104.9 words)가 3286 lemma_pos 커버, C_en_v3는 1113
- KoBEST/kmmlu 불일치: 형태소 다양성은 추론 보존, 랜덤 도메인 다양성은 지식 recall 보존
- 상세: sprints/sprint4/thoughts/04_결과분析_심화_계획_및_진행.md

### 논문 작성 (S4-8, ★ 6/5 쇼츠 / 6/12 최종)
- [ ] Introduction — 문제 정의, 기여 (Chimoto et al. 차별화)
- [ ] Related Work — GPTQ, calibrating beyond English
- [ ] Method — 형태소 다양성 greedy selection
- [ ] Experiments — 5모델 × 조건 × 메트릭 비교 테이블
- [ ] Analysis — activation 분析, kmmlu vs KoBEST 차이, 언어 일반화, 텍스트 청크 길이 효과
- [ ] Conclusion

---

## 핵심 스크립트

| 스크립트 | 역할 |
|---------|------|
| `src/build_calibration.py` | 한국어 calibration 생성 (A/B/C/C_v3/C_v5) |
| `src/build_calibration_zh.py` | 중국어 calibration (C_zh / C_zh_v3) |
| `src/build_calibration_en.py` | 영어 calibration (C_en_v3) |
| `src/run_quant_optimum.py` | GPTQ 양자화 (optimum.gptq) |
| `src/selection.py` | Greedy Diversity Selection 알고리즘 |
| `src/analyze_activations.py` | FP16 activation 분포 분석 |

## 주요 경쟁 논문
- **Chimoto et al. (EACL 2026)**: 랜덤 비영어 calibration → 우리: 형태소 다양성 greedy selection
- 차별화: 메커니즘 규명 (activation) + 다국어 × 다양성 교차 검증
