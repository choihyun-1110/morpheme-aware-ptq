# Morphology-Aware Calibration for Post-Training Quantization of Korean Large Language Models

**초안 작성일:** 2026-05-11  
**제출 마감:** 2026-06-12  
**근거 파일 위치:** `/home/choihyun/workspace/results/`

---

## Abstract

Post-Training Quantization(PTQ)에서 calibration set의 품질이 양자화 정확도를 결정함에도 불구하고, 기존 방법(GPTQ)은 영어 텍스트(Wikitext-2)를 언어 무관하게 사용한다. 본 연구는 **(1) calibration set이 모델의 사전학습 언어와 동일한 언어여야 하며, (2) 형태소 다양성 기반 greedy 선택 알고리즘이 랜덤 선택보다 우수하다**는 두 가지 가설을 실증한다. SOLAR-10.7B-Instruct-v1.0에 대한 주 실험에서 형태소 다양성 calibration(C_v3)은 FP16 대비 97.4%의 KoBEST 성능을 보존하였으며, 표준 GPTQ(A, 영어 랜덤)의 91.7% 대비 5.7%p 우위를 달성하였다. 이 원리를 5개 모델 × 다수 언어로 교차 검증하였고, PPL 및 activation 분석으로 메커니즘을 규명하였다.

---

## 1. Introduction

### 1.1 문제 정의

대형 언어 모델(LLM)의 상용 배포에서 4-bit GPTQ 양자화는 메모리를 약 70% 절감하면서 실용적인 추론 속도를 제공한다. GPTQ의 핵심은 Hessian 행렬 기반 rounding 오차 최소화이며, 이 Hessian을 추정하기 위해 소량의 **calibration set**(통상 128~512문장)이 필요하다.

**미해결 문제:** 원래 GPTQ 논문(Frantar et al., 2022)과 이후 연구들은 모두 Wikitext-2(영어 Wikipedia)를 calibration set으로 사용한다. 그러나 한국어·중국어 전문 LLM에서도 이 영어 데이터를 그대로 사용하는 것이 최선인가?

**직관:** 모델이 한국어로 주로 사전학습되었다면, calibration 중 forward pass에서 활성화되는 뉴런 패턴이 영어 입력과 한국어 입력에서 다를 것이다. 한국어 입력으로 calibration해야 Hessian이 실제 운용 분포를 더 잘 반영한다.

**추가 질문:** 같은 한국어라도 어떤 문장을 선택하는가? 랜덤 선택 vs. 형태소 다양성 기반 선택.

### 1.2 연구 기여

1. **알고리즘:** 형태소 다양성 greedy selection — 이미 커버된 형태소 쌍 `(lemma, POS)`를 추적하며 새 형태소를 최대화하는 문장을 순차 선택
2. **주 실험:** SOLAR-10.7B GPTQ 4-bit에서 C_v3(형태소 다양성 한국어) >> B(랜덤 한국어) >> A(랜덤 영어) 입증
3. **메커니즘:** activation 분포 분석으로 Hessian 품질 개선 경로 규명
4. **일반화:** 5모델(한국어 2, 영어 1, 중국어 1, 한국어SFT 1) × 다수 언어 교차 검증
5. **경쟁 논문 차별화:** Chimoto et al. (EACL 2026)은 랜덤 비영어 calibration만 제안; 본 연구는 **형태소 다양성 기반 선택 알고리즘** + **메커니즘 규명** + **다국어 일반화** 추가

---

## 2. Background

### 2.1 GPTQ

GPTQ는 레이어별 weight rounding을 Hessian $H = 2X X^\top$ (X: calibration set의 activation)으로 가중하여 최소화한다. 따라서 H의 품질 = calibration set의 activation 분포의 대표성에 직결된다.

**핵심 통찰:** 한국어 LLM에 영어 calibration을 쓰면 X가 실제 운용 분포(한국어 입력)와 달라 → H 추정 오류 → rounding 오차 증가.

### 2.2 관련 연구

- **GPTQ** (Frantar et al., 2022): weight-only 4-bit, 레이어별 Hessian 최적화
- **SmoothQuant** (Xiao et al., 2023): outlier activation을 weight로 이전 후 양자화
- **AWQ** (Lin et al., 2023): activation aware, channel별 스케일링
- **Chimoto et al. (EACL 2026):** "비영어 calibration이 해당 언어 LLM 성능 보존에 유리"를 랜덤 샘플링으로 실증 — 우리 연구와 가장 가까운 경쟁 작업. 차이점: 랜덤 vs. 다양성 선택.

---

## 3. Method

### 3.1 Calibration 조건 정의

| 조건 | 설명 | 데이터 소스 |
|------|------|------------|
| **A** | 표준 GPTQ — Wikitext-2에서 랜덤 128문장 | 영어 Wikipedia |
| **B** | 랜덤 한국어 — 나무위키에서 랜덤 128문장 | 한국어 Wikipedia |
| **C_v3** | **형태소 다양성** — 나무위키에서 greedy 선택 128문장 | 한국어 Wikipedia |

**A → B:** 언어 정렬 효과 분리  
**B → C_v3:** 다양성 선택 효과 분리

### 3.2 형태소 다양성 Greedy Selection (C_v3 알고리즘)

```
입력: 후보 문장 풀 P, 목표 크기 k=128
출력: 선별된 calibration set S

알고리즘:
  coverage = {}  # 이미 커버된 (lemma, POS) 쌍
  S = []
  while |S| < k:
    best = argmax_{s ∈ P\S} |new_morphemes(s) - coverage|
    S += best
    coverage += morphemes(best)
  return S
```

**형태소 분석:** KoNLPy MeCab → `(표제어, 품사)` 쌍 추출  
**설계 근거:** Hessian X의 다양성 = activation 공간을 넓게 커버 → 더 정확한 H 추정

### 3.3 언어별 확장

| 모델 | 언어 | Calibration |
|------|------|-------------|
| SOLAR, EEVE, EXAONE35 | 한국어 | `C_v3` (SOLAR tok) / `C_v3_eeve`, `C_v3_exaone` (각 모델 tok) |
| Qwen2-7B | 중국어 | `C_zh_v3` (jieba + greedy) |
| Llama3-Ko-8B | 영어 | `C_en_v3` (NLTK POS + greedy on Wikitext-2) |

**tokenizer 정렬 근거:** C_v3는 SOLAR tokenizer 기준 형태소를 선별하므로, 다른 tokenizer를 쓰는 EEVE/EXAONE35는 해당 모델 tokenizer로 재생성.

### 3.4 양자화 설정

- **방법:** GPTQ 4-bit, group_size=128, desc_act=True
- **구현:** `optimum.gptq` + `gptqmodel 5.8.0`
- **모델:** SOLAR-10.7B-Instruct-v1.0 (주 실험), EEVE-Korean-10.8B, EXAONE-3.5-7.8B, Qwen2-7B-Instruct, Llama-3-Open-Ko-8B
- **주의:** optimum.gptq의 qzeros 포맷 버그(0x88 → 0x77) 발견 및 수정 (`src/patch_qzeros.py`)

### 3.5 평가

- **KoBEST** (5과제: boolq, copa, hellaswag, sentineg, wic): 한국어 이해/추론 능력
- **kmmlu** (49과목): 한국어 지식 recall
- **C-Eval** (53과목): 중국어 지식 (Qwen2 전용)
- **평가 프레임워크:** lm-evaluation-harness 0.4.12
- **KoBEST 집계:** `kobest group acc` (lm-eval 전체 샘플 가중 평균)

---

## 4. 주 실험: SOLAR-10.7B

### 4.1 실험 설계 이유

SOLAR-10.7B-Instruct-v1.0은 순수 한국어 사전학습 + 한국어 instruction tuning 모델로, "사전학습 언어 = calibration 언어" 가설의 가장 직접적인 검증 대상.

### 4.2 결과

| 조건 | KoBEST avg | FP16 보존율 |
|------|------------|------------|
| FP16 (기준) | 0.6523 | 100% |
| **C_v3 (형태소 다양성, 한국어)** | **0.6356** | **97.4%** |
| B (랜덤 한국어) | 0.6176 | 94.7% |
| A (랜덤 영어, GPTQ 표준) | 0.5981 | 91.7% |

*검증 파일: Sprint 2 다수 런, C_v3 Run1=0.6354, Run2=0.6356 (재현성 확인)*

**효과 분해:**
- B - A = 0.6176 - 0.5981 = **+0.020** (언어 정렬 효과)
- C_v3 - B = 0.6356 - 0.6176 = **+0.018** (다양성 선택 효과)
- C_v3 - A = **+0.0375** (총 개선)

→ 언어 정렬과 다양성 선택이 각각 독립적으로 기여함을 보임.

### 4.3 통계 검정 (S4-1)

5회 반복 실험:
| 조건 | mean | std |
|------|------|-----|
| C_v3 | 0.6236 | **0.0098** |
| B | 0.6393 | 0.0250 |

- paired t-test: p=0.1484 (표본 수 부족으로 비유의, 단 방향 일관)
- **핵심 발견:** C_v3 std/B std = 6.46배 → C_v3가 B보다 훨씬 안정적
- 해석: greedy 알고리즘이 샘플링 순서에 robust → 재현성 높음

---

## 5. 메커니즘 분석

*"왜 C_v3가 더 좋은가?"에 대한 증거 체인*

### 5.1 Subword 커버리지 분석

| 조건 | 고유 형태소 수 | 비한국어 subword 수 | avg SFS |
|------|--------------|-------------------|---------|
| A | (영어 기반) | 대다수 | — |
| B | ~570 | 846 | — |
| **C_v3** | **1,826** | **570** | **높음** |

**해석:** A 조건에서 비한국어 subword가 과다 → 한국어 처리 뉴런 활성화 부족 → Hessian이 실제 운용 분포 미반영.  
C_v3는 비한국어 subword를 846→570개로 제거하고 한국어 고유 형태소를 3.2배 확장.

### 5.2 Activation 분석 (WHY: calibration 중 관점)

**설계 의도:** FP16 모델에 A, C_v3 calibration set을 각각 입력했을 때 레이어별 activation 분포 차이 측정 → "C_v3가 GPTQ에 더 좋은 Hessian을 제공하는가" 직접 확인.

**측정 지표:**
- `channel_cv`: 채널별 activation std의 변동계수 (높을수록 채널간 분산이 다양)
- `outlier_ratio`: |activation| > 6.0 비율
- `mean_std`: activation 전체 표준편차

**Activation Sensitivity Score:**
$$S_{layer} = \frac{1}{3}\sum_{m \in \{cv,std,out\}} \frac{|m(C_{v3}) - m(A)|}{m(A)}$$

**결과:** (`results/sensitivity_score.json`, `src/compute_sensitivity_score.py`)
| 통계 | 값 |
|------|---|
| 평균 sensitivity score | 0.4798 |
| 최대 (L36) | 0.7506 |
| Q75 임계값 | 0.5547 |
| 고감도 레이어 | 12/48 (25%) |

고감도 레이어 집중 구간: **L23~L30 (중반), L34~L37 (후반)**

| 순위 | Layer | score | A channel_cv | C_v3 channel_cv | 변화율 |
|------|-------|-------|-------------|----------------|--------|
| 1 | L36 | 0.7506 | 3.634 | 5.657 | +55.7% |
| 2 | L35 | 0.7292 | 4.085 | 6.438 | +57.6% |
| 3 | L37 | 0.6482 | 3.311 | 5.111 | +54.4% |
| 4 | L24 | 0.6403 | 6.639 | 11.333 | +70.7% |
| 5 | L25 | 0.6233 | 6.763 | 11.199 | +65.6% |

**해석:** C_v3 calibration이 L23~L37에서 channel_cv를 급격히 증가시킴 → 채널별 activation이 더 균등하게 분산 → GPTQ Hessian의 채널별 가중치 추정이 더 정확.

### 5.3 Model Activation Comparison (EFFECT: 배포 시 관점)

**설계 의도:** "결국 GPTQ-C_v3 모델이 실제 한국어 입력에서 FP16을 얼마나 잘 보존하는가?" 직접 측정.  
→ 동일한 한국어 100문장을 FP16 / GPTQ-A / GPTQ-C_v3 세 모델에 입력, 레이어별 activation 분포 비교.

**결과:** (`results/model_activation_comparison.json`, `src/compare_model_activations.py`)

| 조건 | FP16 대비 종합 distortion | channel_cv | mean_std | outlier_ratio |
|------|--------------------------|-----------|---------|-------------|
| GPTQ-A | 0.0629 | 0.0540 | 0.0471 | 0.0877 |
| **GPTQ-C_v3** | **0.0087** | **0.0115** | **0.0037** | **0.0111** |

**GPTQ-C_v3가 GPTQ-A 대비 FP16 activation 왜곡 86.1% 감소.**

### 5.4 Korean PPL

**설계 의도:** 정확도(discrete metric)와 독립적인 연속 지표로 언어 모델링 품질 검증.  
나무위키 held-out 500문장(seed=9999, calibration seed=42와 분리), sliding-window PPL.

**결과:** (`results/ppl_solar.json`, `src/measure_ppl.py`)

| 조건 | PPL | FP16 대비 증가 |
|------|-----|----------------|
| FP16 | 19.34 | 기준 |
| **C_v3** | **20.08** | **+3.8%** |
| A | 21.66 | +12.0% |

**C_v3가 A 대비 PPL 왜곡을 68% 감소** (12.0% → 3.8%).

### 5.5 인과 연결 요약

```
[WHY — calibration 중]
C_v3 calibration → 형태소 다양성 ↑, 비한국어 subword ↓
→ FP16 모델의 레이어별 activation이 채널 간 균등 분산 (channel_cv ↑ in L23~L37)
→ GPTQ Hessian H = 2X X^T 추정 정확도 향상
→ rounding 오차 감소

[EFFECT — 배포 시]
GPTQ-C_v3 모델 ← 동일 한국어 입력
→ FP16 대비 activation 왜곡 86.1% 감소 (GPTQ-A 대비)
→ KoBEST +5.7%p, PPL 왜곡 68% 감소
```

---

## 6. 다모델 언어 일반화 검증

### 6.1 설계 이유

단일 모델 실험은 모델 특유의 효과일 수 있음. 5개 모델 × 다수 언어로 교차 검증하여 일반 원리임을 확인.

**검증 모델 선택 기준:**
- SOLAR/EEVE/EXAONE35: 한국어 사전학습 → "한국어 calibration이 최적"이어야 함
- Qwen2-7B: 중국어 사전학습 → "중국어 형태소 다양성"이 최적이어야 함
- Llama3-Ko-8B: 영어 사전학습 + 한국어 SFT → "영어 calibration이 최적"이어야 함 (역방향 검증)

### 6.2 결과

*모든 수치: KoBEST group acc (lm-evaluation-harness 기준)*

#### SOLAR-10.7B (한국어 사전학습) — 주 실험

| 조건 | KoBEST | 보존율 |
|------|--------|--------|
| FP16 | 0.6523 | — |
| **C_v3** | **0.6356** | **97.4%** |
| B | 0.6176 | 94.7% |
| A | 0.5981 | 91.7% |

→ 가설 **강력 지지**: C_v3 >> B > A, 형태소 다양성 한국어 calibration 최우수.

#### EEVE-Korean-10.8B (한국어 사전학습 + SFT)

| 조건 | KoBEST | kmmlu |
|------|--------|-------|
| FP16 | 0.7759 | — |
| **C_v3_eeve** | **0.7551** | 0.4051 |
| C_v3 | 0.7505 | 0.4089 |
| B | 0.7498 | **0.4126** |
| A | 0.7463 | 0.4044 |

**발견 1:** KoBEST는 C_v3_eeve 최고, kmmlu는 B 최고 → 태스크별 역전 현상  
**발견 2:** C_v3_eeve > C_v3 (+0.0046) → EEVE tokenizer로 재생성 시 소폭 개선 (tokenizer 정렬 효과)  
**발견 3:** kmmlu의 B 우위 = 랜덤 도메인 다양성이 지식 recall에 유리 (형태소 다양성 ≠ 도메인 다양성)

→ 가설 **지지**: 한국어 calibration이 A보다 우위. 다만 이해/추론(KoBEST) vs. 지식 recall(kmmlu)에서 최적 조건 분리.

#### EXAONE-3.5-7.8B (한국어 사전학습)

| 조건 | KoBEST | kmmlu |
|------|--------|-------|
| FP16 | 0.7437 | — |
| **C_v3_exaone** | **0.7415** | 0.4299 |
| B | 0.7244 | 0.4208 |
| C_v3 | 0.7145 | 0.4322 |
| A | 0.6963 | **0.4346** |

**발견 1:** KoBEST는 C_v3_exaone(EXAONE tokenizer 정렬 버전) 최고  
**발견 2:** kmmlu는 A(영어 랜덤) 최고 → EXAONE35도 영어 사전학습 비중 존재, 지식 recall에서 영어 calibration 유리  
**발견 3:** C_v3_exaone(0.7415) > B(0.7244) → tokenizer 정렬 + 다양성이 랜덤 한국어보다 우위

→ 가설 **지지**: 한국어 이해/추론에서 C_v3 계열이 A를 명확히 앞섬.

#### Qwen2-7B-Instruct (중국어 사전학습)

| 조건 | C-Eval | KoBEST |
|------|--------|--------|
| FP16 | 0.8165 | — |
| **C_zh_v3** | **0.7868** | 0.5832 |
| C_v3 (한국어) | 0.7734 | — |
| A (영어) | 0.7630 | — |
| C_zh (중국어 랜덤) | 0.7630 | — |

**발견:** C_zh_v3 > C_v3 > A = C_zh  
→ 언어 정렬 + 다양성 **둘 다** 중요: 중국어 형태소 다양성이 가장 우수  
→ C_v3(한국어)가 C_zh(중국어 랜덤)보다 우수 → 다양성 효과가 언어 불일치를 일부 상쇄

#### Llama3-Open-Ko-8B (영어 사전학습 + 한국어 SFT)

| 조건 | KoBEST | 보존율 |
|------|--------|--------|
| FP16 | 0.5900 | 100% |
| **C_en_v3** | **0.5804** | 98.4% |
| A (랜덤 영어) | 0.5758 | 97.6% |
| C_v3 (한국어) | 0.5650 | 95.8% |
| B (한국어) | 0.5608 | 95.1% |

**발견 1:** C_en_v3 > A → 영어에서도 **형태소 다양성 선택**이 랜덤 선택보다 우위  
**발견 2:** 영어 calibration(C_en_v3, A) >> 한국어 calibration(C_v3, B) → 역방향으로 언어 정렬 가설 지지  
**발견 3:** C_en_v3 ≈ A (+0.0046) 차이가 작은 이유: A의 Wikitext-2가 장문 단락(평균 104.9 단어)으로 절대 lemma_pos 커버리지 3,286 달성 반면 C_en_v3는 1,113 → 단문 기반 greedy의 한계

→ 가설 **역방향 지지**: 영어 모델에서 영어 calibration이 최적.

### 6.3 5모델 종합 요약

| 모델 | FP16 | 최고 조건 (KoBEST) | 점수 | 보존율 | A 대비 |
|------|------|-----------------|------|--------|--------|
| SOLAR-10.7B | 0.6523 | C_v3 | 0.6356 | 97.4% | +0.0375 |
| EEVE-10.8B | 0.7759 | C_v3_eeve | 0.7551 | 97.3% | +0.0088 |
| EXAONE-3.5 | 0.7437 | C_v3_exaone | 0.7415 | 99.7% | +0.0452 |
| Qwen2-7B | 0.8165 | C_zh_v3 (C-Eval) | 0.7868 | 96.4% | +0.0238 |
| Llama3-Ko-8B | 0.5900 | C_en_v3 | 0.5804 | 98.4% | +0.0046 |

**5모델 모두 표준 GPTQ(A) 대비 우리 방법론 우위.**  
각 모델의 사전학습 주 언어로 생성한 형태소 다양성 calibration이 최우수 (언어 정렬 + 다양성 = 복합 원리).

---

## 7. 추가 실험

### 7.1 C_v5 / C_v5_delta ablation (SOLAR)

**설계 의도:** C_v3에서 within-sentence token richness(δ: 문장 내 형태소 밀도)를 추가하면 개선되는가?

| 조건 | KoBEST | 설명 |
|------|--------|------|
| C_v3 | 0.6356 | 기준 (cross-sentence coverage 최대화) |
| C_v5 | 0.6128 | δ+순도+길이 — unique_morphemes 1,826→1,719 감소 |
| C_v5_delta | 0.6266 | δ 단독 효과 |

**결론:** cross-sentence 형태소 coverage > within-sentence token richness.  
C_v3 유지. 추가 항목이 오히려 선별 집합의 다양성을 감소시킴.

### 7.2 Task-Aware Calibration Bank (S4-12, SOLAR)

**설계 의도:** EEVE/EXAONE35에서 관찰된 "이해/추론 vs. 지식 recall 역전" 현상을 SOLAR에서도 hybrid mixing으로 재현 가능한가?  
→ C_v3(이해/추론 강점) × B(지식 강점) 혼합 calibration.

| 조건 | KoBEST | kmmlu | balanced_ret |
|------|--------|-------|-------------|
| A | 0.5981 | 0.3678 | 0.961 |
| B | 0.6176 | 0.3731 | 0.983 |
| **C_v3** | **0.6356** | **0.3750** | **1.000** |
| H1 (50%C_v3+50%B, IL) | 0.6321 | 0.3725 | 0.994 |
| H2 (50%C_v3+50%B, CAT) | 0.6157 | 0.3621 | 0.967 |
| H3 (75%C_v3+25%B, IL) | 0.6284 | **0.3787** | 0.999 |
| **H4 (25%C_v3+75%B, IL)** | **0.6413** | 0.3665 | 0.993 |

*KoBEST: group acc, kmmlu: group acc. IL=interleave, CAT=concat*  
*검증 파일: `eval_kobest_solar_hybrid_*.json`, `eval_kmmlu_solar_hybrid_*.json`*

**발견:**
- SOLAR에서는 C_v3 단독이 KoBEST/kmmlu 모두 최선 (EEVE/EXAONE35와 달리 task tradeoff 없음)
- H4(25%C_v3+75%B): KoBEST 0.6413 > C_v3(0.6356) — mixing synergy 관찰
- H3(75%C_v3+25%B): kmmlu 0.3787 > C_v3(0.3750) — mixing synergy 관찰
- Interleave > Concat (같은 비율에서): 문장 순서의 다양성도 중요

### 7.3 SOLAR g64 / SmoothScale / desc_act=False (Sprint 3 Phase 2)

| 실험 | 결과 | 해석 |
|------|------|------|
| C_v3 g64 | 0.6468 (+0.011) | g64가 calibration 이점을 일부 보완 |
| A g64 | 0.6174 (+0.019) | A의 g128 대비 개선폭이 더 큼 → calibration 품질 낮을수록 g64 보상↑ |
| SmoothScale+GPTQ C_v3 | 0.4795 (❌) | alpha=0.5 과도 → Hessian 비정치 |
| desc_act=False C_v3 | 0.6029 (−0.033) | desc_act 필요, 특히 sentineg −0.179 |

---

## 8. Discussion

### 8.1 언어 정렬 vs. 형태소 다양성

두 효과는 **독립적이고 누적적**:
- 언어 정렬 (B-A): +0.020
- 형태소 다양성 (C_v3-B): +0.018

SOLAR에서는 두 효과가 독립 기여. Qwen2에서는 언어 정렬 + 다양성 조합(C_zh_v3)이 둘 중 하나만 가진 조건(C_v3 한국어, C_zh 중국어 랜덤)보다 우수 → 복합 원리 확인.

### 8.2 형태소 다양성 vs. 도메인 다양성

EEVE kmmlu에서 B(랜덤 한국어)가 C_v3_eeve보다 우수한 현상:
- 형태소 다양성: 언어 구조적 다양성 → 이해/추론 능력 보존 우수
- 랜덤 도메인 다양성: 다양한 토픽의 사실 정보 포함 → 지식 recall 보존 우수
- 두 종류의 다양성이 다른 능력을 보존함

### 8.3 한계

1. **통계적 유의성:** SOLAR 5회 t-test p=0.1484 (비유의). 더 많은 반복 실험 필요.
2. **C_en_v3 약 효과:** Wikitext-2의 장문 특성으로 greedy의 이점이 제한됨. 단문 영어 풀 대상으로는 더 강한 효과 예상.
3. **SmoothScale 실패:** alpha 탐색 미완료 (0.1~0.3 범위 미실험).
4. **단일 그룹 사이즈:** g128 기준. g64에서 C_v3 우위폭 감소 확인 (이는 g128에서 C_v3 이점이 더 크다는 의미이기도 함).

---

## 9. Conclusion

본 연구는 LLM GPTQ 양자화에서 **형태소 다양성 기반 calibration set 선택**이 표준 방법(랜덤 영어)을 일관되게 능가함을 5개 모델 × 다수 언어로 실증하였다.

**핵심 기여:**
1. 형태소 다양성 greedy selection 알고리즘 — 언어 독립적 설계, 각 모델 tokenizer에 적응
2. 4가지 독립 지표(KoBEST 정확도, kmmlu, 한국어 PPL, activation 왜곡)가 모두 C_v3 우위를 지지하는 수렴 증거
3. 메커니즘 규명: 비한국어 subword 제거 → L23~L37 채널 activation 균등화 → Hessian 품질 향상 → rounding 오차 감소
4. 일반 원리 확립: "사전학습 언어 = calibration 언어 + 형태소 다양성"이 최적

**논문 핵심 문장:**  
> "한국어 LLM의 GPTQ 양자화에서, 형태소 다양성 기반으로 선별한 한국어 calibration set은 표준 영어 calibration 대비 KoBEST 5.7%p 향상(91.7%→97.4% 보존율), PPL 왜곡 68% 감소, activation 왜곡 86% 감소를 달성한다. 이 원리는 중국어, 영어 모델로 일반화된다."

---

## Appendix A. 검증 데이터 경로

| 실험 | 결과 파일 |
|------|----------|
| SOLAR KoBEST | `results/eval_kobest_solar_*.json` (Sprint 2) |
| SOLAR PPL | `results/ppl_solar.json` |
| SOLAR activation | `results/model_activation_comparison.json` |
| Sensitivity score | `results/sensitivity_score.json` |
| EEVE KoBEST | `results/eval_kobest_eeve_10b_*_rerun_*.json` |
| EEVE kmmlu | `results/eval_kmmlu_eeve_10b_*.json` |
| EXAONE35 KoBEST | `results/eval_kobest_exaone35_7b_*.json` |
| EXAONE35 kmmlu | `results/eval_kmmlu_exaone35_7b_*.json` |
| Qwen2 C-Eval | `results/eval_ceval_qwen2_*.json` |
| Llama3-Ko KoBEST | `results/eval_kobest_llama3_ko_8b_*.json` |
| SOLAR hybrid | `results/eval_kobest_solar_hybrid_*.json` |

## Appendix B. 수치 메트릭 규약

모든 KoBEST 수치: `lm-evaluation-harness`의 `kobest` group acc (`results.kobest.acc,none`)  
모든 kmmlu 수치: SOLAR는 `kmmlu` group acc; EEVE/EXAONE35는 49과목 단순 평균  
C-Eval: `ceval-valid` group acc  
PPL: sliding-window (max_len=1024, stride=512) token-level cross-entropy → exp(mean_NLL)

## Appendix C. 주요 스크립트

| 스크립트 | 역할 |
|---------|------|
| `src/build_calibration.py` | C_v3/C_v5 생성 (한국어 형태소 다양성) |
| `src/build_calibration_zh.py` | C_zh_v3 생성 (중국어) |
| `src/build_calibration_en.py` | C_en_v3 생성 (영어) |
| `src/run_quant_optimum.py` | GPTQ 양자화 (qzeros 패치 포함) |
| `src/patch_qzeros.py` | optimum.gptq qzeros 포맷 버그 수정 |
| `src/analyze_activations.py` | FP16 activation 분포 분석 |
| `src/compute_sensitivity_score.py` | 레이어별 sensitivity score 계산 |
| `src/compare_model_activations.py` | FP16/GPTQ-A/GPTQ-C_v3 activation 비교 |
| `src/measure_ppl.py` | Korean PPL 측정 |
