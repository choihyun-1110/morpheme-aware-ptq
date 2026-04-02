# Sprint 4 방향성 및 갭 분석

**작성일:** 2026-03-29
**작성 주체:** Claude (외부 시점 분석)
**기반 문서:** sprint_overview.md, sprint2/thoughts/12, 14, sprint3/thoughts/01, 05, 09, 10, backlog.md
**외부 자료:** TurboQuant (ICLR 2026), Calibrating Beyond English (EACL 2026), FAQ (arXiv 2601.11200)

---

## 1. 현재 연구의 논리적 허점

### 1-1. Qwen2 실험: "avg_SFS 낮음" vs "언어 불일치" 혼재 미해소

**현상:** Qwen2에서 C_zh(중국어, avg_SFS=2.29)가 최하.
**문제:** 이 결과가 의미하는 바가 두 가지로 갈린다.

- **해석 A (다양성 부족 설명):** C_zh는 jieba 형태소 단위여서 avg_SFS가 2.29로 낮다. C_v3(한국어)는 avg_SFS가 훨씬 높다. 따라서 "형태소 다양성이 낮아서" C_zh가 나쁜 것이지, 중국어여서 나쁜 것이 아니다.
- **해석 B (언어 불일치 설명):** 중국어 calibration이 Qwen2의 한국어 처리 레이어(L12-18류)를 제대로 커버하지 못해서 나쁘다. KoBEST가 한국어 평가이므로 당연히 한국어 calibration이 유리하다.

**왜 이 구분이 중요한가:**
- 해석 A가 맞으면: 논문 주장 = "형태소 다양성이 언어를 초월하는 핵심 요인"
- 해석 B가 맞으면: 논문 주장 = "calibration 언어 = 평가 언어일 때 유리"
- 현재 두 해석을 분리할 실험이 없다.

**해결책:**
`C_zh_v3` 실험 설계 — 중국어로 높은 avg_SFS(≥6.0)를 달성한 calibration set 생성.
- jieba 대신 더 세밀한 분석기(pkuseg 혹은 HanLP) 사용, 또는 문자 단위 SFS 계산 재정의
- C_zh avg_SFS=2.29 → C_zh_v3 avg_SFS≥5.0으로 끌어올려 C_v3와 동일 조건으로 비교
- 만약 C_zh_v3 ≈ C_v3 > A : "다양성이 언어보다 중요" 확정
- 만약 C_zh_v3 > C_v3 : "사전학습 언어 정렬 + 다양성" 모두 필요 확정
- 만약 C_zh_v3 ≈ A < C_v3 : "한국어 C_v3의 서브워드 효율성" 특수 효과

---

### 1-2. EXAONE35: "SOLAR tokenizer 기준 C_v3 효과 감소"인지 "EXAONE 특성"인지 미분리

**현상:** EXAONE35에서 B ≈ C_v3 (0.7196 vs 0.7164, 격차 0.0032).
SOLAR에서는 C_v3 > B가 0.018로 명확했다.

**두 대안 해석:**
- **해석 A (tokenizer 의존성):** C_v3가 SOLAR tokenizer 기준으로 생성되었기 때문에 EXAONE35 tokenizer로 입력 시 subword 분포가 달라져 형태소 다양성 효과가 희석.
- **해석 B (모델 특성):** EXAONE35는 한국어+영어 혼합 사전학습이라 한국어 처리 레이어의 activation 분포가 SOLAR보다 덜 집중적. 따라서 calibration 다양성 효과 자체가 낮다.

**왜 이 구분이 중요한가:**
- 해석 A가 맞으면: tokenizer별 C_v3 재생성 시 효과 복원 가능 → 알고리즘이 tokenizer-agnostic하게 개선될 수 있음
- 해석 B가 맞으면: SOLAR-specific 효과이며 다른 모델에서는 한국어 랜덤(B) 수준이 한계

**해결책:**
`C_v3_exaone` 실험 — EXAONE tokenizer로 C_v3 재생성 후 비교.
EEVE에서 C_v3_eeve(+0.0037 개선)가 부분 효과를 보였으므로, EXAONE35에서도 동일 실험이 필요하다. EEVE 결과와 비교하면 tokenizer 의존성의 일반화 가능 범위를 파악할 수 있다.

---

### 1-3. EEVE: B > C_v3_eeve 원인 설명 불완전

**현상:** tokenizer 정렬 후에도 B(0.7595) > C_v3_eeve(0.7551).
**문제:** tokenizer 불일치가 일부 원인임은 확인했지만, 왜 B가 여전히 우위인지 설명이 없다.

**미탐색 가설들:**
1. **문장 길이 분포:** B(NamuWiki 랜덤)의 avg 어절 수가 C_v3_eeve보다 길 가능성. EEVE는 instruction tuning이 강하게 들어간 모델로, 더 긴 문장으로 calibration 시 attention pattern을 더 잘 커버할 수 있다.
2. **텍스트 도메인 분포:** C_v3는 형태소 다양성 기준으로 선택되어 특수 어휘가 많을 수 있음. EEVE의 SFT 분포와 domain mismatch 가능성.
3. **단순 통계 노이즈:** 격차 0.0044는 단일 런이며 통계적으로 유의하지 않을 수 있음.

**해결책:**
EEVE에서 B vs C_v3_eeve의 문장 길이/도메인 분포 비교 분석 (실험 없이 데이터 분석으로 가능).
이후 C_v3_eeve_long (더 긴 문장 버전) 생성 후 비교로 길이 가설 검증.

---

### 1-4. SmoothScale: alpha=0.5 단일 시도 후 결론

**현상:** alpha=0.5에서 Hessian 비정치 오류 8회, KoBEST 0.4795로 실패.
**문제:** alpha 탐색 없이 "SmoothScale 부적합" 결론. SmoothQuant 원논문도 모델마다 최적 alpha가 다름을 명시했다(0.2~0.8).

**논리적 문제:**
alpha=0.5가 SOLAR에 맞지 않았다는 것이지, SmoothScale 자체가 SOLAR에 적합하지 않다는 것이 아니다. 또한 smooth_A ≈ smooth_C_v3이었는데, 이는 "SmoothScale이 calibration 언어 효과를 지배/흡수"하는지 아니면 "SmoothScale 자체 실패로 두 조건 모두 무너졌는지"를 구분할 수 없다.

**해결책:**
Sprint 4에서 alpha=0.1~0.3 구간 탐색 또는 layer-wise alpha (L12-18에만 낮은 alpha 적용).
activation 분석으로 channel_cv가 낮은 레이어만 SmoothScale 적용하는 selective 방식 고려.

---

### 1-5. 통계적 유의성 미검증

**현상:**
- SOLAR C_v3 vs B 격차 0.018: Run1=0.6354, Run2=0.6356으로 재현성 확인했지만, B는 단일 런
- 나머지 모든 조건(EXAONE, EEVE, Qwen2, Llama3-Ko): 모두 단일 런
- EXAONE35 B-C_v3 격차 0.0032: 통계적으로 유의하지 않을 가능성 높음

**논문 리뷰어 관점에서의 위험:** SOLAR C_v3 vs B 차이(0.018)에 대해 "C_v3의 B 대비 run2에서의 점수가 B의 단일 런과 같을 수 있다"는 반론 가능. 다른 모델에서의 결론도 단일 런에 근거.

**해결책:**
SOLAR C_v3 vs B 최소 5회 런 → 95% CI 산출, p-value 계산.
다른 모델도 핵심 조건(최우수 vs 2위) 3회 이상 반복.

---

## 2. 누락된 실험 목록 (우선순위별)

### 우선순위 1 (논문 핵심 주장 강화, 필수)

| 실험 ID | 내용 | 이유 |
|---------|------|------|
| **P1-A** | SOLAR C_v3 vs B 5회 런 → 95% CI, t-test | B 대비 0.018 격차의 통계적 유의성 확인 필수 |
| **P1-B** | C_zh_v3 생성 (avg_SFS≥5.0) + Qwen2 C-Eval/KoBEST | "다양성 vs 언어 정렬" 혼재 완전 해소 |
| **P1-C** | C_v3_exaone 생성 + EXAONE35 재실험 | tokenizer 의존성 일반화 검증 (EEVE 결과 연장) |

### 우선순위 2 (주장 보강, 중요)

| 실험 ID | 내용 | 이유 |
|---------|------|------|
| **P2-A** | SOLAR B g64 추가 측정 | g64에서 C_v3 > B 격차 변화 확인 (현재 g64는 A vs C_v3만 있음) |
| **P2-B** | desc_act=False에서 A 조건 추가 | desc_act=False 조건에서 C_v3 vs A 격차 확인 불가 상태 |
| **P2-C** | EEVE FP16 베이스라인 + A 조건 측정 | EEVE 보존율 계산 불가 (FP16 미측정, A 미측정) |
| **P2-D** | SmoothScale alpha 탐색 (0.1, 0.2, 0.3) | activation 분석 기반 개선 가능성 + 논문 Table용 데이터 |

### 우선순위 3 (논문 완성도, 선택)

| 실험 ID | 내용 | 이유 |
|---------|------|------|
| **P3-A** | AWQ A vs C_v3 (SOLAR) | calibration 언어 효과가 방법론 독립적인지 확인 |
| **P3-B** | LLaMA-3-8B(영어, non-Korean) + English diverse C_v3_en | 영어 모델에서 영어 형태소 다양성 calibration 효과 확인 |
| **P3-C** | Hessian 근사 오차 측정 (레이어별) | activation 다양성→ Hessian 품질→성능 인과 체인 직접 검증 |
| **P3-D** | C_v3 알고리즘 tokenizer-agnostic 재설계 | tokenizer와 무관하게 작동하는 다양성 점수 개발 (Section 4 참조) |

---

## 3. TurboQuant 및 최신 연구 동향

### 3-1. TurboQuant (Google, ICLR 2026)

**핵심 아이디어:**
- KV 캐시를 3-bit 벡터 양자화로 6× 압축, 정확도 손실 없음
- PolarQuant(기하 회전) + QJL(1-bit 오차 보정) 조합
- KV 캐시 양자화 = 추론 시간 메모리 최적화
- 8× 추론 속도 향상 (H100 기준)

**이 연구와의 관계:**
TurboQuant는 **KV 캐시 양자화**이며, 이 연구는 **가중치 양자화(GPTQ)**다.
두 기법은 완전히 다른 파이프라인에 속한다.

- TurboQuant 대상: inference-time KV cache (동적, 입력별 생성)
- 이 연구 대상: weight PTQ (정적, 모델 파라미터)

→ **novelty에 위협이 아님.** TurboQuant는 weight quantization calibration 문제를 전혀 다루지 않는다.

**인용 방향:**
"가중치 양자화(GPTQ 기반)와 KV 캐시 양자화(TurboQuant)는 상보적이다. 본 연구는 가중치 PTQ 단계의 calibration 최적화에 집중한다"는 방식으로 Related Work에서 인용 가능.

---

### 3-2. Calibrating Beyond English (Chimoto et al., EACL 2026) ← 매우 중요

**핵심 아이디어:**
- GPTQ/AWQ에서 영어 외 언어 calibration이 영어 단독 대비 perplexity를 낮춤 (최대 3.52 포인트)
- 멀티링구얼 calibration mix가 단일 언어보다 전반적으로 우수
- GPTQ가 AWQ보다 calibration 언어 변화에 더 민감
- 모델: Llama3.1-8B, Qwen2.5-7B (이 연구와 겹침)

**이 연구와의 관계:**
이 논문은 **직접 경쟁 논문이자 동시 발견 증거**다.

유사점:
- "영어 외 언어 calibration이 GPTQ 품질에 유리"는 핵심 발견이 동일
- Qwen2.5-7B와 Llama3.1-8B 실험 모델이 겹침
- GPTQ > AWQ 민감도 결론도 일치 (desc_act 실험 결과와 일맥상통)

차이점 (이 연구의 차별성):
- Chimoto et al.은 **언어 다양성 믹스**에 집중, 이 연구는 **형태소 다양성 알고리즘** (greedy selection)에 집중
- Chimoto et al.은 단순 랜덤 비언어 calibration, 이 연구는 SFS 기반 최적 선택 알고리즘 개발
- Chimoto et al.은 perplexity 중심, 이 연구는 downstream task (KoBEST/kmmlu/C-Eval) 직접 평가
- 이 연구는 메커니즘(activation 분석, subword 커버리지) 규명 시도

**novelty 위협 수준:** 중간~높음.
핵심 발견(비영어 calibration 유리)이 선행 발표된 상태이므로, 이 연구는 **형태소 다양성 알고리즘 자체** + **메커니즘 규명** + **한국어 특화** 측면에서 차별화해야 한다.

**대응 전략:**
- Related Work에서 명시적으로 인용: "Chimoto et al.(2026)은 언어 다양성의 중요성을 밝혔으나, 최적 언어 내 선택 알고리즘을 제시하지 않았다"
- 이 연구의 기여: 랜덤 비영어 calibration보다 형태소 다양성 최적화 선택이 추가로 얼마나 유리한지 정량화 (B랜덤 vs C_v3 비교가 이 gap)
- 두 논문이 complementary임을 강조

---

### 3-3. FAQ: Family-Aware Quantization (Xiao et al., arXiv 2601.11200)

**핵심 아이디어:**
- 동일 family의 더 큰 모델(예: Qwen-72B)로 calibration 샘플을 재생성
- 원본 calibration 데이터의 distribution bias를 줄여 PTQ 오차 최대 28.5% 감소

**이 연구와의 관계:**
이 연구와 보완적이다. FAQ는 "더 좋은 모델로 calibration을 증류"하는 방향, 이 연구는 "원본 데이터에서 더 다양한 샘플을 선택"하는 방향이다.

→ novelty에 큰 위협은 아님. FAQ는 데이터 생성 방법, 이 연구는 데이터 선택 알고리즘.

---

### 3-4. 관련 선행 연구 존재 여부 총괄

| 연구 | 내용 | 이 연구와 관계 |
|------|------|--------------|
| Chimoto et al. 2026 | 비영어 calibration 유리 | 부분 중첩 (핵심 발견 선행), 차별화 필요 |
| FAQ 2026 | family-aware calibration 재생성 | 상보적, 인용 가능 |
| TurboQuant 2026 | KV cache 양자화 | 다른 도메인, 위협 아님 |
| 형태소 다양성 + PTQ calibration 직접 연구 | **발견되지 않음** | novelty 유지 |

**결론:** "형태소 다양성 기반 greedy calibration selection 알고리즘" 자체는 선행 연구에서 발견되지 않았다. Chimoto et al.과의 차별화 전략이 논문의 핵심 challenge다.

---

## 4. 새로운 양자화 기술 제안

### 4-1. Layer-Selective Calibration (L12-18 집중)

**근거:** activation 분석에서 L12-18이 calibration 언어에 가장 민감 (channel_cv C_v3-B 격차 최대 +0.9).

**제안 아이디어:**
GPTQ의 calibration forward pass를 모든 레이어에 균등 적용하는 대신, L12-18에 더 많은 calibration 샘플을 할당하거나, L12-18 기준의 Hessian 추정을 집중적으로 개선.

구체적 구현:
- GPTQ의 `nsamples` 파라미터를 레이어별로 달리하는 custom loop
- L12-18만 먼저 calibration하고 다른 레이어는 lighter calibration 사용
- 또는 L12-18 activation 다양성을 직접 최대화하는 calibration 선택 기준으로 사용

**한계:** GPTQ 라이브러리 내부 수정이 필요. 논문 아이디어로는 유효하나 구현 복잡성이 높다.

---

### 4-2. Tokenizer-Agnostic 형태소 다양성 알고리즘

**현재 문제:** C_v3가 SOLAR tokenizer 기준으로 생성되어 다른 모델(EEVE, EXAONE35)에서 효과 감소.

**제안:**
"model tokenizer 기준 SFS 계산"으로 각 모델의 tokenizer를 입력받아 동일 알고리즘으로 최적 calibration set을 자동 생성하는 파이프라인.

```
build_calibration.py --model {target_model} --lang {ko|zh|en|...}
```

이 때 SFS를 target model의 tokenizer로 계산. EEVE에서 C_v3_eeve(+0.0037 개선)가 이미 이 방향의 유효성을 부분 검증했다.

**논문 기여 포인트:** "our algorithm is model-agnostic — given any model tokenizer and language corpus, it produces an optimized calibration set"

---

### 4-3. SmoothScale Layer-Wise Alpha 최적화

**현재 결과:** alpha=0.5 일률 적용 → Hessian 붕괴.

**제안:**
activation 분석 결과를 활용한 layer-wise alpha 자동 설정:
- L0 (임베딩), L46-47 (출력): channel_cv ≈ 1 → alpha=0 (SmoothScale 불필요)
- L1-10: channel_cv=13~27 (고유 다양성 높음) → alpha=0.1~0.2
- L12-18, L30-33: channel_cv=9~13 (calibration 민감) → alpha=0.3~0.4

이 방식이면 SmoothScale+C_v3가 0.48대가 아니라 C_v3+α를 달성할 가능성.

---

### 4-4. "Calibration Sensitivity Score" 새 지표 제안

**배경:** 현재는 SFS, UMR, TTR 등 언어학적 다양성 지표로 calibration set을 평가.

**제안:**
모델의 특정 레이어에서 calibration set A vs B를 실제로 forward pass하여 activation 분포 차이(KL divergence, channel_cv 격차)를 계산하는 "calibration sensitivity score" 정의.

이 점수를 알고리즘 최적화 목표로 삼으면: "calibration set 선택 = 모델 중간 레이어 activation 다양성 최대화 문제"로 재정의 가능.

**장점:** 언어와 무관하게 임의 모델에 적용 가능. 형태소 다양성이 효과적인 이유를 mechanistic하게 설명하는 근거가 됨.

---

## 5. Sprint 4 권장 로드맵

### Phase 1: 통계 강화 및 핵심 갭 보완 (4/14 ~ 4/28)

| 작업 | 기간 | 우선순위 | 예상 시간 |
|------|------|---------|---------|
| SOLAR C_v3 vs B 5회 런 → 95% CI | 4/14~4/16 | 최고 | ~10h GPU |
| SOLAR B g64 측정 | 4/14 | 높음 | ~3h GPU |
| desc_act=False A 조건 추가 | 4/14 | 높음 | ~3h GPU |
| EEVE FP16 + A 조건 측정 | 4/15~4/16 | 높음 | ~4h GPU |
| C_zh_v3 calibration 생성 (pkuseg/HanLP, avg_SFS≥5.0) | 4/17~4/20 | 높음 | ~4h CPU |
| Qwen2 C_zh_v3 실험 | 4/20~4/22 | 높음 | ~6h GPU |

### Phase 2: 알고리즘 개선 및 tokenizer-agnostic 구현 (4/28 ~ 5/15)

| 작업 | 기간 | 우선순위 | 비고 |
|------|------|---------|------|
| C_v3_exaone 생성 + EXAONE35 재실험 | 4/28~5/2 | 중간 | tokenizer 의존성 |
| build_calibration.py tokenizer-agnostic 개선 | 5/1~5/8 | 중간 | `--model` 파라미터 추가 |
| SmoothScale alpha 탐색 (0.1~0.3) | 5/3~5/8 | 중간 | layer-wise 우선 |
| AWQ A vs C_v3 실험 | 5/8~5/12 | 낮음 | 방법론 독립성 |

### Phase 3: 논문 작성 (5/15 ~ 6/4)

| 작업 | 기간 | 비고 |
|------|------|------|
| Related Work 업데이트 (Chimoto et al. 인용, TurboQuant) | 5/15~5/20 | 필수 |
| 메인 결과 테이블 확정 (95% CI 포함) | 5/20~5/25 | 필수 |
| 메커니즘 섹션 (activation 분석, subword 커버리지) | 5/25~6/1 | 핵심 차별화 |
| 최종 교정 + 제출 | 6/1~6/4 | 마감: 6/5 쇼츠, 6/12 최종 |

---

## 6. 논문 주장 권장 수정 방향

### 현재 주장 (Sprint 3 종료 시점)
> "사전학습 언어 정렬이 primary 요인, 형태소 다양성은 secondary 요인"

### Chimoto et al. 인용 후 권장 주장
> "비영어 calibration이 유리함(Chimoto et al.)을 확인하고, **언어 내 형태소 다양성 최적화가 랜덤 비영어 calibration보다 추가적으로 성능을 향상**시킴을 실증한다. SOLAR-10.7B에서 한국어 형태소 다양성 calibration(C_v3)이 한국어 랜덤(B) 대비 KoBEST +1.8% 향상, FP16 보존율 97.4%를 달성했다. 이 효과는 tokenizer 정렬 시 다른 모델로도 일반화된다."

**핵심 novelty 재정립:**
1. 형태소 다양성 기반 greedy selection 알고리즘 (새로운 알고리즘)
2. B 랜덤 대비 C_v3의 정량적 이득 + 메커니즘 (subword 커버리지, activation 다양성)
3. tokenizer-agnostic 확장 (C_v3_eeve, C_v3_exaone 결과 기반)

---

## 7. 요약: Sprint 4 최고 우선순위 3가지

1. **C_v3 vs B 통계 검정 (5회 런)** — 논문의 메인 주장이 단일 런에 근거하는 한 리뷰어 설득력이 낮다.
2. **C_zh_v3 실험** — "다양성 vs 언어 정렬" 혼재를 해소하지 않으면 Qwen2 결과가 논문 서사를 약화한다.
3. **Chimoto et al. 차별화 전략 명확화** — 이미 "비영어 calibration 유리"가 EACL 2026에 발표된 상황에서, 이 연구의 추가 기여를 명확히 포지셔닝해야 한다.
