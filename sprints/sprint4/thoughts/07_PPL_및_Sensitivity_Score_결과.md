# Korean PPL 및 Activation Sensitivity Score 결과

**작성일:** 2026-05-11  
**작성 주체:** Claude (외부 시점 분석)

---

## 1. Korean Perplexity (PPL) 측정

### 설정
- **평가 코퍼스:** 나무위키 held-out 500문장 (seed=9999, calibration seed=42와 분리)
- **측정 방식:** sliding-window (max_len=1024, stride=512) token-level cross-entropy → PPL = exp(mean_NLL)
- **평가 총 토큰:** 39,186 tokens
- **스크립트:** `src/measure_ppl.py`
- **결과 파일:** `results/ppl_solar.json`

### 결과

| 조건 | PPL | NLL | FP16 대비 비율 | FP16 대비 증가 |
|------|-----|-----|-------------|-------------|
| **FP16** | **19.34** | 2.9620 | 1.0000 | 기준 |
| **C_v3 (형태소 다양성)** | **20.08** | 2.9997 | **1.038** | **+3.82%** |
| A (표준 GPTQ, 랜덤 영어) | 21.66 | 3.0756 | 1.120 | +12.01% |

### 해석

**C_v3가 A 대비 PPL 왜곡을 3.1배 줄임.**
- A: FP16 대비 PPL +12.0% 증가 (양자화 왜곡 큼)
- C_v3: FP16 대비 PPL +3.8% 증가 (양자화 왜곡 작음)
- 절대 PPL 차이: C_v3=20.08 vs A=21.66 (△1.58)

KoBEST/kmmlu 정확도 결과(C_v3 97.4% 보존율 vs A 91.7%)와 방향 일치.
PPL은 생성 품질의 연속적 지표로, 정확도와 독립적 증거를 제공.

**논문 활용:**  
"C_v3 calibration은 한국어 생성 perplexity 왜곡을 A 대비 68% 감소시켜 (△12.0% → △3.8%), 다운스트림 정확도 개선이 언어 모델링 품질 보존과 일치함을 보인다."

---

## 2. Activation Sensitivity Score (S4-13)

### 설정
- **입력:** `results/activation_analysis.json` (FP16 SOLAR, A/B/C/C_v3/C_v4 조건별 레이어 통계)
- **비교:** A (기준) → C_v3 (목표)
- **지표:** outlier_ratio, channel_cv, mean_std의 상대 변화량 평균
- **스크립트:** `src/compute_sensitivity_score.py`
- **결과 파일:** `results/sensitivity_score.json`, `results/sensitivity_score.png`

### 요약

| 통계 | 값 |
|------|---|
| 평균 sensitivity score | 0.4798 |
| 최대 score | 0.7506 (L36) |
| Q75 threshold | 0.5547 |
| 고감도 레이어 수 | 12 / 48 |

### 고감도 레이어 (score ≥ Q75)

```
[23, 24, 25, 26, 27, 28, 29, 30, 34, 35, 36, 37]
```

### Top-5 레이어 상세

| 순위 | Layer | score | A channel_cv | C_v3 channel_cv | 변화율 |
|------|-------|-------|-------------|----------------|--------|
| 1 | **L36** | 0.7506 | 3.634 | 5.657 | +55.7% |
| 2 | **L35** | 0.7292 | 4.085 | 6.438 | +57.6% |
| 3 | **L37** | 0.6482 | 3.311 | 5.111 | +54.4% |
| 4 | **L24** | 0.6403 | 6.639 | 11.333 | +70.7% |
| 5 | **L25** | 0.6233 | 6.763 | 11.199 | +65.6% |

### 패턴 해석

1. **L35~L37 (후반부):** channel_cv가 낮은 구간 (A=3.3~4.1) — A calibration에서는 채널 간 분산이 고르지 않음. C_v3에서 channel_cv 급증 → GPTQ Hessian이 채널별 중요도를 더 정확하게 추정.
2. **L23~L30 (중반부):** outlier_ratio 변화가 큰 구간 — A에서 이상치 비율이 높고 C_v3에서 낮아짐. GPTQ rounding 오차 원인 구간.
3. **L0~L22 (초반부):** score 낮음 — calibration 조건과 무관하게 안정적.

### 논문 활용

- **Figure 제안:** x=레이어, y=sensitivity score bar chart (고감도 레이어 빨간색) → `results/sensitivity_score.png`
- **Analysis 섹션:** "고감도 레이어는 모델 중반(L23~L30)과 후반(L34~L37)에 집중되며, 이 구간에서 형태소 다양성 calibration이 channel activation 분포를 더 균등하게 만들어 Hessian 추정 품질을 개선한다"

---

## 3. 같은 입력 → 다른 모델 activation 비교 (신규)

### 분석 개요
기존 sensitivity score는 "같은 FP16 모델에 A/C_v3 calibration set을 입력했을 때" activation 차이 (calibration 중 관점).  
이 분석은 **"동일한 한국어 텍스트 100문장을 FP16 / GPTQ-A / GPTQ-C_v3 세 모델에 각각 통과시켰을 때 activation이 얼마나 다른가"** — 배포 시점 관점.

- **스크립트:** `src/compare_model_activations.py`
- **결과 JSON:** `results/model_activation_comparison.json`
- **시각화:** `results/model_activation_comparison.png`

### 결과: FP16 대비 activation 왜곡 (낮을수록 FP16에 가까움)

| 조건 | 종합 distortion | channel_cv | mean_std | outlier_ratio |
|------|---------------|-----------|---------|-------------|
| **GPTQ-A** | 0.0629 | 0.0540 | 0.0471 | 0.0877 |
| **GPTQ-C_v3** | **0.0087** | **0.0115** | **0.0037** | **0.0111** |

**C_v3가 A 대비 activation 왜곡 86.1% 감소** (0.0629 → 0.0087)

### 논리 강화

두 분석의 인과 연결:

```
[WHY — calibration 중]
C_v3 calibration set을 FP16 모델에 입력
→ 레이어별 activation이 더 균등하게 분산 (channel_cv ↑)
→ Hessian 행렬의 채널별 가중치 추정이 더 정확
→ GPTQ rounding 오차 감소

[EFFECT — 배포 시]
실제 한국어 입력을 GPTQ-C_v3 모델에 통과
→ FP16 대비 activation 왜곡 86.1% 감소 (A 대비)
→ KoBEST +5.7%p, PPL 왜곡 68% 감소
```

### 논문 Figure 활용
`results/model_activation_comparison.png`:
- Panel 1 (channel_cv): 48개 레이어별 FP16/GPTQ-A/GPTQ-C_v3 비교선 — 녹색(C_v3)이 검정(FP16)에 훨씬 가깝게 붙어있음
- Panel 2 (outlier_ratio): C_v3가 FP16 outlier 패턴을 더 잘 보존

---

## 4. 논문 연결 요약

| 지표 | A | C_v3 | FP16 | 비고 |
|------|---|------|------|------|
| KoBEST (acc) | 0.5981 (91.7%) | **0.6356 (97.4%)** | 0.6523 | Sprint 2 |
| kmmlu (acc) | 0.3678 | **0.3750** | — | Sprint 4 |
| Korean PPL | 21.66 (+12.0%) | **20.08 (+3.8%)** | 19.34 | Sprint 4 신규 |
| channel_cv (평균) | 기준 | **+45% 이상** | — | Sprint 2 activation |
| 고감도 레이어 비율 | — | 12/48 (25%) | — | Sprint 4 신규 |

| 지표 | A | C_v3 | FP16 | 비고 |
|------|---|------|------|------|
| KoBEST (acc) | 0.5981 (91.7%) | **0.6356 (97.4%)** | 0.6523 | Sprint 2 |
| kmmlu (acc) | 0.3678 | **0.3750** | — | Sprint 4 |
| Korean PPL | 21.66 (+12.0%) | **20.08 (+3.8%)** | 19.34 | `results/ppl_solar.json` |
| FP16 대비 activation 왜곡 | 0.0629 | **0.0087** (-86.1%) | 0 | `results/model_activation_comparison.json` |
| channel_cv 왜곡 | 0.0540 | **0.0115** | 0 | 同上 |

**4가지 독립 지표(정확도×2, PPL, activation 보존도)가 모두 C_v3 우위를 지지.**

### 시각화 파일 경로
- `results/model_activation_comparison.png` — 레이어별 FP16/A/C_v3 activation 비교 (논문 Figure 후보)
- `results/sensitivity_score.png` — calibration 민감도 레이어 bar chart
- `results/ppl_solar.json` — PPL 수치
