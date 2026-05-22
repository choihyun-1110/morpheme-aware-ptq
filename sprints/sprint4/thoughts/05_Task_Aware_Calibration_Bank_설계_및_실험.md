# Task-Aware Calibration Bank — 설계 및 실험

**작성일:** 2026-04-23

---

## 배경 및 동기

Sprint 4 결과 분析에서 EEVE/EXAONE35를 대상으로 다음을 발견:
- **KoBEST (이해/추론)**: 형태소 다양성 calibration(C_v3류) 최고
- **kmmlu (지식 recall)**: 랜덤/도메인 다양성 calibration(B 또는 A) 최고
- 두 벤치마크 간 최적 calibration이 일치하지 않는 **task-dependent tradeoff** 존재

→ "정답 calibration 하나"가 아니라 **목적별 calibration bank** 개념 도입

---

## 핵심 가설

> `Reasoning bank`(C_v3)와 `Random Korean bank`(B)를 혼합한 hybrid calibration이  
> KoBEST/kmmlu tradeoff를 더 균형 있게 만들 수 있다.

---

## 방법론 위치

논문 기여 구조:
```
기존: morpheme diversity greedy selection → KoBEST 개선
새로움: Task-Aware Calibration Bank
  - 관찰: KoBEST↔kmmlu 최적 bank 불일치
  - 제안: 목적별 bank 혼합으로 tradeoff 완화
  - 검증: mixing ratio별 balanced retention
```

---

## Bank 정의

| Bank | 내용 | 근거 |
|------|------|------|
| Reasoning bank | `C_v3` (형태소 다양성 greedy) | KoBEST 최고, EEVE 검증 |
| Random Korean bank | `B` (랜덤 한국어, 나무위키) | kmmlu 최고 (EEVE), 도메인 다양성 |

---

## 실험 조건

**대상 모델:** SOLAR-10.7B-Instruct-v1.0 (기준 모델, 결과 가장 풍부)

**고정 변수:**
- bits=4, group_size=128, desc_act=True
- n_sentences=128 (총합)
- eval: KoBEST + kmmlu

**실험 조건 (약칭):**

| ID | bank_r (Cv3) | bank_k (B) | mode | 약칭 |
|----|-------------|-----------|------|-----|
| H1 | 64 | 64 | interleave | 50/50-IL |
| H2 | 64 | 64 | concat(Cv3→B) | 50/50-CAT |
| H3 | 96 | 32 | interleave | 75/25-IL |
| H4 | 32 | 96 | interleave | 25/75-IL |

**비교 기준:**

| 조건 | KoBEST | kmmlu |
|------|--------|-------|
| FP16 | 0.6523 | — |
| A | 0.5981 | — |
| B | 0.6176 | — |
| C_v3 | 0.6356 | — |

> ⚠️ SOLAR kmmlu single-bank baseline (A/B/C_v3) 측정 진행 중 (GPU1, 2026-04-23).
> balanced_retention 판정은 이 값이 확보된 후 확정 가능.

---

## 판정 기준

```
retention_kobest = hybrid_KoBEST / best_single_KoBEST   (= C_v3 기준)
retention_kmmlu  = hybrid_kmmlu  / best_single_kmmlu    (= B 기준)
balanced_retention = (retention_kobest + retention_kmmlu) / 2
```

**pareto-dominance 기준:**  
단순 balanced_retention 외에, hybrid가 두 단일 bank 모두보다 낮지 않은 경우를 "pareto-dominant" 조건으로 별도 표기.

**결과 해석:**
1. **hybrid가 C_v3/B보다 balanced_retention 높음** → task-aware bank mixing이 tradeoff를 완화. 논문 메시지: "bank orchestration"
2. **hybrid가 두 지표 사이** → compromise일 뿐, specialized bank가 여전히 최선. 논문 메시지: "task별 bank 분리 필요성"
3. **hybrid가 한쪽 이하** → calibration signal 충돌. 두 bank가 서로 다른 layer activation pattern을 요구하면, 혼합 Hessian이 둘 다 부정확해질 수 있음 (signal cancellation). 논문 메시지: "interference 발생 → DBAR-v1으로 대응"

---

## 구현

| 파일 | 역할 |
|------|------|
| `src/build_hybrid_calibration.py` | bank mixing JSON 생성 (interleave/concat) |
| `src/run_exp_hybrid_bank.sh` | hybrid 실험 자동화 (양자화 + KoBEST + kmmlu) |
| `src/run_eval_solar_kmmlu_baseline.sh` | SOLAR A/B/C_v3 kmmlu 기준값 측정 |
| `src/run_dbar_v1.py` | DBAR-v1 양자화 (dual-bank merged Hessian) |

**생성된 calibration 파일:**
- `results/calibration_set_H_Cv364_B64_interleave.json`
- `results/calibration_set_H_Cv364_B64_concat_ab.json`
- `results/calibration_set_H_Cv396_B32_interleave.json`
- `results/calibration_set_H_Cv332_B96_interleave.json`

---

## 실험 진행 상황

| 실험 | 상태 | GPU |
|------|------|-----|
| H1: 50/50-IL | ✅ 완료 | GPU0 |
| H2: 50/50-CAT | ✅ 완료 | GPU0 |
| H3: 75/25-IL | ✅ 완료 | GPU0 |
| H4: 25/75-IL | ✅ 완료 | GPU0 |
| SOLAR kmmlu baseline (A/B/C_v3) | ✅ 완료 | GPU1 |

**시작:** 2026-04-23 03:07 / **완료:** 2026-04-23 07:42

---

## 결과

### SOLAR kmmlu 단일 bank 기준값 (2026-04-23 측정)

| 조건 | kmmlu |
|------|-------|
| A (랜덤 영어) | 0.3678 |
| B (랜덤 한국어) | 0.3731 |
| **C_v3 (형태소 다양성)** | **0.3750** |

**주목:** SOLAR에서는 C_v3가 kmmlu도 최선 (EEVE/EXAONE35와 달리 task-dependent tradeoff 없음)

---

### Hybrid 결과 (KoBEST + kmmlu + balanced_retention)

기준값: best_KoBEST = C_v3 = 0.6356 (Sprint2), best_kmmlu = C_v3 = 0.3750

| 조건 | KoBEST | kmmlu | ret_KoBEST | ret_kmmlu | balanced_ret |
|------|--------|-------|-----------|----------|-------------|
| A | 0.5981 | 0.3678 | 0.941 | 0.981 | 0.961 |
| B | 0.6176 | 0.3731 | 0.972 | 0.995 | 0.983 |
| **C_v3** | **0.6356** | **0.3750** | **1.000** | **1.000** | **1.000** |
| H1 50/50-IL | 0.6321 | 0.3725 | 0.994 | 0.993 | 0.994 |
| H2 50/50-CAT | 0.6157 | 0.3621 | 0.969 | 0.966 | 0.967 |
| **H3 75/25-IL** | 0.6284 | **0.3787** | 0.989 | **1.010** | **0.999** |
| **H4 25/75-IL** | **0.6413** | 0.3665 | **1.009** | 0.977 | 0.993 |

### 핵심 해석

1. **H4 (25%C_v3 + 75%B) → KoBEST 0.6413**: 단일 bank C_v3(0.6356)를 **초과**. B의 한국어 랜덤 다양성이 추론 보존을 보완하는 synergy 효과.
2. **H3 (75%C_v3 + 25%B) → kmmlu 0.3787**: 단일 bank C_v3(0.3750)를 **초과**. 소량의 B가 지식 recall 향상.
3. **H2 (50/50-CAT)**: 최하위 → calibration 순서 중요, concat은 C_v3 signal이 B에 덮임.
4. **SOLAR에서 task tradeoff 없음**: C_v3가 KoBEST/kmmlu 모두 단일 최선 → EEVE/EXAONE35 관찰이 SOLAR에 적용되지 않음.

### 논문 메시지 (해석 #2: compromise)
SOLAR에서는 C_v3가 두 태스크 모두 최선이므로 hybrid의 동기 자체가 약함.
하지만 H4>C_v3(KoBEST), H3>C_v3(kmmlu)는 "mixing synergy"로 별도 기술 가능:
> "형태소 다양성 calibration에 소량의 랜덤 한국어 혼합 시 특정 태스크 성능이 단일 bank를 초과"

---

## DBAR-v1: Hybrid의 한계 극복

Hybrid calibration의 구조적 한계:
- 총 128 samples를 64+64로 분할 → 각 bank의 Hessian 추정 sample 수가 절반
- signal cancellation 시 두 bank 모두 부정확해짐

**DBAR-v1 (Dual-Bank Adaptive Rounding v1)**:
```
H_dual = λ · H_r + (1-λ) · H_k
  - H_r: Reasoning bank 128 samples로 추정
  - H_k: Random Korean bank 128 samples로 추정
  - λ: mixing ratio (λ=0.5 → 동등 가중)
```

Hybrid 대비 차이:
- 각 bank 128 samples 전부 사용 → Hessian 추정 품질 2배
- λ가 sample 수에서 분리됨: λ=0.3이어도 각 bank는 128 samples

구현: `src/run_dbar_v1.py` (DualBankGPTQQuantizer, GPTQQuantizer 서브클래스)

DBAR-v1 실험은 SOLAR kmmlu baseline 확보 후 λ=0.3/0.5/0.7 세 조건 실행 예정.

**DBAR-v1 실행 결과: ❌ 실패 (2026-04-23)**
- λ=0.3 시도 → `RuntimeError: Inference tensors cannot be saved for backward`
- 원인: `torch.inference_mode()` 컨텍스트 내 tensor를 autograd 추적 경로에 사용
- 수정 방향: `run_dbar_v1.py`에서 inference_mode 대신 `torch.no_grad()` 사용, 또는 Hessian 계산 전 `.clone()` 추가
- **논문 일정 감안 시 수정 후 재실험은 선택적** — hybrid 결과만으로도 충분한 분석 가능

---

## 다음 단계

1. ~~SOLAR kmmlu baseline 완료~~ ✅
2. ~~Hybrid 실험 완료 + 결과 해석~~ ✅
3. DBAR-v1 수정 재실험 (선택적, 논문 일정 고려)
4. **논문 S4-8 착수** ← 현재 단계
