# Activation 분포 분석 기반 양자화 개선 설계

**작성일:** 2026-03-26
**작성 주체:** Claude (외부 시점 분석)

---

## 1. Activation 분석 결과 재해석

### 레이어별 channel_cv 패턴

```
L00 (임베딩):  B=0.98  C_v3=1.02  → 차이 없음 (언어 무관)
L01-L10:       B=13~27  C_v3=14~27  → C_v3 +0.5~0.8 (원래 다양해서 차이 작음)
L12-L18:       B=9.4~12.4  C_v3=10.3~13.3  → C_v3 +0.9 (격차 최대, <<<)
L19-L29:       B=9.4~10.8  C_v3=10.1~11.3  → C_v3 +0.7 (안정적 우위)
L30-L33:       B=9.0~9.7   C_v3=9.8~10.5  → C_v3 +0.8 (<<<)
L34-L45:       점차 감소, 차이 줄어듦
L46-L47:       B=0.7~1.3  C_v3=0.8~1.4  → 거의 동일 (출력 레이어)
```

### 핵심 해석

**임베딩(L0)과 출력(L46-47)은 calibration 언어와 무관.**
→ 이 레이어들은 어떤 calibration을 써도 GPTQ 품질이 비슷할 것.

**L12-18 (한국어 언어 처리 핵심 레이어):**
- C_v3 channel_cv가 B 대비 +0.9 (가장 큰 격차)
- SOLAR의 한국어 morphological/syntactic 처리가 이 레이어에 집중
- C_v3 calibration이 이 레이어들의 Hessian을 가장 잘 추정 → GPTQ 왜곡 최소화

**L01-10 (고유한 높은 다양성):**
- 원래 cv=13~27로 매우 높음 → 어떤 calibration도 충분히 다양한 Hessian 제공
- C_v3의 이점이 상대적으로 작음
- 이 레이어들은 "robust"하여 calibration 품질에 덜 민감

**outlier_ratio 균일 (0.00006):**
- 모든 레이어, 모든 조건에서 동일 → SOLAR에서는 activation outlier 문제가 경미
- SmoothQuant의 주요 이점(outlier 억제)은 적용 여지가 적음
- 그러나 per-channel 스케일링 자체는 GPTQ 균일성 향상에 기여 가능

---

## 2. 새로운 양자화 실험 설계

### 실험 A: Group Size 최적화 (group_size=64)

**가설:** 중간 레이어(L12-18)의 낮은 channel_cv(10-13) 구간에서,
더 세밀한 그룹(64개 열 단위)이 Hessian 근사 오차를 더 줄여줄 것.
C_v3의 이점이 group_size=64에서 더 증폭되거나, 영향이 중립화될 수 있음.

**실험:**
- SOLAR A group_size=64 vs C_v3 group_size=64
- 기존 group_size=128과 비교

**예상 결과:**
- group_size=64가 A에서는 더 도움 (calibration 품질 부족을 보완)
- group_size=64 + C_v3 조합이 최고 (시너지)
- 모델 크기는 128 대비 ~12% 증가 (tradeoff)

---

### 실험 B: SmoothScale + GPTQ 파이프라인

**배경:** activation_analysis.json에서 channel_cv는 높지만 outlier_ratio는 낮음.
일반 SmoothQuant는 outlier 때문에 효과적이나, 여기서는 channel 간 분산 균일화가 목적.

**구현:** `src/smooth_scale.py`
- 각 레이어의 per-channel activation 평균으로 스케일 s_j 계산
- s_j = mean(|X_j|)^0.5 (alpha=0.5)
- weight에 흡수: W_j_new = W_j * s_j
- 스케일된 모델에 GPTQ C_v3 적용

**두 단계 가설:**
1. SmoothScale → channel_cv 감소 → GPTQ 더 균일한 조건
2. C_v3 calibration → 여전히 다양한 한국어 activation → 이중 이점

**대조군:**
- SmoothScale + A (언어 효과 없이 스케일만)
- 기존 GPTQ C_v3 (스케일 없이 calibration만)
- 기존 GPTQ A (베이스라인)

**결과 해석 가이드:**
```
SmoothScale+C_v3 > GPTQ+C_v3 > SmoothScale+A > GPTQ+A → 두 기법 시너지
SmoothScale+C_v3 ≈ GPTQ+C_v3 > A → SmoothScale 효과 미미, calibration이 핵심
SmoothScale+A > GPTQ+A ≈ GPTQ+C_v3 → SmoothScale이 calibration 이점 대체
```

---

### 실험 C: AWQ vs GPTQ (방법론 비교)

**배경:** AWQ는 GPTQ와 전혀 다른 접근:
- GPTQ: Hessian 기반 2차 최적화 (calibration data로 Hessian 추정)
- AWQ: activation magnitude 기반 salient weight 보호 (1% 중요 weight를 FP16 유지)

**가설:** C_v3 calibration의 이점(한국어 channel 다양성)이 GPTQ보다 AWQ에서
다르게 나타날 수 있음.
- GPTQ에서는 Hessian 추정 품질 → calibration 언어 직접 영향
- AWQ에서는 activation magnitude → salient weight 선택 → 언어에 따라 다른 weight 보호

**예상:** AWQ에서도 C_v3 > A 패턴이 유지되면 → "calibration 언어 효과"는 방법론 독립적
C_v3 이점이 AWQ에서 더 크면 → AWQ가 한국어 calibration을 더 잘 활용하는 방법론

---

### 실험 D: desc_act=False (컬럼 재정렬 효과)

**배경:** desc_act=True (현재): activation 크기 기준 weight 컬럼 재정렬
- channel_cv가 높은 레이어에서 중요
- 재정렬 후 quantization이 중요한 채널부터 최적화

**가설:** C_v3의 높은 channel_cv는 desc_act=True의 이점을 증폭
→ desc_act=False에서는 C_v3의 이점이 줄어들 것
→ "C_v3 이점의 일부는 desc_act=True를 통해 발현"

---

## 3. 실험 결과 예상 매트릭스

| 실험 | 모델 크기 | 예상 품질 순위 |
|------|----------|--------------|
| GPTQ C_v3 g=128 (현재) | 기준 | 기준 |
| GPTQ C_v3 g=64 | +12% | 기준+α |
| GPTQ A g=64 | +12% | 기준-β+γ |
| SmoothScale+GPTQ C_v3 | 기준 | 기준+δ |
| SmoothScale+GPTQ A | 기준 | 기준+ε |
| AWQ C_v3 | 유사 | 미지 |
| AWQ A | 유사 | 미지 |
| GPTQ C_v3 desc_act=False | 기준 | 기준-ζ |

α: group_size 세밀화 효과
β: calibration 열화
γ: group_size 보완 효과
δ: smooth scaling 이점
ε: smooth scaling 이점 (A 대비 작을 것)
ζ: desc_act 제거로 인한 손실

---

## 4. 연구 의의

이 실험들은 다음 질문에 답합니다:

**Q1:** Calibration 언어 효과가 특정 quantization 기법(GPTQ)에만 해당하는가,
       아니면 AWQ 같은 다른 방법론에도 일반적으로 적용되는가?

**Q2:** Group size 선택과 calibration 언어 선택은 독립적인가, 상호작용하는가?

**Q3:** Activation 스케일링(SmoothQuant류)과 calibration 언어 선택은
       additive한가, 중복인가?

**Q4:** desc_act가 C_v3의 channel_cv 우위를 mediate하는 메커니즘인가?

이 결과들이 모이면: "언어 정렬 calibration"이 quantization 방법론 전반에
걸쳐 유효한 일반 원리인지, 또는 GPTQ의 Hessian 추정에만 특화된 효과인지
구분할 수 있습니다.

---

## 5. 실행 파일

- `src/run_exp_gpu0_phase2.sh` — GPU 0 Phase 1 완료 후 자동 실행
- `src/run_quant_awq.py` — AutoAWQ 기반 양자화
- `src/smooth_scale.py` — SmoothQuant-inspired 채널 스케일링 전처리
