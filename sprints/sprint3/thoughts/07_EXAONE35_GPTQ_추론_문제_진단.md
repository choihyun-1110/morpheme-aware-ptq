# EXAONE35 GPTQ 추론 문제 진단

**작성일:** 2026-03-26

---

## 현상

EXAONE-3.5-7.8B GPTQ 4bit 모델 (조건 A, B)의 KoBEST 평가 결과:

| 조건 | boolq | copa | hellaswag | sentineg | wic | agg |
|------|-------|------|-----------|----------|-----|-----|
| A | 0.4979 | 0.517 | 0.248 | 0.4962 | 0.5119 | 0.4784 |
| B | 0.4979 | 0.517 | 0.248 | 0.4962 | 0.5119 | 0.4784 |

**모든 태스크에서 랜덤 기준선(≈50% for binary, 25% for 4-choice)**
**조건 A와 B의 결과가 완전히 동일**

---

## 원인 분석

### 확인된 사실
1. 모델 가중치는 서로 다름 (`max_diff` 수백만 이상)
2. lm_eval 로딩 시 `Converting GPTQ v1 to v2` 변환 발생
3. `TritonV2QuantLinear` 커널 사용
4. 점수가 정확히 random baseline과 일치 → 모델이 항상 동일한 선택지 예측

### 가설 1: optimum.gptq 저장 포맷 비호환 (가장 유력)

optimum.gptq로 저장된 모델:
- `model-00001-of-00002.safetensors` 형식 (HF 표준)
- `config.json`에 quantization_config 내장
- `quantize_config.json` 별도 파일 없음

auto_gptq로 저장된 모델:
- `gptq_model-4bit-128g.safetensors` (단일 파일)
- `quantize_config.json` 별도 파일

gptqmodel이 optimum.gptq 포맷을 "GPTQ v1"로 오인식 → v2로 변환 시 가중치 손상 가능

**검증 방법:** SOLAR g64 (optimum.gptq) KoBEST 결과 확인
- 합리적 결과 (>0.55) → EXAONE35 특화 문제
- 랜덤 결과 → optimum.gptq 포맷 자체 문제

### 가설 2: EXAONE35 커스텀 아키텍처와 Triton 커널 비호환

EXAONE35는 GLU(Gated Linear Unit) MLP, 독자적 attention 구현 사용.
gptqmodel의 Triton 커널이 표준 LLaMA attention을 가정하면 EXAONE35 처리 실패 가능.

### 가설 3: desc_act=True 처리 실패

desc_act=True일 때 gptq_post_init 함수가 activation ordering에 따른
weight 재정렬을 수행. EXAONE35 아키텍처에서 이 재정렬이 잘못 적용되면
출력이 완전히 망가질 수 있음.

---

## 현재 대응 계획

### 즉시 진행
- EXAONE35 C_v3 KoBEST 평가 (동일한 결과 예상, 확인 목적)
- EXAONE35 FP16 KoBEST 베이스라인 평가 (step 7, GPU 1 파이프라인 끝에 추가됨)
  - FP16이 정상이면 → GPTQ 문제 확인
  - FP16도 이상하면 → EXAONE35 + lm_eval 호환 문제

### 분기별 대응

**만약 SOLAR g64 (optimum.gptq)도 랜덤 결과:**
→ optimum.gptq 저장 포맷 전체 문제
→ `optimum_to_autogptq.py` 변환 스크립트 필요 또는 평가 방법 변경

**만약 SOLAR g64 (optimum.gptq) 정상:**
→ EXAONE35 특화 문제 (커널 비호환 또는 desc_act 처리)
→ `desc_act=False`로 EXAONE35 재양자화 시도

**공통 대안:**
- lm_eval 평가 시 `autogptq=` 파라미터 대신 다른 방법 사용
- 또는 직접 inference 코드 작성 (lm_eval 없이)

---

## 중간 결론

EXAONE35 실험은 현재 무효 상태. SOLAR g64 결과를 보고 다음 단계 결정 필요.

SOLAR Phase 2 실험 (group_size=64, SmoothScale, desc_act=False)은 계속 진행.
이 실험들은 SOLAR (LLaMA 기반)에서 진행되므로 EXAONE35 문제와 무관.

---

## 시사점

만약 optimum.gptq 포맷이 gptqmodel 로딩과 비호환이면:
- Sprint 3에서 생성한 모든 새 GPTQ 모델 (exaone35, solar_g64 등) 재평가 필요
- Sprint 2 모델들은 auto_gptq로 저장 → 기존 평가 결과 유효

이 경우 Phase 2 실험은 올바른 평가 방법 확립 후 재실행 필요.
