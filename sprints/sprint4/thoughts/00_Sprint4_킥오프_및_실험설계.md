# Sprint 4 킥오프 및 실험 설계

**작성일:** 2026-04-04

---

## 세션 개요

Sprint 3 완료 후 첫 Sprint 4 세션. 실험 재설계 및 GPU 풀가동 체계 확립.

---

## 1. S4-1 재설계: 통계 검정 분산 문제 발견 및 수정

### 문제 발견

이전 S4-1 실험 결과를 확인하니 C_v3 5회 런이 모두 완전히 동일한 값:
- C_v3: `0.6487 × 5` (std=0)
- B: `0.6542 × 5` (std=0)

원인: GPTQ는 동일한 calibration set을 동일한 순서로 처리하면 완전히 결정론적. 분산이 0이므로 t-test 자체가 의미 없음.

### 해결 방법: Seed 기반 셔플

`run_quant_optimum.py`에 `--seed` 파라미터 추가:
- seed에 따라 calibration 텍스트 순서를 셔플
- GPTQ는 순서에 따라 Hessian 근사가 달라지므로 실질적 분산 발생
- C_v3와 B에 동일한 seed 사용 → paired 비교 공정성 확보

```bash
# run마다 다른 seed (42~46)
run_stat "${CALIB_Cv3}" "C_v3" 1 42
run_stat "${CALIB_Cv3}" "C_v3" 2 43
...
run_stat "${CALIB_B}" "B" 1 42  # 동일 seed로 paired 비교
```

### 현재 진행

- 기존 std=0 결과 삭제 후 재실행 중 (GPU0)
- 예상 소요: ~7시간 (10회 런 × ~42분/런)

---

## 2. S4-2: Qwen2 C_zh_v3 실험 설계

### 배경

Sprint 3 Qwen2 결과: C_v3(Korean, 0.6375) > A(English, 0.6238) > C_zh(Chinese, 0.6063)

C_zh가 최하위인 이유가 두 가지 혼재:
1. **언어 불일치**: 중국어 pretraining 모델에 중국어 calibration 줬는데 왜 안 좋음?
2. **다양성 부족**: C_zh avg_SFS=2.29 (C_v3=6.57 대비 매우 낮음)

C_zh_v3로 다양성 문제를 해소하면 혼재가 풀림.

### C_zh_v3 생성 방법

`build_calibration_zh.py`에 `--min-sfs` 파라미터 추가:
- `--min-sfs 3.0`: SFS 3.0 이상인 문장만 후보로 사용
- 100k 후보에서 greedy 선택 → 고다양성 128문장
- 목표 avg_SFS ≥ 4.0

### 결과 해석 매트릭스

| 결과 | 의미 |
|------|------|
| C_zh_v3 ≈ C_v3 > A | 다양성이 언어보다 중요 (language-agnostic) |
| C_zh_v3 > C_v3 > A | 언어정렬 + 다양성 모두 필요 |
| C_zh_v3 > A > C_v3 | 사전학습 언어 정렬이 핵심 |
| C_v3 > C_zh_v3 > A | 한국어 형태소 다양성이 언어 독립적으로 우수 |

### 현재 진행

- GPU1에서 S4-4 완료 후 자동 실행 예정
- KoBEST + C-Eval 두 벤치마크 평가

---

## 3. S4-4: EXAONE35 C_v3_exaone

### 배경

Sprint 3에서 EEVE에 tokenizer-aligned calibration(C_v3_eeve) 적용 시 소폭 개선(+0.0037) 확인.
EXAONE35도 동일하게 EXAONE tokenizer로 C_v3를 재생성하면 B ≈ C_v3 → B < C_v3 개선 가능성.

### 현재 진행

- GPU1에서 C_v3_exaone calibration 생성 중 (~20분, NamuWiki 처리)
- 생성 후 자동으로 양자화 + KoBEST 평가

---

## 4. Upstage 최신 모델 조사

### 조사 결과 요약

| 모델 | 파라미터 | 4-bit VRAM | HuggingFace | 라이선스 |
|------|---------|-----------|------------|---------|
| Solar Pro Preview | 22B | ~13GB | ✅ 공개 | MIT |
| Solar Pro 2 | 31B | ~18GB | ❌ API 전용 | MIT |
| Solar Open 100B | 102B MoE | ~52GB | ✅ 공개 | Upstage License |
| SOLAR-10.7B | 10.7B | ~6GB | ✅ 공개 | CC-BY-NC |

### 결론

TITAN RTX 2×24GB(48GB) 환경에서:
- **Solar Pro Preview (22B)**: 단일 GPU에 여유, MIT, HuggingFace 공개 → 최적 타겟
- Solar Open 100B: 52GB로 48GB 초과 → 불가
- Solar Pro 2/3: API 전용 (가중치 미공개)

블록 단위 양자화(GPTQ block-wise)는 이미 현재 방식에서 CPU 오프로딩으로 구현됨 → 양자화 자체는 VRAM 무관, 추론 시 VRAM이 제약.

**Solar Pro Preview (22B) C_v3 양자화는 S4-1/S4-2 완료 후 진행 예정.**

---

## 5. README 갱신

Sprint 3 결과 전면 반영:
- 5모델 교차검증 테이블 추가
- Phase 2 variants (g64, desc_act=False)
- vLLM 서빙 벤치마크 결과
- 신규 스크립트 (run_quant_optimum.py, benchmark_serving.py 등)

---

## 현재 GPU 상태 (2026-04-04 08:46 기준)

| GPU | 실행 중 | 예상 완료 |
|-----|---------|---------|
| GPU0 | S4-1 SOLAR C_v3/B 5×2 런 | ~16:00 |
| GPU1 | S4-4 EXAONE35 → S4-2 Qwen2 C_zh_v3 | ~19:00 |
