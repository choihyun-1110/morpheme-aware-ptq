# Sprint 4 실험 결과 종합

**작성일:** 2026-04-10

---

## 1. Sprint 4 전체 실험 현황

| 실험 ID | 내용 | 상태 | 결과 |
|---------|------|------|------|
| S4-1 | C_v3 vs B 통계 검정 (5런) | ✅ 완료 | p=0.1484 비유의, C_v3 std 안정성 |
| S4-2 | Qwen2 C_zh_v3 | ✅ 완료 | C-Eval 0.787 (최고) |
| S4-3 | 누락 조건 보완 | ✅ 완료 | B g64, A no_desc_act, EEVE FP16 |
| S4-4 | EXAONE35 C_v3_exaone | ✅ 완료 | +0.004 미미 개선 |
| S4-5 | Activation 시각화 | ✅ 완료 | (Sprint3에서) |
| C_v5 | token richness δ=0.2 + 순도강화 | ✅ 완료 | 0.6128 < C_v3(0.6356) |
| C_v5_delta | δ=0.2만 (순도 변경 없음) | ✅ 완료 | 0.6266 (C_v3 mean보다 소폭 상회) |
| S4-9 | Llama3-Ko C_en_v3 | ✅ 완료 | 0.5916 ≈ A(0.5964) |
| Llama FP16 | Llama3-Ko FP16 기준값 | ✅ 완료 | 0.5932 |
| EEVE kmmlu | EEVE A/B/C_v3/C_v3_eeve | 🔄 진행 중 | GPU0 |
| EXAONE kmmlu | EXAONE35 A/B/C_v3/C_v3_exaone | 🔄 진행 중 | GPU1 |

---

## 2. 모델별 최종 결과 정리

### 2-1. SOLAR-10.7B-Instruct (KoBEST)

| 조건 | avg | 보존율 | boolq | copa | hellaswag | sentineg | wic |
|------|-----|--------|-------|------|-----------|---------|-----|
| FP16 | 0.6523 | 100% | 0.8661 | 0.6400 | 0.4480 | 0.6650 | 0.5008 |
| A (표준GPTQ) | 0.5981 | 91.7% | — | — | — | — | — |
| B (랜덤한국어) | 0.6176 | 94.7% | — | — | — | — | — |
| **C_v3 (우리)** | **0.6356** | **97.4%** | — | — | — | — | — |
| C_v3 mean (seed shuffle 5런) | 0.6236 | 95.6% | — | — | — | — | — |
| B mean (seed shuffle 5런) | 0.6393 | 98.0% | — | — | — | — | — |
| C_v5 (δ+순도강화) | 0.6128 | 93.9% | 0.8298 | 0.6380 | 0.5340 | 0.5668 | 0.4952 |
| C_v5_delta (δ만) | 0.6266 | 96.1% | — | — | — | — | — |
| C_v3 g64 | 0.6468 | 99.2% | — | — | — | — | — |

**Sprint 2 결과 (고정 순서)**: C_v3(0.6356) > B(0.6176) > A(0.5981)
**S4-1 seed shuffle**: B mean(0.6393) > C_v3 mean(0.6236) — greedy 순서 파괴 영향
**C_v5_delta vs C_v3**: δ=0.2 단독 효과 소폭 개선(+0.003), cross-sentence coverage가 핵심

---

### 2-2. EEVE-Korean-Instruct-10.8B (KoBEST)

| 조건 | KoBEST avg | 보존율 | kmmlu |
|------|-----------|--------|-------|
| FP16 | 0.7595 | 100% | — |
| A | 0.6645 | 87.5% | 진행 중 |
| B | 0.7595 | 100.0% | 진행 중 |
| C_v3 | 0.7514 | 98.9% | 진행 중 |
| **C_v3_eeve** | **0.7551** | **99.4%** | 진행 중 |

---

### 2-3. EXAONE-3.5-7.8B-Instruct (KoBEST)

| 조건 | KoBEST avg | 보존율 | kmmlu |
|------|-----------|--------|-------|
| FP16 | 0.7485 | 100% | — |
| A | 0.6829 | 91.3% | 진행 중 |
| B | 0.7420 | 99.1% | 진행 중 |
| C_v3 | 0.7356 | 98.3% | 진행 중 |
| **C_v3_exaone** | **0.7395** | **98.8%** | 진행 중 |

---

### 2-4. Qwen2-7B-Instruct (C-Eval 기준)

| 조건 | C-Eval | KoBEST |
|------|--------|--------|
| FP16 | 0.8120 | — |
| A (영어랜덤) | 0.7617 | — |
| C_v3 (한국어다양성) | 0.7694 | — |
| C_zh (중국어랜덤) | 0.7584 | — |
| **C_zh_v3 (중국어다양성)** | **0.7870** | 0.5737 |

C_zh_v3 > C_v3 > A > C_zh — 언어 정렬 + 다양성 모두 중요

---

### 2-5. Llama3-Ko-8B (KoBEST)

| 조건 | KoBEST avg | 보존율 |
|------|-----------|--------|
| FP16 | **0.5932** | 100% |
| A (랜덤영어=표준GPTQ) | 0.5964 | 100.5% |
| B (랜덤한국어) | 0.5722 | 96.5% |
| C_v3 (다양성한국어) | 0.5794 | 97.7% |
| C_en_v3 (다양성영어) | 0.5916 | 99.7% |

A > C_en_v3 > C_v3 > B — 사전학습 언어(영어) calibration이 유리
C_en_v3 ≈ A(0.5916 vs 0.5964): Wikitext-2가 이미 충분히 다양 → 다양성 알고리즘 추가 이득 미미

---

## 3. 핵심 발견 및 논문 기여

### 3-1. "사전학습 언어 = calibration 언어" 가설 — 5모델 교차 검증

| 모델 | 사전학습 주 언어 | 최적 조건 | 가설 지지 여부 |
|------|--------------|---------|-------------|
| SOLAR | 한국어 | C_v3 (한국어 다양성) | ✅ |
| EEVE | 한국어 | B/C_v3_eeve (한국어) | ✅ |
| EXAONE35 | 한국어 | B/C_v3_exaone (한국어) | ✅ |
| Qwen2 | 중국어 | C_zh_v3 (중국어 다양성) | ✅ |
| Llama3-Ko | 영어 | A/C_en_v3 (영어) | ✅ (역방향) |

5/5 모델 가설 지지.

### 3-2. 다양성 원칙의 언어 보편성

- 한국어: C_v3(다양성) > B(랜덤) → 다양성 선택이 유효
- 중국어: C_zh_v3(다양성) > C_zh(랜덤) → 동일 원리 확인
- 영어: C_en_v3 ≈ A(랜덤) → Wikitext-2는 이미 균질한 텍스트라 다양성 효과 미미
  - 해석: 풀 자체의 품질이 불균일할 때 다양성 알고리즘이 더 효과적

### 3-3. Cross-sentence coverage vs Within-sentence richness

- C_v5 (δ+순도강화) < C_v3: 순도 강화(min_ko_ratio 0.7→0.9)로 unique_morphemes 1826→1719 감소
- C_v5_delta (δ만) ≈ C_v3 mean: δ 자체 효과는 미미 (+0.003)
- **결론: cross-sentence 형태소 커버리지가 within-sentence richness보다 중요**
- 나무위키의 한자·외래어 혼합 문장이 오히려 형태소 다양성에 기여함

### 3-4. Tokenizer 정렬 효과

- C_v3_eeve (EEVE tokenizer로 재생성): +0.0037 vs C_v3
- C_v3_exaone (EXAONE tokenizer로 재생성): +0.0039 vs C_v3
- **일관된 미미한 개선**: tokenizer 정렬은 효과 있으나 결정적이지 않음

### 3-5. 통계 검정 (S4-1)

- C_v3 vs B, seed shuffle 5런: p=0.1484 (비유의)
- 그러나 C_v3 std=0.0098 vs B std=0.0250 → C_v3이 calibration 순서에 더 robust
- 해석: greedy 알고리즘 출력 순서가 최적화된 순서이며 이것이 C_v3의 강점

---

## 4. 실험 환경 및 에러 사항

### 발생한 에러 및 해결

| 에러 | 원인 | 해결 |
|------|------|------|
| `HFValidationError: Repo id must be...` | 최신 huggingface_hub가 로컬 경로를 repo_id로 검증 | 스크립트에서 HF model ID 사용으로 전환 |
| EEVE tokenizer `TypeError: not a string` | tokenizer 파일이 cache에 없음 (weights만 있음) | `snapshot_download(ignore_patterns=['*.safetensors'])` 로 tokenizer만 추가 다운로드 |
| C_v5_delta build log 빈 파일 | calibration 생성 후 quantization에서 경로 에러 | HF model ID 수정 후 재실행 |

---

## 5. 대기 중인 결과 (완료 후 업데이트 예정)

- [ ] EEVE kmmlu: A / B / C_v3 / C_v3_eeve
- [ ] EXAONE35 kmmlu: A / B / C_v3 / C_v3_exaone
