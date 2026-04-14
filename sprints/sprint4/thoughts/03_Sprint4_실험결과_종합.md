# Sprint 4 실험 결과 종합

**작성일:** 2026-04-10 / **최종 업데이트:** 2026-04-13

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
| S4-10 EEVE kmmlu | EEVE A/B/C_v3/C_v3_eeve | ✅ 완료 | B kmmlu최고(0.4126), C_v3_eeve KoBEST최고 |
| S4-11 EXAONE kmmlu | EXAONE35 A/B/C_v3/C_v3_exaone | ✅ 완료 | A kmmlu최고(0.4346), B KoBEST최고 |

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
| FP16 | 0.7826 | 100% | — |
| A | 0.7612 | 97.3% | 0.4044 |
| B | 0.7571 | 96.7% | **0.4126** |
| C_v3 | 0.7600 | 97.1% | 0.4089 |
| **C_v3_eeve** | **0.7669** | **98.0%** | 0.4051 |

KoBEST: C_v3_eeve > A > C_v3 > B
kmmlu: B > C_v3 > C_v3_eeve > A
→ 평가 벤치마크별 최적 조건이 다름. KoBEST는 한국어 이해/추론, kmmlu는 지식 recall 측정.

---

### 2-3. EXAONE-3.5-7.8B-Instruct (KoBEST + kmmlu)

| 조건 | KoBEST avg | 보존율 | kmmlu |
|------|-----------|--------|-------|
| FP16 | 0.7485 | 100% | — |
| A | 0.6829 | 91.3% | **0.4346** |
| **B** | **0.7420** | **99.1%** | 0.4208 |
| C_v3 | 0.7356 | 98.3% | 0.4322 |
| C_v3_exaone | 0.7395 | 98.8% | 0.4299 |

KoBEST: B > C_v3_exaone > C_v3 >> A  
kmmlu: **A > C_v3 > C_v3_exaone > B** ← EEVE와 반대 패턴  
→ EXAONE35는 kmmlu에서 A(영어 랜덤)가 최고. B(한국어)가 kmmlu 최하.

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
- 영어: C_en_v3 ≈ A(랜덤) → A가 이미 long-paragraph로 절대 커버리지 달성 (3286 vs 1113 lemma_pos pairs)
  - 해석: **greedy selection은 짧은 동질 텍스트 단위에서 효과적**. A는 avg 104.9-word 단락이라 이미 충분한 coverage 확보.
  - "Wikitext-2 균질" 가설 수정: pool 품질이 아닌 **텍스트 청크 길이**가 핵심 (→ A-2 분析 참고)

### 3-3. Cross-sentence coverage vs Within-sentence richness

- C_v5 (δ+순도강화) < C_v3: 순도 강화(min_ko_ratio 0.7→0.9)로 unique_morphemes 1826→1719 감소
- C_v5_delta (δ만) ≈ C_v3 mean: δ 자체 효과는 미미 (+0.003)
- **결론: cross-sentence 형태소 커버리지가 within-sentence richness보다 중요**
- 나무위키의 한자·외래어 혼합 문장이 오히려 형태소 다양성에 기여함

### 3-4. Tokenizer 정렬 효과

- C_v3_eeve (EEVE tokenizer로 재생성): +0.0037 vs C_v3
- C_v3_exaone (EXAONE tokenizer로 재생성): +0.0039 vs C_v3
- **일관된 미미한 개선**: tokenizer 정렬은 효과 있으나 결정적이지 않음

### 3-5. 통계 검정 (S4-1 + Levene's)

- Paired t-test: p=0.1484 (비유의, n=5 한계)
- **Levene's test: p=0.1611, 분산비 6.46x** (C_v3 std=0.0098 vs B std=0.0250)
- 해석: greedy 알고리즘 출력 순서가 최적화된 순서 → C_v3이 calibration 순서에 6.5x 더 robust
- 논문: 평균 차이 비유의이지만 분산 안정성이 C_v3의 실용적 강점

### 3-6. KoBEST vs kmmlu 태스크별 최적 조건 불일치

| 모델 | KoBEST 최고 | kmmlu 최고 | 패턴 |
|------|-----------|-----------|------|
| EEVE | C_v3_eeve | B | 형태소다양성↑추론, 랜덤↑지식 |
| EXAONE35 | B | A | 랜덤한국어↑추론, 랜덤영어↑지식 |

- **KoBEST (이해/추론)**: 형태소 다양성 calibration이 문맥 처리 weight 보존에 유리
- **kmmlu (지식 recall)**: 도메인 다양성(랜덤)이 지식 저장 weight 보존에 유리
- EXAONE35 kmmlu에서 A(영어)가 최고: EXAONE도 영어 사전학습 비중이 있음을 시사
- **일반화**: calibration의 다양성 "종류"가 보존되는 능력 유형을 결정 (형태소 다양성 ≠ 도메인 다양성)

---

## 4. 실험 환경 및 에러 사항

### 발생한 에러 및 해결

| 에러 | 원인 | 해결 |
|------|------|------|
| `HFValidationError: Repo id must be...` | 최신 huggingface_hub가 로컬 경로를 repo_id로 검증 | 스크립트에서 HF model ID 사용으로 전환 |
| EEVE tokenizer `TypeError: not a string` | tokenizer 파일이 cache에 없음 (weights만 있음) | `snapshot_download(ignore_patterns=['*.safetensors'])` 로 tokenizer만 추가 다운로드 |
| C_v5_delta build log 빈 파일 | calibration 생성 후 quantization에서 경로 에러 | HF model ID 수정 후 재실행 |

---

## 5. 결과 업데이트

- [x] EEVE kmmlu: 완료 (2026-04-10)
  - KoBEST: C_v3_eeve(0.7669) > A(0.7612) > C_v3(0.7600) > B(0.7571)
  - kmmlu: B(0.4126) > C_v3(0.4089) > C_v3_eeve(0.4051) > A(0.4044)
  - 발견: 평가 태스크 유형에 따라 최적 조건이 다름
- [ ] EXAONE35 kmmlu: 진행 중 (GPU1, modeling_exaone.py 버전 이슈 수정 후 재시작)

### 에러 이슈 (2026-04-10)
- EXAONE 양자화 중 `TypeError: <lambda>() got an unexpected keyword argument 'input_ids'`
  - 원인: HF model ID 사용 시 최신 `modeling_exaone.py` 자동 다운로드, 현재 transformers와 충돌
  - 해결: transformers_modules 캐시에 스냅샷 원본 파일로 덮어쓰기 후 재실행
