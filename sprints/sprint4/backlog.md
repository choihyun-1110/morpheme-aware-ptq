# Sprint 4: 갭 보완 + 논문 작성

**기간:** 2026-04-14 ~ 2026-06-04
**★ 마감:** 6/5 쇼츠, 6/12 최종 제출

---

## 목표

Sprint 3까지의 실험 갭을 보완하고, 논문으로 완성한다.
갭 분석 근거: `sprints/sprint3/thoughts/11_Sprint4_방향성_및_갭분析.md`

---

## 항목

| ID | 항목 | 우선순위 | 상태 |
|----|------|---------|------|
| S4-1 | C_v3 vs B 통계 검정 (5회 런 → 95% CI) | 🔴 최고 | ✅ 완료 |
| S4-2 | C_zh_v3 생성 + Qwen2 재실험 | 🔴 최고 | ✅ 완료 |
| S4-3 | 누락 조건 보완 실험 (B g64 / desc_act=False A / EEVE FP16+A) | 🟡 중간 | ✅ 완료 |
| S4-4 | C_v3_exaone 생성 + EXAONE35 재실험 | 🟡 중간 | ✅ 완료 |
| S4-5 | Activation 시각화 개선 | 🟡 중간 | ✅ 완료 |
| C_v5 | token richness δ 효과 검증 (C_v5 / C_v5_delta ablation) | 🟡 중간 | ✅ 완료 |
| S4-9 | C_en_v3 생성 + Llama3-Ko 재실험 (영어 다양성 검증) | 🔴 최고 | ✅ 완료 |
| S4-10 | EEVE kmmlu (A/B/C_v3/C_v3_eeve) | 🟡 중간 | ✅ 완료 |
| S4-11 | EXAONE35 kmmlu (A/B/C_v3/C_v3_exaone) | 🟡 중간 | ✅ 완료 |
| S4-12 | Task-Aware Calibration Bank (hybrid mixing) | 🔴 최고 | ✅ 완료 (DBAR-v1 실패, hybrid 분석 완료) |
| S4-13 | Activation Sensitivity Score 정의 | 🟡 중간 | ✅ 완료 (Korean PPL + sensitivity score + model activation 비교) |
| S4-14 | Layer-aware g64/g128 (API 확인 후) | 🟢 낮음 | ⬜ 대기 |
| S4-6 | SmoothScale alpha 탐색 (0.1~0.3, layer-wise) | 🟢 낮음 | ⬜ 대기 |
| S4-7 | AWQ A vs C_v3 (방법론 독립성 확인) | 🟢 낮음 | ⬜ 대기 |
| **S4-15** | **Llama-3-8B (순수 영어) + C_en_v3 vs A (MMLU)** | 🔴 최고 | 🔄 실험 중 |
| S4-8 | 논문 작성 | 🔴 최고 (6/12) | ⬜ 대기 |

---

## S4-1: C_v3 vs B 통계 검정

**목적:** 핵심 주장(C_v3 > B, +0.018)의 통계적 유의성 확보
**현재 상태:** C_v3 Run1=0.6354, Run2=0.6356 (2회). B는 단일 런(0.6176).

**실험 계획:**
- B 조건 4회 추가 실행 (총 5회)
- C_v3 3회 추가 실행 (총 5회)
- paired t-test, Cohen's d 계산
- 목표: p < 0.05, 95% CI 비중첩

**예상 소요:** GPU 시간 ~10h (각 런 ~24min × 10회)

---

## S4-2: C_zh_v3 생성 + Qwen2 재실험

**목적:** "다양성 낮음" vs "언어 불일치" 혼재 해소
**현재 C_zh 문제:** jieba 기반, avg_SFS=2.29 (C_v3보다 낮음)

**C_zh_v3 생성 전략:**
- pkuseg 또는 HanLP로 더 세밀한 중국어 형태소 분석
- SFS greedy 선택으로 avg_SFS ≥ 5.0 달성
- 동일 알고리즘, 중국어만 입력

**결과 해석 매트릭스:**

| 결과 | 해석 |
|------|------|
| C_zh_v3 ≈ C_v3 > A | 다양성이 언어보다 중요 |
| C_zh_v3 > C_v3 > A | 언어 정렬 + 다양성 모두 필요 |
| C_zh_v3 > A > C_v3 | 사전학습 언어 정렬이 핵심 |

---

## S4-3: 누락 조건 보완

**3가지 빠른 실험 (각 ~3h):**

1. **SOLAR B g64** — g64에서도 C_v3 > B? (현재 g64는 C_v3=0.6468, A=0.6174만 있음)
2. **desc_act=False + A 조건** — desc_act=False에서 C_v3(0.6029) vs A 격차 확인
3. **EEVE FP16 베이스라인 + A 조건** — EEVE 보존율 계산 (FP16 미측정, A 미측정 상태)

---

## S4-4: C_v3_exaone + EXAONE35 재실험

**목적:** tokenizer 의존성 일반화 (EEVE C_v3_eeve +0.0037 결과 연장)
**예상:** EXAONE35 tokenizer로 C_v3 재생성 시 B ≈ C_v3 → B < C_v3로 개선될 가능성

---

## S4-5: Activation 시각화 개선

**현재:** B/C/C_v3/C_v4 (SOLAR 기준, Sprint 2 데이터)
**필요:**
1. **A 조건 추가** — 현재 B/C/C_v3/C_v4만 있어 A vs C_v3 시각화 불가
2. **다모델 activation 비교** — EXAONE35, EEVE, Qwen2에서 동일 분석
3. **논문용 figure** — 학술지 품질, 영문 라벨, 단일 패널 버전

**스크립트:**
- `src/analyze_activations.py` — FP16 모델에 calibration set 주입 후 activation 수집
- `src/visualize_activations.py` — 시각화 (현재 B/C/C_v3/C_v4)
- **추가 필요:** `--condition A` 옵션, 다모델 비교 패널

---

## S4-9: C_en_v3 + Llama3-Ko 재실험 ← 논문 완결을 위해 필수

**목적:** "다양성 원칙이 언어에 독립적인가" 최종 검증

**배경:**
- 한국어 모델: 랜덤 한국어(B) < 다양성 한국어(C_v3) 확인 ✅
- 중국어 모델: C_zh_v3 실험 중 (2026-04-07)
- Llama3-Ko (영어 사전학습): A(랜덤 영어) > C_v3(한국어 다양성) — 사전학습 언어 가설 역방향 지지
  - 하지만 A = 랜덤 선택. **C_en_v3 = 다양성 선택(영어)** 는 아직 미실험

**C_en_v3 알고리즘:**
- 데이터 소스: Wikitext-2 (A와 동일 풀, 선별만 다름)
- 형태소 분석: NLTK POS 태깅 + WordNet 표제어 → `(lemma, coarse_POS)` 쌍
- 선별: 동일 greedy 알고리즘으로 `(lemma, POS)` 커버리지 최대화

**결과 해석:**
| 결과 | 의미 |
|------|------|
| C_en_v3 > A | 다양성 원칙이 영어에도 유효 → **언어 독립적 일반 원리** 확립 |
| C_en_v3 ≈ A | Wikitext-2가 이미 충분히 다양 (다양성 원칙의 한계 or 천장 효과) |

**구현 완료 (2026-04-07):**
- `src/build_calibration_en.py` — NLTK 기반 영어 calibration 생성
- `src/run_exp_llama3ko_cen.sh` — 실험 자동화 스크립트

**실행 방법:**
```bash
docker exec llm-dev bash /home/choihyun/workspace/src/run_exp_llama3ko_cen.sh
```
(GPU0에서 C_v5 실험 완료 후 이어서 실행)

---

## S4-6: SmoothScale alpha 탐색

**배경:** alpha=0.5 → Hessian 붕괴 (0.4795). alpha=0.1~0.3 탐색 필요.
**Activation 분석 기반 layer-wise 방안:**
- L0, L46-47: alpha=0 (불필요)
- L1-10: alpha=0.1~0.2
- L12-18, L30-33: alpha=0.3~0.4

---

## S4-7: AWQ A vs C_v3

**목적:** calibration 언어 효과가 GPTQ 특유인지 방법론 독립적인지 확인
**전제:** autoawq 설치 필요 (`pip install autoawq`)

---

## S4-8: 논문 작성

**구성안:**

1. Introduction — 문제 정의, 연구 기여 (Chimoto et al. 차별화)
2. Related Work — GPTQ, SmoothQuant, AWQ, Calibrating Beyond English(Chimoto), TurboQuant
3. Method — 형태소 다양성 greedy selection 알고리즘 (tokenizer-agnostic 개선 포함)
4. Experiments — 모델 × 조건 × 메트릭 테이블 (95% CI 포함)
5. Analysis — activation 분석, subword 커버리지, Hessian 연결
6. Conclusion

**마감:**
- 6/5: 쇼츠 제작물
- 6/12: 최종 제출

**경쟁 논문 차별화:**
- Chimoto et al. (EACL 2026): 랜덤 비영어 calibration vs 이 연구: 형태소 다양성 greedy selection
- 추가 기여: 메커니즘 규명 (activation 분석) + 한국어 특화 + tokenizer-agnostic 알고리즘

---

## S4-15: Llama-3-8B (순수 영어) + C_en_v3 vs A — MMLU 검증

**배경 및 동기:**
- S4-9 (Llama3-Ko C_en_v3)에서 C_en_v3 ≈ A (0.5804 vs 0.5758) → 다양성 효과 미미
- 그런데 Llama3-Ko-8B(beomi/Llama-3-Open-Ko-8B)는 사실 한국어 지속학습 모델 → 순수 영어 모델로 보기 어려움
- **수정**: meta-llama/Meta-Llama-3-8B (순수 영어 사전학습) 사용 → 훨씬 깔끔한 실험 설계

**실험 설계:**
- 모델: `meta-llama/Meta-Llama-3-8B` (Llama3 원본, 영어 전용)
- 조건: A (랜덤 Wikitext-2), C_en_v3 (다양성 Wikitext-2)
- 평가: MMLU (57과목, 영어 지식 — 한국어 kmmlu와 대응)
- GPU: TITAN RTX GPU0 또는 1

**가설:**
| 결과 | 해석 |
|------|------|
| C_en_v3 > A | 다양성 원칙이 영어에서도 유효 → **언어 독립적 일반 원리 확립** |
| C_en_v3 ≈ A | Wikitext-2 동질성 효과 (S4-9와 동일 패턴, 데이터 소스 문제) |

**스크립트:** `src/run_exp_llama3_base_mmlu.sh`

```bash
docker exec llm-dev bash /home/choihyun/workspace/src/run_exp_llama3_base_mmlu.sh
```

**논문 반영:**
- S4-9 (Llama3-Ko) 결과는 모델 혼재 문제로 제한적 → S4-15로 대체 또는 보완
- 성공 시 Section 6 ("언어 독립성 검증")의 핵심 근거 강화

---

## 진행 메모

### 2026-03-29

Sprint 4 backlog 생성. Sprint 3 갭 분석 완료:
- `thoughts/11_Sprint4_방향성_및_갭분析.md` 참고
- 최우선: S4-1(통계), S4-2(C_zh_v3), S4-5(activation 시각화)
- 경쟁 논문 Chimoto et al. (EACL 2026) 발견 — 차별화 전략 재정립 필요

### 2026-04-10

주요 실험 완료:
- S4-1~S4-4 모두 완료
- C_v5 / C_v5_delta ablation 완료 → cross-sentence coverage가 within-sentence richness보다 중요
- S4-9 (C_en_v3 + Llama3-Ko) 완료 → C_en_v3(0.5916) ≈ A(0.5964), 다양성 효과 미미
  - 수정된 이유: A가 long-paragraph(avg 104.9 words)으로 이미 절대 lemma_pos 커버리지 3286 달성 (C_en_v3 1113), 텍스트 청크 길이 효과
- Qwen2 C_zh_v3 완료 → C-Eval 0.787 (모든 조건 중 최고)
- Llama3-Ko FP16 기준값: 0.5932
- EEVE / EXAONE35 kmmlu: 진행 중 (GPU0/1)

에러 이슈:
- huggingface_hub 신버전에서 로컬 경로 → HF repo ID 검증 실패 → 모든 스크립트 HF model ID 사용으로 수정
- EEVE tokenizer cache 미완 → snapshot_download로 tokenizer만 추가 다운로드

상세 결과: `thoughts/03_Sprint4_실험결과_종합.md`

### 2026-04-13

실험 완료:
- S4-10 EEVE kmmlu: KoBEST C_v3_eeve(0.7551) 최고, kmmlu B(0.4126) 최고 → 태스크별 역전
- S4-11 EXAONE35 kmmlu: KoBEST C_v3_exaone(0.7415) 최고, kmmlu A(0.4346) 최고 → EXAONE도 영어 지식 recall↑

결과 분석 완료 (thoughts/04_결과분析_심화_계획_및_진행.md):
- A-1 Levene's test: 분산비 6.46x (C_v3가 B보다 6배 안정적)
- A-2 C_en_v3 ≈ A 원인: A long-paragraph 효과 (절대 커버리지 3286 vs 1113)
- A-3 KoBEST/kmmlu 불일치: 형태소 다양성 ≠ 도메인 다양성, 보존 능력 유형 다름

**현재 상태: 모든 계획된 실험 완료. S4-8 논문 작성만 남음.**

### 2026-04-23

S4-12 Task-Aware Calibration Bank 완료:
- Hybrid H1~H4 모두 완료, SOLAR kmmlu baseline 측정 완료
- 핵심 결과: SOLAR에서는 C_v3가 KoBEST/kmmlu 모두 최선 (task tradeoff 없음)
- H4(25%Cv3+75%B): KoBEST=0.6413, C_v3(0.6356) 초과 — mixing synergy
- H3(75%Cv3+25%B): kmmlu=0.3787, C_v3(0.3750) 초과 — mixing synergy
- DBAR-v1: inference tensor 에러로 실패, 논문 일정 감안해 skip 결정
- 상세: `thoughts/05_Task_Aware_Calibration_Bank_설계_및_실험.md`
