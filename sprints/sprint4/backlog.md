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
| S4-1 | C_v3 vs B 통계 검정 (5회 런 → 95% CI) | 🔴 최고 | ⬜ 대기 |
| S4-2 | C_zh_v3 생성 + Qwen2 재실험 (다양성 vs 언어 정렬 혼재 해소) | 🔴 최고 | ⬜ 대기 |
| S4-3 | 누락 조건 보완 실험 (B g64 / desc_act=False A / EEVE FP16+A) | 🟡 중간 | ⬜ 대기 |
| S4-4 | C_v3_exaone 생성 + EXAONE35 재실험 (tokenizer 의존성 일반화) | 🟡 중간 | ⬜ 대기 |
| S4-5 | Activation 시각화 개선 (Sprint3 모델 추가 + 논문용 figure) | 🟡 중간 | ✅ 완료 |
| S4-6 | SmoothScale alpha 탐색 (0.1~0.3, layer-wise) | 🟢 낮음 | ⬜ 대기 |
| S4-7 | AWQ A vs C_v3 (방법론 독립성 확인) | 🟢 낮음 | ⬜ 대기 |
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

## 진행 메모

### 2026-03-29

Sprint 4 backlog 생성. Sprint 3 갭 분석 완료:
- `thoughts/11_Sprint4_방향성_및_갭분析.md` 참고
- 최우선: S4-1(통계), S4-2(C_zh_v3), S4-5(activation 시각화)
- 경쟁 논문 Chimoto et al. (EACL 2026) 발견 — 차별화 전략 재정립 필요
