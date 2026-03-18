# S2-3: Cond C_v3 알고리즘 재설계

**작성일시:** 2026-03-18
**관련 항목:** S2-3
**근거 문서:** `08_calibration_통계비교_BCC_v2.md`

---

## 1. 재설계 배경

통계 비교(08번 문서) 결과, C/C_v2의 핵심 문제는 아래 두 가지였다.

- **비한국어 혼합 문장 비율이 44.5%** (128개 중 57개가 한국어 비율 < 30%)
- **SFS가 B 대비 3배 높음** → 이는 한국어 형태소 다양성이 아니라
  한자/일본어/통화코드 등 비한국어 문자가 tokenizer에서 글자 단위로 분절되기 때문

C 알고리즘이 의도와 달리 "다양한 한국어 형태소"가 아닌 "비한국어 문자가 많이 섞인 문장"을
선호했다는 것이 핵심 진단이다.

이 문제는 `alpha/beta/gamma` 파라미터와 무관하므로,
C_v2 방식의 파라미터 튜닝으로는 해결할 수 없다.

---

## 2. 변경 내용

### 2-1. `src/diversity.py` — SFS 계산 방식 수정

**Before:**
```python
score.sfs = len(token_ids) / score.n_eojeols if score.n_eojeols > 0 else 0.0
```
→ 전체 토큰 수 / 전체 어절 수 (비한국어 어절 포함)

**After:**
```python
ko_eojeols = sum(1 for eo in score.text.split() if _KO_RE.search(eo))
score.sfs = len(token_ids) / ko_eojeols if ko_eojeols > 0 else 0.0
```
→ 전체 토큰 수 / **한국어 문자를 포함하는 어절 수만**

효과: 비한국어 어절이 많은 문장의 SFS 인플레이션 방지

### 2-2. `src/selection.py` — coverage_bonus 한국어 형태소만 사용

**Before:**
```python
new_morphemes = cand.morpheme_set - covered_morphemes
coverage_bonus = len(new_morphemes) / (len(cand.morpheme_set) + 1e-8)
```
→ SL(외국어)/SH(한자)/SN(숫자)/SW(기호) 형태소 포함

**After:**
```python
cand_ko = _ko_morpheme_set(cand.morpheme_set)  # SL/SH/SN/SW 제외
new_morphemes = cand_ko - covered_morphemes
coverage_bonus = len(new_morphemes) / (len(cand_ko) + 1e-8)
covered_morphemes |= _ko_morpheme_set(candidates[best_idx].morpheme_set)
```
→ 한국어 형태소만으로 커버리지 보너스 계산

효과: 비한국어 문자가 많은 문장이 coverage_bonus에서 불이익을 받아 선택 안 됨

### 2-3. `src/selection.py` — `filter_sentence_like_candidates`에 `min_ko_ratio` 추가

```python
def filter_sentence_like_candidates(..., min_ko_ratio: float = 0.0):
    ...
    if min_ko_ratio > 0.0 and _ko_char_ratio(candidate.text) < min_ko_ratio:
        continue
```

효과: 한국어 문자 비율이 낮은 문장을 후보 풀에서 아예 제거

### 2-4. `src/build_calibration.py` — 새 인자 추가

- `--c-min-ko-ratio`: 최소 한국어 문자 비율 (기본 0.0 = 비활성)
- `--suffix`: 출력 파일명 버전 태그 (예: `v3`)

---

## 3. C_v3 실행 파라미터

```bash
python src/build_calibration.py \
  --condition C \
  --model upstage/SOLAR-10.7B-Instruct-v1.0 \
  --n-sentences 128 \
  --n-candidates 100000 \
  --alpha 0.3 \
  --beta 0.15 \
  --gamma 0.15 \
  --c-min-eojeols 5 \
  --c-min-tokens 24 \
  --c-target-eojeols 12 \
  --c-target-tokens 64 \
  --c-min-ko-ratio 0.7 \
  --suffix v3
```

출력: `results/calibration_set_C_v3_upstage_SOLAR-10.7B-Instruct-v1.0.json`

---

## 4. 기대 효과 vs 실제 생성 결과

| 지표 | 기존 C | 예상 C_v3 | **실제 C_v3** |
|------|--------|-----------|--------------|
| 비한국어 비율 > 30% 문장 수 | 57/128 (44.5%) | ~0/128 | **0/128** (min_ko_ratio=0.7 필터로 제거) |
| 평균 SFS | 6.80 | B 수준(2~3)에 가까워질 것 | **5.04** (하락했지만 B보다 여전히 높음) |
| 총 고유 형태소 커버리지 | 1934 (B=1510) | B보다 높게 유지될 것 | **1,685** (B보다 높음, C보다 낮음) |
| KoBEST BoolQ | 0.7564 | B 수준(0.7885) 근접 기대 | 평가 진행 중 |

**C_v3 calibration set 생성 결과 (2026-03-18 완료):**
- 후보 풀: 100,000개 추출 → 필터 후 92,538개 유효 (7,462개 제거)
- 선택된 문장: 128개
- 총 고유 형태소 커버리지: 1,685 (B=1,510, C=1,934 사이)
- 평균 SFS: 5.04 (기존 C=6.80 → 하락, 그러나 B 수준에는 미달)
- 평균 diversity_score: 0.677
- 출력: `results/calibration_set_C_v3_upstage_SOLAR-10.7B-Instruct-v1.0.json`

**해석:** SFS가 5.04로 낮아진 것은 긍정적이나, B(추정 ~1.5~2.5)보다 여전히 높다.
이는 한국어 문장이더라도 복잡한 복합어/전문용어는 subword 분절이 많기 때문으로,
C 알고리즘의 형태소 다양성 최대화 목표 자체가 SFS를 높이는 방향으로 작동하는 구조적 원인이다.

---

## 5. 리스크

- `min_ko_ratio=0.7`이 너무 강해서 후보 풀이 지나치게 줄어들면 필터 완화 필요
- SFS 계산 변경으로 diversity_score의 정규화 범위가 달라질 수 있음
  → 이는 후보 풀 전체에 min-max 정규화를 적용하므로 자동 보정됨

---

## 6. 다음 단계 (생성 후)

1. C_v3 calibration 통계 확인 (비한국어 비율, SFS, 커버리지)
2. C_v3로 양자화 (`src/run_quant.py`)
3. KoBEST 평가 (`src/run_eval.sh`)
4. A / B / C / C_v3 / FP16 전체 비교표 작성
