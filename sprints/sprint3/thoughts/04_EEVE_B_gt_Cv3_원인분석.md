# S3-3: EEVE B > C_v3 원인 분석

**작성일:** 2026-03-26

---

## 현상 요약

| 모델 | A | B | C_v3 | 결과 |
|------|---|---|------|------|
| SOLAR-10.7B | 0.5981 | 0.6176 | **0.6356** | C_v3 최선 |
| EEVE-10B | 0.7172 | **0.7595** | 0.7514 | B 최선, C_v3 < B (-0.008) |

SOLAR에서 C_v3가 B 대비 +0.018 우위였으나,
SOLAR 기반 SFT 모델인 EEVE에서는 B가 C_v3 대비 +0.008 우위.

---

## 가설 분석

### 가설 1: Tokenizer 불일치 (우선순위: 높음)

**설명:**
- C_v3는 SOLAR tokenizer 기준으로 생성됨 (SFS, subword 커버리지 모두 SOLAR vocab 기준)
- EEVE가 SOLAR tokenizer를 그대로 쓰는지 확인 필요
- 만약 EEVE tokenizer ≠ SOLAR tokenizer라면, C_v3의 "비한국어 subword 제거" 효과가 EEVE에서 다르게 작동

**검증 방법:**
```python
from transformers import AutoTokenizer
solar_tok = AutoTokenizer.from_pretrained("upstage/SOLAR-10.7B-Instruct-v1.0")
eeve_tok = AutoTokenizer.from_pretrained("yanolja/EEVE-Korean-Instruct-10.8B-v1.0")

# vocab 크기 비교
print(len(solar_tok))  # SOLAR vocab size
print(len(eeve_tok))   # EEVE vocab size

# C_v3 문장들의 subword 분포 비교
# SOLAR tok vs EEVE tok으로 tokenize했을 때 다른가?
```

**EEVE tokenizer로 C_v3 재생성 후 재평가가 필요한 이유:**
SOLAR vocab 기준 "한국어 형태소" 판단이 EEVE vocab 기준에서 달라질 수 있음.
EEVE tokenizer용 C_v3_eeve를 만들어 EEVE 양자화에 쓰면 C_v3 > B 패턴이 재현될 가능성.

---

### 가설 2: Instruction Tuning 강도 차이 (우선순위: 중간)

**설명:**
- EEVE는 SOLAR 대비 더 강력한 instruction tuning 적용
- Instruction tuning이 강할수록 모델이 다양한 입력 패턴에 robust해짐
- → calibration 데이터 선택의 영향이 희석될 수 있음

**관찰 지지 근거:**
- EEVE의 기본 성능(FP16 추정치 ~0.77 이상)이 SOLAR(0.6523)보다 훨씬 높음
- 높은 기본 성능 모델에서는 양자화 손실 패턴이 달라질 수 있음

**한계:** 직접 검증이 어려움. SFT 데이터와 calibration 상호작용에 관한 이론적 근거 부족.

---

### 가설 3: 통계적 노이즈 (우선순위: 보완적)

**설명:**
- EEVE에서 B - C_v3 = 0.008 (매우 작은 차이)
- SOLAR에서 C_v3 - B = 0.018 (2배 이상 차이)
- EEVE의 차이는 노이즈 범위일 가능성

**SOLAR에서의 재현성:**
- SOLAR C_v3: Run1=0.6354, Run2=0.6356 (차이 0.0002) → 신호 명확
- EEVE는 단일 런 → 재현성 확인 필요

**검증 방법:** EEVE B, C_v3 각각 2회 이상 실행하여 평균과 분산 비교

---

## 권장 실험 순서

1. **[즉시 가능] Tokenizer 비교 스크립트 실행**
   - SOLAR tokenizer vs EEVE tokenizer vocab 비교
   - C_v3 문장들을 각 tokenizer로 tokenize → subword 분포 차이 확인

2. **[Docker 필요] EEVE tokenizer로 C_v3_eeve 생성**
   - `src/build_calibration.py`에서 `tokenizer_name` 파라미터를 EEVE로 변경
   - 새 calibration set 생성 (128문장)

3. **[Docker 필요] EEVE × C_v3_eeve 양자화 및 평가**
   - EEVE + C_v3_eeve로 양자화 → KoBEST 평가
   - B vs C_v3_eeve 비교

---

## 예상 결과 해석

| 결과 | 의미 |
|------|------|
| C_v3_eeve > B | Tokenizer 불일치가 원인 — EEVE에도 형태소 다양성 효과 있음 |
| C_v3_eeve ≈ B | Instruction tuning 강도 또는 모델 특성 차이 |
| C_v3_eeve < B | EEVE 구조적 특성으로 랜덤 calibration이 더 유리 |

---

## SOLAR vs EEVE 차이점 (참고)

| 항목 | SOLAR-10.7B | EEVE-10B |
|------|------------|----------|
| 베이스 | SOLAR | SOLAR (동일) |
| SFT | 기본 instruction | 강화 instruction (Evolve-Instruct) |
| KoBEST FP16 추정 | 0.6523 | ~0.78+ (GPTQ B 기준 역산) |
| Tokenizer | SOLAR vocab | SOLAR 기반 (확인 필요) |
