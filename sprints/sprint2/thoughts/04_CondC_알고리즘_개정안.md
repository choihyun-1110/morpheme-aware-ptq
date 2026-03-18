# S2-3: Cond C 알고리즘 개정안

**작성일시:** 2026-03-17  
**관련 항목:** S2-1, S2-3

## 1. 개정 배경

1차 실험 결과에서 KoBEST 총점은 `Cond B > Cond C > Cond A`였다.

해석 문서와 calibration set 통계를 보면, 기존 Cond C는 형태소/토큰 다양성은 높았지만 다음 문제가 있었다.

- 짧은 문구/표제어형 샘플이 많이 포함됨
- 평균 어절 수와 평균 토큰 수가 B보다 크게 짧음
- 문장당 다양성은 높지만, set-level 형태소 커버리지는 기대보다 넓지 않음

즉, 기존 C objective는 실질적으로 **문장다운 한국어**보다 **짧고 압축된 고다양성 표현**을 선호했을 가능성이 있다.

## 2. 개정 목표

개정된 C는 아래 네 가지를 동시에 노린다.

1. 형태소 다양성 유지
2. 신규 형태소 커버리지 유지
3. 너무 짧은 문구 편향 완화
4. 문장 종결형을 갖춘 자연스러운 문장 우선

## 3. 코드 변경 요약

변경 파일:

- `src/selection.py`
- `src/build_calibration.py`

### 3-1. 문장형 후보 사전 필터 추가

Cond C에만 아래 필터를 적용하도록 수정했다.

- 최소 어절 수: `min_eojeols`
- 최소 subword 토큰 수: `min_subword_tokens`
- 문장 종결형 여부: `require_sentence_final`

문장 종결형 판정은 아래 휴리스틱을 사용한다.

- 문장부호 `. ! ? …` 로 끝남
- 또는 Kiwi 마지막 형태소 태그가 `EF` / `SF`

### 3-2. 선택 점수에 길이 보너스 추가

기존:

```text
final_score = diversity_score + alpha * coverage_bonus
```

개정:

```text
final_score
= diversity_score
+ alpha * coverage_bonus
+ beta * length_bonus
+ gamma * sentence_bonus
```

여기서:

- `length_bonus`: 어절 수와 subword 토큰 수가 목표 길이에 가까울수록 증가
- `sentence_bonus`: 문장 종결형이면 1, 아니면 0

## 4. 현재 기본 파라미터

- `alpha = 0.3`
- `beta = 0.15`
- `gamma = 0.15`
- `min_eojeols = 5`
- `min_subword_tokens = 24`
- `target_eojeols = 12`
- `target_subword_tokens = 64`
- `require_sentence_final = True`

즉, 여전히 diversity/coverage가 중심이지만, 길이와 문장성에 소폭 가중치를 부여하는 형태다.

## 5. 기대 효과

- 기존 C보다 더 자연스러운 한국어 문장이 선택될 가능성
- 짧은 명사구/표제어/문구 편향 완화
- BoolQ/COPA 같은 문장 이해형 태스크에서 B와의 격차 축소 가능성
- HellaSwag에서 보였던 C의 장점은 유지할 수도 있음

## 6. 리스크

- 필터가 너무 강하면 후보 풀이 과도하게 줄어들 수 있음
- 길이 보너스가 강해지면 오히려 일반 랜덤 한국어와 비슷해질 수 있음
- 결과적으로 “형태소 다양성”보다 “문장 길이”에 더 끌려갈 위험이 있음

따라서 현재 개정안은 **완전한 목적함수 교체가 아니라, 문장성 보정 계층을 얹는 수준**으로 설계했다.

## 7. 재실험 권장 순서

1. 개정된 알고리즘으로 Cond C calibration set 재생성
2. 새 C set의 길이/형태소 통계 확인
3. 새 C로 양자화
4. KoBEST 재평가
5. 기존 C와 새 C를 직접 비교

## 8. 재생성 예시 명령

```bash
cd /home/choihyun/workspace
docker exec -it llm-dev bash -lc '
source /opt/conda/etc/profile.d/conda.sh &&
conda activate llm-quant &&
cd /home/choihyun/workspace &&
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
  --c-target-tokens 64
'
```
