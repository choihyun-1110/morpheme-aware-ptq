# S2-2/S2-3: Cond A vs Cond C KoBEST 1차 비교

**작성일시:** 2026-03-17  
**관련 항목:** S2-2, S2-3

## 1. 비교 대상

- Cond A: `quantized_models/SOLAR_10.7B_4bit_cond_A`
- Cond C: `quantized_models/SOLAR_10.7B_4bit_cond_C`

평가 파일:

- Cond A: `results/eval_cond_A_2026-03-17T07-34-26.732750.json`
- Cond C: `results/eval_cond_C_2026-03-17T07-46-33.147056.json`

## 2. 총괄 결과

| 조건 | kobest acc | kobest acc_norm | kobest f1 |
|------|------------|-----------------|-----------|
| Cond A | 0.5981 | 0.5160 | 0.5461 |
| Cond C | 0.6062 | 0.5300 | 0.5582 |

1차적으로는 **Cond C가 Cond A보다 전체 KoBEST에서 우세**하다.

- acc: `+0.0081`
- acc_norm: `+0.0140`
- f1: `+0.0121`

## 3. 태스크별 비교

| 태스크 | Cond A acc | Cond C acc | 차이 (C-A) |
|--------|------------|------------|------------|
| kobest_boolq | 0.7151 | 0.7564 | +0.0413 |
| kobest_copa | 0.6160 | 0.6240 | +0.0080 |
| kobest_hellaswag | 0.4340 | 0.4400 | +0.0060 |
| kobest_sentineg | 0.6952 | 0.6196 | -0.0756 |
| kobest_wic | 0.4881 | 0.4865 | -0.0016 |

추가로 `kobest_hellaswag`의 `acc_norm`은:

- Cond A: `0.5160`
- Cond C: `0.5300`

## 4. 해석 메모

- Cond C는 전체 점수와 BoolQ, COPA, HellaSwag에서 개선을 보였다.
- 특히 `kobest_boolq` 상승폭이 커서 전체 평균 개선에 크게 기여한 것으로 보인다.
- 반면 `kobest_sentineg`는 Cond C가 Cond A보다 뚜렷하게 낮았다.
- 따라서 현재 단계에서는 **Cond C가 전반적으로 유리해 보이지만, 모든 하위 태스크에서 일관되게 우세하다고 보기는 어렵다.**

## 5. 다음 액션

- Cond B 양자화 완료
- Cond B KoBEST 평가 수행
- 이후 A/B/C 전체 비교표 및 간단한 해석 작성
