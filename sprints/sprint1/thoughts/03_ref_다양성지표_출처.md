# 다양성 점수 지표 출처 및 참고문헌

**관련 백로그:** S1-2 (형태소 인식 캘리브레이션 셋 알고리즘 설계)
**작성일:** 2026-03-17

---

## 1. 기존 학문에서 차용한 지표

### ① Type-Token Ratio (TTR)
- **출처:** 전산언어학/코퍼스 언어학의 고전적 어휘 다양성 지표.
- **원저:** Johnson, W. (1944). "Studies in Language Behavior." — 거의 교과서 수준의 기본 개념.
- **용도:** 텍스트의 어휘 풍부도(lexical richness)를 정량화.
- **우리 연구의 적용:** TTR을 LLM의 **subword tokenizer 단위**로 계산하여, 모델이 실제로 바라보는 토큰 수준에서의 다양성을 측정.

### ② Unique Morpheme Ratio (UMR)
- **출처:** TTR의 변형. 형태소 분석 기반 다양성 측정은 한국어 NLP 연구에서 유사 접근이 존재.
- **참고:** 한국어 교착어 특성 연구, 한국어 형태소 분석 기반 텍스트 분류 논문들에서 형태소 집합의 크기를 feature로 쓰는 경우가 있음.
- **우리 연구의 적용:** (형태소, 품사) 쌍을 단위로 하여 TTR보다 더 세밀한 언어적 다양성을 포착.

### ③ Subword Fragmentation Score (SFS)
- **출처:** 다국어 NLP에서 "fertility" 또는 "fragmentation"으로 불리는 토크나이저 품질 지표.
- **참고 논문:**
  - Rust et al. (2021). "How Good is Your Tokenizer?" — 언어별 토크나이저 fertility 비교 분석.
  - Ács (2019). "Exploring BERT's Vocabulary" — subword 분절 패턴 분석.
- **우리 연구의 적용:** 어절 당 평균 subword 수를 계산하여, 해당 문장이 tokenizer에 의해 얼마나 세밀하게 분절되는지를 정량화. 분절이 많을수록 activation 분포가 복잡해질 것이라는 가설.

---

## 2. 본 연구의 Novel Contribution

**개별 지표는 기존 것이지만, 다음 조합과 적용이 새로움:**

1. 세 지표를 **composite score로 합성**하여 문장 선택에 사용하는 것
2. 이를 **PTQ calibration 데이터 선별**이라는 맥락에 적용하는 것
3. Greedy coverage 기반으로 **형태소 커버리지를 극대화**하며 선별하는 것

> **기존 연구와의 관계:**
> - DuQuant (NeurIPS 2024 Oral): calibration 데이터 선택 전략을 **Future Work으로만 언급**
> - QDrop (ICLR 2022): calibration 분포의 중요성을 이론적으로 제시, **언어 차원 분석 없음**
> - 본 연구: 이 gap을 한국어 형태소 다양성 관점에서 **실증적으로 채움**

---

## 3. 논문 작성 시 인용 가이드

```
"We adopt well-established lexical diversity metrics — Type-Token Ratio (Johnson, 1944)
and subword fertility (Rust et al., 2021) — and extend them with a morpheme-level
unique ratio tailored for agglutinative languages. These three indicators are combined
into a composite diversity score, which, to our knowledge, is the first application of
linguistically-motivated data selection for PTQ calibration."
```
