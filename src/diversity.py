"""
다양성 점수 산출 모듈
- UMR, TTR, SFS 계산 및 Composite Diversity Score 산출
"""
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np


@dataclass
class SentenceScore:
    """문장별 다양성 점수 데이터 클래스."""
    text: str
    # 형태소 분석 결과
    morphemes: List[Tuple[str, str]] = field(default_factory=list)  # (형태소, 품사)
    morpheme_set: Set[Tuple[str, str]] = field(default_factory=set)
    # 토크나이저 결과
    token_ids: List[int] = field(default_factory=list)
    n_eojeols: int = 0  # 어절 수
    # 점수
    umr: float = 0.0   # Unique Morpheme Ratio
    ttr: float = 0.0   # Type-Token Ratio
    sfs: float = 0.0   # Subword Fragmentation Score
    diversity_score: float = 0.0  # 최종 합성 점수
    # 정규화 점수
    umr_norm: float = 0.0
    ttr_norm: float = 0.0
    sfs_norm: float = 0.0


def analyze_morphemes(sentences: List[str], num_workers: int = 4) -> List[SentenceScore]:
    """
    Kiwi로 형태소 분석 수행 및 UMR 산출.
    
    Args:
        sentences: 분석할 문장 리스트
        num_workers: Kiwi 멀티스레드 워커 수
    Returns:
        scores: SentenceScore 리스트 (morphemes, morpheme_set, umr 채워짐)
    """
    from kiwipiepy import Kiwi
    
    kiwi = Kiwi(num_workers=num_workers)
    scores = []
    
    for text in sentences:
        result = kiwi.tokenize(text)
        morphemes = [(token.form, token.tag) for token in result]
        morpheme_set = set(morphemes)
        
        umr = len(morpheme_set) / len(morphemes) if morphemes else 0.0
        n_eojeols = len(text.split())
        
        score = SentenceScore(
            text=text,
            morphemes=morphemes,
            morpheme_set=morpheme_set,
            umr=umr,
            n_eojeols=n_eojeols,
        )
        scores.append(score)
    
    print(f"[다양성] 형태소 분석 완료: {len(scores)}문장, 평균 UMR={np.mean([s.umr for s in scores]):.3f}")
    return scores


def analyze_tokens(scores: List[SentenceScore], tokenizer) -> List[SentenceScore]:
    """
    LLM tokenizer로 TTR 및 SFS 산출.

    SFS = 한국어 어절당 subword 토큰 수.
    비한국어(한자/외국어/숫자/기호) 어절은 SFS 분모에서 제외하여
    비한국어 문자 혼합에 의한 SFS 인플레이션을 방지한다.

    Args:
        scores: 형태소 분석 완료된 SentenceScore 리스트
        tokenizer: HuggingFace AutoTokenizer 인스턴스
    Returns:
        scores: TTR, SFS가 추가로 채워진 SentenceScore 리스트
    """
    import re
    _KO_RE = re.compile(r"[가-힣]")

    for score in scores:
        encoding = tokenizer(score.text, add_special_tokens=False)
        token_ids = encoding["input_ids"]
        unique_tokens = set(token_ids)

        score.token_ids = token_ids
        score.ttr = len(unique_tokens) / len(token_ids) if token_ids else 0.0

        # SFS 분모: 한국어 문자를 1자 이상 포함하는 어절 수만 사용
        ko_eojeols = sum(1 for eo in score.text.split() if _KO_RE.search(eo))
        score.sfs = len(token_ids) / ko_eojeols if ko_eojeols > 0 else 0.0

    print(f"[다양성] 토큰 분석 완료: 평균 TTR={np.mean([s.ttr for s in scores]):.3f}, "
          f"평균 SFS={np.mean([s.sfs for s in scores]):.2f}")
    return scores


def compute_diversity_scores(scores: List[SentenceScore],
                              weights: Tuple[float, float, float] = (1/3, 1/3, 1/3)
                              ) -> List[SentenceScore]:
    """
    min-max 정규화 후 Composite Diversity Score 산출.
    
    Args:
        scores: UMR, TTR, SFS가 산출된 SentenceScore 리스트
        weights: (w_umr, w_ttr, w_sfs) 가중치
    Returns:
        scores: diversity_score가 채워진 SentenceScore 리스트
    """
    w1, w2, w3 = weights
    
    # min-max 정규화
    umrs = np.array([s.umr for s in scores])
    ttrs = np.array([s.ttr for s in scores])
    sfss = np.array([s.sfs for s in scores])
    
    def minmax(arr):
        rng = arr.max() - arr.min()
        if rng == 0:
            return np.zeros_like(arr)
        return (arr - arr.min()) / rng
    
    umr_norm = minmax(umrs)
    ttr_norm = minmax(ttrs)
    sfs_norm = minmax(sfss)
    
    for i, score in enumerate(scores):
        score.umr_norm = float(umr_norm[i])
        score.ttr_norm = float(ttr_norm[i])
        score.sfs_norm = float(sfs_norm[i])
        score.diversity_score = w1 * score.umr_norm + w2 * score.ttr_norm + w3 * score.sfs_norm
    
    print(f"[다양성] 점수 산출 완료: 평균 D={np.mean([s.diversity_score for s in scores]):.3f}, "
          f"최대 D={max(s.diversity_score for s in scores):.3f}")
    return scores


if __name__ == "__main__":
    # 간단 테스트 (토크나이저 없이 UMR만)
    test_sentences = [
        "철수가 학교에서 과학을 열심히 공부했다.",
        "나는 나를 나에게 보냈다.",
        "대한민국의 수도는 서울특별시이며 인구가 가장 많은 도시이다.",
    ]
    scores = analyze_morphemes(test_sentences)
    for s in scores:
        print(f"UMR={s.umr:.3f} | 형태소 {len(s.morphemes)}개 | {s.text}")
