"""
데이터 전처리 모듈
- 나무위키 데이터 로드, 문장 분리, 필터링
"""
import re
from typing import List, Optional


def load_namuwiki(max_docs: Optional[int] = None):
    """
    heegyu/namuwiki-extracted 데이터셋을 HuggingFace에서 로드.
    
    Args:
        max_docs: 로드할 최대 문서 수 (None이면 전체)
    Returns:
        dataset: HuggingFace Dataset 객체
    """
    from datasets import load_dataset
    
    dataset = load_dataset("heegyu/namuwiki-extracted", split="train")
    if max_docs is not None:
        dataset = dataset.select(range(min(max_docs, len(dataset))))
    
    print(f"[전처리] 나무위키 문서 {len(dataset)}개 로드 완료")
    return dataset


def clean_text(text: str) -> str:
    """나무위키 잔여 마크업 및 특수문자 정리."""
    # [math(...)] 태그 제거
    text = re.sub(r'\[math\([^\)]*\)\]', '', text)
    # [age(...)] 태그 제거
    text = re.sub(r'\[age\([^\)]*\)\]', '', text)
    # [include(...)] 태그 제거
    text = re.sub(r'\[include\([^\)]*\)\]', '', text)
    # [ruby(...)] 태그 제거
    text = re.sub(r'\[ruby\([^\)]*\)\]', '', text)
    # 위키 링크 [[텍스트]] → 텍스트, [[표시|링크]] → 표시
    text = re.sub(r'\[\[([^\]|]*)\|([^\]]*)\]\]', r'\1', text)
    text = re.sub(r'\[\[([^\]]*)\]\]', r'\1', text)
    # 각주 [* ...] 제거
    text = re.sub(r'\[\*[^\]]*\]', '', text)
    # 연속 공백 정리
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def split_sentences(text: str) -> List[str]:
    """
    텍스트를 문장 단위로 분리.
    kss 라이브러리가 있으면 사용, 없으면 정규식 기반 분리.
    """
    try:
        import kss
        sentences = kss.split_sentences(text)
    except ImportError:
        # kss 미설치 시 정규식 기반 분리
        sentences = re.split(r'(?<=[.!?。])\s+', text)
    
    return [s.strip() for s in sentences if s.strip()]


def is_valid_sentence(sentence: str, 
                      min_len: int = 20, 
                      max_len: int = 500,
                      min_korean_ratio: float = 0.5) -> bool:
    """
    문장 필터링 조건 검사.
    
    Args:
        sentence: 검사할 문장
        min_len: 최소 문자 수
        max_len: 최대 문자 수
        min_korean_ratio: 최소 한국어 비율 (0~1)
    """
    # 길이 검사
    if len(sentence) < min_len or len(sentence) > max_len:
        return False
    
    # 한국어 비율 검사
    korean_chars = len(re.findall(r'[가-힣]', sentence))
    total_chars = len(re.findall(r'[^\s]', sentence))
    if total_chars == 0:
        return False
    
    korean_ratio = korean_chars / total_chars
    if korean_ratio < min_korean_ratio:
        return False
    
    return True


def build_candidate_pool(n_sentences: int = 100_000,
                         max_docs: Optional[int] = None,
                         seed: int = 42) -> List[str]:
    """
    나무위키에서 전처리된 후보 문장 풀을 구축.
    
    Args:
        n_sentences: 최종 후보 문장 수
        max_docs: 처리할 최대 문서 수
        seed: 랜덤 시드 (재현성)
    Returns:
        sentences: 필터링된 문장 리스트
    """
    import random
    random.seed(seed)
    
    dataset = load_namuwiki(max_docs)
    
    all_sentences = []
    for doc in dataset:
        text = clean_text(doc['text'])
        sentences = split_sentences(text)
        valid = [s for s in sentences if is_valid_sentence(s)]
        all_sentences.extend(valid)
    
    print(f"[전처리] 유효 문장 총 {len(all_sentences)}개 추출")
    
    # 후보 풀 크기가 충분하면 랜덤 샘플링
    if len(all_sentences) > n_sentences:
        random.shuffle(all_sentences)
        all_sentences = all_sentences[:n_sentences]
        print(f"[전처리] {n_sentences}개로 샘플링 완료")
    
    return all_sentences


if __name__ == "__main__":
    # 테스트: 소규모로 동작 확인
    sentences = build_candidate_pool(n_sentences=1000, max_docs=100)
    print(f"\n--- 샘플 문장 (상위 5개) ---")
    for i, s in enumerate(sentences[:5]):
        print(f"[{i+1}] {s[:80]}...")
