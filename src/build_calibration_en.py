"""
영어 Calibration Set 구축 파이프라인 (Llama3-Ko 실험용)

한국어 C_v3 / 중국어 C_zh_v3 알고리즘의 영어 버전:
- NLTK POS 태깅 + WordNet 표제어 추출 (lemma, POS) 다양성 최대화
- 데이터: Wikitext-2 (Condition A와 동일 소스, 선별 방식만 다름)
- 가설 검증: "사전학습 언어 = calibration 언어" 원칙이 언어 독립적인가?

Usage:
    python src/build_calibration_en.py --model beomi/Llama-3-Open-Ko-8B --n-sentences 128
"""
import os
import sys
import json
import time
import re
import random
import argparse
from typing import List, Dict, Any, Tuple, Set
from dataclasses import dataclass, field

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("HOME", "/home/choihyun")
os.environ.setdefault("HF_HOME", os.path.join(WORKSPACE, ".cache/huggingface"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(WORKSPACE, ".cache/huggingface/hub"))

_EN_RE = re.compile(r"[a-zA-Z]")

# NLTK POS → WordNet POS 매핑
def _get_wordnet_pos(tag: str) -> str:
    from nltk.corpus import wordnet
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN  # 기본값


def en_char_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(1 for c in chars if _EN_RE.match(c)) / len(chars)


# ── 데이터 로딩 ───────────────────────────────────────────────────────────────

def load_english_candidates(n: int = 100_000, seed: int = 42) -> List[str]:
    """Wikitext-2에서 후보 문장 추출 (Condition A와 동일 소스)."""
    from datasets import load_dataset

    print(f"[데이터] Wikitext-2 로딩 중 (목표 {n}개 문장)...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    random.seed(seed)
    candidates = []

    for item in ds:
        text = item["text"].strip()
        if not text:
            continue
        # 문단 단위 분리 (Wikitext는 줄 단위로 문장이 옴)
        for line in text.split("\n"):
            line = line.strip()
            # 영어 문자 비율 0.7 이상, 길이 40~300자
            if 40 <= len(line) <= 300 and en_char_ratio(line) >= 0.7:
                candidates.append(line)

    random.shuffle(candidates)
    result = candidates[:n]
    print(f"[데이터] {len(result)}개 영어 후보 문장 확보")
    return result


# ── 형태소 분석 ───────────────────────────────────────────────────────────────

@dataclass
class EnScore:
    text: str
    morpheme_set: Set[Tuple[str, str]] = field(default_factory=set)
    n_words: int = 0
    n_unique_words: int = 0
    umr: float = 0.0
    ttr: float = 0.0
    sfs: float = 0.0
    diversity_score: float = 0.0
    n_tokens: int = 0


def analyze_en_morphemes(candidates: List[str]) -> List[EnScore]:
    """NLTK POS 태깅 + WordNet 표제어 추출로 형태소 분석."""
    import nltk
    from nltk.stem import WordNetLemmatizer
    from nltk import pos_tag, word_tokenize

    # 필요한 NLTK 데이터 다운로드 (이미 있으면 스킵)
    for pkg in ["averaged_perceptron_tagger_eng", "punkt_tab", "wordnet"]:
        nltk.download(pkg, quiet=True)

    lemmatizer = WordNetLemmatizer()
    scores = []

    print(f"[형태소] NLTK 영어 형태소 분석 중 ({len(candidates)}개 문장)...")
    for i, text in enumerate(candidates):
        if i % 10000 == 0 and i > 0:
            print(f"  {i}/{len(candidates)} 완료")

        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)

        # 알파벳 단어만 (구두점·숫자 제외)
        words = [(w, t) for w, t in tagged if w.isalpha() and len(w) > 1]

        # (lemma, coarse_pos) 쌍 생성
        morpheme_set = set()
        for word, tag in words:
            wn_pos = _get_wordnet_pos(tag)
            lemma = lemmatizer.lemmatize(word.lower(), wn_pos)
            # coarse POS: N/V/J/R 첫 글자만
            coarse = tag[0] if tag else "X"
            morpheme_set.add((lemma, coarse))

        n_words = len(words)
        n_unique = len(set(w.lower() for w, _ in words))
        umr = len(morpheme_set) / n_words if n_words > 0 else 0.0
        ttr = n_unique / n_words if n_words > 0 else 0.0

        scores.append(EnScore(
            text=text,
            morpheme_set=morpheme_set,
            n_words=n_words,
            n_unique_words=n_unique,
            umr=umr,
            ttr=ttr,
        ))

    return scores


def analyze_en_tokens(scores: List[EnScore], tokenizer) -> List[EnScore]:
    """tokenizer로 SFS 계산 (영어 단어 기준)."""
    for score in scores:
        token_ids = tokenizer.encode(score.text, add_special_tokens=False)
        score.n_tokens = len(token_ids)
        # SFS = BPE 토큰 수 / 영어 단어 수
        score.sfs = score.n_tokens / score.n_words if score.n_words > 0 else 0.0
    return scores


def compute_en_diversity(scores: List[EnScore]) -> List[EnScore]:
    """UMR / TTR / SFS 정규화 → diversity_score."""
    umr_vals = [s.umr for s in scores]
    ttr_vals = [s.ttr for s in scores]
    sfs_vals = [s.sfs for s in scores]

    def normalize(vals):
        mn, mx = min(vals), max(vals)
        if mx == mn:
            return [0.5] * len(vals)
        return [(v - mn) / (mx - mn) for v in vals]

    umr_n = normalize(umr_vals)
    ttr_n = normalize(ttr_vals)
    sfs_n = normalize(sfs_vals)

    for s, u, t, f in zip(scores, umr_n, ttr_n, sfs_n):
        s.diversity_score = (u + t + f) / 3.0
    return scores


# ── Greedy 선택 ───────────────────────────────────────────────────────────────

def filter_en_candidates(scores: List[EnScore],
                          min_words: int = 8,
                          min_tokens: int = 20,
                          min_en_ratio: float = 0.7,
                          min_sfs: float = 0.0) -> List[EnScore]:
    filtered = []
    for s in scores:
        if s.n_words < min_words:
            continue
        if s.n_tokens < min_tokens:
            continue
        if en_char_ratio(s.text) < min_en_ratio:
            continue
        if s.sfs < min_sfs:
            continue
        filtered.append(s)
    print(f"[필터] {len(scores)} → {len(filtered)}개 "
          f"(min_words={min_words}, min_en_ratio={min_en_ratio}, min_sfs={min_sfs})")
    return filtered


def greedy_en_select(scores: List[EnScore], n: int = 128,
                     alpha: float = 0.3) -> List[EnScore]:
    """영어 형태소 다양성 Greedy 선택."""
    covered: Set[Tuple[str, str]] = set()
    selected = []
    remaining = list(scores)

    while len(selected) < n and remaining:
        best_idx = -1
        best_score = -1.0

        for i, cand in enumerate(remaining):
            new_m = cand.morpheme_set - covered
            coverage_bonus = len(new_m) / (len(cand.morpheme_set) + 1e-8)
            score = cand.diversity_score + alpha * coverage_bonus
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx < 0:
            break

        chosen = remaining.pop(best_idx)
        covered |= chosen.morpheme_set
        selected.append(chosen)

        if len(selected) % 32 == 0:
            print(f"  {len(selected)}/{n} 선택 완료 "
                  f"(커버리지: {len(covered)} 형태소)")

    return selected


# ── 저장 ──────────────────────────────────────────────────────────────────────

def export_en_calibration(selected: List[EnScore], model_name: str,
                           output_path: str, params: Dict[str, Any]):
    unique_morphemes = set()
    for s in selected:
        unique_morphemes |= s.morpheme_set

    output = {
        "condition": "C_en_morphology_diverse",
        "model": model_name,
        "n_sentences": len(selected),
        "n_unique_morphemes": len(unique_morphemes),
        "avg_diversity_score": round(
            sum(s.diversity_score for s in selected) / len(selected), 4),
        "avg_sfs": round(sum(s.sfs for s in selected) / len(selected), 4),
        "avg_n_tokens": round(sum(s.n_tokens for s in selected) / len(selected), 2),
        "params": params,
        "sentences": [
            {
                "text": s.text,
                "n_words": s.n_words,
                "n_unique_morphemes": len(s.morpheme_set),
                "umr": round(s.umr, 4),
                "ttr": round(s.ttr, 4),
                "sfs": round(s.sfs, 4),
                "n_tokens": s.n_tokens,
                "diversity_score": round(s.diversity_score, 4),
            }
            for s in selected
        ],
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n[저장] {output_path}")
    print(f"  문장 수: {len(selected)}")
    print(f"  고유 형태소: {len(unique_morphemes)}")
    print(f"  평균 SFS: {output['avg_sfs']}")
    print(f"  평균 토큰 수: {output['avg_n_tokens']}")


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="영어 Calibration Set 구축 (C_en_v3 — Llama3-Ko 실험용)"
    )
    parser.add_argument("--model", default="beomi/Llama-3-Open-Ko-8B",
                        help="대상 모델 (tokenizer 사용)")
    parser.add_argument("--n-sentences", type=int, default=128)
    parser.add_argument("--n-candidates", type=int, default=50_000,
                        help="Wikitext-2 후보 풀 크기 (전체 약 2.5M토큰)")
    parser.add_argument("--alpha", type=float, default=0.3,
                        help="형태소 커버리지 보너스 계수")
    parser.add_argument("--min-words", type=int, default=8,
                        help="최소 영어 단어 수")
    parser.add_argument("--min-tokens", type=int, default=20,
                        help="최소 subword 토큰 수")
    parser.add_argument("--min-en-ratio", type=float, default=0.7,
                        help="최소 영어 문자 비율")
    parser.add_argument("--min-sfs", type=float, default=0.0,
                        help="최소 SFS (높을수록 subword-rich 문장만 선별)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--suffix", type=str, default="en_v3",
                        help="출력 파일명 suffix")
    args = parser.parse_args()

    start = time.time()

    # 1. 후보 로딩 (Wikitext-2 — A와 동일 소스)
    candidates = load_english_candidates(n=args.n_candidates, seed=args.seed)

    # 2. 형태소 분석
    scores = analyze_en_morphemes(candidates)

    # 3. 토큰 분석
    print(f"\n[토큰] {args.model} tokenizer 로딩...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    scores = analyze_en_tokens(scores, tokenizer)

    # 4. 다양성 점수
    scores = compute_en_diversity(scores)

    # 5. 필터
    scores = filter_en_candidates(
        scores,
        min_words=args.min_words,
        min_tokens=args.min_tokens,
        min_en_ratio=args.min_en_ratio,
        min_sfs=args.min_sfs,
    )
    if len(scores) < args.n_sentences:
        raise ValueError(f"필터 후 후보 부족: {len(scores)} < {args.n_sentences}")

    # 6. Greedy 선택
    print(f"\n[선택] Greedy 다양성 선택 (n={args.n_sentences})...")
    selected = greedy_en_select(scores, n=args.n_sentences, alpha=args.alpha)

    # 7. 저장
    safe_model = args.model.replace("/", "_")
    out_path = os.path.join(WORKSPACE, "results",
                            f"calibration_set_C_{args.suffix}_{safe_model}.json")
    export_en_calibration(selected, args.model, out_path, params=vars(args))

    # 8. 통계 요약
    print("\n[통계 요약]")
    print(f"  선택된 문장: {len(selected)}")
    print(f"  고유 (lemma, POS) 쌍: {sum(len(s.morpheme_set) for s in selected)}")
    print(f"  커버된 고유 형태소: {len(set().union(*[s.morpheme_set for s in selected]))}")
    print(f"  평균 단어 수: {sum(s.n_words for s in selected)/len(selected):.1f}")
    print(f"  평균 SFS: {sum(s.sfs for s in selected)/len(selected):.2f}")

    print(f"\n[완료] 총 소요 시간: {time.time()-start:.1f}초")


if __name__ == "__main__":
    main()
