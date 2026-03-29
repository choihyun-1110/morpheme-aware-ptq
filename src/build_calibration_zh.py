"""
중국어 Calibration Set 구축 파이프라인 (Qwen2 실험용)

한국어 C_v3 알고리즘의 중국어 버전:
- jieba 형태소 분석기
- 중국어 Wikipedia 데이터
- 중국어 문자 비율 필터
- 형태소 다양성 greedy 선택

Usage:
    python src/build_calibration_zh.py --model Qwen/Qwen2-7B-Instruct --n-sentences 128
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

# 중국어 한자 범위
_ZH_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
_NON_ZH_TAGS = {"eng", "x"}   # jieba 품사에서 영어/특수문자


def is_chinese_char(c: str) -> bool:
    return bool(_ZH_RE.match(c))


def zh_char_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(1 for c in chars if is_chinese_char(c)) / len(chars)


# ── 데이터 로딩 ──────────────────────────────────────────────────────────────

def load_chinese_candidates(n: int = 100_000, seed: int = 42) -> List[str]:
    """중국어 Wikipedia에서 후보 문장 추출."""
    from datasets import load_dataset

    print(f"[데이터] 중국어 Wikipedia 로딩 중 (목표 {n}개 문장)...")
    # wikipedia 중국어 데이터셋
    ds = load_dataset("wikimedia/wikipedia", "20231101.zh", split="train",
                      streaming=True)

    random.seed(seed)
    candidates = []

    for item in ds:
        if len(candidates) >= n * 3:   # 여유 있게 수집
            break
        text = item["text"]
        # 문단 단위 분리
        for para in text.split("\n"):
            para = para.strip()
            # 중국어 문자 비율 0.5 이상, 길이 10~200자
            if 10 <= len(para) <= 200 and zh_char_ratio(para) >= 0.5:
                candidates.append(para)
            if len(candidates) >= n * 3:
                break

    random.shuffle(candidates)
    result = candidates[:n]
    print(f"[데이터] {len(result)}개 중국어 후보 문장 확보")
    return result


# ── 형태소 분석 ──────────────────────────────────────────────────────────────

@dataclass
class ZhScore:
    text: str
    morpheme_set: Set[Tuple[str, str]] = field(default_factory=set)
    n_words: int = 0
    n_unique_words: int = 0
    umr: float = 0.0
    ttr: float = 0.0
    sfs: float = 0.0
    diversity_score: float = 0.0
    n_tokens: int = 0


def analyze_zh_morphemes(candidates: List[str]) -> List[ZhScore]:
    """jieba로 형태소 분석."""
    try:
        import jieba
        import jieba.posseg as pseg
    except ImportError:
        raise ImportError("jieba가 없습니다. pip install jieba")

    scores = []
    for text in candidates:
        pairs = list(pseg.cut(text))
        words = [(w, f) for w, f in pairs if w.strip()]

        # 중국어 단어만 커버리지 계산 (영어/숫자 제외)
        zh_words = [(w, f) for w, f in words
                    if zh_char_ratio(w) >= 0.5 and f not in _NON_ZH_TAGS]

        morpheme_set = {(w, f) for w, f in zh_words}
        n_words = len(words)
        n_unique = len(set(w for w, f in words))

        umr = len(morpheme_set) / n_words if n_words > 0 else 0.0
        ttr = n_unique / n_words if n_words > 0 else 0.0

        scores.append(ZhScore(
            text=text,
            morpheme_set=morpheme_set,
            n_words=n_words,
            n_unique_words=n_unique,
            umr=umr,
            ttr=ttr,
        ))
    return scores


def analyze_zh_tokens(scores: List[ZhScore], tokenizer) -> List[ZhScore]:
    """tokenizer로 SFS 계산 (중국어 단어 기준)."""
    try:
        import jieba
    except ImportError:
        raise ImportError("jieba가 없습니다. pip install jieba")

    for score in scores:
        token_ids = tokenizer.encode(score.text, add_special_tokens=False)
        score.n_tokens = len(token_ids)

        # 중국어 단어(어절) 수 기준 SFS
        zh_words = [w for w in jieba.cut(score.text)
                    if zh_char_ratio(w) >= 0.5]
        n_zh_words = len(zh_words)
        score.sfs = score.n_tokens / n_zh_words if n_zh_words > 0 else 0.0

    return scores


def compute_zh_diversity(scores: List[ZhScore]) -> List[ZhScore]:
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


# ── Greedy 선택 ──────────────────────────────────────────────────────────────

def filter_zh_candidates(scores: List[ZhScore],
                          min_words: int = 5,
                          min_tokens: int = 20,
                          min_zh_ratio: float = 0.7) -> List[ZhScore]:
    filtered = []
    for s in scores:
        if s.n_words < min_words:
            continue
        if s.n_tokens < min_tokens:
            continue
        if zh_char_ratio(s.text) < min_zh_ratio:
            continue
        filtered.append(s)
    print(f"[필터] {len(scores)} → {len(filtered)}개 (min_words={min_words}, "
          f"min_zh_ratio={min_zh_ratio})")
    return filtered


def greedy_zh_select(scores: List[ZhScore], n: int = 128,
                     alpha: float = 0.3) -> List[ZhScore]:
    """중국어 형태소 다양성 Greedy 선택."""
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


# ── 저장 ─────────────────────────────────────────────────────────────────────

def export_zh_calibration(selected: List[ZhScore], model_name: str,
                           output_path: str, params: Dict[str, Any]):
    unique_morphemes = set()
    for s in selected:
        unique_morphemes |= s.morpheme_set

    output = {
        "condition": "C_zh_morphology_diverse",
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


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2-7B-Instruct")
    parser.add_argument("--n-sentences", type=int, default=128)
    parser.add_argument("--n-candidates", type=int, default=50_000)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--min-words", type=int, default=5)
    parser.add_argument("--min-tokens", type=int, default=20)
    parser.add_argument("--min-zh-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--suffix", type=str, default="zh")
    args = parser.parse_args()

    start = time.time()

    # 1. 후보 로딩
    candidates = load_chinese_candidates(n=args.n_candidates, seed=args.seed)

    # 2. 형태소 분석
    print("\n[형태소] jieba 분석 중...")
    scores = analyze_zh_morphemes(candidates)

    # 3. 토큰 분석
    print(f"\n[토큰] {args.model} tokenizer 로딩...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    scores = analyze_zh_tokens(scores, tokenizer)

    # 4. 다양성 점수
    scores = compute_zh_diversity(scores)

    # 5. 필터
    scores = filter_zh_candidates(
        scores,
        min_words=args.min_words,
        min_tokens=args.min_tokens,
        min_zh_ratio=args.min_zh_ratio,
    )
    if len(scores) < args.n_sentences:
        raise ValueError(f"필터 후 후보 부족: {len(scores)} < {args.n_sentences}")

    # 6. Greedy 선택
    print(f"\n[선택] Greedy 다양성 선택 (n={args.n_sentences})...")
    selected = greedy_zh_select(scores, n=args.n_sentences, alpha=args.alpha)

    # 7. 저장
    safe_model = args.model.replace("/", "_")
    out_path = os.path.join(WORKSPACE, "results",
                            f"calibration_set_C_{args.suffix}_{safe_model}.json")
    export_zh_calibration(selected, args.model, out_path, params=vars(args))

    print(f"\n[완료] 총 소요 시간: {time.time()-start:.1f}초")


if __name__ == "__main__":
    main()
