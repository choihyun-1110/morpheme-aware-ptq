"""
메인 파이프라인: Calibration Set 구축
- 전처리 → 형태소 분석 → 토큰 분석 → 다양성 점수 → Greedy 선별 → 저장
"""
import os
import sys
import argparse
import time

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE_ROOT = os.path.join(WORKSPACE_ROOT, ".cache")
HF_HOME_ROOT = os.path.join(CACHE_ROOT, "huggingface")

os.environ.setdefault("HOME", "/home/choihyun")
os.environ.setdefault("XDG_CACHE_HOME", CACHE_ROOT)
os.environ.setdefault("HF_HOME", HF_HOME_ROOT)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(HF_HOME_ROOT, "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(HF_HOME_ROOT, "transformers"))

for path in (
    os.environ["XDG_CACHE_HOME"],
    os.environ["HF_HOME"],
    os.environ["HUGGINGFACE_HUB_CACHE"],
    os.environ["TRANSFORMERS_CACHE"],
):
    os.makedirs(path, exist_ok=True)

# 프로젝트 루트의 src를 path에 추가
sys.path.insert(0, os.path.dirname(__file__))

from preprocess import build_candidate_pool
from diversity import analyze_morphemes, analyze_tokens, compute_diversity_scores
from selection import (
    filter_sentence_like_candidates,
    greedy_diversity_select,
    export_calibration_set,
)


def build_condition_A(n: int = 128, seed: int = 42) -> str:
    """조건 A: Wikitext-2 영어 랜덤 128문장."""
    from datasets import load_dataset
    import random
    import json
    
    random.seed(seed)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    
    # 빈 줄 제거 및 최소 길이 필터링
    sentences = [row['text'].strip() for row in dataset 
                 if row['text'].strip() and len(row['text'].strip()) > 20]
    
    selected = random.sample(sentences, min(n, len(sentences)))
    
    output_path = os.path.join(os.path.dirname(__file__), "..", "results", "calibration_set_A.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    output = {
        "condition": "A_english_baseline",
        "n_sentences": len(selected),
        "source": "wikitext-2-raw-v1",
        "sentences": [{"text": s} for s in selected],
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"[조건 A] 영어 기준 {len(selected)}문장 → {output_path}")
    return output_path


def build_condition_B(candidate_pool, n: int = 128, seed: int = 42) -> str:
    """조건 B: 나무위키 한국어 랜덤 128문장."""
    import random
    import json
    
    random.seed(seed)
    selected = random.sample(candidate_pool, min(n, len(candidate_pool)))
    
    output_path = os.path.join(os.path.dirname(__file__), "..", "results", "calibration_set_B.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    output = {
        "condition": "B_korean_random",
        "n_sentences": len(selected),
        "source": "heegyu/namuwiki-extracted",
        "sentences": [{"text": s} for s in selected],
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"[조건 B] 한국어 랜덤 {len(selected)}문장 → {output_path}")
    return output_path


def build_condition_C(candidate_pool, tokenizer, model_name: str,
                       n: int = 128,
                       alpha: float = 0.3,
                       beta: float = 0.15,
                       gamma: float = 0.15,
                       min_eojeols: int = 5,
                       min_subword_tokens: int = 24,
                       require_sentence_final: bool = True,
                       target_eojeols: int = 12,
                       target_subword_tokens: int = 64,
                       min_ko_ratio: float = 0.0,
                       suffix: str = "") -> str:
    """조건 C: 한국어 형태소 다양성 최대화 128문장."""
    
    print(f"\n{'='*60}")
    print(f"  조건 C: 형태소 다양성 최대화 (n={n}, α={alpha})")
    print(f"  모델: {model_name}")
    print(f"{'='*60}\n")
    
    # Step 1: 형태소 분석 → UMR
    scores = analyze_morphemes(candidate_pool)
    
    # Step 2: 토큰 분석 → TTR, SFS
    scores = analyze_tokens(scores, tokenizer)
    
    # Step 3: C 조건 전용 문장형 후보 필터링
    scores = filter_sentence_like_candidates(
        scores,
        min_eojeols=min_eojeols,
        min_subword_tokens=min_subword_tokens,
        require_sentence_final=require_sentence_final,
        min_ko_ratio=min_ko_ratio,
    )
    if len(scores) < n:
        raise ValueError(
            f"문장형 필터 후 후보가 부족합니다: {len(scores)}개 < {n}개. "
            "필터 조건을 완화해 주세요."
        )

    # Step 4: 다양성 점수 산출
    scores = compute_diversity_scores(scores)
    
    # Step 5: Greedy 선별
    selected = greedy_diversity_select(
        scores,
        n=n,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        min_eojeols=min_eojeols,
        target_eojeols=target_eojeols,
        min_subword_tokens=min_subword_tokens,
        target_subword_tokens=target_subword_tokens,
    )
    
    # Step 6: 저장
    safe_model_name = model_name.replace("/", "_")
    version_tag = f"_{suffix}" if suffix else ""
    output_path = os.path.join(os.path.dirname(__file__), "..",
                               "results", f"calibration_set_C{version_tag}_{safe_model_name}.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    export_calibration_set(
        selected, model_name, output_path,
        params={
            "weights": [1/3, 1/3, 1/3],
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "candidate_pool_size": len(candidate_pool),
            "filtered_candidate_size": len(scores),
            "min_eojeols": min_eojeols,
            "min_subword_tokens": min_subword_tokens,
            "require_sentence_final": require_sentence_final,
            "target_eojeols": target_eojeols,
            "target_subword_tokens": target_subword_tokens,
            "min_ko_ratio": min_ko_ratio,
            "ko_only_coverage_bonus": True,
            "ko_only_sfs": True,
        }
    )
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Calibration Set 구축 파이프라인")
    parser.add_argument("--condition", type=str, default="C",
                       choices=["A", "B", "C", "all"],
                       help="구축할 조건 (A/B/C/all)")
    parser.add_argument("--model", type=str, 
                       default="upstage/SOLAR-10.7B-Instruct-v1.0",
                       help="대상 모델 (조건 C에서 tokenizer 사용)")
    parser.add_argument("--n-sentences", type=int, default=128,
                       help="선택할 문장 수")
    parser.add_argument("--n-candidates", type=int, default=100_000,
                       help="후보 풀 크기")
    parser.add_argument("--max-docs", type=int, default=None,
                       help="처리할 최대 나무위키 문서 수 (테스트용)")
    parser.add_argument("--alpha", type=float, default=0.3,
                       help="커버리지 보너스 계수")
    parser.add_argument("--beta", type=float, default=0.15,
                       help="길이 보너스 계수 (조건 C)")
    parser.add_argument("--gamma", type=float, default=0.15,
                       help="문장 종결 보너스 계수 (조건 C)")
    parser.add_argument("--c-min-eojeols", type=int, default=5,
                       help="조건 C 최소 어절 수")
    parser.add_argument("--c-min-tokens", type=int, default=24,
                       help="조건 C 최소 subword 토큰 수")
    parser.add_argument("--c-target-eojeols", type=int, default=12,
                       help="조건 C 목표 어절 수")
    parser.add_argument("--c-target-tokens", type=int, default=64,
                       help="조건 C 목표 subword 토큰 수")
    parser.add_argument("--allow-fragments", action="store_true",
                       help="조건 C에서 문장 종결형 필터를 끔")
    parser.add_argument("--c-min-ko-ratio", type=float, default=0.0,
                       help="조건 C 최소 한국어 문자 비율 (0.0이면 비활성, 예: 0.7)")
    parser.add_argument("--suffix", type=str, default="",
                       help="출력 파일명 suffix (예: v3 → calibration_set_C_v3_...)")
    parser.add_argument("--seed", type=int, default=42,
                       help="랜덤 시드")
    args = parser.parse_args()
    
    start_time = time.time()
    
    # 후보 풀 구축 (조건 B, C에서 공통 사용)
    candidate_pool = None
    if args.condition in ("B", "C", "all"):
        print("\n[1단계] 후보 문장 풀 구축 중...")
        candidate_pool = build_candidate_pool(
            n_sentences=args.n_candidates,
            max_docs=args.max_docs,
            seed=args.seed
        )
    
    # 조건별 구축
    if args.condition in ("A", "all"):
        build_condition_A(n=args.n_sentences, seed=args.seed)
    
    if args.condition in ("B", "all"):
        build_condition_B(candidate_pool, n=args.n_sentences, seed=args.seed)
    
    if args.condition in ("C", "all"):
        from transformers import AutoTokenizer
        print(f"\n[로드] Tokenizer: {args.model}")
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        build_condition_C(
            candidate_pool, tokenizer, args.model,
            n=args.n_sentences,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            min_eojeols=args.c_min_eojeols,
            min_subword_tokens=args.c_min_tokens,
            require_sentence_final=not args.allow_fragments,
            target_eojeols=args.c_target_eojeols,
            target_subword_tokens=args.c_target_tokens,
            min_ko_ratio=args.c_min_ko_ratio,
            suffix=args.suffix,
        )
    
    elapsed = time.time() - start_time
    print(f"\n[완료] 총 소요 시간: {elapsed:.1f}초")


if __name__ == "__main__":
    main()
