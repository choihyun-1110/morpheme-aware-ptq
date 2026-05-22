"""
build_hybrid_calibration.py
Task-Aware Calibration Bank Mixing

두 개의 calibration bank를 지정한 비율로 혼합한 hybrid calibration set 생성.

사용 예:
    python3 build_hybrid_calibration.py \
        --bank-a results/calibration_set_B.json \
        --bank-b results/calibration_set_C_v3_upstage_SOLAR-10.7B-Instruct-v1.0.json \
        --n-a 64 --n-b 64 \
        --mode interleave \
        --label-a B --label-b Cv3 \
        --out-dir results/
"""

import argparse
import json
import os
import random
from datetime import datetime


def load_sentences(path):
    with open(path) as f:
        d = json.load(f)
    sents = d.get("sentences", [])
    return [s["text"] if isinstance(s, dict) else s for s in sents], d


def sample_sentences(sents, n, seed=42):
    """앞에서 n개 slice (greedy 순서 유지) or 랜덤 샘플링."""
    if n >= len(sents):
        return sents[:]
    # greedy 알고리즘으로 생성된 bank는 순서가 의미 있으므로 앞에서 slice
    return sents[:n]


def interleave(sents_a, sents_b):
    """A와 B를 교차 배치: A[0], B[0], A[1], B[1], ..."""
    result = []
    for a, b in zip(sents_a, sents_b):
        result.append(a)
        result.append(b)
    # 길이 차이 처리
    if len(sents_a) > len(sents_b):
        result.extend(sents_a[len(sents_b):])
    elif len(sents_b) > len(sents_a):
        result.extend(sents_b[len(sents_a):])
    return result


def concat(sents_a, sents_b, order="ab"):
    """A 다음 B (또는 B 다음 A) 순서로 이어 붙이기."""
    if order == "ab":
        return sents_a + sents_b
    else:
        return sents_b + sents_a


def build_hybrid(sents_a, sents_b, n_a, n_b, mode, seed=42):
    selected_a = sample_sentences(sents_a, n_a, seed)
    selected_b = sample_sentences(sents_b, n_b, seed)

    if mode == "interleave":
        mixed = interleave(selected_a, selected_b)
    elif mode == "concat_ab":
        mixed = concat(selected_a, selected_b, order="ab")
    elif mode == "concat_ba":
        mixed = concat(selected_a, selected_b, order="ba")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return mixed, selected_a, selected_b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-a", required=True, help="reasoning bank JSON (e.g. C_v3)")
    parser.add_argument("--bank-b", required=True, help="domain bank JSON (e.g. B)")
    parser.add_argument("--n-a", type=int, default=64, help="bank-a에서 선택할 문장 수")
    parser.add_argument("--n-b", type=int, default=64, help="bank-b에서 선택할 문장 수")
    parser.add_argument("--mode", choices=["interleave", "concat_ab", "concat_ba"],
                        default="interleave", help="mixing 방식")
    parser.add_argument("--label-a", default="A", help="bank-a 레이블 (파일명용)")
    parser.add_argument("--label-b", default="B", help="bank-b 레이블 (파일명용)")
    parser.add_argument("--out-dir", default="results/")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sents_a, meta_a = load_sentences(args.bank_a)
    sents_b, meta_b = load_sentences(args.bank_b)

    print(f"Bank A ({args.label_a}): {len(sents_a)} sentences → {args.n_a} selected")
    print(f"Bank B ({args.label_b}): {len(sents_b)} sentences → {args.n_b} selected")
    print(f"Mode: {args.mode}")

    mixed, sel_a, sel_b = build_hybrid(
        sents_a, sents_b, args.n_a, args.n_b, args.mode, args.seed
    )

    label = f"H_{args.label_a}{args.n_a}_{args.label_b}{args.n_b}_{args.mode}"
    filename = f"calibration_set_{label}.json"
    out_path = os.path.join(args.out_dir, filename)

    output = {
        "condition": f"hybrid_{args.mode}",
        "label": label,
        "n_sentences": len(mixed),
        "mixing": {
            "bank_a": {
                "label": args.label_a,
                "source_file": os.path.basename(args.bank_a),
                "n_selected": len(sel_a),
                "original_condition": meta_a.get("condition", ""),
            },
            "bank_b": {
                "label": args.label_b,
                "source_file": os.path.basename(args.bank_b),
                "n_selected": len(sel_b),
                "original_condition": meta_b.get("condition", ""),
            },
            "mode": args.mode,
            "seed": args.seed,
        },
        "created_at": datetime.now().isoformat(),
        "sentences": [{"text": s} for s in mixed],
    }

    os.makedirs(args.out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n생성 완료: {out_path}")
    print(f"총 {len(mixed)}문장 (A:{len(sel_a)} + B:{len(sel_b)})")
    print(f"\n샘플 (앞 3개):")
    for i, s in enumerate(mixed[:3]):
        print(f"  [{i}] {s[:80]}")


if __name__ == "__main__":
    main()
