"""
Activation Sensitivity Score (ASS) 계산 — S4-13

기존 activation_analysis.json(레이어별 통계)을 읽어서
각 레이어가 calibration 조건 변화에 얼마나 민감한지 점수화.

핵심 지표:
    sensitivity_score(layer) = Σ_metric  |metric(C_v3) - metric(A)| / metric(A)

    metric: outlier_ratio, channel_cv, mean_std (entropy 제외 — noise 큼)

결과:
    - 레이어별 sensitivity score 순위
    - "고감도 레이어" (score > threshold) 목록
    - 논문 Figure용 bar chart (레이어 vs sensitivity)

사용법:
    python src/compute_sensitivity_score.py
    python src/compute_sensitivity_score.py \\
        --input results/activation_analysis.json \\
        --ref A --target C_v3 \\
        --out results/sensitivity_score.json \\
        --plot results/sensitivity_score.png
"""
import os
import json
import argparse
import numpy as np

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

METRICS = ["outlier_ratio", "channel_cv", "mean_std"]
THRESHOLD_QUANTILE = 0.75   # 상위 25% 레이어를 "고감도"로 분류


def compute_scores(data: dict, ref: str, target: str) -> dict:
    """
    레이어별 sensitivity score 계산.

    score = 평균 상대 변화량 over METRICS
    """
    ref_layers   = data[ref]["per_layer"]
    tgt_layers   = data[target]["per_layer"]

    layer_ids = sorted(ref_layers.keys(), key=int)
    scores = {}

    for lid in layer_ids:
        r = ref_layers[lid]
        t = tgt_layers[lid]

        rel_changes = []
        for m in METRICS:
            base = abs(r.get(m, 0))
            diff = abs(t.get(m, 0) - r.get(m, 0))
            rel = diff / (base + 1e-8)
            rel_changes.append(rel)

        scores[int(lid)] = {
            "score":            round(float(np.mean(rel_changes)), 6),
            "rel_outlier":      round(float(rel_changes[0]), 6),
            "rel_channel_cv":   round(float(rel_changes[1]), 6),
            "rel_mean_std":     round(float(rel_changes[2]), 6),
            "ref_channel_cv":   r.get("channel_cv"),
            "tgt_channel_cv":   t.get("channel_cv"),
            "ref_outlier":      r.get("outlier_ratio"),
            "tgt_outlier":      t.get("outlier_ratio"),
        }

    return scores


def summarize(scores: dict) -> dict:
    vals = [v["score"] for v in scores.values()]
    threshold = float(np.quantile(vals, THRESHOLD_QUANTILE))
    high_sensitivity = [lid for lid, v in scores.items() if v["score"] >= threshold]

    return {
        "mean_score":        round(float(np.mean(vals)), 6),
        "max_score":         round(float(np.max(vals)), 6),
        "threshold_q75":     round(threshold, 6),
        "n_high_sensitivity": len(high_sensitivity),
        "high_sensitivity_layers": sorted(high_sensitivity),
        "top5_layers":       sorted(scores, key=lambda l: scores[l]["score"], reverse=True)[:5],
    }


def plot(scores: dict, summary: dict, out_path: str, ref: str, target: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib 없음 — 시각화 생략")
        return

    layer_ids = sorted(scores.keys())
    vals = [scores[lid]["score"] for lid in layer_ids]
    colors = ["#e74c3c" if scores[lid]["score"] >= summary["threshold_q75"] else "#3498db"
              for lid in layer_ids]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(layer_ids, vals, color=colors, width=0.8, edgecolor="none")
    ax.axhline(summary["threshold_q75"], color="gray", linestyle="--", linewidth=1,
               label=f"Q75 threshold ({summary['threshold_q75']:.4f})")
    ax.set_xlabel("Layer index", fontsize=11)
    ax.set_ylabel("Sensitivity score\n(mean relative change)", fontsize=11)
    ax.set_title(f"Activation Sensitivity Score: {ref} → {target}  "
                 f"(SOLAR-10.7B FP16)", fontsize=12)
    ax.legend(fontsize=9)

    # 상위 5개 레이어 라벨
    for lid in summary["top5_layers"]:
        ax.text(lid, scores[lid]["score"] + 0.001, str(lid),
                ha="center", va="bottom", fontsize=7, color="#c0392b")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[plot] 저장: {out_path}")
    plt.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  default=os.path.join(WORKSPACE, "results/activation_analysis.json"))
    p.add_argument("--ref",    default="A",    help="기준 조건")
    p.add_argument("--target", default="C_v3", help="비교 조건")
    p.add_argument("--out",    default=os.path.join(WORKSPACE, "results/sensitivity_score.json"))
    p.add_argument("--plot",   default=os.path.join(WORKSPACE, "results/sensitivity_score.png"))
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    if args.ref not in data:
        print(f"[오류] ref='{args.ref}' 가 {args.input}에 없음. 있는 조건: {list(data.keys())}")
        return
    if args.target not in data:
        print(f"[오류] target='{args.target}' 가 {args.input}에 없음. 있는 조건: {list(data.keys())}")
        return

    print(f"[ASS] {args.ref} → {args.target} 레이어별 sensitivity score 계산")

    scores  = compute_scores(data, args.ref, args.target)
    summary = summarize(scores)

    # 출력
    print(f"\n{'='*55}")
    print(f"Activation Sensitivity Score 요약 ({args.ref} → {args.target})")
    print(f"{'='*55}")
    print(f"  mean score:           {summary['mean_score']:.4f}")
    print(f"  max score:            {summary['max_score']:.4f}")
    print(f"  Q75 threshold:        {summary['threshold_q75']:.4f}")
    print(f"  고감도 레이어 수:      {summary['n_high_sensitivity']} / {len(scores)}")
    print(f"  고감도 레이어:         {summary['high_sensitivity_layers']}")
    print(f"  Top-5 레이어:         {summary['top5_layers']}")

    print(f"\n{'Layer':>6} | {'score':>8} | {'rel_cv':>8} | {'rel_out':>8} | "
          f"{'A_cv':>7} | {'Cv3_cv':>7}")
    print("-" * 60)
    for lid in sorted(scores.keys()):
        v = scores[lid]
        marker = " ←" if lid in summary["top5_layers"] else ""
        print(f"{lid:>6} | {v['score']:>8.4f} | {v['rel_channel_cv']:>8.4f} | "
              f"{v['rel_outlier']:>8.4f} | {v['ref_channel_cv']:>7.3f} | "
              f"{v['tgt_channel_cv']:>7.3f}{marker}")

    # 저장
    result = {"summary": summary, "per_layer": scores}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n[ASS] 결과 저장: {args.out}")

    # 시각화
    plot(scores, summary, args.plot, args.ref, args.target)


if __name__ == "__main__":
    main()
