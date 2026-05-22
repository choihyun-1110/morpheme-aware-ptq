"""
FP16 vs GPTQ-A vs GPTQ-C_v3: 같은 입력에 대한 activation 분포 비교

동일한 한국어 텍스트를 세 모델에 통과시켜 레이어별 activation 분포를 비교.
"GPTQ-C_v3가 FP16 activation을 A보다 얼마나 더 잘 보존하는가"를 직접 측정.

측정:
  - 레이어별 channel_cv 차이: |GPTQ - FP16| (낮을수록 FP16에 가까움)
  - 레이어별 outlier_ratio 차이
  - 레이어별 mean_std 차이
  - 종합 distortion score = 레이어 평균 상대 오차

결과:
  - results/model_activation_comparison.json
  - results/model_activation_comparison.png  (논문용 figure)
"""
import os, sys, json, argparse, math
import numpy as np

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("HOME", "/home/choihyun")
os.environ.setdefault("HF_HOME", os.path.join(WORKSPACE, ".cache/huggingface"))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODELS = {
    "FP16":   ("fp16",  "upstage/SOLAR-10.7B-Instruct-v1.0"),
    "GPTQ-A":   ("gptq",  os.path.join(WORKSPACE, "quantized_models/SOLAR_10.7B_4bit_cond_A")),
    "GPTQ-C_v3":("gptq",  os.path.join(WORKSPACE, "quantized_models/SOLAR_10.7B_4bit_cond_C_v3")),
}
MAX_SEQ_LEN = 512
N_SENTENCES = 100   # 빠른 측정, 논문용 figure 충분


def load_sentences(json_path, n):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    sents = [item["text"] for item in data["sentences"]]
    return sents[:n]


def channel_cv(t: torch.Tensor) -> float:
    if t.dim() == 3:
        t = t.reshape(-1, t.shape[-1])
    ch_std = t.float().std(dim=0)
    return float(ch_std.std() / (ch_std.mean() + 1e-8))


def mean_std(t: torch.Tensor) -> float:
    return float(t.float().std().item())


def outlier_ratio(t: torch.Tensor, thresh=6.0) -> float:
    return float((t.float().abs() > thresh).float().mean().item())


def collect_activations(model, tokenizer, sentences, device):
    """레이어별 activation 통계 수집."""
    layer_stats = {}
    handles = []

    def make_hook(idx):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            layer_stats.setdefault(idx, []).append(h.detach().cpu().squeeze(0))
        return hook

    for idx, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(make_hook(idx)))

    model.eval()
    with torch.no_grad():
        for text in sentences:
            enc = tokenizer(text, return_tensors="pt",
                            truncation=True, max_length=MAX_SEQ_LEN).to(device)
            model(**enc)

    for h in handles:
        h.remove()

    results = {}
    for idx, tensors in sorted(layer_stats.items()):
        cat = torch.cat(tensors, dim=0)
        if cat.dim() == 3:
            cat = cat.reshape(-1, cat.shape[-1])
        results[idx] = {
            "channel_cv":    round(channel_cv(cat), 6),
            "mean_std":      round(mean_std(cat), 6),
            "outlier_ratio": round(outlier_ratio(cat), 6),
        }
    return results


def load_model(name, mode, path, device):
    print(f"\n[모델 로드] {name}")
    if mode == "fp16":
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.float16,
            device_map={"": device}, trust_remote_code=True)
    else:
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            path, device_map={"": device}, trust_remote_code=True)
    model.eval()
    mem = torch.cuda.memory_allocated(int(device.split(":")[-1])) / 1024**3
    print(f"  GPU: {mem:.1f} GB")
    return model, tok


def compute_distortion(fp16_stats, gptq_stats):
    """레이어별 FP16 대비 상대 오차."""
    metrics = ["channel_cv", "mean_std", "outlier_ratio"]
    layer_dist = {}
    for lid in sorted(fp16_stats.keys()):
        f = fp16_stats[lid]
        g = gptq_stats.get(lid, {})
        rel = []
        for m in metrics:
            base = abs(f.get(m, 0)) + 1e-8
            diff = abs(g.get(m, 0) - f.get(m, 0))
            rel.append(diff / base)
        layer_dist[lid] = {
            "distortion": round(float(np.mean(rel)), 6),
            "rel_channel_cv":    round(rel[0], 6),
            "rel_mean_std":      round(rel[1], 6),
            "rel_outlier_ratio": round(rel[2], 6),
            "fp16_channel_cv":   f.get("channel_cv"),
            "gptq_channel_cv":   g.get("channel_cv"),
        }
    mean_dist = float(np.mean([v["distortion"] for v in layer_dist.values()]))
    return layer_dist, round(mean_dist, 6)


def plot(fp16_stats, gptq_a_stats, gptq_cv3_stats, dist_a, dist_cv3, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("[plot] matplotlib 없음")
        return

    layers = sorted(fp16_stats.keys())
    fp16_cv  = [fp16_stats[l]["channel_cv"]    for l in layers]
    a_cv     = [gptq_a_stats[l]["channel_cv"]  for l in layers]
    cv3_cv   = [gptq_cv3_stats[l]["channel_cv"] for l in layers]
    fp16_out = [fp16_stats[l]["outlier_ratio"]    for l in layers]
    a_out    = [gptq_a_stats[l]["outlier_ratio"]  for l in layers]
    cv3_out  = [gptq_cv3_stats[l]["outlier_ratio"] for l in layers]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # ── Panel 1: channel_cv ─────────────────────────────────────────────
    ax = axes[0]
    ax.plot(layers, fp16_cv,  color="#2c3e50", lw=2, label="FP16 (reference)", zorder=3)
    ax.plot(layers, a_cv,     color="#e74c3c", lw=1.5, alpha=0.85,
            label=f"GPTQ-A  (mean distortion={dist_a['channel_cv']:.3f})")
    ax.plot(layers, cv3_cv,   color="#27ae60", lw=1.5, alpha=0.85,
            label=f"GPTQ-C_v3  (mean distortion={dist_cv3['channel_cv']:.3f})")
    ax.fill_between(layers, fp16_cv, a_cv,   alpha=0.12, color="#e74c3c")
    ax.fill_between(layers, fp16_cv, cv3_cv, alpha=0.12, color="#27ae60")
    ax.set_ylabel("Channel CV\n(activation diversity)", fontsize=11)
    ax.set_title("GPTQ Activation Preservation: same Korean input → FP16 / GPTQ-A / GPTQ-C_v3\n"
                 "(SOLAR-10.7B-Instruct, 100 Korean sentences)", fontsize=12)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Panel 2: outlier_ratio ──────────────────────────────────────────
    ax = axes[1]
    ax.plot(layers, fp16_out,  color="#2c3e50", lw=2, label="FP16", zorder=3)
    ax.plot(layers, a_out,     color="#e74c3c", lw=1.5, alpha=0.85, label="GPTQ-A")
    ax.plot(layers, cv3_out,   color="#27ae60", lw=1.5, alpha=0.85, label="GPTQ-C_v3")
    ax.fill_between(layers, fp16_out, a_out,   alpha=0.12, color="#e74c3c")
    ax.fill_between(layers, fp16_out, cv3_out, alpha=0.12, color="#27ae60")
    ax.set_xlabel("Layer index", fontsize=11)
    ax.set_ylabel("Outlier ratio\n(|activation| > 6.0)", fontsize=11)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot] 저장: {out_path}")
    plt.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--eval-json",
                   default=os.path.join(WORKSPACE, "results/eval_korean_held_out.json"))
    p.add_argument("--n-sentences", type=int, default=N_SENTENCES)
    p.add_argument("--out-json",
                   default=os.path.join(WORKSPACE, "results/model_activation_comparison.json"))
    p.add_argument("--out-plot",
                   default=os.path.join(WORKSPACE, "results/model_activation_comparison.png"))
    return p.parse_args()


def main():
    args = parse_args()
    sentences = load_sentences(args.eval_json, args.n_sentences)
    print(f"[비교] {len(sentences)}문장으로 FP16/GPTQ-A/GPTQ-C_v3 activation 비교")

    all_stats = {}
    for name, (mode, path) in MODELS.items():
        model, tok = load_model(name, mode, path, args.device)
        stats = collect_activations(model, tok, sentences, args.device)
        all_stats[name] = stats
        print(f"  [{name}] {len(stats)}개 레이어 수집 완료")
        del model
        torch.cuda.empty_cache()

    # ── distortion 계산 ─────────────────────────────────────────────────
    fp16 = all_stats["FP16"]
    dist_a_layers,   mean_a   = compute_distortion(fp16, all_stats["GPTQ-A"])
    dist_cv3_layers, mean_cv3 = compute_distortion(fp16, all_stats["GPTQ-C_v3"])

    # metric별 mean distortion
    def mean_metric(dist_layers, key):
        return round(float(np.mean([v[key] for v in dist_layers.values()])), 6)

    dist_a_summary   = {m: mean_metric(dist_a_layers,   f"rel_{m}")
                        for m in ["channel_cv", "mean_std", "outlier_ratio"]}
    dist_cv3_summary = {m: mean_metric(dist_cv3_layers, f"rel_{m}")
                        for m in ["channel_cv", "mean_std", "outlier_ratio"]}

    # ── 출력 ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FP16 대비 activation 왜곡 (낮을수록 FP16에 가까움)")
    print(f"{'='*60}")
    print(f"{'조건':12} | {'종합 distortion':>16} | {'channel_cv':>12} | {'mean_std':>10} | {'outlier':>8}")
    print("-" * 65)
    print(f"{'GPTQ-A':12} | {mean_a:>16.4f} | {dist_a_summary['channel_cv']:>12.4f} | "
          f"{dist_a_summary['mean_std']:>10.4f} | {dist_a_summary['outlier_ratio']:>8.4f}")
    print(f"{'GPTQ-C_v3':12} | {mean_cv3:>16.4f} | {dist_cv3_summary['channel_cv']:>12.4f} | "
          f"{dist_cv3_summary['mean_std']:>10.4f} | {dist_cv3_summary['outlier_ratio']:>8.4f}")
    improvement = (mean_a - mean_cv3) / mean_a * 100
    print(f"\nC_v3 왜곡 감소: {improvement:.1f}%  ({mean_a:.4f} → {mean_cv3:.4f})")

    # ── 저장 ──────────────────────────────────────────────────────────────
    result = {
        "n_sentences": len(sentences),
        "summary": {
            "GPTQ-A":    {"mean_distortion": mean_a,   **dist_a_summary},
            "GPTQ-C_v3": {"mean_distortion": mean_cv3, **dist_cv3_summary},
            "improvement_pct": round(improvement, 2),
        },
        "per_layer": {
            "GPTQ-A":    {str(k): v for k, v in dist_a_layers.items()},
            "GPTQ-C_v3": {str(k): v for k, v in dist_cv3_layers.items()},
        },
        "raw_stats": {k: {str(l): v for l, v in s.items()}
                      for k, s in all_stats.items()},
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[완료] 결과 저장: {args.out_json}")

    plot(fp16, all_stats["GPTQ-A"], all_stats["GPTQ-C_v3"],
         dist_a_summary, dist_cv3_summary, args.out_plot)


if __name__ == "__main__":
    main()
