"""
Calibration Set별 Activation 분포 분석

FP16 모델에 B/C/C_v3/C_v4 calibration set을 통과시켜
레이어별 activation 통계를 비교한다.

측정 지표:
  - mean/std: 활성화 분포의 중심과 퍼짐
  - outlier_ratio: |x| > 6.0 비율 (GPTQ 품질과 직결)
  - entropy: 활성화 값의 분포 엔트로피 (다양성)
  - coverage: 레이어 출력 채널별 분산의 균등도 (coefficient of variation)

결과: results/activation_analysis.json
"""
import os
import sys
import json
import math

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("HOME", "/home/choihyun")
os.environ.setdefault("HF_HOME", os.path.join(WORKSPACE, ".cache/huggingface"))

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "upstage/SOLAR-10.7B-Instruct-v1.0"
DEVICE = "cuda:0"
OUTLIER_THRESH = 6.0
MAX_SEQ_LEN = 512

CALIB_FILES = {
    "B":    "results/calibration_set_B.json",
    "C":    "results/calibration_set_C_upstage_SOLAR-10.7B-Instruct-v1.0.json",
    "C_v3": "results/calibration_set_C_v3_upstage_SOLAR-10.7B-Instruct-v1.0.json",
    "C_v4": "results/calibration_set_C_v4_upstage_SOLAR-10.7B-Instruct-v1.0.json",
}


def load_sentences(json_path):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return [item["text"] for item in data["sentences"]]


def entropy_from_tensor(t: torch.Tensor) -> float:
    """activation 값들의 히스토그램 엔트로피."""
    vals = t.float().cpu().numpy().flatten()
    counts, _ = np.histogram(vals, bins=100, range=(-15, 15))
    counts = counts + 1e-10
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log(probs)))


def channel_cv(t: torch.Tensor) -> float:
    """채널(hidden dim)별 분산의 coefficient of variation — 고르게 활성화되는지."""
    # t: (batch, seq, hidden) or (batch, hidden)
    if t.dim() == 3:
        t = t.reshape(-1, t.shape[-1])
    channel_std = t.float().std(dim=0)   # (hidden,)
    mean_std = channel_std.mean().item()
    std_std = channel_std.std().item()
    return std_std / (mean_std + 1e-8)


def analyze_condition(model, tokenizer, sentences, condition_name):
    print(f"\n  [{condition_name}] {len(sentences)}문장 분석 중...")

    # 레이어별 activation 누적용
    layer_stats = {}   # layer_idx -> list of tensors (cpu)
    handles = []

    def make_hook(layer_idx):
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # squeeze batch dim → (seq_len, hidden) 으로 저장해야 concat 가능
            layer_stats.setdefault(layer_idx, []).append(hidden.detach().cpu().squeeze(0))
        return hook

    # 모든 decoder layer의 self_attn output hook
    for idx, layer in enumerate(model.model.layers):
        h = layer.register_forward_hook(make_hook(idx))
        handles.append(h)

    model.eval()
    with torch.no_grad():
        for i, text in enumerate(sentences):
            enc = tokenizer(
                text, return_tensors="pt",
                truncation=True, max_length=MAX_SEQ_LEN
            ).to(DEVICE)
            model(**enc)
            if (i + 1) % 32 == 0:
                print(f"    {i+1}/{len(sentences)} done")

    for h in handles:
        h.remove()

    # 레이어별 통계 계산
    results = {}
    for layer_idx, tensors in sorted(layer_stats.items()):
        cat = torch.cat(tensors, dim=0)   # (total_tokens, hidden)
        if cat.dim() == 3:
            cat = cat.reshape(-1, cat.shape[-1])

        flat = cat.float()
        abs_flat = flat.abs()

        mean_val   = flat.mean().item()
        std_val    = flat.std().item()
        outlier_r  = (abs_flat > OUTLIER_THRESH).float().mean().item()
        ent        = entropy_from_tensor(flat)
        cv         = channel_cv(flat)

        results[layer_idx] = {
            "mean":          round(mean_val, 6),
            "std":           round(std_val, 6),
            "outlier_ratio": round(outlier_r, 6),
            "entropy":       round(ent, 6),
            "channel_cv":    round(cv, 6),
        }

    # 전체 평균 (레이어 평균)
    avg = lambda key: float(np.mean([v[key] for v in results.values()]))
    summary = {
        "mean_outlier_ratio": round(avg("outlier_ratio"), 6),
        "mean_entropy":       round(avg("entropy"), 6),
        "mean_channel_cv":    round(avg("channel_cv"), 6),
        "mean_std":           round(avg("std"), 6),
    }

    print(f"    outlier_ratio={summary['mean_outlier_ratio']:.4f}  "
          f"entropy={summary['mean_entropy']:.4f}  "
          f"channel_cv={summary['mean_channel_cv']:.4f}")

    return {"summary": summary, "per_layer": results}


def main():
    print("=" * 60)
    print("Activation 분포 분석 시작")
    print("=" * 60)

    print(f"\n[1/2] FP16 모델 로드 중: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map={"": DEVICE},
        trust_remote_code=True,
    )
    model.eval()
    print(f"  GPU 메모리: {torch.cuda.memory_allocated(0)/1024**3:.1f} GB")

    print("\n[2/2] Calibration별 activation 분석")
    all_results = {}
    for cond, path in CALIB_FILES.items():
        full_path = os.path.join(WORKSPACE, path)
        if not os.path.exists(full_path):
            print(f"  [{cond}] 파일 없음, 건너뜀: {full_path}")
            continue
        sentences = load_sentences(full_path)
        all_results[cond] = analyze_condition(model, tokenizer, sentences, cond)

    # 저장
    out_path = os.path.join(WORKSPACE, "results/activation_analysis.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n[완료] 결과 저장: {out_path}")

    # 비교 요약 출력
    print("\n" + "=" * 60)
    print("조건별 요약 비교")
    print("=" * 60)
    header = f"{'조건':8} | {'outlier_ratio':>14} | {'entropy':>10} | {'channel_cv':>12} | {'mean_std':>10}"
    print(header)
    print("-" * 60)
    for cond, res in all_results.items():
        s = res["summary"]
        print(f"{cond:8} | {s['mean_outlier_ratio']:>14.4f} | "
              f"{s['mean_entropy']:>10.4f} | "
              f"{s['mean_channel_cv']:>12.4f} | "
              f"{s['mean_std']:>10.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
