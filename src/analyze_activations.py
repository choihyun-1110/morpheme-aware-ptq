"""
Calibration Set별 Activation 분포 분석

FP16 모델에 calibration set을 통과시켜 레이어별 activation 통계를 비교한다.

측정 지표:
  - mean/std: 활성화 분포의 중심과 퍼짐
  - outlier_ratio: |x| > 6.0 비율 (GPTQ 품질과 직결)
  - entropy: 활성화 값의 분포 엔트로피 (다양성)
  - channel_cv: 레이어 출력 채널별 분산의 coefficient of variation

사용법:
  # SOLAR 기본 (A/B/C/C_v3/C_v4)
  python src/analyze_activations.py

  # 모델+조건 지정
  python src/analyze_activations.py --model Qwen/Qwen2-7B-Instruct --device cuda:1 \\
    --conditions A B C_v3 --out results/activation_qwen2.json

결과: results/activation_analysis.json (기본)
"""
import os
import sys
import json
import math
import argparse

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("HOME", "/home/choihyun")
os.environ.setdefault("HF_HOME", os.path.join(WORKSPACE, ".cache/huggingface"))

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

OUTLIER_THRESH = 6.0
MAX_SEQ_LEN = 512
DEVICE = "cuda:0"

# 모델별 기본 calibration 파일 매핑
DEFAULT_CALIB_FILES = {
    "upstage/SOLAR-10.7B-Instruct-v1.0": {
        "A":    "results/calibration_set_A.json",
        "B":    "results/calibration_set_B.json",
        "C":    "results/calibration_set_C_upstage_SOLAR-10.7B-Instruct-v1.0.json",
        "C_v3": "results/calibration_set_C_v3_upstage_SOLAR-10.7B-Instruct-v1.0.json",
        "C_v4": "results/calibration_set_C_v4_upstage_SOLAR-10.7B-Instruct-v1.0.json",
    },
    "Qwen/Qwen2-7B-Instruct": {
        "A":    "results/calibration_set_A.json",
        "C_v3": "results/calibration_set_C_v3_Qwen_Qwen2-7B-Instruct.json",
        "C_zh": "results/calibration_set_C_zh_Qwen_Qwen2-7B-Instruct.json",
    },
    "yanolja/EEVE-Korean-Instruct-10.8B-v1.0": {
        "A":        "results/calibration_set_A.json",
        "B":        "results/calibration_set_B.json",
        "C_v3":     "results/calibration_set_C_v3_upstage_SOLAR-10.7B-Instruct-v1.0.json",
        "C_v3_eeve":"results/calibration_set_C_v3_eeve_yanolja_EEVE-Korean-Instruct-10.8B-v1.0.json",
    },
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


def parse_args():
    p = argparse.ArgumentParser(description="Calibration set별 activation 분포 분석")
    p.add_argument("--model", default="upstage/SOLAR-10.7B-Instruct-v1.0",
                   help="HuggingFace 모델 ID")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--conditions", nargs="+", default=None,
                   help="분석할 조건 목록 (기본: 모델에 따라 자동)")
    p.add_argument("--calib-map", nargs="+", default=None,
                   help="조건=파일경로 형식으로 직접 지정. 예: A=results/calib_A.json")
    p.add_argument("--out", default=None,
                   help="출력 JSON 경로 (기본: results/activation_analysis.json)")
    p.add_argument("--merge", action="store_true",
                   help="기존 출력 파일에 새 조건을 병합 (덮어쓰지 않음)")
    return p.parse_args()


def main():
    global DEVICE
    args = parse_args()
    model_id = args.model
    device = args.device
    DEVICE = device

    # calibration 파일 맵 결정
    if args.calib_map:
        calib_files = {}
        for item in args.calib_map:
            cond, path = item.split("=", 1)
            calib_files[cond] = path
    elif model_id in DEFAULT_CALIB_FILES:
        calib_files = DEFAULT_CALIB_FILES[model_id]
    else:
        print(f"[경고] 모델 '{model_id}'의 기본 calibration 매핑 없음. --calib-map 으로 지정하세요.")
        sys.exit(1)

    # 조건 필터
    if args.conditions:
        calib_files = {k: v for k, v in calib_files.items() if k in args.conditions}

    out_path = args.out or os.path.join(WORKSPACE, "results/activation_analysis.json")

    print("=" * 60)
    print(f"Activation 분포 분석: {model_id}")
    print(f"조건: {list(calib_files.keys())}")
    print("=" * 60)

    print(f"\n[1/2] FP16 모델 로드 중: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map={"": device},
        trust_remote_code=True,
    )
    model.eval()
    dev_idx = int(device.split(":")[-1]) if ":" in device else 0
    print(f"  GPU 메모리: {torch.cuda.memory_allocated(dev_idx)/1024**3:.1f} GB")

    # 기존 결과 로드 (--merge 모드)
    all_results = {}
    if args.merge and os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            all_results = json.load(f)
        print(f"  기존 결과 로드: {list(all_results.keys())}")

    print("\n[2/2] Calibration별 activation 분석")
    for cond, path in calib_files.items():
        if args.merge and cond in all_results:
            print(f"  [{cond}] 이미 존재 — 건너뜀 (--merge)")
            continue
        full_path = os.path.join(WORKSPACE, path) if not os.path.isabs(path) else path
        if not os.path.exists(full_path):
            print(f"  [{cond}] 파일 없음, 건너뜀: {full_path}")
            continue
        sentences = load_sentences(full_path)
        all_results[cond] = analyze_condition(model, tokenizer, sentences, cond)

    # 저장
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n[완료] 결과 저장: {out_path}")

    # 비교 요약 출력
    print("\n" + "=" * 60)
    print("조건별 요약 비교")
    print("=" * 60)
    header = f"{'조건':10} | {'outlier_ratio':>14} | {'entropy':>10} | {'channel_cv':>12} | {'mean_std':>10}"
    print(header)
    print("-" * 62)
    for cond, res in all_results.items():
        s = res["summary"]
        print(f"{cond:10} | {s['mean_outlier_ratio']:>14.4f} | "
              f"{s['mean_entropy']:>10.4f} | "
              f"{s['mean_channel_cv']:>12.4f} | "
              f"{s['mean_std']:>10.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
