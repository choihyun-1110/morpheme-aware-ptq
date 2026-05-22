"""
Korean Perplexity (PPL) 측정

FP16 / GPTQ-A / GPTQ-C_v3 세 조건의 PPL을 한국어 held-out 텍스트로 비교.

- 평가 텍스트: 나무위키 held-out 500문장 (seed=9999, calibration과 다른 시드)
- 측정 방식: token-level cross-entropy → PPL = exp(mean_loss)
- stride sliding-window으로 긴 문서도 공정하게 처리

사용법:
    # 기본 (SOLAR FP16 + A + C_v3, GPU0)
    python src/measure_ppl.py

    # 조건 직접 지정
    python src/measure_ppl.py \\
        --conditions fp16 A C_v3 \\
        --device cuda:0 \\
        --out results/ppl_solar.json

    # 평가 문장 파일 직접 지정 (calibration set과 동일 JSON 포맷)
    python src/measure_ppl.py --eval-json results/eval_korean_500.json
"""
import os
import sys
import json
import argparse
import math

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("HOME", "/home/choihyun")
os.environ.setdefault("HF_HOME", os.path.join(WORKSPACE, ".cache/huggingface"))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── 경로 설정 ────────────────────────────────────────────────────────────────
FP16_MODEL_ID   = "upstage/SOLAR-10.7B-Instruct-v1.0"
GPTQ_A_DIR      = os.path.join(WORKSPACE, "quantized_models/SOLAR_10.7B_4bit_cond_A")
GPTQ_C_V3_DIR   = os.path.join(WORKSPACE, "quantized_models/SOLAR_10.7B_4bit_cond_C_v3")

MAX_SEQ_LEN     = 1024   # sliding window 크기
STRIDE          = 512    # 50% overlap
EVAL_N          = 500    # held-out 문장 수
EVAL_SEED       = 9999   # calibration(seed=42)과 다른 시드


def build_eval_corpus(n: int = EVAL_N, seed: int = EVAL_SEED, save_path: str = None) -> list[str]:
    """나무위키 held-out 문장 구축 (calibration과 다른 시드)."""
    sys.path.insert(0, WORKSPACE)
    from src.preprocess import build_candidate_pool

    print(f"[PPL] 나무위키 held-out 코퍼스 구축 중 (seed={seed}, n={n})...")
    sentences = build_candidate_pool(n_sentences=n * 5, max_docs=2000, seed=seed)

    # calibration set B(seed=42)와 내용 중복 최소화를 위해 뒤쪽 슬라이스 사용
    held_out = sentences[-n:] if len(sentences) >= n else sentences
    print(f"[PPL] held-out {len(held_out)}문장 확보")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump({"sentences": [{"text": s} for s in held_out]}, f,
                      ensure_ascii=False, indent=2)
        print(f"[PPL] 저장: {save_path}")

    return held_out


def load_eval_corpus(json_path: str) -> list[str]:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return [item["text"] for item in data["sentences"]]


def compute_ppl(model, tokenizer, sentences: list[str], device: str,
                max_len: int = MAX_SEQ_LEN, stride: int = STRIDE) -> dict:
    """sliding-window PPL 계산."""
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in sentences:
            enc = tokenizer(text, return_tensors="pt")
            input_ids = enc["input_ids"][0]  # (seq_len,)
            seq_len = input_ids.size(0)

            if seq_len <= 1:
                continue

            prev_end = 0
            for begin in range(0, seq_len, stride):
                end = min(begin + max_len, seq_len)
                chunk = input_ids[begin:end].unsqueeze(0).to(device)

                # 이미 이전 window에서 처리된 토큰은 loss 집계에서 제외
                target_len = end - max(begin, prev_end)
                if target_len <= 0:
                    prev_end = end
                    continue

                with torch.no_grad():
                    out = model(chunk, labels=chunk)
                    # out.loss는 전체 chunk의 평균 NLL
                    # target_len 토큰 분만 집계
                    nll = out.loss.item() * (end - begin)
                    # 앞쪽 context 토큰 제외
                    context_len = end - begin - target_len
                    nll -= out.loss.item() * context_len  # 근사 제거

                total_nll += nll
                total_tokens += target_len
                prev_end = end

                if end >= seq_len:
                    break

    if total_tokens == 0:
        return {"ppl": float("inf"), "nll": float("inf"), "n_tokens": 0}

    mean_nll = total_nll / total_tokens
    ppl = math.exp(min(mean_nll, 100))  # overflow 방지
    return {"ppl": round(ppl, 4), "nll": round(mean_nll, 6), "n_tokens": total_tokens}


def load_model_fp16(model_id: str, device: str):
    print(f"  FP16 모델 로드: {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map={"": device},
        trust_remote_code=True,
    )
    return model, tok


def load_model_gptq(model_dir: str, device: str):
    print(f"  GPTQ 모델 로드: {model_dir}")
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map={"": device},
        trust_remote_code=True,
    )
    model.eval()
    return model, tok


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--conditions", nargs="+", default=["fp16", "A", "C_v3"],
                   help="측정할 조건 (fp16, A, C_v3)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--eval-json", default=None,
                   help="사전 생성된 held-out JSON 경로 (없으면 자동 생성)")
    p.add_argument("--out", default=None,
                   help="결과 JSON 경로 (기본: results/ppl_solar.json)")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device
    out_path = args.out or os.path.join(WORKSPACE, "results/ppl_solar.json")

    # ── 평가 코퍼스 ──────────────────────────────────────────────────────────
    eval_json_path = os.path.join(WORKSPACE, "results/eval_korean_held_out.json")
    if args.eval_json:
        sentences = load_eval_corpus(args.eval_json)
        print(f"[PPL] 외부 평가 파일 사용: {len(sentences)}문장")
    elif os.path.exists(eval_json_path):
        sentences = load_eval_corpus(eval_json_path)
        print(f"[PPL] 기존 held-out 파일 사용: {len(sentences)}문장")
    else:
        sentences = build_eval_corpus(n=EVAL_N, seed=EVAL_SEED, save_path=eval_json_path)

    # ── 조건별 PPL 측정 ──────────────────────────────────────────────────────
    cond_map = {
        "fp16":  ("fp16",  FP16_MODEL_ID),
        "A":     ("gptq",  GPTQ_A_DIR),
        "C_v3":  ("gptq",  GPTQ_C_V3_DIR),
    }

    results = {}
    for cond in args.conditions:
        if cond not in cond_map:
            print(f"[경고] 알 수 없는 조건: {cond}, 건너뜀")
            continue

        mode, path = cond_map[cond]
        print(f"\n{'='*50}")
        print(f"[PPL] 조건: {cond}")

        if mode == "fp16":
            model, tok = load_model_fp16(path, device)
        else:
            model, tok = load_model_gptq(path, device)

        gpu_mem = torch.cuda.memory_allocated(int(device.split(":")[-1])) / 1024**3
        print(f"  GPU 메모리: {gpu_mem:.1f} GB")

        res = compute_ppl(model, tok, sentences, device)
        results[cond] = res
        print(f"  PPL={res['ppl']:.2f}  NLL={res['nll']:.4f}  tokens={res['n_tokens']}")

        # 메모리 해제
        del model
        torch.cuda.empty_cache()

    # ── 결과 출력 및 저장 ────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("조건별 PPL 비교 (SOLAR-10.7B, 한국어 held-out)")
    print(f"{'조건':10} | {'PPL':>8} | {'NLL':>8} | {'tokens':>8}")
    print("-" * 42)
    for cond, res in results.items():
        print(f"{cond:10} | {res['ppl']:>8.2f} | {res['nll']:>8.4f} | {res['n_tokens']:>8}")

    if "fp16" in results and "C_v3" in results:
        ratio = results["C_v3"]["ppl"] / results["fp16"]["ppl"]
        print(f"\nPPL 비율 C_v3/FP16: {ratio:.4f}  ({(ratio-1)*100:+.2f}%)")
    if "fp16" in results and "A" in results:
        ratio = results["A"]["ppl"] / results["fp16"]["ppl"]
        print(f"PPL 비율 A/FP16:    {ratio:.4f}  ({(ratio-1)*100:+.2f}%)")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[PPL] 결과 저장: {out_path}")


if __name__ == "__main__":
    main()
