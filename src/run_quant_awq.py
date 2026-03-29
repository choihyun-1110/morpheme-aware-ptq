"""
AutoAWQ 기반 양자화 — calibration 언어 효과 비교용
AWQ: activation 크기 기반 salient weight 보호 → GPTQ와 다른 메커니즘
"""
import os
import sys
import json
import time
import argparse

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE_ROOT = os.path.join(WORKSPACE_ROOT, ".cache")
HF_HOME_ROOT = os.path.join(CACHE_ROOT, "huggingface")

os.environ.setdefault("HOME", "/home/choihyun")
os.environ.setdefault("XDG_CACHE_HOME", CACHE_ROOT)
os.environ.setdefault("HF_HOME", HF_HOME_ROOT)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(HF_HOME_ROOT, "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(HF_HOME_ROOT, "transformers"))


def load_texts(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = [item["text"] for item in data["sentences"]]
    print(f"[calibration] {len(texts)} 문장 로드: {json_path}")
    return texts


def run_awq(model_id: str, calibration_json: str, output_dir: str,
            bits: int = 4, group_size: int = 128, zero_point: bool = True):
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    print(f"\n{'='*60}")
    print(f"[AutoAWQ] 양자화 시작: {model_id}")
    print(f"Calibration: {calibration_json}")
    print(f"Bits: {bits}, group_size: {group_size}")
    print(f"{'='*60}\n")

    os.makedirs(output_dir, exist_ok=True)

    print("[1/3] Tokenizer 및 모델 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoAWQForCausalLM.from_pretrained(
        model_id, trust_remote_code=True,
        safetensors=True
    )

    print("[2/3] Calibration 데이터 로드 및 AWQ 실행...")
    texts = load_texts(calibration_json)

    quant_config = {
        "zero_point": zero_point,
        "q_group_size": group_size,
        "w_bit": bits,
        "version": "GEMM",
    }

    start_time = time.time()
    model.quantize(tokenizer, quant_config=quant_config, calib_data=texts)
    elapsed = time.time() - start_time
    print(f"AWQ 완료! (소요: {elapsed:.1f}초)")

    print(f"[3/3] 저장 중: {output_dir}")
    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\n[성공] AWQ 모델 저장 완료: {output_dir}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--calib-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    args = parser.parse_args()

    run_awq(
        model_id=args.model,
        calibration_json=args.calib_path,
        output_dir=args.out_dir,
        bits=args.bits,
        group_size=args.group_size,
    )
