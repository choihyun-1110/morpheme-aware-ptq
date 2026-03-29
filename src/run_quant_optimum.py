"""
optimum.gptq 기반 양자화 — auto_gptq 미지원 모델(EXAONE 등) 전용
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

for path in (
    os.environ["XDG_CACHE_HOME"],
    os.environ["HF_HOME"],
    os.environ["HUGGINGFACE_HUB_CACHE"],
    os.environ["TRANSFORMERS_CACHE"],
):
    os.makedirs(path, exist_ok=True)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from optimum.gptq import GPTQQuantizer


def load_texts(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = [item["text"] for item in data["sentences"]]
    print(f"[calibration] {len(texts)} 문장 로드 완료: {json_path}")
    return texts


def run_quantization(model_id: str, calibration_json: str, output_dir: str,
                     bits: int = 4, group_size: int = 128, desc_act: bool = True):
    print(f"\n{'='*60}")
    print(f"[optimum.gptq] 양자화 시작: {model_id}")
    print(f"Calibration: {calibration_json}")
    print(f"Target Dir : {output_dir}")
    print(f"{'='*60}\n")

    os.makedirs(output_dir, exist_ok=True)

    print("[1/4] Tokenizer 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    print("[2/4] Calibration 텍스트 로드 중...")
    texts = load_texts(calibration_json)

    print("[3/4] 원본 모델 로드 중 (FP16, CPU)...")
    # CPU 로드: optimum.gptq가 layer별로 GPU 이동 처리 → OOM 방지
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )

    print("[4/4] GPTQQuantizer 양자화 실행 중...")
    quantizer = GPTQQuantizer(
        bits=bits,
        dataset=texts,
        group_size=group_size,
        desc_act=desc_act,
        act_group_aware=False,  # gptqmodel 5.x: disable to allow desc_act=True
        model_seqlen=2048,
    )

    start_time = time.time()
    quantized_model = quantizer.quantize_model(model, tokenizer)
    elapsed = time.time() - start_time
    print(f"양자화 완료! (소요 시간: {elapsed:.1f}초)")

    quantized_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\n[성공] 저장 완료: {output_dir}\n")

    # qzeros 포맷 패치: optimum.gptq는 qzeros에 zero_point(=8)를 직접 저장하지만,
    # gptqmodel/auto_gptq는 (zero_point - 1 = 7)을 기대함.
    # 각 nibble에서 1을 빼서 0x88888888 → 0x77777777 변환.
    print("[qzeros 패치] 0x88...→0x77... 변환 중...")
    from safetensors import safe_open
    from safetensors.torch import save_file
    import glob
    shard_files = sorted(glob.glob(os.path.join(output_dir, "model*.safetensors")))
    for sf_path in shard_files:
        tensors = {}
        with safe_open(sf_path, framework="pt") as f:
            meta = dict(f.metadata()) if f.metadata() else {}
            for key in f.keys():
                t = f.get_tensor(key)
                if "qzeros" in key:
                    t = t - 0x11111111
                tensors[key] = t
        save_file(tensors, sf_path, metadata=meta)
        print(f"  [패치 완료] {os.path.basename(sf_path)}")

    # quantize_config.json 생성 (gptqmodel이 FORMAT.GPTQ로 인식하도록)
    import json as _json
    qc_path = os.path.join(output_dir, "quantize_config.json")
    if not os.path.exists(qc_path):
        qc = {
            "bits": bits, "group_size": group_size, "damp_percent": 0.1,
            "desc_act": desc_act, "static_groups": False, "sym": True,
            "true_sequential": True, "model_name_or_path": None,
            "model_file_base_name": None, "is_marlin_format": False,
            "quant_method": "gptq",
        }
        with open(qc_path, "w") as f:
            _json.dump(qc, f, indent=2)
        print(f"  [생성] quantize_config.json")
    print("[qzeros 패치] 완료!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--calib-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--no-desc-act", action="store_true")
    args = parser.parse_args()

    run_quantization(
        model_id=args.model,
        calibration_json=args.calib_path,
        output_dir=args.out_dir,
        bits=args.bits,
        group_size=args.group_size,
        desc_act=not args.no_desc_act,
    )
