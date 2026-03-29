"""
vLLM 기반 추론 벤치마크

FP16 vs GPTQ 4-bit 모델의 추론 성능 비교:
- throughput (tokens/sec)
- latency (TTFT, avg per-token)
- VRAM 사용량

Usage:
    python src/benchmark_serving.py --model quantized_models/SOLAR_10.7B_4bit_cond_C_v3 --label C_v3
    python src/benchmark_serving.py --model upstage/SOLAR-10.7B-Instruct-v1.0 --label FP16
"""
import os
import sys
import json
import time
import argparse
import subprocess

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("HOME", "/home/choihyun")
os.environ.setdefault("HF_HOME", os.path.join(WORKSPACE, ".cache/huggingface"))

import torch

PROMPTS = [
    "한국어 자연어처리의 최근 동향에 대해 설명해주세요.",
    "양자화란 무엇이며 딥러닝 모델에서 어떻게 활용되나요?",
    "서울의 역사적인 관광지를 다섯 곳 추천해주세요.",
    "인공지능이 의료 분야에 미치는 영향에 대해 논하세요.",
    "파이썬과 자바의 차이점을 설명해주세요.",
    "기후 변화가 한국 농업에 미치는 영향은 무엇인가요?",
    "머신러닝과 딥러닝의 차이를 초보자에게 설명해주세요.",
    "한국의 반도체 산업 현황에 대해 서술하세요.",
] * 4  # 32 prompts


def get_gpu_memory_mb(device=0):
    result = subprocess.run(
        ["nvidia-smi", f"--id={device}", "--query-gpu=memory.used",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    try:
        return int(result.stdout.strip())
    except Exception:
        return -1


def run_benchmark(model_path: str, label: str, quantized: bool, device: int = 0):
    print(f"\n{'='*60}")
    print(f"벤치마크: {label}")
    print(f"모델 경로: {model_path}")
    print(f"{'='*60}")

    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("vLLM이 설치되지 않았습니다. pip install vllm")
        sys.exit(1)

    gpu_before = get_gpu_memory_mb(device)

    # 모델 로드
    print("\n[1/3] 모델 로딩 중...")
    load_start = time.time()

    kwargs = dict(
        model=model_path,
        trust_remote_code=True,
        dtype="float16",
        gpu_memory_utilization=0.92,
        max_model_len=512,
        enforce_eager=True,
    )
    if quantized:
        kwargs["quantization"] = "gptq"

    llm = LLM(**kwargs)
    load_time = time.time() - load_start
    gpu_after_load = get_gpu_memory_mb(device)
    vram_used = gpu_after_load - gpu_before

    print(f"  로드 시간: {load_time:.1f}s")
    print(f"  VRAM 사용: {vram_used} MiB")

    # 추론 벤치마크
    print("\n[2/3] 추론 벤치마크 ({} 프롬프트)...".format(len(PROMPTS)))
    sampling_params = SamplingParams(temperature=0.0, max_tokens=256)

    # warmup
    llm.generate(PROMPTS[:2], sampling_params)

    infer_start = time.time()
    outputs = llm.generate(PROMPTS, sampling_params)
    infer_time = time.time() - infer_start

    total_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput = total_output_tokens / infer_time
    avg_latency_ms = (infer_time / len(PROMPTS)) * 1000

    print(f"  총 출력 토큰: {total_output_tokens}")
    print(f"  소요 시간: {infer_time:.2f}s")
    print(f"  throughput: {throughput:.1f} tokens/sec")
    print(f"  avg latency: {avg_latency_ms:.0f} ms/prompt")

    result = {
        "label": label,
        "model_path": model_path,
        "quantized": quantized,
        "vram_used_mib": vram_used,
        "load_time_s": round(load_time, 2),
        "num_prompts": len(PROMPTS),
        "total_output_tokens": total_output_tokens,
        "infer_time_s": round(infer_time, 2),
        "throughput_tok_per_sec": round(throughput, 2),
        "avg_latency_ms": round(avg_latency_ms, 1),
    }

    print(f"\n[3/3] 결과:")
    for k, v in result.items():
        print(f"  {k}: {v}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--quantized", action="store_true", default=False)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    result = run_benchmark(args.model, args.label, args.quantized, args.device)

    out_path = args.out or os.path.join(WORKSPACE, f"results/benchmark_{args.label}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
