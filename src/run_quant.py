"""
양자화 파이프라인 (GPTQ)
- JSON 포맷의 Calibration Set(A/B/C) 로드
- AutoGPTQ를 이용한 4-bit 양자화 수행
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
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig


def load_calibration_data(json_path: str, tokenizer, max_seq_len: int = 2048):
    """
    JSON 결과 파일에서 텍스트를 읽어와 Tokenizer로 인코딩한 데이터셋 반환.
    AutoGPTQ의 quantize()는 입력으로 List[Dict[str, torch.Tensor]] 형태를 요구함.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Calibration JSON 파일을 찾을 수 없습니다: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    sentences = [item["text"] for item in data["sentences"]]
    print(f"[{data.get('condition', 'unknown')}] {len(sentences)} 문장 로드 완료.")
    
    # AutoGPTQ quantize()에서 내부적으로 필요한 디바이스로 옮기므로
    # calibration 텐서는 CPU에 둬서 device mismatch를 피한다.
    examples = []
    for text in sentences:
        encodings = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_len,
            padding=False,
            return_tensors="pt"
        )
        examples.append({
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"]
        })
        
    return examples


def run_quantization(model_id: str, 
                     calibration_json: str, 
                     output_dir: str,
                     bits: int = 4, 
                     group_size: int = 128, 
                     desc_act: bool = True):
    """
    AutoGPTQ를 이용한 양자화 수행.
    """
    print(f"\n{'='*60}")
    print(f"양자화 시작: {model_id}")
    print(f"Calibration: {calibration_json}")
    print(f"Target Dir : {output_dir}")
    print(f"{'='*60}\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Tokenizer Load
    print("[1/4] Tokenizer 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # 2. Calibration Data Load
    print(f"[2/4] Calibration 데이터 로드 중... ({calibration_json})")
    examples = load_calibration_data(calibration_json, tokenizer)
    
    # 3. Model Load & Quantization
    print("[3/4] 원본 모델 로드 및 양자화 준비 중...")
    quantize_config = BaseQuantizeConfig(
        bits=bits,
        group_size=group_size,
        desc_act=desc_act,
    )

    device = "cuda:0"
    print(f"양자화 대상 디바이스: {device}")
    
    # 양자화 중에는 모델이 여러 디바이스/CPU로 분산되면
    # rotary embedding 등에서 device mismatch가 발생할 수 있어
    # 단일 GPU에 명시적으로 올린다.
    model = AutoGPTQForCausalLM.from_pretrained(
        model_id,
        quantize_config,
        torch_dtype=torch.float16,
        device_map={"": device},
        trust_remote_code=True,
    )

    base_model = getattr(model.model, "model", None)
    rotary_emb = getattr(base_model, "rotary_emb", None) if base_model is not None else None
    if rotary_emb is not None:
        if hasattr(rotary_emb, "inv_freq"):
            rotary_emb.inv_freq = rotary_emb.inv_freq.to(device)
        if hasattr(rotary_emb, "original_inv_freq"):
            rotary_emb.original_inv_freq = rotary_emb.original_inv_freq.to(device)
        print(f"rotary_emb.inv_freq device: {rotary_emb.inv_freq.device}")
        if hasattr(rotary_emb, "original_inv_freq"):
            print(f"rotary_emb.original_inv_freq device: {rotary_emb.original_inv_freq.device}")
    
    print(f"양자화 실행 전 GPU 메모리 할당: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
    print(f"양자화 진행 중 (수십 분 소요 가능)...")
    
    start_time = time.time()
    
    # 실제 양자화 수행
    model.quantize(examples)
    
    quant_time = time.time() - start_time
    print(f"양자화 완료! (소요 시간: {quant_time:.1f}초)")
    print(f"양자화 실행 후 GPU 메모리 할당: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
    
    # 4. Save
    print(f"[4/4] 양자화 모델 저장 중... ({output_dir})")
    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"\n[성공] 양자화 모델 저장 완료: {output_dir}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoGPTQ 양자화 실행 스크립트")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace 모델 ID (원본)")
    parser.add_argument("--calib", type=str, required=True, help="어떤 조건을 입력할 것인가 (A, B, C)")
    parser.add_argument("--calib-path", type=str, required=True, help="입력 JSON 파일 경로")
    parser.add_argument("--out-dir", type=str, required=True, help="저장할 디렉토리 경로")
    parser.add_argument("--bits", type=int, default=4, help="양자화 비트 수")
    parser.add_argument("--group-size", type=int, default=128, help="GPTQ Group Size")
    parser.add_argument("--no-desc-act", action="store_true", help="desc_act (act_order) 끄기 (속도 이점, 성능 하락)")
    
    args = parser.parse_args()
    
    run_quantization(
        model_id=args.model,
        calibration_json=args.calib_path,
        output_dir=args.out_dir,
        bits=args.bits,
        group_size=args.group_size,
        desc_act=not args.no_desc_act
    )
