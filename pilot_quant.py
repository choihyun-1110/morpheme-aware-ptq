import os
import time
import torch
import transformers.modeling_utils
if not hasattr(transformers.modeling_utils, "no_init_weights"):
    class no_init_weights:
        def __init__(self, _=False): pass
        def __enter__(self): pass
        def __exit__(self, *args): pass
    transformers.modeling_utils.no_init_weights = no_init_weights

from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

os.environ["HF_HOME"] = "/home/choihyun/workspace/.cache/huggingface"
model_id = "HuggingFaceTB/SmolLM-135M" 
quant_path = "/home/choihyun/workspace/quantized_models/SmolLM-135M-GPTQ-pilot"
print(f"[{time.strftime('%H:%M:%S')}] 파일럿 양자화 시작: {model_id}")

tokenizer = AutoTokenizer.from_pretrained(model_id)
examples = [tokenizer(t) for t in ", "테스트 둘", "테스트 셋", "테스트 넷", "테스트 다섯"]]["테스트 하

quantize_config = BaseQuantizeConfig(bits=4, group_size=128, desc_act=False)
print("모델 로드 중...")
model = AutoGPTQForCausalLM.from_pretrained(model_id, quantize_config, torch_dtype=torch.float16, device_map="auto")
print("양자화 시작...")
model.quantize(examples)
print("저장 중...")
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
print("완료!")
