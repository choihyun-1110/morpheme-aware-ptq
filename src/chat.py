"""
대화형 chat 스크립트 — FP16 원본 or 양자화 모델 비교

사용법:
  # FP16 원본
  python3 chat.py --model upstage/SOLAR-10.7B-Instruct-v1.0

  # 양자화 (gptqmodel)
  python3 chat.py --model ../quantized_models/SOLAR_10.7B_4bit_cond_C_v3

  # 두 모델 동시 비교 (GPU 메모리 충분할 때)
  python3 chat.py --model upstage/SOLAR-10.7B-Instruct-v1.0 \
                  --model-b ../quantized_models/SOLAR_10.7B_4bit_cond_C_v3

  # vllm 백엔드 사용 (빠른 생성)
  python3 chat.py --model upstage/SOLAR-10.7B-Instruct-v1.0 --backend vllm

환경변수:
  CUDA_VISIBLE_DEVICES=0  (기본 GPU 0)

특수 명령 (채팅 중):
  /quit   종료
  /clear  대화 기록 초기화
  /info   모델 정보 출력
  /swap   모델 A↔B 전환 (비교 모드)
"""

import os, sys, argparse, textwrap

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE_ROOT = os.path.join(WORKSPACE_ROOT, ".cache")
HF_HOME_ROOT = os.path.join(CACHE_ROOT, "huggingface")

os.environ.setdefault("HOME", "/home/choihyun")
os.environ.setdefault("XDG_CACHE_HOME", CACHE_ROOT)
os.environ.setdefault("HF_HOME", HF_HOME_ROOT)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(HF_HOME_ROOT, "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(HF_HOME_ROOT, "transformers"))

import torch

SOLAR_SYSTEM = "당신은 도움이 되는 AI 어시스턴트입니다."


# ──────────────────────────────────────────────────────────────────────
# 백엔드: transformers (HF)
# ──────────────────────────────────────────────────────────────────────

class HFBackend:
    def __init__(self, model_path: str, device: str = "cuda:0"):
        from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

        print(f"[HF] 로드 중: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()
        self.device = device
        self.path = model_path
        mem = torch.cuda.memory_allocated(device) / 1e9
        print(f"[HF] 로드 완료 ({mem:.1f} GB 사용)")

    def generate(self, messages: list[dict], max_new_tokens: int = 512,
                 temperature: float = 0.7, stream: bool = True) -> str:
        from transformers import TextStreamer

        if hasattr(self.tokenizer, "apply_chat_template"):
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self.model.device)
        else:
            # fallback: 단순 연결
            prompt = "\n".join(
                f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
                for m in messages
            ) + "\nAssistant:"
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)

        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True) if stream else None

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            pad_token_id=self.tokenizer.eos_token_id,
            attention_mask=torch.ones_like(input_ids),
        )
        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False

        with torch.inference_mode():
            output = self.model.generate(input_ids, **gen_kwargs)

        if not stream:
            new_tokens = output[0][input_ids.shape[-1]:]
            return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return ""

    def unload(self):
        del self.model
        torch.cuda.empty_cache()


# ──────────────────────────────────────────────────────────────────────
# 백엔드: vllm
# ──────────────────────────────────────────────────────────────────────

class VLLMBackend:
    def __init__(self, model_path: str, gpu_id: int = 0,
                 quantization: str = None, max_model_len: int = 4096):
        from vllm import LLM, SamplingParams

        self._SamplingParams = SamplingParams

        print(f"[vllm] 로드 중: {model_path}")
        kwargs = dict(
            model=model_path,
            trust_remote_code=True,
            dtype="float16",
            gpu_memory_utilization=0.90,
            max_model_len=max_model_len,
            tensor_parallel_size=1,
        )
        if quantization:
            kwargs["quantization"] = quantization

        self.llm = LLM(**kwargs)
        self.path = model_path
        print(f"[vllm] 로드 완료")

    def generate(self, messages: list[dict], max_new_tokens: int = 512,
                 temperature: float = 0.7, stream: bool = True) -> str:
        # vllm은 apply_chat_template 직접 처리
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True)

        if hasattr(tok, "apply_chat_template"):
            prompt = tok.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        else:
            prompt = "\n".join(
                f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
                for m in messages
            ) + "\nAssistant:"

        params = self._SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        outputs = self.llm.generate([prompt], params)
        text = outputs[0].outputs[0].text
        if stream:
            print(text, end="", flush=True)
            print()
        return text

    def unload(self):
        del self.llm
        torch.cuda.empty_cache()


# ──────────────────────────────────────────────────────────────────────
# Chat 루프
# ──────────────────────────────────────────────────────────────────────

def short_name(path: str) -> str:
    return os.path.basename(path.rstrip("/")) or path


def chat_loop(backends: list, names: list, system_prompt: str,
              max_new_tokens: int, temperature: float):

    histories = [
        [{"role": "system", "content": system_prompt}]
        for _ in backends
    ]

    active = 0  # 비교 모드에서 현재 활성 모델 인덱스

    print("\n" + "="*60)
    if len(backends) == 1:
        print(f"모델: {names[0]}")
    else:
        print(f"[비교 모드]  A: {names[0]}  |  B: {names[1]}")
        print(f"현재 활성: A  (/swap으로 전환)")
    print("="*60)
    print("특수 명령: /quit  /clear  /info  /swap")
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료합니다.")
            break

        if not user_input:
            continue

        if user_input == "/quit":
            print("종료합니다.")
            break

        if user_input == "/clear":
            for i in range(len(histories)):
                histories[i] = [{"role": "system", "content": system_prompt}]
            print("[대화 기록 초기화]")
            continue

        if user_input == "/info":
            for i, (b, n) in enumerate(zip(backends, names)):
                marker = "← 활성" if i == active and len(backends) > 1 else ""
                print(f"  {n}  {marker}")
                if hasattr(b, "model"):
                    mem = sum(
                        p.numel() * p.element_size()
                        for p in b.model.parameters()
                    ) / 1e9
                    print(f"    파라미터 메모리: {mem:.1f} GB")
            continue

        if user_input == "/swap":
            if len(backends) < 2:
                print("[단일 모델 모드 — /swap 불가]")
            else:
                active = 1 - active
                print(f"[전환] 현재 활성: {'A' if active == 0 else 'B'} ({names[active]})")
            continue

        # 비교 모드: 현재 활성 모델에만 입력
        targets = [active] if len(backends) > 1 else [0]

        for i in targets:
            histories[i].append({"role": "user", "content": user_input})

            if len(backends) > 1:
                print(f"\n[{'A' if i == 0 else 'B'}: {names[i]}]")

            print(f"Assistant: ", end="", flush=True)
            reply = backends[i].generate(
                histories[i],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                stream=True,
            )
            if reply:
                print(reply)
            print()

            if reply:
                histories[i].append({"role": "assistant", "content": reply})
            # stream 모드에서는 streamer가 출력하므로 history에는 별도 저장 필요
            # (간단히 처리: stream=True 시 reply 빈 문자열 → 히스토리에 placeholder 저장)
            else:
                histories[i].append({"role": "assistant", "content": "[stream output]"})


# ──────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__
    )
    p.add_argument("--model", required=True,
                   help="모델 경로 또는 HuggingFace ID (모델 A)")
    p.add_argument("--model-b", default=None,
                   help="비교할 두 번째 모델 (선택)")
    p.add_argument("--backend", choices=["hf", "vllm"], default="hf",
                   help="추론 백엔드 (기본: hf)")
    p.add_argument("--gpu", type=int, default=0,
                   help="사용할 GPU 번호 (기본: 0)")
    p.add_argument("--gpu-b", type=int, default=1,
                   help="모델 B GPU 번호 (비교 모드, 기본: 1)")
    p.add_argument("--same-gpu", action="store_true",
                   help="A/B 모두 같은 GPU에 올리기 (--gpu 값 사용)")
    p.add_argument("--quant", default=None,
                   help="vllm quantization 옵션 (gptq, awq, ...)")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--system", default=SOLAR_SYSTEM,
                   help="시스템 프롬프트")
    return p.parse_args()


def load_backend(model_path, backend, gpu_id, quant=None):
    if backend == "vllm":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        return VLLMBackend(model_path, gpu_id=gpu_id, quantization=quant)
    else:
        device = f"cuda:{gpu_id}"
        return HFBackend(model_path, device=device)


def main():
    args = parse_args()

    backends, names = [], []

    print(f"\n{'='*60}")
    print(f"백엔드: {args.backend}  |  온도: {args.temperature}")
    print(f"{'='*60}")

    b_a = load_backend(args.model, args.backend, args.gpu, args.quant)
    backends.append(b_a)
    names.append(short_name(args.model))

    if args.model_b:
        gpu_b = args.gpu if args.same_gpu else args.gpu_b
        b_b = load_backend(args.model_b, args.backend, gpu_b, args.quant)
        backends.append(b_b)
        names.append(short_name(args.model_b))

    try:
        chat_loop(
            backends=backends,
            names=names,
            system_prompt=args.system,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    finally:
        for b in backends:
            b.unload()


if __name__ == "__main__":
    main()
