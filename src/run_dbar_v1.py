"""
DBAR-v1: Dual-Bank Adaptive Rounding (Merged Hessian)

핵심 아이디어:
    표준 GPTQ는 단일 calibration set으로 Hessian H를 추정한다.
    DBAR-v1은 두 개의 bank (Reasoning: C_v3, Domain: B)에서
    Hessian을 각각 추정한 후 가중 평균으로 병합한다:

        H_dual = λ · H_r + (1-λ) · H_k

    이렇게 하면 KoBEST(추론) 과 kmmlu(지식 recall) 양쪽을
    동시에 고려한 rounding이 가능하다.

차이점 (vs hybrid calibration):
    - hybrid: 128 samples를 64+64로 나눔 → 각 bank에서 H 추정 sample 수 절반
    - DBAR-v1: 각 bank에서 128 samples 전부 사용 → H 추정 품질 2배
    - λ와 sample 수가 분리됨: λ=0.3도 각 bank는 128 samples

사용법:
    python3 run_dbar_v1.py \
        --model upstage/SOLAR-10.7B-Instruct-v1.0 \
        --bank-r results/calibration_set_C_v3_upstage_SOLAR-10.7B-Instruct-v1.0.json \
        --bank-k results/calibration_set_B.json \
        --lambda-weight 0.5 \
        --out-dir quantized_models/solar_dbar_v1_lambda05

실행 (docker):
    docker exec llm-dev bash -c "CUDA_VISIBLE_DEVICES=0 \\
        /opt/conda/envs/llm-quant/bin/python3.11 \\
        /home/choihyun/workspace/src/run_dbar_v1.py ..."
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
from torch import nn
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from optimum.gptq import GPTQQuantizer
from optimum.gptq.utils import (
    get_block_name_with_pattern,
    get_device,
    get_layers,
    get_preceding_modules,
    get_seqlen,
    nested_move_to,
)
from optimum.utils.modeling_utils import recurse_getattr
from optimum.gptq.data import prepare_dataset
from transformers.utils.quantization_config import QuantizationMethod

from gptqmodel.quantization import GPTQ


def load_texts(json_path: str) -> list[str]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [item["text"] for item in data["sentences"]]


def _has_device_more_than_cpu():
    return torch.cuda.is_available() or (hasattr(torch, "xpu") and torch.xpu.is_available())


class DualBankGPTQQuantizer(GPTQQuantizer):
    """
    DBAR-v1: per-block dual-bank Hessian merging.

    각 transformer block에서:
      1. Reasoning bank (bank_r) 128 samples로 H_r 추정
      2. Domain bank    (bank_k) 128 samples로 H_k 추정
      3. H_dual = λ·H_r + (1-λ)·H_k
      4. H_dual로 fasterquant() 호출
    """

    def __init__(
        self,
        bank_r_texts: list[str],
        bank_k_texts: list[str],
        lambda_weight: float = 0.5,
        **kwargs,
    ):
        # GPTQQuantizer에 reasoning bank를 dataset으로 전달 (tokenization 재사용)
        super().__init__(dataset=bank_r_texts, **kwargs)
        self._bank_r_texts = bank_r_texts
        self._bank_k_texts = bank_k_texts
        self.lambda_weight = lambda_weight
        print(
            f"[DBAR-v1] λ={lambda_weight:.2f}  "
            f"bank_r={len(bank_r_texts)} samples, "
            f"bank_k={len(bank_k_texts)} samples"
        )

    # ------------------------------------------------------------------
    # Override quantize_model to implement dual-bank Hessian merging
    # ------------------------------------------------------------------
    def quantize_model(self, model: nn.Module, tokenizer=None):
        """
        GPTQQuantizer.quantize_model()에서 Hessian 축적 부분만 교체.
        나머지 (tokenization, input capture, pack_model 등)는 동일.
        """
        from gptqmodel import QuantizeConfig
        from gptqmodel.quantization import FORMAT, METHOD
        from gptqmodel.utils.importer import hf_select_quant_linear_v2
        from gptqmodel.utils.model import (
            hf_convert_gptq_v1_to_v2_format,
            hf_gptqmodel_post_init as gptq_post_init,
        )

        model.eval()
        if self.format != "gptq_v2":
            self.format = "gptq_v2"

        has_config = False
        has_device_map = False
        if hasattr(model, "config"):
            has_config = True
            use_cache = model.config.use_cache
            model.config.use_cache = False

        if hasattr(model, "hf_device_map"):
            has_device_map = True
            devices = list(model.hf_device_map.values())
            if "disk" in devices:
                raise ValueError("disk offload is not supported with DBAR-v1")

        if hasattr(model, "dtype"):
            self.use_cuda_fp16 = model.dtype == torch.float16

        if self.model_seqlen is None:
            self.model_seqlen = min(4028, get_seqlen(model))

        device = get_device(model)

        # ── Step 1: Tokenize both banks ────────────────────────────────
        def tokenize(texts):
            if isinstance(tokenizer, str):
                tok = AutoTokenizer.from_pretrained(tokenizer)
            else:
                tok = tokenizer
            raw = [tok(t, return_tensors="pt") for t in texts]
            return prepare_dataset(raw, pad_token_id=self.pad_token_id, batch_size=self.batch_size)

        print("[DBAR-v1] Tokenizing reasoning bank (bank_r)...")
        dataset_r = tokenize(self._bank_r_texts)
        print("[DBAR-v1] Tokenizing domain bank (bank_k)...")
        dataset_k = tokenize(self._bank_k_texts)

        # ── Step 2: Find blocks, capture block-0 inputs for both banks ─
        if self.block_name_to_quantize is None:
            self.block_name_to_quantize = get_block_name_with_pattern(model)
        if self.module_name_preceding_first_block is None:
            self.module_name_preceding_first_block = get_preceding_modules(
                model, self.block_name_to_quantize
            )

        blocks = recurse_getattr(model, self.block_name_to_quantize)
        cur_layer_device = get_device(blocks[0])

        if not has_device_map:
            to_device = cur_layer_device
            for module_name in self.module_name_preceding_first_block:
                module = recurse_getattr(model, module_name)
                if module is None:
                    raise ValueError(f"Module {module_name} not found")
                module = module.to(to_device)
            blocks[0] = blocks[0].to(to_device)

        def capture_inputs(dataset):
            """Run model, capture block-0 inputs via pre-hook."""
            layer_inputs, layer_input_kwargs = [], []

            def store_input_hook(module, args, kwargs):
                li = []
                if kwargs.get("hidden_states") is not None:
                    li.append(nested_move_to(kwargs["hidden_states"], device=cur_layer_device))
                else:
                    li.append(nested_move_to(args[0], device=cur_layer_device))
                layer_inputs.append(li)
                other_kwargs = {
                    k: nested_move_to(v, cur_layer_device)
                    for k, v in kwargs.items()
                    if k != "hidden_states"
                }
                layer_input_kwargs.append(other_kwargs)
                raise ValueError

            handle = blocks[0].register_forward_pre_hook(store_input_hook, with_kwargs=True)
            for data in dataset:
                for k, v in data.items():
                    data[k] = nested_move_to(v, cur_layer_device)
                try:
                    model(**data)
                except ValueError:
                    pass
            handle.remove()
            return layer_inputs, layer_input_kwargs

        print("[DBAR-v1] Capturing block-0 inputs for bank_r...")
        layer_inputs_r, layer_input_kwargs_r = capture_inputs(dataset_r)
        print("[DBAR-v1] Capturing block-0 inputs for bank_k...")
        layer_inputs_k, layer_input_kwargs_k = capture_inputs(dataset_k)

        if not has_device_map:
            blocks[0].to(device)
            for module_name in self.module_name_preceding_first_block:
                module = recurse_getattr(model, module_name)

        torch.cuda.empty_cache()

        # ── Step 3: Block-by-block dual-bank quantization ──────────────
        quantizers = {}

        for i, block in enumerate(tqdm(blocks, desc=f"DBAR-v1 {self.block_name_to_quantize}")):
            print(f"\n[DBAR-v1] Block {i+1}/{len(blocks)}")

            if not _has_device_more_than_cpu() or get_device(block) != torch.device("cpu"):
                pass  # already on GPU
            else:
                block = block.to(0)

            layers = get_layers(block)
            block_device = get_device(block)

            # Build subset list (true_sequential: one layer at a time)
            if self.true_sequential:
                layers_name_list = [[key] for key in layers.keys()]
            else:
                layers_name_list = [list(layers.keys())]

            for subset_name_list in tqdm(
                layers_name_list, leave=False, desc=f"Block {i+1} layers"
            ):
                subset_layers = {name: layers[name] for name in subset_name_list}

                # ── Pass 1: Accumulate H_r ──────────────────────────────
                gptq_r = {}
                handles_r = []
                for name in subset_layers:
                    gptq_r[name] = GPTQ(subset_layers[name], qcfg=self.quantizeConfig)
                    gptq_r[name].quantizer.configure(
                        bits=self.bits, sym=self.sym, perchannel=True
                    )

                    def _add_r(nm):
                        def _hook(_, inp, out):
                            gptq_r[nm].add_batch(inp[0].data, out.data)
                        return _hook

                    handles_r.append(
                        subset_layers[name].register_forward_hook(_add_r(name))
                    )

                for j in range(len(layer_inputs_r)):
                    inp_r = nested_move_to(layer_inputs_r[j], block_device)
                    kw_r = {
                        k: nested_move_to(v, block_device)
                        for k, v in layer_input_kwargs_r[j].items()
                    }
                    block(*inp_r, **kw_r)

                for h in handles_r:
                    h.remove()

                # Materialize H_r
                for name in subset_name_list:
                    gptq_r[name].materialize_global_hessian()

                # ── Pass 2: Accumulate H_k ──────────────────────────────
                gptq_k = {}
                handles_k = []
                for name in subset_layers:
                    gptq_k[name] = GPTQ(subset_layers[name], qcfg=self.quantizeConfig)
                    gptq_k[name].quantizer.configure(
                        bits=self.bits, sym=self.sym, perchannel=True
                    )

                    def _add_k(nm):
                        def _hook(_, inp, out):
                            gptq_k[nm].add_batch(inp[0].data, out.data)
                        return _hook

                    handles_k.append(
                        subset_layers[name].register_forward_hook(_add_k(name))
                    )

                for j in range(len(layer_inputs_k)):
                    inp_k = nested_move_to(layer_inputs_k[j], block_device)
                    kw_k = {
                        k: nested_move_to(v, block_device)
                        for k, v in layer_input_kwargs_k[j].items()
                    }
                    block(*inp_k, **kw_k)

                for h in handles_k:
                    h.remove()

                # Materialize H_k
                for name in subset_name_list:
                    gptq_k[name].materialize_global_hessian()

                # ── Merge and Quantize ──────────────────────────────────
                lam = self.lambda_weight
                for name in subset_name_list:
                    H_r = gptq_r[name].H
                    H_k = gptq_k[name].H
                    if H_r is None or H_k is None:
                        print(f"  [경고] {name}: H_r 또는 H_k가 None — bank_r Hessian 단독 사용")
                        # fallback: use H_r only
                    else:
                        H_dual = lam * H_r + (1.0 - lam) * H_k
                        gptq_r[name].H = H_dual
                        gptq_r[name]._hessian_dirty = False
                        del H_r, H_k, H_dual
                        torch.cuda.empty_cache()

                    print(f"  fasterquant: {name}  (block {i+1}, λ={lam})")
                    quant_outputs = gptq_r[name].fasterquant(
                        percdamp=self.damp_percent,
                        group_size=self.group_size,
                        actorder=self.desc_act,
                    )
                    scale, zero, g_idx = quant_outputs[0], quant_outputs[1], quant_outputs[2]
                    quantizers[f"{self.block_name_to_quantize}.{i}.{name}"] = (
                        gptq_r[name].quantizer,
                        scale,
                        zero,
                        g_idx,
                    )
                    gptq_r[name].free()
                    gptq_k[name].free()

                del subset_layers

            # ── Update cached inputs for next block ─────────────────────
            # Run BOTH banks through the (now quantized) block
            new_r, new_k = [], []
            for j in range(len(layer_inputs_r)):
                inp = nested_move_to(layer_inputs_r[j], block_device)
                kw = {k: nested_move_to(v, block_device) for k, v in layer_input_kwargs_r[j].items()}
                out = block(*inp, **kw)
                primary = out[0] if isinstance(out, tuple) else out
                primary = nested_move_to(primary, device=cur_layer_device)
                new_r.append([primary])
            for j in range(len(layer_inputs_k)):
                inp = nested_move_to(layer_inputs_k[j], block_device)
                kw = {k: nested_move_to(v, block_device) for k, v in layer_input_kwargs_k[j].items()}
                out = block(*inp, **kw)
                primary = out[0] if isinstance(out, tuple) else out
                primary = nested_move_to(primary, device=cur_layer_device)
                new_k.append([primary])

            if not has_device_map:
                blocks[i] = block.to(device)

            del layers
            layer_inputs_r, layer_inputs_k = new_r, new_k
            torch.cuda.empty_cache()

        # ── Step 4: Pack model ─────────────────────────────────────────
        self.pack_model(model=model, quantizers=quantizers)
        model.is_quantized = True
        model.quantization_method = QuantizationMethod.GPTQ
        if has_config:
            model.config.use_cache = use_cache
            model.config.quantization_config = self.to_dict()

        model = self.post_init_model(model)
        torch.cuda.empty_cache()
        return model


def run_dbar_v1(
    model_id: str,
    bank_r_path: str,
    bank_k_path: str,
    output_dir: str,
    lambda_weight: float = 0.5,
    bits: int = 4,
    group_size: int = 128,
    desc_act: bool = True,
):
    print(f"\n{'='*60}")
    print(f"[DBAR-v1] 양자화 시작")
    print(f"  model   : {model_id}")
    print(f"  bank_r  : {bank_r_path}")
    print(f"  bank_k  : {bank_k_path}")
    print(f"  λ       : {lambda_weight}")
    print(f"  bits/gs : {bits}b / g{group_size}")
    print(f"  out_dir : {output_dir}")
    print(f"{'='*60}\n")

    os.makedirs(output_dir, exist_ok=True)

    bank_r_texts = load_texts(bank_r_path)
    bank_k_texts = load_texts(bank_k_path)
    print(f"bank_r: {len(bank_r_texts)} sentences")
    print(f"bank_k: {len(bank_k_texts)} sentences")

    print("\nTokenizer 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    print("모델 로드 중 (FP16, CPU)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )

    quantizer = DualBankGPTQQuantizer(
        bank_r_texts=bank_r_texts,
        bank_k_texts=bank_k_texts,
        lambda_weight=lambda_weight,
        bits=bits,
        group_size=group_size,
        desc_act=desc_act,
        act_group_aware=False,
        model_seqlen=2048,
    )

    start = time.time()
    quantized_model = quantizer.quantize_model(model, tokenizer)
    elapsed = time.time() - start
    print(f"\n양자화 완료! ({elapsed:.1f}초 = {elapsed/60:.1f}분)")

    quantized_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"저장 완료: {output_dir}")

    # qzeros 패치 (optimum.gptq 호환성)
    print("[qzeros 패치] 0x88...→0x77... 변환 중...")
    from safetensors import safe_open
    from safetensors.torch import save_file
    import glob as _glob

    shard_files = sorted(_glob.glob(os.path.join(output_dir, "model*.safetensors")))
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
        print(f"  패치 완료: {os.path.basename(sf_path)}")

    # quantize_config.json
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

    # DBAR-v1 메타데이터 저장
    meta_path = os.path.join(output_dir, "dbar_v1_config.json")
    meta = {
        "method": "DBAR-v1",
        "model": model_id,
        "bank_r": bank_r_path,
        "bank_k": bank_k_path,
        "bank_r_size": len(bank_r_texts),
        "bank_k_size": len(bank_k_texts),
        "lambda_weight": lambda_weight,
        "bits": bits,
        "group_size": group_size,
        "desc_act": desc_act,
        "elapsed_sec": round(elapsed, 1),
    }
    with open(meta_path, "w") as f:
        _json.dump(meta, f, indent=2)
    print(f"메타데이터 저장: {meta_path}")
    print("\n[DBAR-v1] 완료!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DBAR-v1: Dual-Bank Adaptive Rounding")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--bank-r", required=True, help="Reasoning bank JSON (C_v3)")
    parser.add_argument("--bank-k", required=True, help="Domain bank JSON (B)")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--lambda-weight", type=float, default=0.5,
                        help="λ: reasoning bank weight (0=domain only, 1=reasoning only)")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--no-desc-act", action="store_true")
    args = parser.parse_args()

    run_dbar_v1(
        model_id=args.model,
        bank_r_path=args.bank_r,
        bank_k_path=args.bank_k,
        output_dir=args.out_dir,
        lambda_weight=args.lambda_weight,
        bits=args.bits,
        group_size=args.group_size,
        desc_act=not args.no_desc_act,
    )
