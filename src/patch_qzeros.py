"""
optimum.gptq 저장 모델의 qzeros 패치:
  optimum.gptq: zero_point=8 직접 저장 (0x88888888)
  gptqmodel: (zero_point - 1) = 7 기대 (0x77777777)
  → 각 nibble에서 1을 빼서 보정 (0x88...→0x77...)
"""
import os
import sys
import json
import glob
import argparse
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def patch_model_qzeros(model_dir: str, group_size: int = 128, desc_act: bool = True):
    shard_files = sorted(glob.glob(os.path.join(model_dir, "model*.safetensors")))
    if not shard_files:
        print(f"[스킵] safetensors 없음: {model_dir}")
        return False

    # 이미 패치됐는지 확인 (임시: qzeros 값으로 체크)
    with safe_open(shard_files[0], framework="pt") as f:
        keys = list(f.keys())
        qz_keys = [k for k in keys if "qzeros" in k]
        if not qz_keys:
            print(f"[스킵] qzeros 없음: {model_dir}")
            return False
        sample = f.get_tensor(qz_keys[0])
        # 0x88888888 = -2004318072 (signed int32) → 패치 필요
        # 0x77777777 = 2004318071 → 이미 패치됨
        if sample[0, 0].item() == 2004318071:
            print(f"[스킵] 이미 패치됨: {model_dir}")
            return False

    print(f"[패치] {model_dir}")
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
        print(f"  [완료] {os.path.basename(sf_path)}")

    # quantize_config.json 생성/갱신
    qc_path = os.path.join(model_dir, "quantize_config.json")
    if not os.path.exists(qc_path):
        qc = {
            "bits": 4, "group_size": group_size, "damp_percent": 0.1,
            "desc_act": desc_act, "static_groups": False, "sym": True,
            "true_sequential": True, "model_name_or_path": None,
            "model_file_base_name": None, "is_marlin_format": False,
            "quant_method": "gptq",
        }
        with open(qc_path, "w") as f:
            json.dump(qc, f, indent=2)
        print(f"  [생성] quantize_config.json (group_size={group_size})")

    print(f"  [완료] {model_dir}\n")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", default="/home/choihyun/workspace/quantized_models")
    args = parser.parse_args()

    models_dir = args.models_dir

    # optimum.gptq로 저장된 모델 목록 (model*.safetensors 파일 있는 것들)
    model_dirs = []
    for d in os.listdir(models_dir):
        full = os.path.join(models_dir, d)
        if os.path.isdir(full) and glob.glob(os.path.join(full, "model*.safetensors")):
            model_dirs.append(full)

    model_dirs.sort()
    print(f"optimum.gptq 포맷 모델 {len(model_dirs)}개 발견:")
    for d in model_dirs:
        print(f"  {os.path.basename(d)}")
    print()

    # group_size 추정: 이름에 g64 포함이면 64, 아니면 128
    for d in model_dirs:
        name = os.path.basename(d)
        gs = 64 if "g64" in name else 128
        # desc_act=False 모델은 qzeros 없을 수 있음
        da = "desc_act_false" not in name
        patch_model_qzeros(d, group_size=gs, desc_act=da)

    print("모든 패치 완료!")
