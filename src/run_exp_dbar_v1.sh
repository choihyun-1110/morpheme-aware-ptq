#!/bin/bash
# DBAR-v1 실험: λ=0.3 / 0.5 / 0.7
#
# Reasoning bank: C_v3 (128 samples)
# Domain bank:    B    (128 samples)
# 각 bank 128 samples 전부 사용 → hybrid 64+64 대비 Hessian 추정 품질 2배
#
# 실행: docker exec llm-dev bash /home/choihyun/workspace/src/run_exp_dbar_v1.sh

set -euo pipefail

WORKSPACE="/home/choihyun/workspace"
RESULTS="${WORKSPACE}/results"
QUANT_MODELS="${WORKSPACE}/quantized_models"
SRC="${WORKSPACE}/src"

SOLAR_MODEL="upstage/SOLAR-10.7B-Instruct-v1.0"
BANK_R="${RESULTS}/calibration_set_C_v3_upstage_SOLAR-10.7B-Instruct-v1.0.json"
BANK_K="${RESULTS}/calibration_set_B.json"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HOME="/home/choihyun"
export HF_HOME="${WORKSPACE}/.cache/huggingface"
export XDG_CACHE_HOME="${WORKSPACE}/.cache"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

cd "${WORKSPACE}"
source /opt/conda/etc/profile.d/conda.sh
conda activate llm-quant

LM_EVAL="/opt/conda/envs/llm-quant/bin/lm_eval"
PYTHON="/opt/conda/envs/llm-quant/bin/python3.11"
GPU_DEVICE="cuda:${CUDA_VISIBLE_DEVICES:-0}"

log() { echo "[$(date '+%H:%M')] $*"; }

dbar_eval() {
    local lambda="$1"
    local label="dbar_v1_lambda${lambda/./}"  # e.g. 0.5 → lambda05
    local quant_dir="${QUANT_MODELS}/solar_${label}"

    log "====== DBAR-v1 λ=${lambda} ======"

    # 양자화
    if [ ! -f "${quant_dir}/quantize_config.json" ]; then
        log "양자화 중: λ=${lambda}"
        "${PYTHON}" "${SRC}/run_dbar_v1.py" \
            --model "${SOLAR_MODEL}" \
            --bank-r "${BANK_R}" \
            --bank-k "${BANK_K}" \
            --lambda-weight "${lambda}" \
            --out-dir "${quant_dir}" \
            2>&1 | tee "${RESULTS}/quant_solar_${label}.log"
    else
        log "스킵 (이미 양자화됨): ${label}"
    fi

    # KoBEST
    if ! compgen -G "${RESULTS}/eval_kobest_solar_${label}*.json" > /dev/null 2>&1; then
        log "KoBEST: ${label}"
        "${LM_EVAL}" --model hf \
            --model_args "pretrained=${quant_dir},trust_remote_code=True,dtype=float16" \
            --tasks kobest --device "${GPU_DEVICE}" --batch_size 4 \
            --output_path "${RESULTS}/eval_kobest_solar_${label}.json" --log_samples \
            2>&1 | tee "${RESULTS}/eval_kobest_solar_${label}.log"
    fi

    # kmmlu
    if ! compgen -G "${RESULTS}/eval_kmmlu_solar_${label}*.json" > /dev/null 2>&1; then
        log "kmmlu: ${label}"
        "${LM_EVAL}" --model hf \
            --model_args "pretrained=${quant_dir},trust_remote_code=True,dtype=float16" \
            --tasks kmmlu --device "${GPU_DEVICE}" --batch_size 4 \
            --output_path "${RESULTS}/eval_kmmlu_solar_${label}.json" --log_samples \
            2>&1 | tee "${RESULTS}/eval_kmmlu_solar_${label}.log"
    fi

    rm -rf "${quant_dir}"
    log "${label} 완료"
}

log "======================================================"
log "SOLAR DBAR-v1 실험 (GPU${CUDA_VISIBLE_DEVICES:-0})"
log "======================================================"

dbar_eval "0.3"   # reasoning 약화
dbar_eval "0.5"   # 동등
dbar_eval "0.7"   # reasoning 강화

log "======================================================"
log "결과 비교"
log "======================================================"

"${PYTHON}" - <<'PYEOF'
import json, glob, os

R = "/home/choihyun/workspace/results"

def agg_kobest(path):
    if not path or not os.path.exists(path): return None
    with open(path) as f: d = json.load(f)
    r = d.get("results", {})
    task_keys = [("kobest_copa","acc_norm,none","acc,none"),
                 ("kobest_hellaswag","acc_norm,none","acc,none"),
                 ("kobest_sentineg","acc_norm,none","acc,none"),
                 ("kobest_wic","acc,none",None),
                 ("kobest_boolq","acc,none",None)]
    scores = []
    for t,p,fb in task_keys:
        if t in r:
            v = r[t].get(p) or (r[t].get(fb) if fb else None)
            if v is not None: scores.append(v)
    return round(sum(scores)/len(scores),4) if scores else None

def agg_kmmlu(path):
    if not path or not os.path.exists(path): return None
    with open(path) as f: d = json.load(f)
    r = d.get("results", {})
    scores = [v.get("acc,none") for k,v in r.items() if "acc,none" in v]
    return round(sum(scores)/len(scores),4) if scores else None

def latest(pat):
    f = sorted(glob.glob(pat))
    return f[-1] if f else None

# 기준값
FP16_KB = 0.6523
baselines = {
    "B":   {"kb": 0.6176, "km": agg_kmmlu(latest(R+"/eval_kmmlu_solar_cond_B*.json"))},
    "C_v3": {"kb": 0.6356, "km": agg_kmmlu(latest(R+"/eval_kmmlu_solar_cond_C_v3*.json"))},
}
best_kb = baselines["C_v3"]["kb"]
best_km = baselines["B"]["km"]

def balanced(kb, km):
    if not kb or not km or not best_km: return None
    return round((kb/best_kb + km/best_km)/2, 4)

print(f"\n{'조건':<22} {'KoBEST':>8} {'보존율':>7} {'kmmlu':>7} {'balanced':>9}")
print("-" * 56)

for label, vals in baselines.items():
    kb, km = vals["kb"], vals["km"]
    bal = balanced(kb, km)
    print("{:<22} {:>8} {:>7} {:>7} {:>9}".format(
        label, f"{kb:.4f}", f"{kb/FP16_KB*100:.1f}%",
        f"{km:.4f}" if km else "—", f"{bal:.4f}" if bal else "—"
    ))

print()
for lam_str in ["03", "05", "07"]:
    lam = float(lam_str[0] + "." + lam_str[1])
    lbl = f"dbar_v1_lambda{lam_str}"
    kb_path = latest(R+f"/eval_kobest_solar_{lbl}*.json")
    km_path = latest(R+f"/eval_kmmlu_solar_{lbl}*.json")
    kb = agg_kobest(kb_path)
    km = agg_kmmlu(km_path)
    bal = balanced(kb, km)
    print("{:<22} {:>8} {:>7} {:>7} {:>9}".format(
        f"DBAR-v1 λ={lam}",
        f"{kb:.4f}" if kb else "없음",
        f"{kb/FP16_KB*100:.1f}%" if kb else "—",
        f"{km:.4f}" if km else "없음",
        f"{bal:.4f}" if bal else "—"
    ))
PYEOF

log "DBAR-v1 실험 완료"
