#!/bin/bash
# GPU0: Llama3-Ko FP16 baseline + EEVE kmmlu 실험
#
# 1. Llama3-Ko FP16 KoBEST (보존율 기준값, ~30min)
# 2. EEVE A/B/C_v3/C_v3_eeve kmmlu (재양자화 + 평가, ~각 40min)

set -euo pipefail

WORKSPACE="/home/choihyun/workspace"
RESULTS="${WORKSPACE}/results"
QUANT_MODELS="${WORKSPACE}/quantized_models"
SRC="${WORKSPACE}/src"

LLAMA_MODEL="beomi/Llama-3-Open-Ko-8B"
EEVE_MODEL="yanolja/EEVE-Korean-Instruct-10.8B-v1.0"

CALIB_A="${RESULTS}/calibration_set_A.json"
CALIB_B="${RESULTS}/calibration_set_B.json"
CALIB_CV3="${RESULTS}/calibration_set_C_v3_upstage_SOLAR-10.7B-Instruct-v1.0.json"
CALIB_CV3_EEVE="${RESULTS}/calibration_set_C_v3_eeve_yanolja_EEVE-Korean-Instruct-10.8B-v1.0.json"

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

result_exists() {
    compgen -G "${RESULTS}/${1}*.json" > /dev/null 2>&1
}

eval_kobest() {
    local model_path="$1"; local out_name="$2"
    log "KoBEST: ${out_name}"
    "${LM_EVAL}" --model hf \
        --model_args "pretrained=${model_path},trust_remote_code=True,dtype=float16" \
        --tasks kobest --device "${GPU_DEVICE}" --batch_size 4 \
        --output_path "${RESULTS}/eval_kobest_${out_name}.json" --log_samples \
        2>&1 | tee "${RESULTS}/eval_kobest_${out_name}.log"
}

eval_kmmlu() {
    local model_path="$1"; local out_name="$2"
    log "kmmlu: ${out_name}"
    "${LM_EVAL}" --model hf \
        --model_args "pretrained=${model_path},trust_remote_code=True,dtype=float16" \
        --tasks kmmlu --device "${GPU_DEVICE}" --batch_size 4 \
        --output_path "${RESULTS}/eval_kmmlu_${out_name}.json" --log_samples \
        2>&1 | tee "${RESULTS}/eval_kmmlu_${out_name}.log"
}

quant_and_eval_eeve_kmmlu() {
    local calib="$1"; local cond="$2"
    local quant_dir="${QUANT_MODELS}/eeve_10b_cond_${cond}"

    if result_exists "eval_kmmlu_eeve_10b_${cond}"; then
        log "스킵: eeve kmmlu ${cond} 이미 완료"
        return
    fi

    log "=== EEVE ${cond} 양자화 ==="
    "${PYTHON}" "${SRC}/run_quant_optimum.py" \
        --model "${EEVE_MODEL}" \
        --calib-path "${calib}" \
        --out-dir "${quant_dir}" \
        2>&1 | tee "${RESULTS}/quant_eeve_${cond}.log"

    eval_kobest "${quant_dir}" "eeve_10b_${cond}_rerun"
    eval_kmmlu  "${quant_dir}" "eeve_10b_${cond}"
    rm -rf "${quant_dir}"
    log "완료: EEVE ${cond}"
}

log "======================================================"
log "GPU0: Llama3-Ko FP16 + EEVE kmmlu"
log "======================================================"

# ── 1. Llama3-Ko FP16 KoBEST (보존율 기준값) ──
if ! result_exists "eval_kobest_llama3_ko_8b_fp16"; then
    log "Llama3-Ko FP16 KoBEST 평가..."
    eval_kobest "${LLAMA_MODEL}" "llama3_ko_8b_fp16"
else
    log "스킵: Llama3-Ko FP16 이미 완료"
fi

# ── 2. EEVE kmmlu (A / B / C_v3 / C_v3_eeve) ──
quant_and_eval_eeve_kmmlu "${CALIB_A}"        "A"
quant_and_eval_eeve_kmmlu "${CALIB_B}"        "B"
quant_and_eval_eeve_kmmlu "${CALIB_CV3}"      "C_v3"
quant_and_eval_eeve_kmmlu "${CALIB_CV3_EEVE}" "C_v3_eeve"

# ── 결과 요약 ──
log "======================================================"
log "결과 요약"
log "======================================================"

"${PYTHON}" - <<'PYEOF'
import json, glob, os

R = "/home/choihyun/workspace/results"

def agg_kobest(path):
    if not path or not os.path.exists(path): return None
    with open(path) as f: d = json.load(f)
    r = d.get("results", {})
    task_keys = [
        ("kobest_copa","acc_norm,none","acc,none"),
        ("kobest_hellaswag","acc_norm,none","acc,none"),
        ("kobest_sentineg","acc_norm,none","acc,none"),
        ("kobest_wic","acc,none",None),
        ("kobest_boolq","acc,none",None),
    ]
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

def latest(pat): f=sorted(glob.glob(pat)); return f[-1] if f else None

# Llama3-Ko FP16
fp16_path = latest(R+"/eval_kobest_llama3_ko_8b_fp16*.json")
fp16 = agg_kobest(fp16_path)
print("\nLlama3-Ko FP16 KoBEST: {}".format(fp16 if fp16 else "없음"))
if fp16:
    print("  (기존 C_en_v3=0.5916이므로 보존율={:.1f}%)".format(0.5916/fp16*100))

# EEVE kmmlu
print("\nEEVE kmmlu 결과:")
fp16_eeve_kb = agg_kobest(latest(R+"/eval_kobest_eeve_10b_fp16*.json"))
entries = [
    ("A",        latest(R+"/eval_kmmlu_eeve_10b_A*.json")),
    ("B",        latest(R+"/eval_kmmlu_eeve_10b_B*.json")),
    ("C_v3",     latest(R+"/eval_kmmlu_eeve_10b_C_v3_2*.json")),
    ("C_v3_eeve",latest(R+"/eval_kmmlu_eeve_10b_C_v3_eeve*.json")),
]
print("{:<14} {:>8}".format("조건","kmmlu"))
print("-"*24)
for label,path in entries:
    v = agg_kmmlu(path)
    print("{:<14} {:>8}".format(label, "{:.4f}".format(v) if v else "없음"))
PYEOF

log "======================================================"
log "GPU0 실험 완료"
log "======================================================"
