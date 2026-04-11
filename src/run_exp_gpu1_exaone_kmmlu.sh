#!/bin/bash
# GPU1: EXAONE35 kmmlu (A / B / C_v3 / C_v3_exaone)

set -euo pipefail

WORKSPACE="/home/choihyun/workspace"
RESULTS="${WORKSPACE}/results"
QUANT_MODELS="${WORKSPACE}/quantized_models"
SRC="${WORKSPACE}/src"

EXAONE_MODEL="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"

CALIB_A="${RESULTS}/calibration_set_A.json"
CALIB_B="${RESULTS}/calibration_set_B.json"
CALIB_CV3="${RESULTS}/calibration_set_C_v3_upstage_SOLAR-10.7B-Instruct-v1.0.json"
CALIB_CV3_EXAONE=$(ls "${RESULTS}"/calibration_set_C_exaone*.json 2>/dev/null | head -1 || true)

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
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
GPU_DEVICE="cuda:${CUDA_VISIBLE_DEVICES:-1}"

log() { echo "[$(date '+%H:%M')] $*"; }

result_exists() {
    compgen -G "${RESULTS}/${1}*.json" > /dev/null 2>&1
}

eval_kmmlu() {
    local model_path="$1"; local out_name="$2"
    log "kmmlu: ${out_name}"
    "${LM_EVAL}" --model hf \
        --model_args "pretrained=${model_path},trust_remote_code=True,dtype=float16" \
        --tasks kmmlu --device "${GPU_DEVICE}" --batch_size 4 \
        --output_path "${RESULTS}/eval_kmmlu_${out_name}.json" --log_samples \
        2>&1 | tee "${RESULTS}/eval_kmmlu_${out_name}.log"
    log "완료: ${out_name}"
}

quant_and_eval() {
    local calib="$1"; local cond="$2"
    local quant_dir="${QUANT_MODELS}/exaone35_7b_cond_${cond}"

    if result_exists "eval_kmmlu_exaone35_7b_${cond}"; then
        log "스킵: exaone kmmlu ${cond} 이미 완료"
        return
    fi

    log "=== EXAONE35 ${cond} 양자화 ==="
    "${PYTHON}" "${SRC}/run_quant_optimum.py" \
        --model "${EXAONE_MODEL}" \
        --calib-path "${calib}" \
        --out-dir "${quant_dir}" \
        2>&1 | tee "${RESULTS}/quant_exaone35_${cond}.log"

    eval_kmmlu "${quant_dir}" "exaone35_7b_${cond}"
    rm -rf "${quant_dir}"
}

log "======================================================"
log "GPU1: EXAONE35 kmmlu (A/B/C_v3/C_v3_exaone)"
log "======================================================"

quant_and_eval "${CALIB_A}"         "A"
quant_and_eval "${CALIB_B}"         "B"
quant_and_eval "${CALIB_CV3}"       "C_v3"
quant_and_eval "${CALIB_CV3_EXAONE}" "C_v3_exaone"

# ── 결과 요약 ──
"${PYTHON}" - <<'PYEOF'
import json, glob, os

R = "/home/choihyun/workspace/results"

def agg_kmmlu(path):
    if not path or not os.path.exists(path): return None
    with open(path) as f: d = json.load(f)
    r = d.get("results", {})
    scores = [v.get("acc,none") for k,v in r.items() if "acc,none" in v]
    return round(sum(scores)/len(scores),4) if scores else None

def agg_kobest(path):
    if not path or not os.path.exists(path): return None
    with open(path) as f: d = json.load(f)
    r = d.get("results", {})
    task_keys = [("kobest_copa","acc_norm,none","acc,none"),("kobest_hellaswag","acc_norm,none","acc,none"),
                 ("kobest_sentineg","acc_norm,none","acc,none"),("kobest_wic","acc,none",None),("kobest_boolq","acc,none",None)]
    scores = []
    for t,p,fb in task_keys:
        if t in r:
            v = r[t].get(p) or (r[t].get(fb) if fb else None)
            if v is not None: scores.append(v)
    return round(sum(scores)/len(scores),4) if scores else None

def latest(pat): f=sorted(glob.glob(pat)); return f[-1] if f else None

fp16_kb = agg_kobest(latest(R+"/eval_kobest_exaone35_7b_fp16*.json"))

print("\nEXAONE35 KoBEST+kmmlu 비교 (FP16 KoBEST={})".format(fp16_kb))
print("{:<16} {:>8} {:>8}".format("조건","KoBEST","kmmlu"))
print("-"*36)
entries = [
    ("A",         latest(R+"/eval_kobest_exaone35_7b_A*.json"),         latest(R+"/eval_kmmlu_exaone35_7b_A*.json")),
    ("B",         latest(R+"/eval_kobest_exaone35_7b_B*.json"),         latest(R+"/eval_kmmlu_exaone35_7b_B*.json")),
    ("C_v3",      latest(R+"/eval_kobest_exaone35_7b_C_v3_2*.json"),    latest(R+"/eval_kmmlu_exaone35_7b_C_v3_2*.json")),
    ("C_v3_exaone",latest(R+"/eval_kobest_exaone35_7b_C_v3_exaone*.json"),latest(R+"/eval_kmmlu_exaone35_7b_C_v3_exaone*.json")),
]
for label,kp,mp in entries:
    kb = agg_kobest(kp); km = agg_kmmlu(mp)
    print("{:<16} {:>8} {:>8}".format(label,
        "{:.4f}".format(kb) if kb else "없음",
        "{:.4f}".format(km) if km else "없음"))
PYEOF

log "======================================================"
log "EXAONE35 kmmlu 완료"
log "======================================================"
