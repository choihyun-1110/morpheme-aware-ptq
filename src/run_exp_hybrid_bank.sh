#!/bin/bash
# Task-Aware Calibration Bank Mixing 실험
#
# 목적: Reasoning bank(C_v3) + Domain bank(B) hybrid calibration이
#       KoBEST/kmmlu tradeoff를 균형 있게 만드는지 검증
#
# 실험 조건:
#   1. H_Cv3_64_B_64_interleave  (기본, C_v3 앞)
#   2. H_Cv3_64_B_64_concat_ab   (C_v3 64 → B 64)
#   3. H_Cv3_96_B_32             (reasoning 강화)
#   4. H_Cv3_32_B_96             (domain 강화)
#
# 고정 변수: SOLAR, g128, desc_act=True, 4bit
# 비교 기준: 기존 B(0.6176/KoBEST), C_v3(0.6356/KoBEST) Sprint2 결과
#
# 실행: docker exec llm-dev bash /home/choihyun/workspace/src/run_exp_hybrid_bank.sh

set -euo pipefail

WORKSPACE="/home/choihyun/workspace"
RESULTS="${WORKSPACE}/results"
QUANT_MODELS="${WORKSPACE}/quantized_models"
SRC="${WORKSPACE}/src"

SOLAR_MODEL="upstage/SOLAR-10.7B-Instruct-v1.0"
CALIB_CV3="${RESULTS}/calibration_set_C_v3_upstage_SOLAR-10.7B-Instruct-v1.0.json"
CALIB_B="${RESULTS}/calibration_set_B.json"

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

result_exists_kobest() {
    compgen -G "${RESULTS}/eval_kobest_solar_hybrid_${1}*.json" > /dev/null 2>&1
}

result_exists_kmmlu() {
    compgen -G "${RESULTS}/eval_kmmlu_solar_hybrid_${1}*.json" > /dev/null 2>&1
}

eval_kobest() {
    local model_path="$1"; local label="$2"
    log "KoBEST: ${label}"
    "${LM_EVAL}" --model hf \
        --model_args "pretrained=${model_path},trust_remote_code=True,dtype=float16" \
        --tasks kobest --device "${GPU_DEVICE}" --batch_size 4 \
        --output_path "${RESULTS}/eval_kobest_solar_hybrid_${label}.json" --log_samples \
        2>&1 | tee "${RESULTS}/eval_kobest_solar_hybrid_${label}.log"
}

eval_kmmlu() {
    local model_path="$1"; local label="$2"
    log "kmmlu: ${label}"
    "${LM_EVAL}" --model hf \
        --model_args "pretrained=${model_path},trust_remote_code=True,dtype=float16" \
        --tasks kmmlu --device "${GPU_DEVICE}" --batch_size 4 \
        --output_path "${RESULTS}/eval_kmmlu_solar_hybrid_${label}.json" --log_samples \
        2>&1 | tee "${RESULTS}/eval_kmmlu_solar_hybrid_${label}.log"
}

quant_and_eval() {
    local calib="$1"; local label="$2"
    local quant_dir="${QUANT_MODELS}/solar_hybrid_${label}"

    log "====== ${label} ======"

    if result_exists_kobest "${label}" && result_exists_kmmlu "${label}"; then
        log "스킵: ${label} 이미 완료"
        return
    fi

    log "양자화 중..."
    "${PYTHON}" "${SRC}/run_quant_optimum.py" \
        --model "${SOLAR_MODEL}" \
        --calib-path "${calib}" \
        --out-dir "${quant_dir}" \
        2>&1 | tee "${RESULTS}/quant_solar_hybrid_${label}.log"

    eval_kobest "${quant_dir}" "${label}"
    eval_kmmlu  "${quant_dir}" "${label}"
    rm -rf "${quant_dir}"
    log "${label} 완료"
}

# ── Step 1: Hybrid calibration JSON 생성 ──────────────────────────────────────
log "======================================================"
log "Hybrid Calibration Bank 생성"
log "======================================================"

# 파일명 형식: calibration_set_H_{labelA}{nA}_{labelB}{nB}_{mode}.json
# 1. interleave 64+64
CALIB_H_64_64_IL="${RESULTS}/calibration_set_H_Cv364_B64_interleave.json"
if [ ! -f "${CALIB_H_64_64_IL}" ]; then
    "${PYTHON}" "${SRC}/build_hybrid_calibration.py" \
        --bank-a "${CALIB_CV3}" --bank-b "${CALIB_B}" \
        --n-a 64 --n-b 64 --mode interleave \
        --label-a Cv3 --label-b B --out-dir "${RESULTS}"
fi

# 2. concat_ab 64+64
CALIB_H_64_64_CAB="${RESULTS}/calibration_set_H_Cv364_B64_concat_ab.json"
if [ ! -f "${CALIB_H_64_64_CAB}" ]; then
    "${PYTHON}" "${SRC}/build_hybrid_calibration.py" \
        --bank-a "${CALIB_CV3}" --bank-b "${CALIB_B}" \
        --n-a 64 --n-b 64 --mode concat_ab \
        --label-a Cv3 --label-b B --out-dir "${RESULTS}"
fi

# 3. reasoning 강화: C_v3 96 + B 32
CALIB_H_96_32="${RESULTS}/calibration_set_H_Cv396_B32_interleave.json"
if [ ! -f "${CALIB_H_96_32}" ]; then
    "${PYTHON}" "${SRC}/build_hybrid_calibration.py" \
        --bank-a "${CALIB_CV3}" --bank-b "${CALIB_B}" \
        --n-a 96 --n-b 32 --mode interleave \
        --label-a Cv3 --label-b B --out-dir "${RESULTS}"
fi

# 4. domain 강화: C_v3 32 + B 96
CALIB_H_32_96="${RESULTS}/calibration_set_H_Cv332_B96_interleave.json"
if [ ! -f "${CALIB_H_32_96}" ]; then
    "${PYTHON}" "${SRC}/build_hybrid_calibration.py" \
        --bank-a "${CALIB_CV3}" --bank-b "${CALIB_B}" \
        --n-a 32 --n-b 96 --mode interleave \
        --label-a Cv3 --label-b B --out-dir "${RESULTS}"
fi

log "Hybrid calibration 생성 완료"
ls -la "${RESULTS}"/calibration_set_H_*.json

# ── Step 2: 실험 실행 ──────────────────────────────────────────────────────────
log "======================================================"
log "SOLAR Hybrid Bank 실험 시작 (GPU${CUDA_VISIBLE_DEVICES:-0})"
log "======================================================"

quant_and_eval "${CALIB_H_64_64_IL}"  "Cv364_B64_interleave"
quant_and_eval "${CALIB_H_64_64_CAB}" "Cv364_B64_concat_ab"
quant_and_eval "${CALIB_H_96_32}"     "Cv396_B32_interleave"
quant_and_eval "${CALIB_H_32_96}"     "Cv332_B96_interleave"

# ── Step 3: 결과 집계 ─────────────────────────────────────────────────────────
log "======================================================"
log "결과 비교"
log "======================================================"

"${PYTHON}" - <<'PYEOF'
import json, glob, os, statistics

R = "/home/choihyun/workspace/results"
FP16_KB = 0.6523

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

def balanced(kb, km, best_kb, best_km):
    if not kb or not km or not best_kb or not best_km: return None
    return round((kb/best_kb + km/best_km)/2, 4)

# 기존 결과 (Sprint2/S4 기준)
baselines = [
    ("FP16",    0.6523, None),
    ("A",       0.5981, None),
    ("B",       0.6176, None),
    ("C_v3",    0.6356, None),
]

# kmmlu baseline (SOLAR - S4에서 측정한 값)
KB_KMMLU = {
    "B":   latest(R+"/eval_kmmlu_solar_cond_B*.json"),
    "C_v3": latest(R+"/eval_kmmlu_solar_cond_C_v3*.json"),
    "A":   latest(R+"/eval_kmmlu_solar_cond_A*.json"),
}

hybrid_labels = [
    ("H Cv364+B64 IL",  "Cv364_B64_interleave"),
    ("H Cv364+B64 CAB", "Cv364_B64_concat_ab"),
    ("H Cv396+B32 IL",  "Cv396_B32_interleave"),
    ("H Cv332+B96 IL",  "Cv332_B96_interleave"),
]

print(f"\n{'조건':<26} {'KoBEST':>8} {'보존율':>7} {'kmmlu':>7} {'balanced':>9}")
print("-" * 62)

best_kb = 0.6356  # C_v3
best_km_path = KB_KMMLU.get("B")
best_km = agg_kmmlu(best_km_path)

for label, kb_val, km_path in baselines:
    km_val = agg_kmmlu(KB_KMMLU.get(label)) if label in KB_KMMLU else None
    bal = balanced(kb_val, km_val, best_kb, best_km)
    print("{:<26} {:>8} {:>7} {:>7} {:>9}".format(
        label,
        f"{kb_val:.4f}",
        f"{kb_val/FP16_KB*100:.1f}%",
        f"{km_val:.4f}" if km_val else "—",
        f"{bal:.4f}" if bal else "—"
    ))

print()
for display_label, file_label in hybrid_labels:
    kb_path = latest(R+f"/eval_kobest_solar_hybrid_{file_label}*.json")
    km_path = latest(R+f"/eval_kmmlu_solar_hybrid_{file_label}*.json")
    kb = agg_kobest(kb_path)
    km = agg_kmmlu(km_path)
    bal = balanced(kb, km, best_kb, best_km)
    print("{:<26} {:>8} {:>7} {:>7} {:>9}".format(
        display_label,
        f"{kb:.4f}" if kb else "없음",
        f"{kb/FP16_KB*100:.1f}%" if kb else "—",
        f"{km:.4f}" if km else "없음",
        f"{bal:.4f}" if bal else "—"
    ))

print()
print("판정 기준:")
print("  balanced_retention = (KoBEST/best_single_KoBEST + kmmlu/best_single_kmmlu) / 2")
print("  best_single_KoBEST = C_v3 (0.6356), best_single_kmmlu = B (위 값)")
PYEOF

log "======================================================"
log "Hybrid Bank 실험 완료"
log "======================================================"
