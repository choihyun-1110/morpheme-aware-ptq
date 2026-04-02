#!/bin/bash
# Sprint 4 GPU 1 파이프라인
# S4-3: 누락 조건 보완 (B g64 / A no_desc_act / EEVE FP16+A)
# S4-4: C_v3_exaone 생성 + EXAONE35 재실험 (tokenizer 의존성 일반화)
#
# 실행: docker exec llm-dev bash /home/choihyun/workspace/src/run_exp_s4_gpu1.sh

set -euo pipefail

WORKSPACE="/home/choihyun/workspace"
RESULTS="${WORKSPACE}/results"
QUANT_MODELS="${WORKSPACE}/quantized_models"
SRC="${WORKSPACE}/src"
SOLAR_MODEL="upstage/SOLAR-10.7B-Instruct-v1.0"
EEVE_MODEL="yanolja/EEVE-Korean-Instruct-10.8B-v1.0"
EXAONE_MODEL="${WORKSPACE}/.cache/huggingface/hub/models--LGAI-EXAONE--EXAONE-3.5-7.8B-Instruct/snapshots"
CALIB_A="${RESULTS}/calibration_set_A.json"
CALIB_B="${RESULTS}/calibration_set_B.json"
CALIB_Cv3="${RESULTS}/calibration_set_C_v3_upstage_SOLAR-10.7B-Instruct-v1.0.json"

export CUDA_VISIBLE_DEVICES=1
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

log() { echo "[$(date '+%H:%M')] $*"; }

eval_kobest() {
    local model_path="$1"
    local out_name="$2"
    local extra="${3:-}"
    log "KoBEST 평가: ${out_name}"
    "${LM_EVAL}" --model hf \
        --model_args "pretrained=${model_path},trust_remote_code=True,dtype=float16${extra}" \
        --tasks kobest \
        --device cuda:1 \
        --batch_size 4 \
        --output_path "${RESULTS}/eval_kobest_${out_name}.json" \
        --log_samples \
        2>&1 | tee "${RESULTS}/eval_kobest_${out_name}.log"
    log "완료: ${out_name}"
}

quant_optimum() {
    local model="$1"; local calib="$2"; local out="$3"; local extra="${4:-}"
    if [ -f "${out}/config.json" ]; then
        log "이미 양자화됨 (스킵): ${out}"; return 0
    fi
    "${PYTHON}" "${SRC}/run_quant_optimum.py" \
        --model "${model}" --calib-path "${calib}" --out-dir "${out}" ${extra} \
        2>&1 | tee "${RESULTS}/quant_$(basename ${out}).log"
}

# EXAONE 모델 경로 동적 탐색
find_exaone_path() {
    local snap
    snap=$(ls "${WORKSPACE}/.cache/huggingface/hub/models--LGAI-EXAONE--EXAONE-3.5-7.8B-Instruct/snapshots/" 2>/dev/null | head -1)
    if [ -n "${snap}" ]; then
        echo "${WORKSPACE}/.cache/huggingface/hub/models--LGAI-EXAONE--EXAONE-3.5-7.8B-Instruct/snapshots/${snap}"
    else
        echo "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
    fi
}

log "======================================================"
log "Sprint 4 GPU 1: S4-3 누락 조건 + S4-4 EXAONE35"
log "======================================================"

# ─────────────────────────────────────────────────────────
# S4-3-1: SOLAR B group_size=64
# ─────────────────────────────────────────────────────────
log "=== [S4-3-1] SOLAR B g64 ==="
if [ ! -f "${RESULTS}/eval_kobest_solar_B_g64.json" ]; then
    quant_optimum "${SOLAR_MODEL}" "${CALIB_B}" "${QUANT_MODELS}/solar_B_g64" "--group-size 64"
    eval_kobest "${QUANT_MODELS}/solar_B_g64" "solar_B_g64"
else
    log "스킵: solar_B_g64 이미 완료"
fi

# ─────────────────────────────────────────────────────────
# S4-3-2: SOLAR A desc_act=False
# ─────────────────────────────────────────────────────────
log "=== [S4-3-2] SOLAR A desc_act=False ==="
if [ ! -f "${RESULTS}/eval_kobest_solar_A_no_desc_act.json" ]; then
    quant_optimum "${SOLAR_MODEL}" "${CALIB_A}" "${QUANT_MODELS}/solar_A_no_desc_act" "--no-desc-act"
    eval_kobest "${QUANT_MODELS}/solar_A_no_desc_act" "solar_A_no_desc_act"
else
    log "스킵: solar_A_no_desc_act 이미 완료"
fi

# ─────────────────────────────────────────────────────────
# S4-3-3: EEVE FP16 baseline
# ─────────────────────────────────────────────────────────
log "=== [S4-3-3] EEVE FP16 베이스라인 ==="
if [ ! -f "${RESULTS}/eval_kobest_eeve_10b_fp16.json" ]; then
    eval_kobest "${EEVE_MODEL}" "eeve_10b_fp16"
else
    log "스킵: eeve_10b_fp16 이미 완료"
fi

# ─────────────────────────────────────────────────────────
# S4-4: C_v3_exaone 생성 + EXAONE35 재실험
# ─────────────────────────────────────────────────────────
log "=== [S4-4] C_v3_exaone calibration 생성 ==="
CALIB_Cv3_EXAONE="${RESULTS}/calibration_set_C_v3_exaone_LGAI-EXAONE-3.5-7.8B-Instruct.json"
EXAONE_PATH=$(find_exaone_path)
log "EXAONE 경로: ${EXAONE_PATH}"

if [ ! -f "${CALIB_Cv3_EXAONE}" ]; then
    # EXAONE 패치 먼저 적용
    "${PYTHON}" "${SRC}/patch_exaone.py" 2>&1 | tail -5 || true
    log "C_v3_exaone calibration 생성 중 (나무위키 처리, ~20분)..."
    "${PYTHON}" "${SRC}/build_calibration.py" \
        --condition C \
        --model "${EXAONE_PATH}" \
        --suffix "exaone_LGAI-EXAONE-3.5-7.8B-Instruct" \
        2>&1 | tee "${RESULTS}/build_C_v3_exaone.log"
    log "C_v3_exaone calibration 생성 완료"
else
    log "스킵: C_v3_exaone 이미 존재"
fi

log "=== [S4-4] EXAONE35 C_v3_exaone 양자화 + 평가 ==="
if [ ! -f "${RESULTS}/eval_kobest_exaone35_7b_C_v3_exaone.json" ]; then
    "${PYTHON}" "${SRC}/patch_exaone.py" 2>&1 | tail -5 || true
    quant_optimum "${EXAONE_PATH}" "${CALIB_Cv3_EXAONE}" "${QUANT_MODELS}/exaone35_7b_cond_C_v3_exaone"
    eval_kobest "${QUANT_MODELS}/exaone35_7b_cond_C_v3_exaone" "exaone35_7b_C_v3_exaone"
else
    log "스킵: exaone35_7b_C_v3_exaone 이미 완료"
fi

# ─────────────────────────────────────────────────────────
# 결과 요약
# ─────────────────────────────────────────────────────────
log "======================================================"
log "결과 요약"
log "======================================================"

"${PYTHON}" - <<'PYEOF'
import json, os

results_dir = "/home/choihyun/workspace/results"

def kobest_agg(path):
    if not os.path.exists(path): return None
    with open(path) as f:
        d = json.load(f)
    r = d.get("results", {})
    tasks = {
        "kobest_copa":      ("acc_norm,none", "acc,none"),
        "kobest_hellaswag": ("acc_norm,none", "acc,none"),
        "kobest_sentineg":  ("acc_norm,none", "acc,none"),
        "kobest_wic":       ("acc,none",      None),
        "kobest_boolq":     ("acc,none",      None),
    }
    scores = []
    for task, (primary, fallback) in tasks.items():
        if task in r:
            val = r[task].get(primary) or (r[task].get(fallback) if fallback else None)
            if val is not None:
                scores.append(val)
    return sum(scores) / len(scores) if scores else None

checks = [
    ("solar_B_g64",             "S4-3-1 SOLAR B g64"),
    ("solar_A_no_desc_act",     "S4-3-2 SOLAR A no_desc_act"),
    ("eeve_10b_fp16",           "S4-3-3 EEVE FP16"),
    ("exaone35_7b_C_v3_exaone", "S4-4   EXAONE35 C_v3_exaone"),
]
for fname, label in checks:
    p = f"{results_dir}/eval_kobest_{fname}.json"
    v = kobest_agg(p)
    status = f"{v:.4f}" if v is not None else "MISSING"
    print(f"  {label}: {status}")
PYEOF

log "======================================================"
log "GPU 1 Sprint 4 파이프라인 완료"
log "======================================================"
