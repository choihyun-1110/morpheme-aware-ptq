#!/bin/bash
# GPU 0 Phase 2 v2 — optimum.gptq + gptqmodel 사용 (auto_gptq position_embeddings 호환성 문제 우회)
# activation 분석 기반 새 양자화 실험:
#   1. SOLAR C_v3 group_size=64
#   2. SOLAR A    group_size=64 (대조군)
#   3. SOLAR C_v3 + SmoothScale 전처리 → GPTQ 4bit
#   4. SOLAR A    AWQ 4bit
#   5. SOLAR C_v3 AWQ 4bit
#   6. SOLAR C_v3 desc_act=False

set -euo pipefail

WORKSPACE="/home/choihyun/workspace"
RESULTS="${WORKSPACE}/results"
QUANT_MODELS="${WORKSPACE}/quantized_models"
SRC="${WORKSPACE}/src"
MODEL="upstage/SOLAR-10.7B-Instruct-v1.0"

export CUDA_VISIBLE_DEVICES=0
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
    log "KoBEST 평가: ${out_name}"
    "${LM_EVAL}" --model hf \
        --model_args "pretrained=${model_path},trust_remote_code=True,dtype=float16" \
        --tasks kobest \
        --device cuda:0 \
        --batch_size 4 \
        --output_path "${RESULTS}/eval_kobest_${out_name}.json" \
        --log_samples \
        2>&1 | tee "${RESULTS}/eval_kobest_${out_name}.log"
    log "완료: ${out_name}"
}

quant_solar() {
    local calib_path="$1"
    local out_dir="$2"
    local log_name="$3"
    local extra_args="${4:-}"
    if [ -f "${out_dir}/config.json" ]; then
        log "이미 양자화됨 (스킵): ${log_name}"
        return 0
    fi
    log "SOLAR 양자화 (optimum.gptq): ${log_name}"
    "${PYTHON}" "${SRC}/run_quant_optimum.py" \
        --model "${MODEL}" \
        --calib-path "${calib_path}" \
        --out-dir "${out_dir}" \
        ${extra_args} \
        2>&1 | tee "${RESULTS}/quant_${log_name}.log"
    log "양자화 완료: ${log_name}"
}

CALIB_A="${RESULTS}/calibration_set_A.json"
CALIB_Cv3="${RESULTS}/calibration_set_C_v3_upstage_SOLAR-10.7B-Instruct-v1.0.json"

# ─────────────────────────────────────────────────────────
# 실험 1: group_size=64
# ─────────────────────────────────────────────────────────
log "=== [EXP1] SOLAR C_v3 group_size=64 ==="
quant_solar "${CALIB_Cv3}" "${QUANT_MODELS}/solar_C_v3_g64" "solar_C_v3_g64" "--group-size 64"
eval_kobest "${QUANT_MODELS}/solar_C_v3_g64" "solar_C_v3_g64"

log "=== [EXP1b] SOLAR A group_size=64 (대조군) ==="
quant_solar "${CALIB_A}" "${QUANT_MODELS}/solar_A_g64" "solar_A_g64" "--group-size 64"
eval_kobest "${QUANT_MODELS}/solar_A_g64" "solar_A_g64"

# ─────────────────────────────────────────────────────────
# 실험 2: SmoothScale + GPTQ
# ─────────────────────────────────────────────────────────
log "=== [EXP2] SmoothScale + GPTQ C_v3 ==="
SMOOTH_DIR="${QUANT_MODELS}/solar_smooth_scaled"
"${PYTHON}" "${SRC}/smooth_scale.py" \
    --model "${MODEL}" \
    --calib-path "${CALIB_Cv3}" \
    --out-dir "${SMOOTH_DIR}" \
    --alpha 0.5 \
    2>&1 | tee "${RESULTS}/smooth_scale_C_v3.log"

# SmoothScale 후 변환된 모델(SMOOTH_DIR)을 GPTQ 양자화 (원본 MODEL 아님)
if [ -f "${SMOOTH_DIR}/config.json" ]; then
    log "SmoothScale 모델 발견: ${SMOOTH_DIR}"
    if [ ! -f "${QUANT_MODELS}/solar_smooth_C_v3/config.json" ]; then
        log "SOLAR SmoothScale+GPTQ C_v3 양자화"
        "${PYTHON}" "${SRC}/run_quant_optimum.py" \
            --model "${SMOOTH_DIR}" \
            --calib-path "${CALIB_Cv3}" \
            --out-dir "${QUANT_MODELS}/solar_smooth_C_v3" \
            2>&1 | tee "${RESULTS}/quant_solar_smooth_C_v3.log"
    else
        log "이미 양자화됨 (스킵): solar_smooth_C_v3"
    fi
    eval_kobest "${QUANT_MODELS}/solar_smooth_C_v3" "solar_smooth_C_v3"

    if [ ! -f "${QUANT_MODELS}/solar_smooth_A/config.json" ]; then
        log "SOLAR SmoothScale+GPTQ A 양자화"
        "${PYTHON}" "${SRC}/run_quant_optimum.py" \
            --model "${SMOOTH_DIR}" \
            --calib-path "${CALIB_A}" \
            --out-dir "${QUANT_MODELS}/solar_smooth_A" \
            2>&1 | tee "${RESULTS}/quant_solar_smooth_A.log"
    else
        log "이미 양자화됨 (스킵): solar_smooth_A"
    fi
    eval_kobest "${QUANT_MODELS}/solar_smooth_A" "solar_smooth_A"
else
    log "경고: SmoothScale 실패 (SMOOTH_DIR 없음) — EXP2 스킵"
fi

# ─────────────────────────────────────────────────────────
# 실험 3: AWQ
# ─────────────────────────────────────────────────────────
if "${PYTHON}" -c "import awq" 2>/dev/null; then
    log "=== [EXP3] AWQ A ==="
    "${PYTHON}" "${SRC}/run_quant_awq.py" \
        --model "${MODEL}" --calib-path "${CALIB_A}" \
        --out-dir "${QUANT_MODELS}/solar_awq_A" \
        2>&1 | tee "${RESULTS}/quant_solar_awq_A.log"
    eval_kobest "${QUANT_MODELS}/solar_awq_A" "solar_awq_A"

    log "=== [EXP3b] AWQ C_v3 ==="
    "${PYTHON}" "${SRC}/run_quant_awq.py" \
        --model "${MODEL}" --calib-path "${CALIB_Cv3}" \
        --out-dir "${QUANT_MODELS}/solar_awq_C_v3" \
        2>&1 | tee "${RESULTS}/quant_solar_awq_C_v3.log"
    eval_kobest "${QUANT_MODELS}/solar_awq_C_v3" "solar_awq_C_v3"
else
    log "=== [EXP3] AutoAWQ 미설치 — 스킵 ==="
fi

# ─────────────────────────────────────────────────────────
# 실험 4: desc_act=False
# ─────────────────────────────────────────────────────────
log "=== [EXP4] SOLAR C_v3 desc_act=False ==="
quant_solar "${CALIB_Cv3}" "${QUANT_MODELS}/solar_C_v3_no_desc_act" "solar_C_v3_no_desc_act" "--no-desc-act"
eval_kobest "${QUANT_MODELS}/solar_C_v3_no_desc_act" "solar_C_v3_no_desc_act"

log "=== GPU 0 Phase 2 모든 실험 완료 ==="
