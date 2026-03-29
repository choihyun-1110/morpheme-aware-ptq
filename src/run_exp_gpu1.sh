#!/bin/bash
# GPU 1 실험 파이프라인
# EXAONE-3.5-7.8B 양자화 3조건 (A / B / C_v3) + KoBEST 평가

set -euo pipefail

WORKSPACE="/home/choihyun/workspace"
RESULTS="${WORKSPACE}/results"
QUANT_MODELS="${WORKSPACE}/quantized_models"
SRC="${WORKSPACE}/src"

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

log() { echo "[$(date '+%H:%M')] $*"; }

# ── 헬퍼: 양자화 + KoBEST 평가 ───────────────────────────
quant_and_eval() {
    local model_id="$1"
    local calib_label="$2"
    local calib_path="$3"
    local out_dir="${QUANT_MODELS}/$4"
    local log_tag="$5"

    log "양자화 시작 (optimum.gptq): ${log_tag}"
    python "${SRC}/run_quant_optimum.py" \
        --model "${model_id}" \
        --calib-path "${calib_path}" \
        --out-dir "${out_dir}" \
        2>&1 | tee "${RESULTS}/quant_${log_tag}.log"
    log "양자화 완료: ${log_tag}"

    log "KoBEST 평가 시작: ${log_tag}"
    # optimum GPTQ 형식은 transformers가 자동 인식하므로 autogptq= 불필요
    local model_args="pretrained=${out_dir},trust_remote_code=True"
    "${LM_EVAL}" --model hf \
        --model_args "${model_args}" \
        --tasks kobest \
        --device cuda:0 \
        --batch_size 4 \
        --output_path "${RESULTS}/eval_kobest_${log_tag}.json" \
        --log_samples \
        2>&1 | tee "${RESULTS}/eval_kobest_${log_tag}.log"
    log "KoBEST 평가 완료: ${log_tag}"
}

MODEL="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"

# ── 1. EXAONE35 × A ───────────────────────────────────────
log "=== [1/3] EXAONE35 조건 A ==="
quant_and_eval \
    "${MODEL}" \
    "A" \
    "${RESULTS}/calibration_set_A.json" \
    "exaone35_7b_cond_A" \
    "exaone35_7b_A"

# ── 2. EXAONE35 × B ───────────────────────────────────────
log "=== [2/3] EXAONE35 조건 B ==="
quant_and_eval \
    "${MODEL}" \
    "B" \
    "${RESULTS}/calibration_set_B.json" \
    "exaone35_7b_cond_B" \
    "exaone35_7b_B"

# ── 3. EXAONE35 × C_v3 ────────────────────────────────────
log "=== [3/3] EXAONE35 조건 C_v3 ==="
quant_and_eval \
    "${MODEL}" \
    "C_v3" \
    "${RESULTS}/calibration_set_C_v3_upstage_SOLAR-10.7B-Instruct-v1.0.json" \
    "exaone35_7b_cond_C_v3" \
    "exaone35_7b_C_v3"

log "=== GPU 1 모든 실험 완료 ==="
