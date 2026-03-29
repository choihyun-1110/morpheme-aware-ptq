#!/bin/bash
# GPU 0 실험 파이프라인
# 1. Qwen2 FP16 베이스라인 C-Eval
# 2. Qwen2 GPTQ (A/C_v3/C_zh) C-Eval
# 3. [완료 후] EEVE C_v3_eeve calibration 생성 → 양자화 → KoBEST

set -euo pipefail

WORKSPACE="/home/choihyun/workspace"
RESULTS="${WORKSPACE}/results"
QUANT_MODELS="${WORKSPACE}/quantized_models"
SRC="${WORKSPACE}/src"

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

log() { echo "[$(date '+%H:%M')] $*"; }

# ── 헬퍼: GPTQ 모델 eval ──────────────────────────────────
eval_ceval() {
    local model_path="$1"
    local out_name="$2"
    log "C-Eval 평가 시작: ${out_name}"
    local model_args="pretrained=${model_path},trust_remote_code=True"
    if [ -f "${model_path}/quantize_config.json" ]; then
        local sft
        sft="$(find "${model_path}" -maxdepth 1 -name '*.safetensors' | head -n 1)"
        model_args="${model_args},autogptq=$(basename "${sft}")"
    fi
    "${LM_EVAL}" --model hf \
        --model_args "${model_args}" \
        --tasks ceval-valid \
        --device cuda:0 \
        --batch_size 4 \
        --output_path "${RESULTS}/eval_ceval_${out_name}.json" \
        --log_samples \
        2>&1 | tee "${RESULTS}/eval_ceval_${out_name}.log"
    log "C-Eval 평가 완료: ${out_name}"
}

# ── 1. Qwen2 FP16 베이스라인 C-Eval ──────────────────────
log "=== [1/6] Qwen2 FP16 베이스라인 C-Eval ==="
eval_ceval "Qwen/Qwen2-7B-Instruct" "qwen2_fp16"

# ── 2. Qwen2 GPTQ A C-Eval ───────────────────────────────
log "=== [2/6] Qwen2 GPTQ A C-Eval ==="
eval_ceval "${QUANT_MODELS}/qwen2_7b_cond_A" "qwen2_gptq_A"

# ── 3. Qwen2 GPTQ C_v3 C-Eval ────────────────────────────
log "=== [3/6] Qwen2 GPTQ C_v3 C-Eval ==="
eval_ceval "${QUANT_MODELS}/qwen2_7b_cond_C_v3" "qwen2_gptq_C_v3"

# ── 4. Qwen2 GPTQ C_zh C-Eval ────────────────────────────
log "=== [4/6] Qwen2 GPTQ C_zh C-Eval ==="
eval_ceval "${QUANT_MODELS}/qwen2_7b_cond_C_zh" "qwen2_gptq_C_zh"

log "=== Qwen2 C-Eval 모든 평가 완료 ==="

# ── 5. EEVE C_v3_eeve calibration 생성 ────────────────────
log "=== [5/6] EEVE tokenizer C_v3_eeve calibration 생성 ==="
python "${SRC}/build_calibration.py" \
    --condition C \
    --model "yanolja/EEVE-Korean-Instruct-10.8B-v1.0" \
    --n-sentences 128 \
    --n-candidates 100000 \
    --c-min-ko-ratio 0.7 \
    --suffix "v3_eeve" \
    2>&1 | tee "${RESULTS}/build_C_v3_eeve.log"

EEVE_CALIB="${RESULTS}/calibration_set_C_v3_eeve_yanolja_EEVE-Korean-Instruct-10.8B-v1.0.json"
log "calibration 생성 완료: ${EEVE_CALIB}"

# ── 6. EEVE C_v3_eeve 양자화 ──────────────────────────────
log "=== [6/6] EEVE C_v3_eeve 양자화 ==="
python "${SRC}/run_quant.py" \
    --model "yanolja/EEVE-Korean-Instruct-10.8B-v1.0" \
    --calib "C_v3_eeve" \
    --calib-path "${EEVE_CALIB}" \
    --out-dir "${QUANT_MODELS}/eeve_10b_cond_C_v3_eeve" \
    2>&1 | tee "${RESULTS}/quant_eeve_10b_C_v3_eeve.log"
log "EEVE C_v3_eeve 양자화 완료"

# ── 7. EEVE C_v3_eeve KoBEST 평가 ─────────────────────────
log "=== [7/7] EEVE C_v3_eeve KoBEST 평가 ==="
EEVE_GPTQ="${QUANT_MODELS}/eeve_10b_cond_C_v3_eeve"
eeve_args="pretrained=${EEVE_GPTQ},trust_remote_code=True"
eeve_sft="$(find "${EEVE_GPTQ}" -maxdepth 1 -name '*.safetensors' | head -n 1)"
eeve_args="${eeve_args},autogptq=$(basename "${eeve_sft}")"

"${LM_EVAL}" --model hf \
    --model_args "${eeve_args}" \
    --tasks kobest \
    --device cuda:0 \
    --batch_size 4 \
    --output_path "${RESULTS}/eval_kobest_eeve_10b_C_v3_eeve.json" \
    --log_samples \
    2>&1 | tee "${RESULTS}/eval_kobest_eeve_10b_C_v3_eeve.log"
log "EEVE C_v3_eeve KoBEST 평가 완료"

log "=== GPU 0 모든 실험 완료 ==="
