#!/bin/bash
# GPU 1 실험 파이프라인 v2 — GPU 최대 활용
# 순서: EEVE C_v3_eeve → EXAONE35 A/B/C_v3
# (EXAONE35 모델은 백그라운드 다운로드로 병행)

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

eval_kobest() {
    local model_path="$1"
    local out_name="$2"
    log "KoBEST 평가: ${out_name}"
    local model_args="pretrained=${model_path},trust_remote_code=True"
    if [ -f "${model_path}/quantize_config.json" ]; then
        local sft
        sft="$(find "${model_path}" -maxdepth 1 -name '*.safetensors' | head -n 1)"
        model_args="${model_args},autogptq=$(basename "${sft}")"
    fi
    "${LM_EVAL}" --model hf \
        --model_args "${model_args}" \
        --tasks kobest \
        --device cuda:0 \
        --batch_size 4 \
        --output_path "${RESULTS}/eval_kobest_${out_name}.json" \
        --log_samples \
        2>&1 | tee "${RESULTS}/eval_kobest_${out_name}.log"
    log "KoBEST 완료: ${out_name}"
}

eval_kobest_optimum() {
    local model_path="$1"
    local out_name="$2"
    log "KoBEST 평가 (optimum): ${out_name}"
    # optimum GPTQ 형식은 autogptq= 불필요
    local model_args="pretrained=${model_path},trust_remote_code=True"
    "${LM_EVAL}" --model hf \
        --model_args "${model_args}" \
        --tasks kobest \
        --device cuda:0 \
        --batch_size 4 \
        --output_path "${RESULTS}/eval_kobest_${out_name}.json" \
        --log_samples \
        2>&1 | tee "${RESULTS}/eval_kobest_${out_name}.log"
    log "KoBEST 완료: ${out_name}"
}

# ── 백그라운드: EXAONE35 모델 다운로드 ────────────────────
log "=== EXAONE35 모델 다운로드 백그라운드 시작 ==="
python -c "
import os
os.environ['HF_HOME'] = '${HF_HOME}'
os.environ['HOME'] = '/home/choihyun'
from huggingface_hub import snapshot_download
print('EXAONE-3.5-7.8B 다운로드 시작...')
path = snapshot_download('LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct')
print('다운로드 완료:', path)
" > "${RESULTS}/exaone35_download.log" 2>&1 &
DOWNLOAD_PID=$!
log "EXAONE35 다운로드 PID: ${DOWNLOAD_PID}"

# ── 1. EEVE C_v3_eeve calibration 생성 (CPU) ──────────────
log "=== [1/7] EEVE tokenizer C_v3_eeve calibration 생성 ==="
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

# ── 2. EEVE C_v3_eeve 양자화 ──────────────────────────────
log "=== [2/7] EEVE C_v3_eeve 양자화 ==="
python "${SRC}/run_quant.py" \
    --model "yanolja/EEVE-Korean-Instruct-10.8B-v1.0" \
    --calib "C_v3_eeve" \
    --calib-path "${EEVE_CALIB}" \
    --out-dir "${QUANT_MODELS}/eeve_10b_cond_C_v3_eeve" \
    2>&1 | tee "${RESULTS}/quant_eeve_10b_C_v3_eeve.log"
log "EEVE C_v3_eeve 양자화 완료"

# ── 3. EEVE C_v3_eeve KoBEST 평가 ─────────────────────────
log "=== [3/7] EEVE C_v3_eeve KoBEST 평가 ==="
eval_kobest "${QUANT_MODELS}/eeve_10b_cond_C_v3_eeve" "eeve_10b_C_v3_eeve"

# ── EXAONE35 다운로드 완료 대기 ───────────────────────────
log "=== EXAONE35 다운로드 완료 대기 중 (PID: ${DOWNLOAD_PID}) ==="
wait "${DOWNLOAD_PID}" && log "EXAONE35 다운로드 완료!" || log "EXAONE35 다운로드 경고 (이미 완료되었을 수 있음)"

# ── 4. EXAONE35 × A 양자화 + KoBEST ──────────────────────
log "=== [4/7] EXAONE35 조건 A 양자화 ==="
python "${SRC}/run_quant_optimum.py" \
    --model "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct" \
    --calib-path "${RESULTS}/calibration_set_A.json" \
    --out-dir "${QUANT_MODELS}/exaone35_7b_cond_A" \
    2>&1 | tee "${RESULTS}/quant_exaone35_7b_A.log"
log "=== [5/7] EXAONE35 조건 A KoBEST 평가 ==="
eval_kobest_optimum "${QUANT_MODELS}/exaone35_7b_cond_A" "exaone35_7b_A"

# ── 5. EXAONE35 × B 양자화 + KoBEST ──────────────────────
log "=== [5/7] EXAONE35 조건 B 양자화 ==="
python "${SRC}/run_quant_optimum.py" \
    --model "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct" \
    --calib-path "${RESULTS}/calibration_set_B.json" \
    --out-dir "${QUANT_MODELS}/exaone35_7b_cond_B" \
    2>&1 | tee "${RESULTS}/quant_exaone35_7b_B.log"
log "=== [6/7] EXAONE35 조건 B KoBEST 평가 ==="
eval_kobest_optimum "${QUANT_MODELS}/exaone35_7b_cond_B" "exaone35_7b_B"

# ── 6. EXAONE35 × C_v3 양자화 + KoBEST ───────────────────
log "=== [6/7] EXAONE35 조건 C_v3 양자화 ==="
python "${SRC}/run_quant_optimum.py" \
    --model "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct" \
    --calib-path "${RESULTS}/calibration_set_C_v3_upstage_SOLAR-10.7B-Instruct-v1.0.json" \
    --out-dir "${QUANT_MODELS}/exaone35_7b_cond_C_v3" \
    2>&1 | tee "${RESULTS}/quant_exaone35_7b_C_v3.log"
log "=== [7/7] EXAONE35 조건 C_v3 KoBEST 평가 ==="
eval_kobest_optimum "${QUANT_MODELS}/exaone35_7b_cond_C_v3" "exaone35_7b_C_v3"

log "=== GPU 1 모든 실험 완료 ==="
