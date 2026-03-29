#!/bin/bash
# GPU 1 실험 파이프라인 v3
# 순서:
#   1. EXAONE35 모델 다운로드 완료
#   2. EXAONE35 패치 재적용
#   3. EXAONE35 A/B/C_v3 양자화 + KoBEST
#   4. EEVE C_v3_eeve 양자화 + KoBEST (GPU 1에서 직접 수행)

set -euo pipefail

WORKSPACE="/home/choihyun/workspace"
RESULTS="${WORKSPACE}/results"
QUANT_MODELS="${WORKSPACE}/quantized_models"
SRC="${WORKSPACE}/src"

EXAONE_SNAPSHOT="${WORKSPACE}/.cache/huggingface/hub/models--LGAI-EXAONE--EXAONE-3.5-7.8B-Instruct/snapshots/553ea250b9a5317231459279d5847d6cf955b9aa"

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

# ── 1. EXAONE35 다운로드 완료 ────────────────────────────
log "=== [1] EXAONE-3.5-7.8B 다운로드 완료 대기 ==="
"${PYTHON}" - <<'PYEOF'
import os
os.environ['HF_HOME'] = '/home/choihyun/workspace/.cache/huggingface'
os.environ['HOME'] = '/home/choihyun'
from huggingface_hub import snapshot_download
print('EXAONE-3.5-7.8B-Instruct 다운로드 (재시도/완료)...')
path = snapshot_download(
    'LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct',
    ignore_patterns=["*.py"],  # Python 파일은 이미 패치됨 — 덮어쓰기 방지
)
print(f'다운로드 완료: {path}')
PYEOF
log "EXAONE35 다운로드 완료"

# ── 2. 패치 재적용 (Python 파일이 갱신된 경우 대비) ──────
log "=== [2] EXAONE35 호환성 패치 적용 ==="
"${PYTHON}" "${SRC}/patch_exaone.py"

eval_kobest_optimum() {
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
    log "KoBEST 완료: ${out_name}"
}

quant_exaone() {
    local calib_path="$1"
    local out_dir="$2"
    local log_name="$3"
    # 이미 저장된 경우 스킵
    if [ -f "${out_dir}/config.json" ]; then
        log "이미 양자화됨 (스킵): ${log_name}"
        return 0
    fi
    log "EXAONE35 양자화: ${log_name}"
    # 로컬 경로 사용 → HF 재다운로드 없음
    "${PYTHON}" "${SRC}/run_quant_optimum.py" \
        --model "${EXAONE_SNAPSHOT}" \
        --calib-path "${calib_path}" \
        --out-dir "${out_dir}" \
        2>&1 | tee "${RESULTS}/quant_${log_name}.log"
    log "양자화 완료: ${log_name}"
}

# ── 3. EXAONE35 × A ───────────────────────────────────────
log "=== [3] EXAONE35 조건 A 양자화 ==="
quant_exaone \
    "${RESULTS}/calibration_set_A.json" \
    "${QUANT_MODELS}/exaone35_7b_cond_A" \
    "exaone35_7b_A"
eval_kobest_optimum "${QUANT_MODELS}/exaone35_7b_cond_A" "exaone35_7b_A"

# ── 4. EXAONE35 × B ───────────────────────────────────────
log "=== [4] EXAONE35 조건 B 양자화 ==="
quant_exaone \
    "${RESULTS}/calibration_set_B.json" \
    "${QUANT_MODELS}/exaone35_7b_cond_B" \
    "exaone35_7b_B"
eval_kobest_optimum "${QUANT_MODELS}/exaone35_7b_cond_B" "exaone35_7b_B"

# ── 5. EXAONE35 × C_v3 ────────────────────────────────────
log "=== [5] EXAONE35 조건 C_v3 양자화 ==="
quant_exaone \
    "${RESULTS}/calibration_set_C_v3_upstage_SOLAR-10.7B-Instruct-v1.0.json" \
    "${QUANT_MODELS}/exaone35_7b_cond_C_v3" \
    "exaone35_7b_C_v3"
eval_kobest_optimum "${QUANT_MODELS}/exaone35_7b_cond_C_v3" "exaone35_7b_C_v3"

# ── 6. EEVE C_v3_eeve 양자화 + KoBEST (GPU 1에서 직접 수행) ──
EEVE_MODEL="yanolja/EEVE-Korean-Instruct-10.8B-v1.0"
CALIB_EEVE="${RESULTS}/calibration_set_C_v3_eeve_yanolja_EEVE-Korean-Instruct-10.8B-v1.0.json"
EEVE_QUANT="${QUANT_MODELS}/eeve_10b_cond_C_v3_eeve"

log "=== [6] EEVE C_v3_eeve 양자화 ==="
"${PYTHON}" "${SRC}/run_quant_optimum.py" \
    --model "${EEVE_MODEL}" \
    --calib-path "${CALIB_EEVE}" \
    --out-dir "${EEVE_QUANT}" \
    2>&1 | tee "${RESULTS}/quant_eeve_10b_C_v3_eeve.log"
log "EEVE C_v3_eeve 양자화 완료"

log "=== [6b] EEVE C_v3_eeve KoBEST 평가 ==="
eval_kobest_optimum "${EEVE_QUANT}" "eeve_10b_C_v3_eeve"

# ── 7. EXAONE35 FP16 KoBEST 베이스라인 (보존율 계산용) ──────
log "=== [7] EXAONE35 FP16 KoBEST 베이스라인 ==="
"${LM_EVAL}" --model hf \
    --model_args "pretrained=${EXAONE_SNAPSHOT},trust_remote_code=True,dtype=float16" \
    --tasks kobest \
    --device cuda:0 \
    --batch_size 4 \
    --output_path "${RESULTS}/eval_kobest_exaone35_7b_fp16.json" \
    --log_samples \
    2>&1 | tee "${RESULTS}/eval_kobest_exaone35_7b_fp16.log"
log "EXAONE35 FP16 KoBEST 완료"

log "=== GPU 1 v3 모든 실험 완료 ==="
