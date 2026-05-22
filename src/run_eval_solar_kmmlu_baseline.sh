#!/bin/bash
# SOLAR kmmlu baseline 측정 (기존 양자화 모델 재사용)
#
# 목적: A/B/C_v3 단일 bank kmmlu 기준값 확보
#       → hybrid balanced_retention 판정 및 DBAR-v1 비교 기준
#
# 전제: 기존 양자화 모델 존재
#   - quantized_models/SOLAR_10.7B_4bit_cond_A
#   - quantized_models/SOLAR_10.7B_4bit_cond_B
#   - quantized_models/SOLAR_10.7B_4bit_cond_C_v3
#
# 실행: docker exec llm-dev bash /home/choihyun/workspace/src/run_eval_solar_kmmlu_baseline.sh

set -euo pipefail

WORKSPACE="/home/choihyun/workspace"
RESULTS="${WORKSPACE}/results"
QUANT_MODELS="${WORKSPACE}/quantized_models"

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
GPU_DEVICE="cuda:${CUDA_VISIBLE_DEVICES:-1}"

log() { echo "[$(date '+%H:%M')] $*"; }

eval_kmmlu() {
    local model_path="$1"; local label="$2"
    local out="${RESULTS}/eval_kmmlu_solar_cond_${label}.json"
    if [ -f "${out}" ]; then
        log "스킵: ${label} 이미 완료"
        return
    fi
    log "kmmlu: ${label} (${model_path})"
    "${LM_EVAL}" --model hf \
        --model_args "pretrained=${model_path},trust_remote_code=True,dtype=float16" \
        --tasks kmmlu --device "${GPU_DEVICE}" --batch_size 4 \
        --output_path "${out}" --log_samples \
        2>&1 | tee "${RESULTS}/eval_kmmlu_solar_cond_${label}.log"
    log "${label} kmmlu 완료"
}

log "======================================================"
log "SOLAR kmmlu baseline (A / B / C_v3) — GPU${CUDA_VISIBLE_DEVICES:-1}"
log "======================================================"

eval_kmmlu "${QUANT_MODELS}/SOLAR_10.7B_4bit_cond_A"    "A"
eval_kmmlu "${QUANT_MODELS}/SOLAR_10.7B_4bit_cond_B"    "B"
eval_kmmlu "${QUANT_MODELS}/SOLAR_10.7B_4bit_cond_C_v3" "C_v3"

log "======================================================"
log "SOLAR kmmlu baseline 완료"
log "======================================================"

# 결과 요약
/opt/conda/envs/llm-quant/bin/python3.11 - <<'PYEOF'
import json, glob, os

R = "/home/choihyun/workspace/results"

def agg_kmmlu(path):
    if not path or not os.path.exists(path): return None
    with open(path) as f: d = json.load(f)
    r = d.get("results", {})
    scores = [v.get("acc,none") for k,v in r.items() if "acc,none" in v]
    return round(sum(scores)/len(scores), 4) if scores else None

labels = ["A", "B", "C_v3"]
print(f"\n{'조건':<10} {'kmmlu':>8}")
print("-" * 20)
for label in labels:
    path = f"{R}/eval_kmmlu_solar_cond_{label}.json"
    val = agg_kmmlu(path)
    print(f"{label:<10} {f'{val:.4f}' if val else '—':>8}")
PYEOF
