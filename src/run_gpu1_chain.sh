#!/bin/bash
# GPU1 체인 실행: kmmlu baseline 완료 후 DBAR-v1 자동 시작
#
# 실행: docker exec -d llm-dev bash /home/choihyun/workspace/src/run_gpu1_chain.sh

set -euo pipefail

WORKSPACE="/home/choihyun/workspace"
RESULTS="${WORKSPACE}/results"
SRC="${WORKSPACE}/src"

export CUDA_VISIBLE_DEVICES=1
export HOME="/home/choihyun"
export HF_HOME="${WORKSPACE}/.cache/huggingface"
export XDG_CACHE_HOME="${WORKSPACE}/.cache"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

log() { echo "[$(date '+%H:%M')] $*"; }

log "GPU1 체인 시작 (kmmlu baseline → DBAR-v1)"

# ── Step 1: kmmlu baseline (A / B / C_v3) ────────────────────────────
log "kmmlu baseline 실행..."
bash "${SRC}/run_eval_solar_kmmlu_baseline.sh"

# ── Step 2: baseline 결과 확인 ────────────────────────────────────────
log "baseline 결과 확인"
for cond in A B C_v3; do
    if compgen -G "${RESULTS}/eval_kmmlu_solar_cond_${cond}*.json" > /dev/null 2>&1; then
        log "  ${cond}: ✅"
    else
        log "  ${cond}: ❌ — 없음"
    fi
done

# ── Step 3: DBAR-v1 λ=0.3 / 0.5 / 0.7 ───────────────────────────────
log "DBAR-v1 실험 시작 (GPU1)..."
bash "${SRC}/run_exp_dbar_v1.sh"

log "GPU1 체인 완료"
