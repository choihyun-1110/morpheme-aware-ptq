#!/bin/bash
# S4-15: Llama-3-8B (영어 베이스) + C_en_v3 실험
#
# 목적: "순수 영어 사전학습 모델 + 영어 다양성 calibration" 검증
#   - Llama3-Ko는 한국어 지속학습 모델 → 영어 대표 케이스로 부적합
#   - Meta-Llama-3-8B: 순수 영어 사전학습 → 영어 calibration이 최적이어야 함
#   - A   = 랜덤 Wikitext-2 (표준 GPTQ 기본값)
#   - C_en_v3 = 다양성 최적화 Wikitext-2 (우리 greedy 알고리즘 영어 버전)
#
# 가설:
#   C_en_v3 > A: 다양성 원칙이 영어에서도 유효 → 언어 독립적 일반 원리 확립
#   (Llama3-Ko 실험의 C_en_v3 ≈ A는 모델 혼재 문제였을 가능성)
#
# 평가: MMLU (57과목, 영어 지식 이해) — 한국어 모델의 kmmlu에 대응
#
# 실행: docker exec llm-dev bash /home/choihyun/workspace/src/run_exp_llama3_base_mmlu.sh

set -euo pipefail

WORKSPACE="/home/choihyun/workspace"
RESULTS="${WORKSPACE}/results"
QUANT_MODELS="${WORKSPACE}/quantized_models"
SRC="${WORKSPACE}/src"
MODEL_ID="meta-llama/Meta-Llama-3-8B"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HOME="/home/choihyun"
export HF_HOME="${WORKSPACE}/.cache/huggingface"
export XDG_CACHE_HOME="${WORKSPACE}/.cache"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_OFFLINE=0
export HF_TOKEN="$(cat ${HF_HOME}/token 2>/dev/null || echo '')"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"

cd "${WORKSPACE}"
source /opt/conda/etc/profile.d/conda.sh
conda activate llm-quant

LM_EVAL="/opt/conda/envs/llm-quant/bin/lm_eval"
PYTHON="/opt/conda/envs/llm-quant/bin/python3.11"
GPU_DEVICE="cuda:${CUDA_VISIBLE_DEVICES:-0}"

log() { echo "[$(date '+%H:%M')] $*"; }

result_exists() {
    compgen -G "${RESULTS}/eval_mmlu_${1}*.json" > /dev/null 2>&1
}

eval_mmlu() {
    local model_path="$1"
    local out_name="$2"
    log "MMLU 평가: ${out_name}"
    "${LM_EVAL}" --model hf \
        --model_args "pretrained=${model_path},trust_remote_code=True,dtype=float16" \
        --tasks mmlu \
        --device "${GPU_DEVICE}" \
        --batch_size 4 \
        --output_path "${RESULTS}/eval_mmlu_${out_name}.json" \
        --log_samples \
        2>&1 | tee "${RESULTS}/eval_mmlu_${out_name}.log"
    log "완료: ${out_name}"
}

log "======================================================"
log "S4-15: Llama-3-8B (영어 베이스) MMLU 실험"
log "GPU: ${CUDA_VISIBLE_DEVICES:-0} / Model: ${MODEL_ID}"
log "======================================================"

# ─────────────────────────────────────────────────────────
# Step 0: C_en_v3 calibration 확인 또는 생성
# (이미 Llama3-Ko 실험에서 생성된 파일 재사용 가능)
# ─────────────────────────────────────────────────────────
CALIB_A=$(ls "${RESULTS}"/calibration_set_A*.json 2>/dev/null | head -1 || true)
CALIB_EN_V3=$(ls "${RESULTS}"/calibration_set_C_en_v3*.json 2>/dev/null | head -1 || true)

if [ -z "${CALIB_EN_V3}" ]; then
    log "C_en_v3 calibration 생성 중 (Wikitext-2 다양성 선별)..."
    "${PYTHON}" "${SRC}/build_calibration_en.py" \
        --model "${MODEL_ID}" \
        --n-sentences 128 \
        --n-candidates 50000 \
        --alpha 0.3 \
        --min-words 8 \
        --min-tokens 20 \
        --suffix "en_v3" \
        2>&1 | tee "${RESULTS}/build_C_en_v3_llama3base.log"
    CALIB_EN_V3=$(ls "${RESULTS}"/calibration_set_C_en_v3*.json 2>/dev/null | head -1 || true)
fi
log "C_en_v3 calibration: ${CALIB_EN_V3}"

if [ -z "${CALIB_A}" ]; then
    log "ERROR: calibration_set_A not found. 먼저 build_calibration.py 실행 필요."
    exit 1
fi
log "A calibration: ${CALIB_A}"

# ─────────────────────────────────────────────────────────
# Step 1: FP16 베이스라인 MMLU
# ─────────────────────────────────────────────────────────
if ! result_exists "llama3_base_8b_fp16"; then
    log "=== FP16 베이스라인 ==="
    eval_mmlu "${MODEL_ID}" "llama3_base_8b_fp16"
else
    log "스킵: llama3_base_8b_fp16 이미 완료"
fi

# ─────────────────────────────────────────────────────────
# Step 2: A 조건 양자화 + MMLU
# ─────────────────────────────────────────────────────────
if ! result_exists "llama3_base_8b_A"; then
    log "=== A (랜덤 영어) 양자화 ==="
    "${PYTHON}" "${SRC}/run_quant_optimum.py" \
        --model "${MODEL_ID}" \
        --calib-path "${CALIB_A}" \
        --out-dir "${QUANT_MODELS}/llama3_base_8b_cond_A" \
        2>&1 | tee "${RESULTS}/quant_llama3base_A.log"
    eval_mmlu "${QUANT_MODELS}/llama3_base_8b_cond_A" "llama3_base_8b_A"
    rm -rf "${QUANT_MODELS}/llama3_base_8b_cond_A"
else
    log "스킵: llama3_base_8b_A 이미 완료"
fi

# ─────────────────────────────────────────────────────────
# Step 3: C_en_v3 조건 양자화 + MMLU
# ─────────────────────────────────────────────────────────
if ! result_exists "llama3_base_8b_C_en_v3"; then
    log "=== C_en_v3 (다양성 영어) 양자화 ==="
    "${PYTHON}" "${SRC}/run_quant_optimum.py" \
        --model "${MODEL_ID}" \
        --calib-path "${CALIB_EN_V3}" \
        --out-dir "${QUANT_MODELS}/llama3_base_8b_cond_C_en_v3" \
        2>&1 | tee "${RESULTS}/quant_llama3base_C_en_v3.log"
    eval_mmlu "${QUANT_MODELS}/llama3_base_8b_cond_C_en_v3" "llama3_base_8b_C_en_v3"
    rm -rf "${QUANT_MODELS}/llama3_base_8b_cond_C_en_v3"
else
    log "스킵: llama3_base_8b_C_en_v3 이미 완료"
fi

# ─────────────────────────────────────────────────────────
# 결과 요약
# ─────────────────────────────────────────────────────────
log "======================================================"
log "결과 요약: Llama-3-8B MMLU 비교"
log "======================================================"

"${PYTHON}" - <<'PYEOF'
import json, glob, os, sys

R = "/home/choihyun/workspace/results"

def get_mmlu_avg(path):
    if not path or not os.path.exists(path):
        return None, {}
    with open(path) as f:
        d = json.load(f)
    r = d.get("results", {})
    # mmlu group acc 또는 subtask 단순 평균
    if "mmlu" in r:
        v = r["mmlu"].get("acc,none")
        if v is not None:
            return round(v, 4), {"mmlu_group": round(v, 4)}
    # subtask 단순 평균
    scores = []
    for k, v in r.items():
        if k.startswith("mmlu_") and "acc,none" in v:
            scores.append(v["acc,none"])
    if scores:
        avg = sum(scores) / len(scores)
        return round(avg, 4), {"subtask_avg": round(avg, 4), "n_tasks": len(scores)}
    return None, {}

def latest(pat):
    f = sorted(glob.glob(pat))
    return f[-1] if f else None

entries = [
    ("FP16 (베이스라인)",      latest(f"{R}/eval_mmlu_llama3_base_8b_fp16*.json")),
    ("A   (랜덤 영어=표준GPTQ)", latest(f"{R}/eval_mmlu_llama3_base_8b_A*.json")),
    ("C_en_v3 (다양성 영어)",    latest(f"{R}/eval_mmlu_llama3_base_8b_C_en_v3*.json")),
]

fp16_score = None
print(f"\n{'조건':<28} {'MMLU avg':>9} {'보존율':>8}  메모")
print("-" * 65)
for label, path in entries:
    if not path:
        print(f"{label:<28} {'(없음)':>9}")
        continue
    avg, meta = get_mmlu_avg(path)
    if avg is None:
        print(f"{label:<28} {'파싱오류':>9}  {path}")
        continue
    if "FP16" in label:
        fp16_score = avg
    ret = f"{avg/fp16_score*100:.1f}%" if fp16_score else "N/A"
    note = str(meta)
    print(f"{label:<28} {avg:>9.4f} {ret:>8}  {note}")

print()
if fp16_score:
    print("[해석]")
    print("  C_en_v3 > A: 다양성 원칙이 영어에서도 유효 → 언어 독립적 일반 원리")
    print("  C_en_v3 ≈ A: Wikitext-2 동질성으로 인한 천장 효과 (Llama3-Ko와 동일 패턴)")
    print("  (기대값: C_en_v3 > A, 순수 영어 모델이라 더 뚜렷한 차이 예상)")
PYEOF

log "======================================================"
log "S4-15 실험 완료"
log "======================================================"
