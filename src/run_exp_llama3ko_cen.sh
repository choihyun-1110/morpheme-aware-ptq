#!/bin/bash
# Llama3-Ko C_en_v3 실험
#
# 목적: "사전학습 언어 = calibration 언어" 가설의 언어 독립성 검증
#   - Llama3-Ko는 영어 사전학습 모델 → 영어 calibration이 유리했음 (A > C_v3 > B)
#   - A = 랜덤 Wikitext-2 (표준 GPTQ 기본값)
#   - C_en_v3 = 다양성 최적화 Wikitext-2 (우리 알고리즘, 영어 버전)
#   - 가설: C_en_v3 > A 이면 → 다양성 원칙이 언어에 독립적임을 증명
#
# 비교표:
#   A(랜덤 영어) vs C_en_v3(다양성 영어) vs B(랜덤 한국어) vs C_v3(다양성 한국어)
#
# 실행: docker exec llm-dev bash /home/choihyun/workspace/src/run_exp_llama3ko_cen.sh

set -euo pipefail

WORKSPACE="/home/choihyun/workspace"
RESULTS="${WORKSPACE}/results"
QUANT_MODELS="${WORKSPACE}/quantized_models"
SRC="${WORKSPACE}/src"
LLAMA3KO_MODEL="${WORKSPACE}/.cache/huggingface/hub/models--beomi--Llama-3-Open-Ko-8B/snapshots/a8e8214f79f0d2cea817020a93dca48e8a6be18b"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HOME="/home/choihyun"
export HF_HOME="${WORKSPACE}/.cache/huggingface"
export XDG_CACHE_HOME="${WORKSPACE}/.cache"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_OFFLINE=0

cd "${WORKSPACE}"
source /opt/conda/etc/profile.d/conda.sh
conda activate llm-quant

LM_EVAL="/opt/conda/envs/llm-quant/bin/lm_eval"
PYTHON="/opt/conda/envs/llm-quant/bin/python3.11"
GPU_DEVICE="cuda:${CUDA_VISIBLE_DEVICES:-0}"

log() { echo "[$(date '+%H:%M')] $*"; }

result_exists() {
    compgen -G "${RESULTS}/eval_kobest_${1}*.json" > /dev/null 2>&1
}

eval_kobest() {
    local model_path="$1"
    local out_name="$2"
    log "KoBEST 평가: ${out_name}"
    "${LM_EVAL}" --model hf \
        --model_args "pretrained=${model_path},trust_remote_code=True,dtype=float16" \
        --tasks kobest \
        --device "${GPU_DEVICE}" \
        --batch_size 4 \
        --output_path "${RESULTS}/eval_kobest_${out_name}.json" \
        --log_samples \
        2>&1 | tee "${RESULTS}/eval_kobest_${out_name}.log"
    log "완료: ${out_name}"
}

log "======================================================"
log "Llama3-Ko C_en_v3 실험 (GPU: ${CUDA_VISIBLE_DEVICES:-0})"
log "가설: 다양성 기반 영어 calibration > 랜덤 영어(A)"
log "======================================================"

# ─────────────────────────────────────────────────────────
# Step 1: C_en_v3 calibration 생성
# ─────────────────────────────────────────────────────────
CALIB_EN_V3=$(ls "${RESULTS}"/calibration_set_C_en_v3*.json 2>/dev/null | head -1 || true)

if [ -z "${CALIB_EN_V3}" ]; then
    log "C_en_v3 calibration 생성 중 (Wikitext-2 다양성 선별, ~15분)..."
    "${PYTHON}" "${SRC}/build_calibration_en.py" \
        --model "${LLAMA3KO_MODEL}" \
        --n-sentences 128 \
        --n-candidates 50000 \
        --alpha 0.3 \
        --min-words 8 \
        --min-tokens 20 \
        --suffix "en_v3" \
        2>&1 | tee "${RESULTS}/build_C_en_v3.log"
    CALIB_EN_V3=$(ls "${RESULTS}"/calibration_set_C_en_v3*.json 2>/dev/null | head -1 || true)
    log "C_en_v3 calibration 생성 완료: ${CALIB_EN_V3}"
else
    log "스킵: C_en_v3 calibration 이미 존재 (${CALIB_EN_V3})"
fi

# ─────────────────────────────────────────────────────────
# Step 2: Llama3-Ko C_en_v3 양자화 + KoBEST 평가
# ─────────────────────────────────────────────────────────
log "=== Llama3-Ko C_en_v3 양자화 ==="
if ! result_exists "llama3_ko_8b_C_en_v3"; then
    "${PYTHON}" "${SRC}/run_quant_optimum.py" \
        --model "${LLAMA3KO_MODEL}" \
        --calib-path "${CALIB_EN_V3}" \
        --out-dir "${QUANT_MODELS}/llama3_ko_8b_cond_C_en_v3" \
        2>&1 | tee "${RESULTS}/quant_llama3ko_C_en_v3.log"
    eval_kobest "${QUANT_MODELS}/llama3_ko_8b_cond_C_en_v3" "llama3_ko_8b_C_en_v3"
    rm -rf "${QUANT_MODELS}/llama3_ko_8b_cond_C_en_v3"
else
    log "스킵: llama3_ko_8b_C_en_v3 이미 완료"
fi

# ─────────────────────────────────────────────────────────
# 결과 요약: Llama3-Ko 전체 비교
# ─────────────────────────────────────────────────────────
log "======================================================"
log "결과 요약: Llama3-Ko KoBEST 전체 비교"
log "======================================================"

"${PYTHON}" - <<'PYEOF'
import json, glob, os

R = "/home/choihyun/workspace/results"

def agg(path):
    if not path or not os.path.exists(path): return None
    with open(path) as f: d = json.load(f)
    r = d.get("results", {})
    task_keys = [
        ("kobest_copa",      "acc_norm,none", "acc,none"),
        ("kobest_hellaswag", "acc_norm,none", "acc,none"),
        ("kobest_sentineg",  "acc_norm,none", "acc,none"),
        ("kobest_wic",       "acc,none",      None),
        ("kobest_boolq",     "acc,none",      None),
    ]
    per_task = {}
    scores = []
    for t, p, fb in task_keys:
        if t in r:
            v = r[t].get(p) or (r[t].get(fb) if fb else None)
            if v is not None:
                scores.append(v)
                per_task[t.replace("kobest_","")] = round(v, 4)
    avg = round(sum(scores)/len(scores), 4) if scores else None
    return avg, per_task

def latest(pat):
    f = sorted(glob.glob(pat))
    return f[-1] if f else None

# Llama3-Ko FP16 기준값 (Sprint 2에서 측정)
fp16 = 0.5561  # Llama3-Ko-8B FP16 KoBEST avg (sprints/sprint2 결과 기준)

entries = [
    ("A  (랜덤 영어=표준GPTQ)", latest(f"{R}/eval_kobest_llama3_ko_8b_A*.json")),
    ("B  (랜덤 한국어)",        latest(f"{R}/eval_kobest_llama3_ko_8b_B*.json")),
    ("C_v3 (다양성 한국어)",    latest(f"{R}/eval_kobest_llama3_ko_8b_C_v3*.json")),
    ("C_en_v3 (다양성 영어)",   latest(f"{R}/eval_kobest_llama3_ko_8b_C_en_v3*.json")),
]

tasks = ["boolq","copa","hellaswag","sentineg","wic"]
print(f"\n{'조건':<22} {'avg':>7} {'보존율':>7}  " + "  ".join(f"{t[:6]:>7}" for t in tasks))
print("-" * 88)
for label, path in entries:
    if not path:
        print(f"{label:<22} {'없음':>7}")
        continue
    result = agg(path)
    if not result:
        print(f"{label:<22} {'파싱오류':>7}")
        continue
    avg, per = result
    ret = f"{avg/fp16*100:.1f}%" if fp16 else "N/A"
    task_vals = "  ".join(f"{per.get(t, 0):>7.4f}" for t in tasks)
    print(f"{label:<22} {avg:>7.4f} {ret:>7}  {task_vals}")

print("\n[해석 기준]")
print("  C_en_v3 > A: 다양성 원칙이 영어에서도 유효 → 언어 독립적 원리 확인")
print("  C_en_v3 ≈ A: 랜덤도 충분 (Wikitext-2가 이미 충분히 다양)")
print("  C_en_v3 < A: 예상치 못한 결과 → 추가 분석 필요")
PYEOF

log "======================================================"
log "Llama3-Ko C_en_v3 실험 완료"
log "======================================================"
