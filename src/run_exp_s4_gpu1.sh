#!/bin/bash
# Sprint 4 GPU 1 파이프라인 v2
# S4-3: 누락 조건 보완 (B g64 / A no_desc_act / EEVE FP16) — 이미 완료, 스킵
# S4-4: C_v3_exaone 생성 + EXAONE35 재실험
# S4-2: C_zh_v3 생성 (min_sfs 필터) + Qwen2 재실험
#
# 실행: docker exec llm-dev bash /home/choihyun/workspace/src/run_exp_s4_gpu1.sh

set -euo pipefail

WORKSPACE="/home/choihyun/workspace"
RESULTS="${WORKSPACE}/results"
QUANT_MODELS="${WORKSPACE}/quantized_models"
SRC="${WORKSPACE}/src"
SOLAR_MODEL="upstage/SOLAR-10.7B-Instruct-v1.0"
EEVE_MODEL="yanolja/EEVE-Korean-Instruct-10.8B-v1.0"
QWEN2_MODEL="${WORKSPACE}/.cache/huggingface/hub/models--Qwen--Qwen2-7B-Instruct/snapshots/f2826a00ceef68f0f2b946d945ecc0477ce4450c"
CALIB_A="${RESULTS}/calibration_set_A.json"
CALIB_B="${RESULTS}/calibration_set_B.json"

export CUDA_VISIBLE_DEVICES=1
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

log() { echo "[$(date '+%H:%M')] $*"; }

# glob 매칭으로 이미 완료된 결과 스킵
result_exists() {
    compgen -G "${RESULTS}/eval_kobest_${1}*.json" > /dev/null 2>&1
}

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

eval_ceval() {
    local model_path="$1"
    local out_name="$2"
    local extra="${3:-}"
    log "C-Eval 평가: ${out_name}"
    "${LM_EVAL}" --model hf \
        --model_args "pretrained=${model_path},trust_remote_code=True,dtype=float16${extra}" \
        --tasks ceval-valid \
        --device cuda:1 \
        --batch_size 4 \
        --output_path "${RESULTS}/eval_ceval_${out_name}.json" \
        --log_samples \
        2>&1 | tee "${RESULTS}/eval_ceval_${out_name}.log"
    log "완료 C-Eval: ${out_name}"
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
log "Sprint 4 GPU 1: S4-3(스킵확인) + S4-4 EXAONE35 + S4-2 Qwen2 C_zh_v3"
log "======================================================"

# ─────────────────────────────────────────────────────────
# S4-3: 누락 조건 — 이미 완료된 것만 확인 후 스킵
# ─────────────────────────────────────────────────────────
log "=== [S4-3 확인] 이미 완료된 조건 스킵 ==="
result_exists "solar_B_g64"         && log "스킵: solar_B_g64"         || log "경고: solar_B_g64 결과 없음"
result_exists "solar_A_no_desc_act" && log "스킵: solar_A_no_desc_act" || log "경고: solar_A_no_desc_act 결과 없음"
result_exists "eeve_10b_fp16"       && log "스킵: eeve_10b_fp16"       || log "경고: eeve_10b_fp16 결과 없음"

# ─────────────────────────────────────────────────────────
# S4-4: C_v3_exaone 생성 + EXAONE35 재실험
# ─────────────────────────────────────────────────────────
log "=== [S4-4] C_v3_exaone calibration + EXAONE35 재실험 ==="
# build_calibration.py는 model 경로 전체를 파일명에 넣으므로 glob으로 탐색
CALIB_Cv3_EXAONE=$(ls "${RESULTS}"/calibration_set_C_*exaone*EXAONE*.json 2>/dev/null | head -1)
CALIB_Cv3_EXAONE="${CALIB_Cv3_EXAONE:-${RESULTS}/calibration_set_C_v3_exaone_LGAI-EXAONE-3.5-7.8B-Instruct.json}"
EXAONE_PATH=$(find_exaone_path)
log "EXAONE 경로: ${EXAONE_PATH}"

if [ ! -f "${CALIB_Cv3_EXAONE}" ]; then
    "${PYTHON}" "${SRC}/patch_exaone.py" 2>&1 | tail -5 || true
    log "C_v3_exaone calibration 생성 중 (~20분)..."
    "${PYTHON}" "${SRC}/build_calibration.py" \
        --condition C \
        --model "${EXAONE_PATH}" \
        --suffix "exaone_LGAI-EXAONE-3.5-7.8B-Instruct" \
        2>&1 | tee "${RESULTS}/build_C_v3_exaone.log"
    log "C_v3_exaone calibration 생성 완료"
else
    log "스킵: C_v3_exaone calibration 이미 존재"
fi

if ! result_exists "exaone35_7b_C_v3_exaone"; then
    "${PYTHON}" "${SRC}/patch_exaone.py" 2>&1 | tail -5 || true
    quant_optimum "${EXAONE_PATH}" "${CALIB_Cv3_EXAONE}" \
        "${QUANT_MODELS}/exaone35_7b_cond_C_v3_exaone"
    eval_kobest "${QUANT_MODELS}/exaone35_7b_cond_C_v3_exaone" "exaone35_7b_C_v3_exaone"
    rm -rf "${QUANT_MODELS}/exaone35_7b_cond_C_v3_exaone"
else
    log "스킵: exaone35_7b_C_v3_exaone 이미 완료"
fi

# ─────────────────────────────────────────────────────────
# S4-2: C_zh_v3 생성 (min_sfs=3.0) + Qwen2 재실험
# 목적: "언어 불일치 vs 다양성 부족" 혼재 해소
# ─────────────────────────────────────────────────────────
log "=== [S4-2] C_zh_v3 calibration 생성 (min_sfs=3.0) ==="
CALIB_C_ZH_V3="${RESULTS}/calibration_set_C_zh_v3_Qwen_Qwen2-7B-Instruct.json"

if [ ! -f "${CALIB_C_ZH_V3}" ]; then
    log "C_zh_v3 calibration 생성 중 (min_sfs=3.0, ~15분)..."
    "${PYTHON}" "${SRC}/build_calibration_zh.py" \
        --model "${QWEN2_MODEL}" \
        --n-candidates 100000 \
        --min-sfs 3.0 \
        --suffix "zh_v3_Qwen_Qwen2-7B-Instruct" \
        2>&1 | tee "${RESULTS}/build_C_zh_v3.log"
    log "C_zh_v3 calibration 생성 완료"
else
    log "스킵: C_zh_v3 calibration 이미 존재"
fi

log "=== [S4-2] Qwen2 C_zh_v3 양자화 + 평가 ==="
if ! result_exists "qwen2_7b_C_zh_v3"; then
    quant_optimum "${QWEN2_MODEL}" "${CALIB_C_ZH_V3}" \
        "${QUANT_MODELS}/qwen2_7b_cond_C_zh_v3"
    eval_kobest "${QUANT_MODELS}/qwen2_7b_cond_C_zh_v3" "qwen2_7b_C_zh_v3"
    eval_ceval  "${QUANT_MODELS}/qwen2_7b_cond_C_zh_v3" "qwen2_7b_C_zh_v3"
    rm -rf "${QUANT_MODELS}/qwen2_7b_cond_C_zh_v3"
else
    log "스킵: qwen2_7b_C_zh_v3 이미 완료"
fi

# ─────────────────────────────────────────────────────────
# 결과 요약
# ─────────────────────────────────────────────────────────
log "======================================================"
log "결과 요약"
log "======================================================"

"${PYTHON}" - <<'PYEOF'
import json, os, glob

results_dir = "/home/choihyun/workspace/results"

def kobest_agg(path):
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

def latest(pattern):
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None

checks = [
    ("solar_B_g64",             "S4-3-1 SOLAR B g64"),
    ("solar_A_no_desc_act",     "S4-3-2 SOLAR A no_desc_act"),
    ("eeve_10b_fp16",           "S4-3-3 EEVE FP16"),
    ("exaone35_7b_C_v3_exaone", "S4-4   EXAONE35 C_v3_exaone"),
    ("qwen2_7b_C_zh_v3",        "S4-2   Qwen2 C_zh_v3"),
]
for fname, label in checks:
    p = latest(f"{results_dir}/eval_kobest_{fname}*.json")
    v = kobest_agg(p) if p else None
    status = f"{v:.4f}" if v is not None else "MISSING"
    print(f"  {label}: {status}")

# Qwen2 전체 비교
print("\n  === Qwen2 KoBEST 전체 비교 ===")
for cond, tag in [("A", "A"), ("C_v3", "C_v3"), ("C_zh", "C_zh"), ("C_zh_v3", "C_zh_v3")]:
    p = latest(f"{results_dir}/eval_kobest_qwen2_7b_{tag}*.json") or \
        latest(f"{results_dir}/eval_kobest_qwen2*{tag}*.json")
    v = kobest_agg(p) if p else None
    status = f"{v:.4f}" if v is not None else "없음"
    print(f"    Qwen2 {cond}: {status}")
PYEOF

log "======================================================"
log "GPU 1 Sprint 4 파이프라인 완료"
log "======================================================"
