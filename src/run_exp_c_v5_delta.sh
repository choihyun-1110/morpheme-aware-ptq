#!/bin/bash
# C_v5_delta 실험: δ 단독 효과 분리
#
# 목적: C_v5(0.6128) < C_v3(0.6356) 원인 규명
#
#   C_v5 = C_v3 + δ=0.2 + min_ko_ratio=0.9 + min_tokens=40  → 0.6128
#   C_v3                                                      → 0.6356
#
#   C_v5는 두 변수가 동시에 변했음 → confound 문제
#   이 실험: C_v3 파라미터 그대로 + δ=0.2만 추가
#
#   결과 해석:
#     C_v5_delta > C_v3  → δ(token richness)가 실제로 유효, 순도 강화가 역효과였음
#     C_v5_delta ≈ C_v3  → δ 효과 미미, 순도 강화가 핵심 문제
#     C_v5_delta < C_v3  → token richness 자체가 오히려 해로움
#
# 실행: docker exec llm-dev bash /home/choihyun/workspace/src/run_exp_c_v5_delta.sh

set -euo pipefail

WORKSPACE="/home/choihyun/workspace"
RESULTS="${WORKSPACE}/results"
QUANT_MODELS="${WORKSPACE}/quantized_models"
SRC="${WORKSPACE}/src"
SOLAR_MODEL="upstage/SOLAR-10.7B-Instruct-v1.0"

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
PYTHON="/opt/conda/envs/llm-quant/bin/python3.11"
GPU_DEVICE="cuda:${CUDA_VISIBLE_DEVICES:-1}"

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
log "SOLAR C_v5_delta 실험 (GPU: ${CUDA_VISIBLE_DEVICES:-1})"
log "목적: δ=0.2 단독 효과 분리 (순도/길이 변경 없음)"
log "======================================================"

# ─────────────────────────────────────────────────────────
# Step 1: C_v5_delta calibration 생성
#   C_v3 파라미터 그대로 + delta=0.2만 추가
#   min_ko_ratio=0.7 (C_v3와 동일, C_v5의 0.9 아님)
#   min_tokens=24 (C_v3와 동일, C_v5의 40 아님)
# ─────────────────────────────────────────────────────────
CALIB_V5D=$(ls "${RESULTS}"/calibration_set_C_v5_delta*.json 2>/dev/null | head -1 || true)

if [ -z "${CALIB_V5D}" ]; then
    log "C_v5_delta calibration 생성 중..."
    log "  파라미터: delta=0.2, min_ko_ratio=0.7(C_v3와 동일), min_tokens=24(C_v3와 동일)"
    "${PYTHON}" "${SRC}/build_calibration.py" \
        --model "upstage/SOLAR-10.7B-Instruct-v1.0" \
        --n-sentences 128 \
        --delta 0.2 \
        --c-min-ko-ratio 0.7 \
        --c-min-tokens 24 \
        --suffix "v5_delta" \
        2>&1 | tee "${RESULTS}/build_C_v5_delta.log"
    CALIB_V5D=$(ls "${RESULTS}"/calibration_set_C_v5_delta*.json 2>/dev/null | head -1 || true)
    log "C_v5_delta 생성 완료: ${CALIB_V5D}"
else
    log "스킵: C_v5_delta 이미 존재 (${CALIB_V5D})"
fi

# ─────────────────────────────────────────────────────────
# Step 2: SOLAR C_v5_delta 양자화 + KoBEST 평가
# ─────────────────────────────────────────────────────────
log "=== SOLAR C_v5_delta 양자화 ==="
if ! result_exists "solar_cond_C_v5_delta"; then
    "${PYTHON}" "${SRC}/run_quant_optimum.py" \
        --model "${SOLAR_MODEL}" \
        --calib-path "${CALIB_V5D}" \
        --out-dir "${QUANT_MODELS}/solar_cond_C_v5_delta" \
        2>&1 | tee "${RESULTS}/quant_solar_C_v5_delta.log"
    eval_kobest "${QUANT_MODELS}/solar_cond_C_v5_delta" "solar_cond_C_v5_delta"
    rm -rf "${QUANT_MODELS}/solar_cond_C_v5_delta"
else
    log "스킵: solar_cond_C_v5_delta 이미 완료"
fi

# ─────────────────────────────────────────────────────────
# 결과 비교 출력
# ─────────────────────────────────────────────────────────
log "======================================================"
log "결과: δ 단독 효과 분석"
log "======================================================"

"${PYTHON}" - <<'PYEOF'
import json, glob, os, statistics

R = "/home/choihyun/workspace/results"

def agg(path):
    if not path or not os.path.exists(path): return None, {}
    with open(path) as f: d = json.load(f)
    r = d.get("results", {})
    task_keys = [
        ("kobest_copa",      "acc_norm,none", "acc,none"),
        ("kobest_hellaswag", "acc_norm,none", "acc,none"),
        ("kobest_sentineg",  "acc_norm,none", "acc,none"),
        ("kobest_wic",       "acc,none",      None),
        ("kobest_boolq",     "acc,none",      None),
    ]
    per, scores = {}, []
    for t, p, fb in task_keys:
        if t in r:
            v = r[t].get(p) or (r[t].get(fb) if fb else None)
            if v is not None:
                scores.append(v)
                per[t.replace("kobest_","")] = round(v,4)
    avg = round(sum(scores)/len(scores),4) if scores else None
    return avg, per

def latest(pat):
    f = sorted(glob.glob(pat))
    return f[-1] if f else None

fp16 = 0.6523

# C_v3 stat runs mean
cv3_runs = sorted(glob.glob(R+"/eval_kobest_solar_C_v3_stat_run*.json"))
cv3_vals = [agg(p)[0] for p in cv3_runs if agg(p)[0]]
cv3_mean = round(statistics.mean(cv3_vals),4) if cv3_vals else None

entries = [
    ("FP16 (기준)",       None,  0.6523),
    ("C_v3 Sprint2",      latest(R+"/eval_kobest_solar_C_v3_stat_run1*.json"), None),  # 단일 대표
    ("C_v3 mean(5런)",    None,  cv3_mean),
    ("C_v5 (δ+순도+길이)",latest(R+"/eval_kobest_solar_cond_C_v5*.json"), None),
    ("C_v5_delta (δ만)", latest(R+"/eval_kobest_solar_cond_C_v5_delta*.json"), None),
    ("B mean(5런)",        None,  0.6393),
    ("A (랜덤영어)",       None,  0.5981),
]

print("\n{:<24} {:>7} {:>7}".format("조건", "avg", "보존율"))
print("-"*42)
for label, path, preset in entries:
    if preset is not None:
        ret = "{:.1f}%".format(preset/fp16*100)
        print("{:<24} {:>7.4f} {:>7}".format(label, preset, ret))
        continue
    avg, _ = agg(path)
    if not avg:
        print("{:<24} {:>7}".format(label, "없음"))
        continue
    ret = "{:.1f}%".format(avg/fp16*100)
    print("{:<24} {:>7.4f} {:>7}".format(label, avg, ret))

print()
print("[해석]")
print("  C_v5_delta > C_v3 → δ(token richness)가 유효, C_v5의 실패는 순도강화 탓")
print("  C_v5_delta ≈ C_v3 → δ 효과 미미")
print("  C_v5_delta < C_v3 → token richness 자체가 GPTQ에 불리")
PYEOF

log "======================================================"
log "C_v5_delta 실험 완료"
log "======================================================"
