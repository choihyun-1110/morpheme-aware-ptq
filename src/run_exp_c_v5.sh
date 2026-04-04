#!/bin/bash
# C_v5 실험: token richness + 순수 한국어 필터 강화
# - delta=0.2 (within-sentence token unique 비율 보너스)
# - min_ko_ratio=0.9 (한국어 순도 강화, C_v3=0.7 대비)
# - min_tokens=40 (충분한 context 보장, C_v3=24 대비)
#
# 비교: A(standard GPTQ) / B(랜덤 한국어) / C_v3(형태소) / C_v5(형태소+richness)
#
# GPU: CUDA_VISIBLE_DEVICES 환경변수로 지정
# 실행: docker exec llm-dev bash /home/choihyun/workspace/src/run_exp_c_v5.sh

set -euo pipefail

WORKSPACE="/home/choihyun/workspace"
RESULTS="${WORKSPACE}/results"
QUANT_MODELS="${WORKSPACE}/quantized_models"
SRC="${WORKSPACE}/src"
MODEL="upstage/SOLAR-10.7B-Instruct-v1.0"

# GPU 자동 선택: 지정 없으면 1번 사용
GPU="${CUDA_VISIBLE_DEVICES:-1}"
export CUDA_VISIBLE_DEVICES="${GPU}"
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
        --device "cuda:0" \
        --batch_size 4 \
        --output_path "${RESULTS}/eval_kobest_${out_name}.json" \
        --log_samples \
        2>&1 | tee "${RESULTS}/eval_kobest_${out_name}.log"
    log "완료: ${out_name}"
}

log "======================================================"
log "C_v5 실험: token richness + 순수 한국어 강화"
log "GPU: ${GPU}"
log "======================================================"

# ─────────────────────────────────────────────────────────
# Step 1: C_v5 calibration set 생성
# ─────────────────────────────────────────────────────────
CALIB_V5="${RESULTS}/calibration_set_C_v5_upstage_SOLAR-10.7B-Instruct-v1.0.json"

if [ ! -f "${CALIB_V5}" ]; then
    log "C_v5 calibration 생성 중 (NamuWiki 100k 처리, ~20분)..."
    log "설정: delta=0.2, min_ko_ratio=0.9, min_tokens=40"
    "${PYTHON}" "${SRC}/build_calibration.py" \
        --condition C \
        --model "${MODEL}" \
        --n-sentences 128 \
        --n-candidates 100000 \
        --alpha 0.3 \
        --beta 0.15 \
        --gamma 0.15 \
        --delta 0.2 \
        --c-min-ko-ratio 0.9 \
        --c-min-tokens 40 \
        --suffix v5 \
        2>&1 | tee "${RESULTS}/build_C_v5.log"
    log "C_v5 calibration 생성 완료"
else
    log "스킵: C_v5 calibration 이미 존재"
fi

# ─────────────────────────────────────────────────────────
# Step 2: SOLAR C_v5 양자화 + KoBEST 평가
# ─────────────────────────────────────────────────────────
log "=== SOLAR C_v5 양자화 ==="
if ! result_exists "solar_cond_C_v5"; then
    "${PYTHON}" "${SRC}/run_quant_optimum.py" \
        --model "${MODEL}" \
        --calib-path "${CALIB_V5}" \
        --out-dir "${QUANT_MODELS}/solar_10.7b_cond_C_v5" \
        2>&1 | tee "${RESULTS}/quant_solar_C_v5.log"
    eval_kobest "${QUANT_MODELS}/solar_10.7b_cond_C_v5" "solar_cond_C_v5"
    rm -rf "${QUANT_MODELS}/solar_10.7b_cond_C_v5"
else
    log "스킵: solar_cond_C_v5 이미 완료"
fi

# ─────────────────────────────────────────────────────────
# 결과 요약: A / B / C_v3 / C_v5 비교
# ─────────────────────────────────────────────────────────
log "======================================================"
log "결과 요약: 표준 GPTQ(A) vs B vs C_v3 vs C_v5"
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

fp16 = 0.6523
entries = [
    ("FP16 (상한)",    None,                                              fp16),
    ("A  (표준GPTQ)",  f"{R}/eval_cond_A_2026-03-17T07-34-26.732750.json", None),
    ("B  (랜덤한국어)", f"{R}/eval_cond_B_2026-03-17T09-58-30.515789.json", None),
    ("C_v3",           f"{R}/eval_cond_C_v3_2026-03-18T06-35-39.561864.json", None),
    ("C_v5 (신규)",    latest(f"{R}/eval_kobest_solar_cond_C_v5*.json"),   None),
]

tasks = ["boolq","copa","hellaswag","sentineg","wic"]
print(f"\n{'조건':<16} {'avg':>7} {'보존율':>7}  " + "  ".join(f"{t[:6]:>7}" for t in tasks))
print("-" * 80)
for label, path, override in entries:
    if override is not None:
        avg = override
        per = {}
        print(f"{label:<16} {avg:>7.4f} {'100.0%':>7}")
        continue
    if not path: continue
    result = agg(path)
    if not result: continue
    avg, per = result
    ret = f"{avg/fp16*100:.1f}%"
    task_vals = "  ".join(f"{per.get(t, 0):>7.4f}" for t in tasks)
    print(f"{label:<16} {avg:>7.4f} {ret:>7}  {task_vals}")

print("\n[참고] 표준 GPTQ = Condition A (Wikitext-2 영어, GPTQ 논문 기본 설정)")
PYEOF

log "======================================================"
log "C_v5 실험 완료"
log "======================================================"
