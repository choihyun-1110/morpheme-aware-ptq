#!/bin/bash
# Sprint 4 GPU 0 파이프라인
# S4-1: SOLAR C_v3 vs B 통계 검정 (각 5회 런 → 95% CI, paired t-test)
#
# 결과: results/eval_kobest_solar_C_v3_stat_run{1..5}.json
#       results/eval_kobest_solar_B_stat_run{1..5}.json
#
# 실행: docker exec llm-dev bash /home/choihyun/workspace/src/run_exp_s4_gpu0.sh

set -euo pipefail

WORKSPACE="/home/choihyun/workspace"
RESULTS="${WORKSPACE}/results"
QUANT_MODELS="${WORKSPACE}/quantized_models"
SRC="${WORKSPACE}/src"
MODEL="upstage/SOLAR-10.7B-Instruct-v1.0"
CALIB_Cv3="${RESULTS}/calibration_set_C_v3_upstage_SOLAR-10.7B-Instruct-v1.0.json"
CALIB_B="${RESULTS}/calibration_set_B.json"

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
PYTHON="/opt/conda/envs/llm-quant/bin/python3.11"

log() { echo "[$(date '+%H:%M')] $*"; }

eval_kobest() {
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
    log "완료: ${out_name}"
}

run_stat() {
    local calib_path="$1"
    local cond_name="$2"
    local run_idx="$3"
    local seed="$4"          # 셔플 시드 (run마다 다름 → 분산 도입)
    local out_tag="${cond_name}_stat_run${run_idx}"
    local model_dir="${QUANT_MODELS}/solar_${out_tag}"

    # 이미 결과 파일이 존재하면 스킵
    if compgen -G "${RESULTS}/eval_kobest_solar_${out_tag}*.json" > /dev/null 2>&1; then
        log "[스킵] 이미 완료: ${out_tag}"
        return 0
    fi

    log "=== [${out_tag}] 양자화 시작 (seed=${seed}) ==="
    "${PYTHON}" "${SRC}/run_quant_optimum.py" \
        --model "${MODEL}" \
        --calib-path "${calib_path}" \
        --out-dir "${model_dir}" \
        --seed "${seed}" \
        2>&1 | tee "${RESULTS}/quant_solar_${out_tag}.log"
    log "양자화 완료: ${out_tag}"

    eval_kobest "${model_dir}" "solar_${out_tag}"

    # 디스크 절약: 평가 완료 후 모델 삭제 (eval JSON은 보존)
    log "모델 디렉토리 삭제 (eval JSON 보존): ${model_dir}"
    rm -rf "${model_dir}"
}

log "======================================================"
log "Sprint 4 S4-1: SOLAR C_v3 vs B 통계 검정 (5회 런, 셔플 시드 도입)"
log "======================================================"

# 기존 결과(seed 없이 돌려서 std=0인 것들) 삭제 후 재실행
log "[정리] 기존 stat 결과 파일 삭제 중..."
rm -f "${RESULTS}"/eval_kobest_solar_C_v3_stat_run*.json
rm -f "${RESULTS}"/eval_kobest_solar_B_stat_run*.json
log "[정리] 완료"

# C_v3 5회 (seed 42,43,44,45,46)
for i in 1 2 3 4 5; do
    seed=$((41 + i))
    run_stat "${CALIB_Cv3}" "C_v3" "${i}" "${seed}"
done

# B 5회 (동일한 seed 사용 → paired 비교 공정성 확보)
for i in 1 2 3 4 5; do
    seed=$((41 + i))
    run_stat "${CALIB_B}" "B" "${i}" "${seed}"
done

log "======================================================"
log "S4-1 모든 런 완료 — 통계 계산"
log "======================================================"

# 결과 요약 출력
"${PYTHON}" - <<'PYEOF'
import json, glob, os, math

results_dir = "/home/choihyun/workspace/results"

def kobest_agg(path):
    with open(path) as f:
        d = json.load(f)
    r = d.get("results", {})
    # kobest aggregate: mean of (copa acc_norm, hellaswag acc_norm, sentineg acc_norm, wic acc, boolq acc)
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

for cond in ["C_v3", "B"]:
    files = sorted(glob.glob(f"{results_dir}/eval_kobest_solar_{cond}_stat_run*.json"))
    vals = []
    for p in files:
        v = kobest_agg(p)
        if v is not None:
            vals.append(v)
            print(f"  {os.path.basename(p)}: {v:.4f}")
    if vals:
        mean = sum(vals) / len(vals)
        std  = math.sqrt(sum((x - mean)**2 for x in vals) / (len(vals) - 1)) if len(vals) > 1 else 0
        se   = std / math.sqrt(len(vals))
        ci95 = 1.96 * se
        print(f"  [{cond}] n={len(vals)}, mean={mean:.4f}, std={std:.4f}, 95%CI=[{mean-ci95:.4f}, {mean+ci95:.4f}]")
        print()
PYEOF

log "======================================================"
log "GPU 0 Sprint 4 파이프라인 완료"
log "======================================================"
