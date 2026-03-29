#!/bin/bash
# GPU 0 Phase 2 — Qwen2 C-Eval 완료 후 자동 실행
# activation 분석 기반 새 양자화 실험:
#   1. SOLAR C_v3 group_size=64 (세밀한 그룹)
#   2. SOLAR A    group_size=64 (대조군)
#   3. SOLAR C_v3 + SmoothScale 전처리 → GPTQ 4bit
#   4. SOLAR A    AWQ 4bit (GPTQ vs AWQ 방법론 비교)
#   5. SOLAR C_v3 AWQ 4bit
#   6. SOLAR C_v3 desc_act=False (빠른 추론 모드)

set -euo pipefail

WORKSPACE="/home/choihyun/workspace"
RESULTS="${WORKSPACE}/results"
QUANT_MODELS="${WORKSPACE}/quantized_models"
SRC="${WORKSPACE}/src"
MODEL="upstage/SOLAR-10.7B-Instruct-v1.0"

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

log() { echo "[$(date '+%H:%M')] $*"; }

eval_kobest() {
    local model_path="$1"
    local out_name="$2"
    local extra_args="${3:-}"
    log "KoBEST 평가: ${out_name}"
    local model_args="pretrained=${model_path},trust_remote_code=True${extra_args}"
    if [ -f "${model_path}/quantize_config.json" ] && [[ "$extra_args" != *"autogptq"* ]]; then
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
    log "완료: ${out_name}"
}

CALIB_A="${RESULTS}/calibration_set_A.json"
CALIB_Cv3="${RESULTS}/calibration_set_C_v3_upstage_SOLAR-10.7B-Instruct-v1.0.json"

# ─────────────────────────────────────────────────────────
# 실험 1: group_size=64 (현재 128 대비 더 세밀한 양자화)
# 가설: 중간 레이어(L12-18)의 낮은 channel_cv 구간에서
#       세밀한 그룹이 Hessian 근사 오차를 더 줄여줄 것
# ─────────────────────────────────────────────────────────
log "=== [EXP1] SOLAR C_v3 group_size=64 ==="
python "${SRC}/run_quant.py" \
    --model "${MODEL}" --calib "C_v3" --calib-path "${CALIB_Cv3}" \
    --out-dir "${QUANT_MODELS}/solar_C_v3_g64" \
    --group-size 64 \
    2>&1 | tee "${RESULTS}/quant_solar_C_v3_g64.log"
eval_kobest "${QUANT_MODELS}/solar_C_v3_g64" "solar_C_v3_g64"

log "=== [EXP1b] SOLAR A group_size=64 (대조군) ==="
python "${SRC}/run_quant.py" \
    --model "${MODEL}" --calib "A" --calib-path "${CALIB_A}" \
    --out-dir "${QUANT_MODELS}/solar_A_g64" \
    --group-size 64 \
    2>&1 | tee "${RESULTS}/quant_solar_A_g64.log"
eval_kobest "${QUANT_MODELS}/solar_A_g64" "solar_A_g64"

# ─────────────────────────────────────────────────────────
# 실험 2: SmoothScale + GPTQ (활성화 분포 균일화 전처리)
# 가설: channel_cv가 높은 레이어에서 activation을 weight로
#       이전하면 GPTQ의 Hessian 추정 오차가 감소
# C_v3 calibration으로 scale 계산 후 GPTQ 적용
# ─────────────────────────────────────────────────────────
log "=== [EXP2] SmoothScale + GPTQ C_v3 ==="
SMOOTH_DIR="${QUANT_MODELS}/solar_smooth_scaled"
python "${SRC}/smooth_scale.py" \
    --model "${MODEL}" \
    --calib-path "${CALIB_Cv3}" \
    --out-dir "${SMOOTH_DIR}" \
    --alpha 0.5 \
    2>&1 | tee "${RESULTS}/smooth_scale_C_v3.log"

# SmoothScale된 모델에 GPTQ C_v3 적용
python "${SRC}/run_quant.py" \
    --model "${SMOOTH_DIR}" \
    --calib "C_v3_smooth" --calib-path "${CALIB_Cv3}" \
    --out-dir "${QUANT_MODELS}/solar_smooth_C_v3" \
    2>&1 | tee "${RESULTS}/quant_solar_smooth_C_v3.log"
eval_kobest "${QUANT_MODELS}/solar_smooth_C_v3" "solar_smooth_C_v3"

# SmoothScale + A calibration (대조)
python "${SRC}/run_quant.py" \
    --model "${SMOOTH_DIR}" \
    --calib "A_smooth" --calib-path "${CALIB_A}" \
    --out-dir "${QUANT_MODELS}/solar_smooth_A" \
    2>&1 | tee "${RESULTS}/quant_solar_smooth_A.log"
eval_kobest "${QUANT_MODELS}/solar_smooth_A" "solar_smooth_A"

# ─────────────────────────────────────────────────────────
# 실험 3: AWQ (AutoAWQ 설치 완료 전제)
# 가설: AWQ의 salient weight 보호가 calibration 언어에 얼마나
#       민감한지 → GPTQ와 다른 패턴이 나타날 수 있음
# ─────────────────────────────────────────────────────────
if python -c "import awq" 2>/dev/null; then
    log "=== [EXP3] AWQ A ==="
    python "${SRC}/run_quant_awq.py" \
        --model "${MODEL}" --calib-path "${CALIB_A}" \
        --out-dir "${QUANT_MODELS}/solar_awq_A" \
        2>&1 | tee "${RESULTS}/quant_solar_awq_A.log"
    eval_kobest "${QUANT_MODELS}/solar_awq_A" "solar_awq_A"

    log "=== [EXP3b] AWQ C_v3 ==="
    python "${SRC}/run_quant_awq.py" \
        --model "${MODEL}" --calib-path "${CALIB_Cv3}" \
        --out-dir "${QUANT_MODELS}/solar_awq_C_v3" \
        2>&1 | tee "${RESULTS}/quant_solar_awq_C_v3.log"
    eval_kobest "${QUANT_MODELS}/solar_awq_C_v3" "solar_awq_C_v3"
else
    log "=== [EXP3] AutoAWQ 미설치 — 스킵 ==="
fi

# ─────────────────────────────────────────────────────────
# 실험 4: desc_act=False (추론 속도 vs 품질 trade-off)
# 기존 실험: desc_act=True (activation 크기 기준 column 재정렬)
# desc_act=False: 재정렬 없음 → 빠르지만 이론상 정확도 낮음
# 질문: C_v3의 높은 channel_cv가 desc_act=False에서도 이점을 주나?
# ─────────────────────────────────────────────────────────
log "=== [EXP4] SOLAR C_v3 desc_act=False ==="
python "${SRC}/run_quant.py" \
    --model "${MODEL}" --calib "C_v3" --calib-path "${CALIB_Cv3}" \
    --out-dir "${QUANT_MODELS}/solar_C_v3_no_desc_act" \
    --no-desc-act \
    2>&1 | tee "${RESULTS}/quant_solar_C_v3_no_desc_act.log"
eval_kobest "${QUANT_MODELS}/solar_C_v3_no_desc_act" "solar_C_v3_no_desc_act"

log "=== GPU 0 Phase 2 모든 실험 완료 ==="
