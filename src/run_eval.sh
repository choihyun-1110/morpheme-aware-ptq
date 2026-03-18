#!/bin/bash
# 양자화된 모델을 lm-evaluation-harness 로 평가하는 스크립트

MODEL_PATH=$1
OUTPUT_NAME=$2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CACHE_ROOT="${WORKSPACE_DIR}/.cache"
HF_HOME_DIR="${CACHE_ROOT}/huggingface"

mkdir -p "${HF_HOME_DIR}/hub" "${HF_HOME_DIR}/transformers"

export HOME="${HOME:-/home/choihyun}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}}"
export HF_HOME="${HF_HOME:-${HF_HOME_DIR}}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME_DIR}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME_DIR}/transformers}"

if [ -z "$MODEL_PATH" ] || [ -z "$OUTPUT_NAME" ]; then
    echo "사용법: bash src/run_eval.sh <모델경로> <저장이름(예: cond_A)>"
    exit 1
fi

echo "================================================="
echo "평가 시작: $MODEL_PATH"
echo "================================================="

# Sprint 문서 기준 한국어 종합 벤치마크(KoBEST)
TASKS="kobest"
MODEL_ARGS="pretrained=$MODEL_PATH,trust_remote_code=True"

if [ -f "$MODEL_PATH/quantize_config.json" ]; then
    GPTQ_MODEL_FILE="$(find "$MODEL_PATH" -maxdepth 1 -name '*.safetensors' | head -n 1)"
    if [ -z "$GPTQ_MODEL_FILE" ]; then
        echo "오류: GPTQ 양자화 모델로 보이지만 .safetensors 파일을 찾지 못했습니다: $MODEL_PATH"
        exit 1
    fi
    GPTQ_MODEL_FILE="$(basename "$GPTQ_MODEL_FILE")"
    MODEL_ARGS="${MODEL_ARGS},autogptq=${GPTQ_MODEL_FILE}"
fi

if command -v lm_eval >/dev/null 2>&1; then
    LM_EVAL_BIN="lm_eval"
elif [ -x "/opt/conda/envs/llm-quant/bin/lm_eval" ]; then
    LM_EVAL_BIN="/opt/conda/envs/llm-quant/bin/lm_eval"
elif command -v docker >/dev/null 2>&1 && [ -z "${RUN_EVAL_IN_DOCKER:-}" ]; then
    echo "로컬에 lm_eval 이 없어 llm-dev 컨테이너의 llm-quant 환경에서 재실행합니다."
    docker exec \
        -u "$(id -u):$(id -g)" \
        -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
        -e HOME="${HOME}" \
        -e XDG_CACHE_HOME="${XDG_CACHE_HOME}" \
        -e HF_HOME="${HF_HOME}" \
        -e HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE}" \
        -e TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}" \
        -e RUN_EVAL_IN_DOCKER=1 \
        llm-dev \
        bash -lc "source /opt/conda/etc/profile.d/conda.sh && conda activate llm-quant && cd /home/choihyun/workspace && bash src/run_eval.sh \"$MODEL_PATH\" \"$OUTPUT_NAME\""
    exit $?
else
    echo "오류: lm_eval 실행 파일을 찾을 수 없습니다."
    echo "호스트에서는 llm-dev 컨테이너가 실행 중이어야 하며, 컨테이너 내부에는 /opt/conda/envs/llm-quant/bin/lm_eval 이 있어야 합니다."
    exit 1
fi

DEVICE="${CUDA_VISIBLE_DEVICES:+cuda:0}"
DEVICE="${DEVICE:-cuda:0}"

"$LM_EVAL_BIN" --model hf \
    --model_args "$MODEL_ARGS" \
    --tasks $TASKS \
    --device "$DEVICE" \
    --batch_size 4 \
    --output_path results/eval_${OUTPUT_NAME}.json \
    --log_samples
