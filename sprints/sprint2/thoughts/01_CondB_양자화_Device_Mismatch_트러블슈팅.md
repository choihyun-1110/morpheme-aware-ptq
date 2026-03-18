# S2-1/S2-3: Cond B 양자화 Device Mismatch 트러블슈팅

**작성일시:** 2026-03-17  
**관련 항목:** S2-1, S2-3

## 1. 상황

`cond_B` 양자화를 `llm-dev` 컨테이너의 `llm-quant` 환경에서 재개했으나, 다음 오류가 반복 발생했다.

```text
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0!
```

발생 지점은 `transformers.models.llama.modeling_llama.LlamaRotaryEmbedding.forward()` 내부였다.

## 2. 초기 가설

처음에는 `src/run_quant.py`가 calibration example을 무조건 `cuda:0`로 올리는 부분이 원인이라고 보았다.

```python
"input_ids": encodings["input_ids"].to("cuda:0")
"attention_mask": encodings["attention_mask"].to("cuda:0")
```

이에 따라 calibration 입력을 CPU 상태로 유지하도록 수정했다.

## 3. 추가 확인

그러나 동일 오류가 재발했다. 그래서 컨테이너 안에서 모델 로드 직후 실제 버퍼 디바이스를 점검했다.

확인 결과:

- `wrapper`: `LlamaForCausalLM`
- `base`: `LlamaModel`
- `rotary_type`: `LlamaRotaryEmbedding`
- `inv_freq_device`: `cpu`
- `original_inv_freq_device`: `cpu`

즉, 문제의 핵심은 입력 텐서보다도 **rotary embedding 버퍼가 CPU에 남아 있었다는 점**이다.

## 4. 원인 해석

현재 조합은 다음과 같다.

- `transformers 4.44.2`
- `auto_gptq 0.7.1`
- `torch 2.5.1+cu121`

`AutoGPTQForCausalLM.from_pretrained()` 이후에도 `rotary_emb.inv_freq` 계열 버퍼가 GPU로 옮겨지지 않았고, 양자화 중 첫 forward에서 `hidden_states`는 `cuda:0`, `inv_freq`는 `cpu` 상태로 만나며 실패한 것으로 보인다.

## 5. 적용한 조치

`src/run_quant.py`에 아래 두 가지 수정 적용:

1. calibration example은 CPU 상태로 유지
2. 모델 로드 직후 `rotary_emb.inv_freq`, `rotary_emb.original_inv_freq`를 `cuda:0`로 명시 이동

또한 `device_map`도 `auto` 대신 `{"": "cuda:0"}`로 고정했다.

### 구현상 주의점

처음에는 `register_buffer()`로 기존 버퍼를 다시 등록하려 했는데, 이 방식은 이미 존재하는 이름(`original_inv_freq`)에 대해 `KeyError`를 발생시켰다.

```text
KeyError: "attribute 'original_inv_freq' already exists"
```

따라서 최종적으로는 **버퍼 재등록이 아니라 기존 버퍼 값을 같은 이름으로 덮어쓰는 방식**으로 수정했다.

```python
rotary_emb.inv_freq = rotary_emb.inv_freq.to(device)
rotary_emb.original_inv_freq = rotary_emb.original_inv_freq.to(device)
```

## 6. 다음 액션

- 수정 후 `cond_B` 양자화를 재실행
- 성공 시 `quantized_models/SOLAR_10.7B_4bit_cond_B/` 생성 확인
- 이후 `run_eval.sh`로 `cond_B` KoBEST 평가 수행
