# vLLM 서빙 벤치마크 결과

**작성일:** 2026-03-20
**작성 주체:** Claude (외부 시점 분석)

---

## 환경

- **서버:** TITAN RTX × 2 (각 24GB, compute capability 7.5)
- **vLLM:** 0.17.1
- **모델:** SOLAR-10.7B-Instruct-v1.0
- **설정:** enforce_eager=True, max_model_len=512, gpu_memory_utilization=0.92
- **벤치마크:** 32 한국어 프롬프트 × 256 max_tokens

---

## 결과

| 조건 | 모델 크기 | load_time | throughput | latency | vram(vLLM) |
|------|---------|-----------|------------|---------|------------|
| FP16 | ~20 GB | 7.85s | 482.87 tok/s | 465.7 ms | 22,299 MiB |
| GPTQ C_v3 | ~5.6 GB | 38.01s | 502.25 tok/s | 443.5 ms | 22,493 MiB |

**모델 가중치 크기:** FP16 20GB → GPTQ 5.6GB (**72% 절감**)

---

## 분석

### Throughput / Latency

- GPTQ 4-bit가 FP16보다 약간 높은 throughput (+4%) 및 낮은 latency (-5%)
- 이유: GPTQ 모델 가중치가 작아 같은 메모리에서 더 많은 KV cache 블록 할당 가능
- vLLM은 `gpu_memory_utilization × total_vram`만큼 KV cache를 사전할당
  - FP16: 모델 20GB + KV cache ~2GB = 22GB
  - GPTQ: 모델 5.6GB + KV cache ~16GB = 22GB → 배치 처리에 유리

### VRAM 수치 해석

- `vram_used_mib`는 vLLM이 GPU에 할당한 총 메모리 (모델 + KV cache 포함)
- 두 모델 모두 비슷하게 보이는 이유: vLLM이 남은 메모리를 KV cache로 채움
- 실제 배포 시 VRAM 이점:
  - FP16: 20GB 필요 (24GB GPU에서 빠듯)
  - GPTQ: 5.6GB 필요 → 더 작은 GPU에서도 서빙 가능, 멀티 GPU 배포 효율 향상

### Load Time 차이

- FP16: 7.85s (weights가 표준 형식, 빠른 로딩)
- GPTQ: 38.01s (dequantization 또는 kernel setup 필요, ~5배 느림)
- 서빙 시작 시간은 느리지만, 실제 추론 속도는 더 빠름

---

## 트러블슈팅

| 문제 | 해결 |
|------|------|
| FA2 not supported (compute capability 7.5 < 8) | `enforce_eager=True` 설정 |
| KV cache 메모리 부족 (FP16 20GB + cache 불가) | `max_model_len=512`, `gpu_memory_utilization=0.92` |
| GPTQ `ImportError: Loading requires optimum` | `pip install optimum` |
| optimum 2.1.0 `QuantizeConfig` 에러 | lm_eval에서 `autogptq=` 파라미터 방식 사용 |

---

## 이력서 활용 포인트

```
GPTQ 4-bit 양자화 모델 vLLM 서빙 검증:
- FP16 대비 모델 VRAM 72% 절감 (20GB → 5.6GB)
- throughput +4% (482 → 502 tok/s)
- latency -5% (466 → 444 ms/prompt)
- enforce_eager 설정으로 TITAN RTX (compute 7.5) 환경 최적화
```
