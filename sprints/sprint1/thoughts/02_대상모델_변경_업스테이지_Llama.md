# S1-1. 대상 모델 우선순위 변경 및 업스테이지(Solar) / Llama 스터디

## 1. 모델 선정 방향 변경
초기 계획했던 `EXAONE 3.5` 모델은 현재 `auto-gptq` 라이브러리에서 아키텍처가 기본 지원되지 않아 커스텀 코드를 작성해야 하는 호환성 이슈가 발견되었습니다.
연구의 핵심 목적은 "형태소 기반 데이터 교정이 양자화 성능 향상에 기여하는가"를 증명하는 것이므로, 특정 모델 호환성 문제에 너무 많은 시간을 쏟기보다는 **검증된 범용 아키텍처**를 우선 사용하여 실험 파이프라인(Calibration -> Quantization -> PPL/KoBEST 평가)을 끝까지 관철시키는 것이 유리합니다.

이에 따라 사용자의 의견을 수렴하여 다음과 같이 모델 우선순위를 조정합니다.

1. **1순위 (Primary): 업스테이지 (Solar 등) 모델**
   - 이유: 사용자가 업스테이지 앰배서더로 활동 중이며, Llama 아키텍처를 기반으로 깊이(Depth) 확장을 적용하여 기본적으로 `LlamaForCausalLM` 구조로 로드됨. 따라서 Hf/GPTQ 호환성이 매우 우수함.
   - 예시 모델: `upstage/SOLAR-10.7B-Instruct-v1.0` 등

2. **2순위 (Secondary): 오픈 Llama 3 / 3.1 기반 한국어 모델**
   - 이유: `auto-gptq` 및 `transformers` 생태계가 가장 완벽하게 지원하는 모델. 에지 케이스(버그) 없이 연구 방법론 자체의 차이를 검증하기 좋음.
   - 예시 모델: `Bllossom/llama-3.1-Korean-8B-Instruct` 또는 `beomi/Llama-3-Open-Ko-8B`

3. **3순위 (Tertiary): EXAONE 3.5 등 로컬/커스텀 아키텍처**
   - 이유: 프레임워크와 스크립트가 완성된 후, EXAONE 커스텀 매핑 코드를 작성하여 추가 검증 모델로 편입.

---

## 2. Solar 10.7B 아키텍처 상세 스터디

### 2-1. Depth Up-Scaling (DUS)
Solar 10.7B의 핵심은 **Depth Up-Scaling(DUS)** 기법입니다. 기존 MoE(Mixture of Experts)와 달리, 네트워크의 **너비(width)가 아닌 깊이(depth)를 확장**하여 성능을 올리는 방식입니다.

| 항목 | 상세 |
|------|------|
| **베이스 모델** | Mistral 7B (Llama 2 구조의 32-layer 모델) 가중치로 초기화 |
| **확장 방법** | 모델을 복제 → 원본의 마지막 8레이어 제거 + 복제본의 처음 8레이어 제거 → 나머지를 concatenation |
| **최종 레이어 수** | **48 layers** (32-8 = 24개 + 32-8 = 24개) |
| **파라미터 수** | **10.7B** |
| **후처리** | 결합부(junction)의 이질성 해소를 위한 **continued pretraining** 수행 |
| **장점** | MoE 대비 학습/추론 파이프라인이 단순. 30B급 모델(Mixtral 8x7B 등)보다 작은 크기로 동등 이상 성능 |

### 2-2. 핵심 아키텍처 구성 요소 (Llama 2 계열)
| 구성 요소 | 사양 |
|-----------|------|
| **Attention** | Multi-Head Attention (MHA) |
| **Positional Encoding** | RoPE (Rotary Positional Embedding) |
| **FFN** | SwiGLU activation |
| **Normalization** | RMSNorm (pre-norm) |
| **Vocab Size** | 32,000 (Llama 2 tokenizer) |
| **Hidden Dim** | 4,096 |
| **Intermediate Dim** | 14,336 |
| **Attention Heads** | 32 |

### 2-3. AutoGPTQ 호환성
- `model.config.model_type == "llama"` → AutoGPTQ의 내부 `GPTQ_CAUSAL_LM_MODEL_MAP`에 **완벽 매핑**.
- `AutoGPTQForCausalLM.from_pretrained()` / `.from_quantized()` 정상 동작 확인됨.
- GPTQ 4-bit (group_size=128) 양자화 후 약 **6~7GB VRAM** 소요 → TITAN RTX(24GB) 1장에서 여유롭게 추론 가능.
- ✅ **Sprint 0에서 서버 호환성 검증 완료:** AutoGPTQ 내부에서 `llama` 아키텍처로 완벽하게 인식됨.

### 2-4. 한국어 학습 비중
- 공식 수치 비공개. 그러나 Upstage 기술 블로그 및 벤치마크 결과에 따르면 한국어 성능이 동급 모델 대비 우수.
- Llama 2 tokenizer(vocab 32K)를 사용하므로, 한국어의 subword fragmentation이 상대적으로 높을 수 있음 → **본 연구의 calibration 다양성 가설 검증에 적합한 조건**.

---

## 3. Llama 3.1 8B 한국어 모델 아키텍처 스터디

### 3-1. 베이스: Llama 3.1 8B 구조
| 구성 요소 | 사양 |
|-----------|------|
| **레이어 수** | **32 layers** |
| **Attention** | **GQA (Grouped-Query Attention)** — KV head를 그룹화하여 추론 효율 향상 |
| **Positional Encoding** | RoPE (base freq = **500,000**) → 최대 128K 토큰 컨텍스트 |
| **FFN** | SwiGLU activation |
| **Normalization** | RMSNorm |
| **Vocab Size** | ~128,256 (Llama 3 tokenizer, 대폭 확장) |
| **Hidden Dim** | 4,096 |
| **Attention Heads** | 32 (KV heads: 8, GQA ratio 4:1) |

### 3-2. 주요 한국어 파인튜닝 모델

#### (a) Bllossom/llama-3.1-Korean-8B-Instruct
- 서울과기대 MLPLab + Teddysum + 연세대 공동 개발
- **한국어-영어 이중 언어 최적화**: 한국어 어휘 확장, 한국어 문화 기반 instruction tuning
- Llama 3.1 tokenizer 기반이므로 vocab ~128K → 한국어 fragmentation이 Solar 대비 낮을 것으로 예상

#### (b) beomi/Llama-3-Open-Ko-8B
- 개발자: 이준범 (Beomi)
- **60GB+ 한국어 공개 텍스트** (17.7B+ 토큰)로 continued pretraining
- Google TRC 프로그램 지원, TPUv5e-256으로 학습
- 표준 Llama 3 tokenizer 사용

### 3-3. AutoGPTQ 호환성
- `model.config.model_type == "llama"` → Solar와 마찬가지로 AutoGPTQ에 **완벽 호환**.
- 8B 모델이므로 양자화(4-bit) 후 약 **5GB VRAM** → 실험 환경에 매우 적합.
- GQA 구조가 양자화 시 KV cache 효율에 영향을 줄 수 있음 — 분석 포인트.

### 3-4. 연구 관점에서의 비교 포인트 (Solar vs Llama 3.1 Ko)

| 비교 항목 | Solar 10.7B | Llama 3.1 Ko 8B |
|-----------|-------------|-----------------|
| **Tokenizer vocab** | 32K (Llama 2) | ~128K (Llama 3) |
| **한국어 fragmentation 예상** | 높음 (작은 vocab) | 중간 (확장된 vocab) |
| **레이어 수** | 48 (DUS) | 32 |
| **Attention 방식** | MHA | GQA (4:1) |
| **파라미터 수** | 10.7B | 8B |
| **양자화 후 VRAM** | ~6-7GB | ~5GB |
| **AutoGPTQ 호환** | ✅ 완벽 | ✅ 완벽 |

> 💡 **Insight:** Tokenizer vocab 크기 차이(32K vs 128K)는 같은 한국어 문장에 대해 서로 다른 subword fragmentation 패턴을 만들어낸다. 이 차이가 calibration set 구성 알고리즘(S1-2)에서 형태소-토큰 정렬(alignment)에 어떤 영향을 미치는지가 핵심 분석 포인트가 될 것이다.

---

## 4. EXAONE 3.5 — 현재 상태 요약 (3순위 보류)

| 항목 | 상태 |
|------|------|
| 아키텍처 | `ExaoneForCausalLM` — 커스텀 decoder-only Transformer |
| AutoGPTQ 지원 | ❌ `GPTQ_CAUSAL_LM_MODEL_MAP`에 미등록. 커스텀 패치 필요 |
| 공식 양자화 | AWQ(W4A16g128), GGUF만 제공 |
| EXAONE 4.0 | GPTQ 공식 지원 시작 → 향후 EXAONE 3.5에도 커뮤니티 패치 기대 |
| **결론** | 실험 파이프라인 완성 후 3순위로 편입. 현 단계에서는 호환성 해결에 시간 투자 불필요 |

---

## 5. 최종 결론 및 다음 단계

### ✅ 확정 사항
1. **1순위 모델: `upstage/SOLAR-10.7B-Instruct-v1.0`** — AutoGPTQ 완벽 호환, 서버 검증 완료
2. **2순위 모델: `beomi/Llama-3-Open-Ko-8B` 또는 `Bllossom/llama-3.1-Korean-8B-Instruct`** — Llama 3 기반 완벽 호환
3. **3순위 모델: `LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct`** — 파이프라인 완성 후 추후 편입

### 📌 다음 단계 (→ S1-2)
- [x] 서버에서 `upstage/SOLAR-10.7B-Instruct-v1.0` 모델의 architecture 호환성 확인 완료 (결과: AutoGPTQ 내부에서 `llama` 아키텍처로 완벽하게 인식됨)
- [x] 모델 크기(10.7B)로 인해 다운로드 시간이 기므로 본격적인 파일럿/양자화 실행은 본 실험(데이터 구성 후)과 연계하여 진행.
- [x] Solar 10.7B / Llama 3.1 Ko 아키텍처 상세 스터디 완료 — DUS, GQA, tokenizer 차이점 정리
- [x] EXAONE 3.5 현재 상태 정리 및 3순위 보류 결정 확정
- **→ S1-2 "형태소 인식 캘리브레이션 셋 알고리즘 설계"로 전환**
  - Kiwi 형태소 분석 → LLM tokenizer 경계 정렬(alignment) 알고리즘 설계
  - Solar(32K vocab)와 Llama 3(128K vocab) 각각의 fragmentation 패턴 고려
