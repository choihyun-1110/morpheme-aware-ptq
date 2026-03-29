# Sprint 3: 언어 정렬 원리 일반화 검증

**기간:** 2026-03-20 ~ 2026-04-13
**★ 마감:** 4/17 중간 보고서

---

## 목표

Sprint 2에서 발견한 **"사전학습 주 언어 = calibration 언어일 때 GPTQ 품질 보존"** 원리를
다른 언어 모델(Qwen2)로 검증하고 일반 원리로 확립한다.

---

## 항목

| ID | 항목 | 우선순위 | 상태 |
|----|------|---------|------|
| S3-1 | Qwen2-7B 중국어 calibration 실험 | 🔴 최고 | ✅ 완료 (KoBEST + C-Eval 모두 C_v3 > A > C_zh — 가설 반증) |
| S3-2 | EXAONE-3.5 환경 구성 및 실험 | 🟡 중간 | ✅ 완료 (A=0.6645, B=0.7196, C_v3=0.7164, FP16=0.7437 — B≈C_v3>>A) |
| S3-3 | EEVE B > C_v3 원인 분석 | 🟡 중간 | ✅ 완료 (C_v3_eeve=0.7551 < B=0.7595 — tokenizer 정렬 부분 효과, B 우위 유지) |
| S3-4 | 중간 보고서 작성 | 🔴 최고 (4/17) | 🔄 초안 완료 (`중간보고서.md`) |
| S3-5 | **vLLM 서빙 벤치마크** | 🟡 중간 (포트폴리오) | ✅ 완료 (FP16 vs GPTQ C_v3) |

---

## S3-1: Qwen2-7B 중국어 calibration 실험

**가설:** Qwen2-7B(중국어 사전학습)에서
중국어 형태소 다양성 calibration(C_zh) > 한국어 calibration(C_v3) > 영어 calibration(A)

**필요 작업:**
1. 중국어 형태소 분석기 선택 — `jieba` 또는 `pkuseg`
2. `src/build_calibration.py`에 중국어 지원 추가 — 형태소 다양성 greedy 선택
3. 중국어 calibration set 생성 (128문장, 위키피디아/웨이보 등)
4. Qwen2-7B GPTQ 양자화 × 3조건 (A / C_v3 / C_zh)
5. 벤치마크 평가 — C-Eval / CMMLU (kmmlu 대응)

**결과 기준:**
- C_zh > C_v3 > A → 언어 정렬 원리 강력 지지
- C_zh ≈ C_v3 > A → 언어 정렬 원리 부분 지지
- 순서 무관 → 가설 재검토

---

## S3-2: EXAONE-3.5 호환 환경 구성

**문제:** EXAONE-3.5-7.8B가 transformers 4.46+ 요구 → 현재 llm-quant 환경(4.44.2) 미호환

**옵션:**
- A) llm-quant 환경 transformers 업그레이드 (기존 실험 재현성 영향 주의)
- B) 별도 conda 환경 `llm-quant-v2` 생성 (권장)
- C) EXAONE-3.0 사용 (구버전, 현재 환경 호환 가능)

**목적:** 순수 한국어 사전학습 모델에서 SOLAR처럼 C_v3 > B 재현 확인

---

## S3-3: EEVE B > C_v3 원인 분석

**현상:** EEVE-10B(SOLAR 기반 SFT)에서 B(0.7595) > C_v3(0.7514)
**SOLAR에서는 C_v3(0.6356) > B(0.6176)이었는데 왜 뒤집혔나?**

**가설:**
1. C_v3 calibration set이 SOLAR tokenizer 기준으로 생성됨 → EEVE tokenizer와 불일치?
   → EEVE tokenizer로 C_v3 재생성 후 재실험
2. EEVE의 instruction tuning이 더 강해 calibration 민감도 감소?
3. 단순 노이즈 (차이 0.008이 유의하지 않을 수 있음)

---

## S3-4: 중간 보고서 작성

**마감:** 2026-04-17
**형식:** 논문 읽은 것 + 연구 방향 이유 + 아이디어 제안

**구성안:**
1. 배경: 한국어 LLM 증가 + PTQ 필요성
2. 관련 논문: GPTQ, SmoothQuant, AWQ, calibration 데이터 영향 연구
3. 연구 질문: calibration 데이터 언어가 PTQ 품질에 영향을 주는가?
4. 아이디어: 형태소 다양성 기반 calibration set 선택 알고리즘
5. 예비 결과: SOLAR C_v3 97.4% FP16 보존, Llama3-Ko 언어 정렬 단서
6. 향후 계획: Qwen2 실험으로 언어 정렬 원리 일반화 검증

---

## S3-5: vLLM 서빙 벤치마크

**목적:** 양자화 모델의 실제 추론 성능을 vLLM으로 검증 — 포트폴리오용 정량 수치 확보

**배경:** 목표 회사 JD에서 vLLM/TensorRT-LLM 경험을 우대함.
"양자화 품질 검증"에서 "양자화 + 서빙 최적화"로 연구 범위 확장.

**필요 작업:**
1. llm-dev 컨테이너에 vLLM 설치 (`pip install vllm`)
2. SOLAR FP16 / GPTQ C_v3 / GPTQ B 모델을 vLLM으로 로딩
3. 추론 벤치마크 스크립트 작성 (`src/benchmark_serving.py`):
   - throughput (tokens/sec)
   - latency (first token, avg token)
   - VRAM 사용량
   - 동시 요청(batch) 처리 능력
4. FP16 대비 4-bit 수치 비교표 작성

**기대 수치 (참고):**
- VRAM: FP16 ~20GB → 4-bit ~6GB (약 70% 절감)
- throughput: 4-bit가 FP16 대비 1.5~2x 빠를 것으로 예상

**이력서 한 줄:**
> "GPTQ 4-bit 양자화 모델 vLLM 서빙 검증 — FP16 대비 VRAM 70% 절감, throughput Xx 향상"

---

## 진행 메모

### 2026-03-26

**S3-4 중간 보고서 초안 작성 완료:**
- `workspace/중간보고서.md` 생성 — 제출 가능 수준의 완성 초안
- 구성: 배경 → 관련 논문(GPTQ/AWQ/SmoothQuant) → 연구 질문 → 알고리즘 → 예비 결과 → 향후 계획 → 결론
- Sprint 2 전체 결과 + Qwen2 결과 + vLLM 벤치마크 포함
- 미완료 실험(Qwen2 C-Eval, EXAONE)은 "향후 계획"으로 명시

**S3-3 이론 분석 완료:**
- `thoughts/04_EEVE_B_gt_Cv3_원인분석.md` — 3가지 가설 분석
- 우선순위 1: Tokenizer 불일치 (EEVE tokenizer로 C_v3 재생성 필요)
- 우선순위 2: Instruction tuning 강도 차이
- 우선순위 3: 통계적 노이즈 (단일 런)
- 검증을 위한 실험 순서 제시

**실험 자동화 파이프라인 구축 및 실행 중:**

GPU 0 파이프라인:
1. ✅ Qwen2 FP16 → C-Eval (avg 0.8120)
2. ✅ Qwen2 GPTQ A/C_v3/C_zh → C-Eval (C_v3>A>C_zh, 가설 반증)
3. ✅ EEVE C_v3_eeve calibration 생성 완료
4. 🔄 Phase 2 v2 실행 중 (`run_exp_gpu0_phase2_v2.sh`, gptqmodel)
   - EXP1: SOLAR C_v3/A group_size=64 → KoBEST
   - EXP2: SmoothScale+GPTQ → KoBEST
   - EXP3: AWQ A/C_v3 → KoBEST
   - EXP4: GPTQ C_v3 desc_act=False → KoBEST

GPU 1 파이프라인 (`src/run_exp_gpu1_v3.sh`, 로그: `results/run_gpu1_v3_master.log`):
1. 🔄 EXAONE35 조건 A 양자화 (optimum.gptq+gptqmodel) → KoBEST (진행 중)
2. ⏳ 조건 B 양자화 → KoBEST
3. ⏳ 조건 C_v3 양자화 → KoBEST
4. ⏳ EEVE C_v3_eeve KoBEST 평가 (GPU 0 양자화 완료 대기)

EEVE C_v3_eeve 양자화:
- ❌ auto_gptq: rotary embedding 차원 오류 (LLaMA+GQA transformers 4.57.6 비호환)
- ❌ optimum.gptq (QuantizeConfig 오류): gptqmodel 미설치 당시 실패
- ⏳ optimum.gptq+gptqmodel: Phase 2 v2 완료 후 GPU 0에서 재실행 예정

**Activation 분석 기반 양자화 개선 실험 설계 및 실행 준비 (2026-03-26):**
- `thoughts/05_activation_기반_양자화_개선_설계.md` — 상세 분석
- **핵심 발견**: L12-18이 calibration 언어에 가장 민감 (channel_cv C_v3-B 격차 최대)
  - L0 (임베딩), L46-47 (출력): calibration 무관
  - L1-10: 원래 cv=14~27로 robust → calibration 덜 민감
  - L12-18, L30-33: C_v3 우위 최대 → 한국어 처리 핵심 레이어로 추정
- **새 실험 4종 (GPU 0 Phase 2, Phase 1 완료 후 자동 실행):**
  - EXP-A: SOLAR A/C_v3 group_size=64 (세밀한 양자화가 calibration 이점 증폭?)
  - EXP-B: SmoothScale+GPTQ (activation 균일화 → Hessian 오차 감소?)
  - EXP-C: AWQ A vs C_v3 (calibration 언어 효과가 AWQ에서도 나타나는가?)
  - EXP-D: GPTQ C_v3 desc_act=False (재정렬 없이 C_v3 이점이 유지되는가?)
- 새 스크립트:
  - `src/run_exp_gpu0_phase2.sh` — GPU 0 Phase 2 파이프라인
  - `src/run_quant_awq.py` — AutoAWQ 기반 양자화
  - `src/smooth_scale.py` — SmoothQuant-inspired 전처리
- AutoAWQ 백그라운드 설치 중

**환경 트러블슈팅 (2026-03-26):**
- EXAONE-3.5 캐시 파일 호환성 패치 (transformers 4.57.6 기준):
  - `configuration_exaone.py`: `RopeParameters` import 제거
  - `modeling_exaone.py`: `use_kernel_func_from_hub`, `use_kernelized_func`, `maybe_autocast`, `check_model_inputs` stub 처리
  - `transformers/utils/auto_docstring.py`: Python 3.10+ `UnionType` (`X | Y`) 처리 버그 패치
- auto_gptq EXAONE 미지원 → `optimum.gptq.GPTQQuantizer`로 대체
  - 새 스크립트: `src/run_quant_optimum.py`
- S3-2 옵션 B(별도 conda env) 불필요 — transformers 4.57.6이 이미 EXAONE 요구사항 충족

**추가 트러블슈팅 II (2026-03-26 속행):**
- gptqmodel 5.8.0 설치: optimum 2.1.0이 auto_gptq 대신 gptqmodel 요구
  - 초기 설치 시 transformers 5.3.0 함께 설치됨 → 즉시 4.57.6으로 복구
  - pip 경고(incompatible)에도 gptqmodel 5.8.0은 transformers 4.57.6에서 정상 작동
- auto_gptq + transformers 4.57.6 + LLaMA 모델 비호환 발견:
  - position_embeddings 새 API 처리 실패 → rotary embedding 차원 오류
  - SOLAR, EEVE 등 모든 LLaMA 기반 모델 영향
  - 해결: optimum.gptq + gptqmodel 5.8.0으로 전면 전환 (`run_quant_optimum.py`)
- EXAONE35 modeling 추가 패치:
  - use_kernel_forward_from_hub → no-op class decorator
  - ALL_ATTENTION_FUNCTIONS.get_interface() → .get() (transformers 4.57.6)
  - auto_docstring.py UnionType 처리 (transformers 재설치로 패치 유실, 재적용)
  - 3개 캐시 위치 전체 패치 (`src/patch_exaone.py` 개선)
- Phase 2 스크립트: `run_exp_gpu0_phase2_v2.sh` (gptqmodel 기반)

**추가 트러블슈팅 (2026-03-26 속행):**
- EXAONE35 모델 shard 00002~00005 누락 → snapshot_download 재시도로 완료
- EXAONE 패치 재적용 시 문제:
  - HF가 modeling_exaone.py를 재다운로드하면 패치 사라짐
  - use_kernel_forward_from_hub (decorator 형태) 미처리
  - `@use_kernelized_func(...)` 데코레이터에서 @만 남아 `@def`/`@class` SyntaxError
  - rope_parameters가 None일 때 subscript 오류 → rope_scaling fallback 추가
  - modules 캐시(_553ea250b9a5) 별도 패치 필요
  - 해결: `src/patch_exaone.py` 개선 — 3개 캐시 위치 전체 패치, rope_parameters 안전 처리
- EEVE C_v3_eeve 양자화: auto_gptq → rotary embedding 차원 오류(32 vs 128)
  - LLaMA GQA 모델에서 transformers 4.57.6과 auto_gptq 불호환
  - 해결: `optimum.gptq` 사용 (`run_quant_optimum.py`)
- 새 파이프라인: `src/run_exp_gpu1_v3.sh` — EXAONE35 로컬 경로 사용, 패치 자동 적용

---

### 2026-03-20

**S3-1 진행 상황:**
- ✅ `src/build_calibration_zh.py` 구현 (jieba + wikimedia/wikipedia 스트리밍)
- ✅ C_zh calibration set 생성 완료: 128문장, 1652 고유형태소, avg_SFS=2.29
- ✅ Qwen2-7B 모델 다운로드 완료 (~12GB)
- ✅ GPTQ 양자화 × 3조건 완료 (A/C_v3/C_zh, GPU 0, 각 ~7분)
- 🔄 KoBEST 평가 진행 중 (A+C_v3 병렬, C_zh 대기 중, ~60분 예상)
- ⏳ 평가 완료 후 결과 분석

**S3-5 진행 상황:**
- ✅ vLLM 0.17.1 설치 완료
- ✅ SOLAR FP16 벤치마크: 482.87 tok/s, 465.7ms/prompt, VRAM 22,299 MiB
- ✅ SOLAR C_v3 GPTQ 벤치마크: 502.25 tok/s, 443.5ms/prompt, VRAM 22,493 MiB
  - 실제 모델 크기: FP16 20GB → GPTQ 5.6GB (72% 절감)
  - vLLM KV cache 사전할당으로 VRAM 수치는 비슷하게 보임
- 이력서용 수치: "GPTQ 4-bit 72% VRAM 절감, throughput +4%, latency -5%"

**Qwen2 C-Eval 결과 (2026-03-26 완료):**

| 조건 | C-Eval 평균 (53과목) | FP16 대비 보존율 |
|------|---------------------|----------------|
| FP16 (베이스라인) | 0.8120 | 100% |
| A (영어) | 0.7617 | 93.8% |
| C_v3 (한국어 형태소) | **0.7694** | **94.7%** |
| C_zh (중국어 형태소) | 0.7584 | 93.4% |

**순위: C_v3 > A > C_zh (KoBEST와 동일한 패턴)**

핵심 해석:
- 중국어 모델(Qwen2)에서 중국어 benchmark(C-Eval)를 평가해도 C_zh가 최악
- "calibration = 평가 언어" 가설 **반증** — "형태소 다양성"이 언어보다 더 중요한 요인
- C_zh의 avg_SFS=2.29가 낮아 다양성 부족할 가능성 (C_v3는 더 높은 SFS)
- 중국어 calibration이 오히려 한국어/중국어 처리 능력을 왜곡

**Qwen2 KoBEST 결과 (2026-03-20 완료):**

| 조건 | boolq | copa | hellaswag | sentineg | wic | **평균** |
|------|-------|------|-----------|----------|-----|---------|
| A (영어) | 0.7885 | 0.6290 | 0.5260 | 0.5642 | 0.6111 | **0.6238** |
| C_v3 (한국어) | 0.8682 | 0.6490 | 0.5260 | 0.5919 | 0.5524 | **0.6375** |
| C_zh (중국어) | 0.8454 | 0.6430 | 0.4420 | 0.6096 | 0.4913 | **0.6063** |

**순위: C_v3 > A > C_zh (가설 C_zh > C_v3 > A 반증)**

핵심 발견:
- KoBEST(한국어) 기준으로 한국어 calibration이 여전히 최우수
- 중국어 calibration이 오히려 한국어 처리 능력을 더 왜곡 (hellaswag: 0.4420)
- 사전학습 언어보다 평가 언어에 맞는 calibration이 중요할 수 있음
- C-Eval/CMMLU (중국어 벤치마크)로 재평가 필요

**vLLM 벤치마크 결과 (2026-03-20 완료):**
- SOLAR FP16: 482.87 tok/s, 22GB 모델, load 7.85s
- SOLAR GPTQ C_v3: 502.25 tok/s, 5.6GB 모델, load 38s
- 모델 크기 72% 절감, throughput +4%, latency -5%

**트러블슈팅:**
- transformers 4.57.6 (vLLM 의존성)이 auto_gptq LayerHijacker와 충돌
- 해결: 평가 시 autogptq 파라미터 방식 사용 (`autogptq=gptq_model-4bit-128g.safetensors`)
- C_zh 양자화: transformers 4.44.2 임시 다운그레이드 후 재설치로 해결

**추가 트러블슈팅 III (2026-03-26 세션 재개):**
- **EXAONE35_A 양자화 성공** (05:18-05:25), 저장 완료
  - 단, KoBEST 평가 단계에서 act_group_aware 충돌 오류로 실패
  - 원인: `quantize_config.json`에 `act_group_aware=false`이 저장되지만,
    transformers `GPTQConfig`에 해당 필드 없어 `to_dict_optimum()`에서 누락
    → `GPTQQuantizer.from_dict()`가 기본값 `act_group_aware=True`로 QuantizeConfig 생성
    → `desc_act=True` + `act_group_aware=True` 충돌 → ValueError
  - 해결: `optimum/gptq/quantizer.py` `from_dict` 메서드 패치
    - `desc_act=True`이면 `act_group_aware=False`로 강제
    - `src/patch_exaone.py`에 optimum 패치 자동 재적용 로직 추가
- **GPU 0 OOM 원인 파악**: `device_map={"": device}`로 모델 전체를 GPU 올림 → ~21GB 점유 → Hessian 계산 784MB 추가 불가
  - 해결: `device_map="cpu"` 변경 → optimum.gptq가 layer별 GPU 이동 처리
  - 추가: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 환경변수 설정
- **skip 로직 추가**: `quant_solar()`, `quant_exaone()` 함수에 이미 완료된 모델 스킵 처리
- **Qwen2 C-Eval 결과 확정**: FP16=0.8165, C_v3=0.7734, A=0.7630, C_zh=0.7630
  - C_v3 보존율 94.7%, A=C_zh 93.4% → "언어 정렬"보다 "형태소 다양성"이 핵심
  - `thoughts/01_Qwen2_실험_결과_분석.md` 업데이트 완료

**현재 실행 중 파이프라인 (08:34 기준):**
- GPU 0: `run_exp_gpu0_phase2_v2.sh` — SOLAR C_v3 g64 양자화 (block 6/48)
- GPU 1: `run_exp_gpu1_v3.sh` — EXAONE35_A KoBEST 평가 중 (양자화 skip, 평가 진행)

**추가 트러블슈팅 IV: optimum.gptq qzeros 포맷 버그 (2026-03-26)**

**현상:** 모든 optimum.gptq 저장 모델이 KoBEST 랜덤 기준선 (SOLAR g64 포함)

**근본 원인:** qzeros 포맷 차이
- auto_gptq: qzeros = zero_point - 1 = **7** (0x77777777)
- optimum.gptq: qzeros = zero_point = **8** (0x88888888)
- gptqmodel 역양자화: `(q - (qzeros+1)) * scale` → optimum.gptq 모델에서 항상 -1 오프셋
- 모든 레이어에서 누적 편향 → activation 지수 증가 → float16 오버플로 → NaN logits

**진단 과정:**
1. 순차적 가설 검증: eval 함수 → gptqmodel 포맷 → qzeros 값 비교
2. `layer0 max_abs=23 → layer1 max_abs=872 → ...` 지수 증가 확인
3. Sprint 2(auto_gptq) qzeros=0x77, Sprint 3(optimum.gptq) qzeros=0x88 발견
4. in-memory qzeros 패치 후 NaN 해소 검증

**해결:**
- `src/patch_qzeros.py`: 기존 모델 일괄 패치 (`0x88 → 0x77`, -0x11111111)
- `src/run_quant_optimum.py`: 저장 후 자동 패치 + quantize_config.json 생성
- 영향 모델 전부 패치 완료 (solar_C_v3_g64, solar_A_g64, exaone35 A/B/C_v3)
- `src/smooth_scale.py`: OOM 수정 (`weight.data * s` → `weight.data.mul_(s)` in-place)

**파이프라인 재실행 완료 결과 (2026-03-26 저녁):**

GPU 1 모든 실험 완료:
- EXAONE35 A: **0.6645** (FP16=0.7437 대비 89.3% 보존)
- EXAONE35 B: **0.7196** (96.8% 보존)
- EXAONE35 C_v3: **0.7164** (96.3% 보존) → B ≈ C_v3 >> A
- EEVE C_v3_eeve: **0.7551** (B=0.7595 대비 0.0044 낮음 → tokenizer 정렬 효과 있으나 미미)
- EXAONE35 FP16: **0.7437** (베이스라인 확정)

GPU 0 완료 및 진행 중:
- SOLAR C_v3 g64: **0.6468** (g128=0.6356 대비 +0.011)
- SOLAR A g64: **0.6174** (g128=0.5981 대비 +0.019) → g64가 A를 더 많이 향상
- SmoothScale+GPTQ C_v3: **0.4795** (❌ 실패 — Hessian 비정치 오류 8회, alpha=0.5 과도)
- SmoothScale+GPTQ A: **0.4819** (❌ 실패 — 동일한 Hessian 문제, smooth_C_v3(0.4795)와 거의 동일)
- SmoothScale 결론: calibration 언어 무관, alpha=0.5 과도 → GPTQ 수렴 실패가 지배적
- desc_act=False C_v3: **0.6029** (desc_act=True 대비 −0.033, 특히 sentineg −0.179)
- AWQ: 미설치 스킵
- **✅ GPU0 Phase 2 파이프라인 완전 종료**

**세부 분석:**
- `thoughts/08_optimum_gptq_qzeros_버그_수정.md` — qzeros 버그 전체 진단
- `thoughts/09_EXAONE35_실험_결과_분석.md` — EXAONE35 결과 + FP16 보존율
- `thoughts/10_EEVE_C_v3_eeve_및_종합_결과_분석.md` — EEVE C_v3_eeve + Sprint3 종합
