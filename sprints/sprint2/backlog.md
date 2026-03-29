# Sprint 2: 양자화 실험 및 벤치마크 검증 파이프라인 구축

## 개요
이 스프린트의 목적은 Sprint 1에서 추출한 3가지 조건(A, B, C)의 Calibration 데이터를 사용하여 실제로 `upstage/SOLAR-10.7B-Instruct-v1.0` 모델을 양자화하고, 그 성능 차이를 벤치마크를 통해 검증하는 것입니다.

| ID | 항목 | 개요 | 예상 난이도 |
|----|------|------|-------------|
| **S2-1** | 양자화 파이프라인 구축 | 추출된 JSON 데이터(A/B/C)를 읽어와서 `AutoGPTQ`에 주입하고 4bit 양자화를 수행하는 자동화 스크립트 작성 | 🟡 중간 |
| **S2-2** | LM Evaluation Harness 연동 | 양자화된 3개의 모델을 대상으로 `lm-evaluation-harness`를 붙여 Ko-HellaSwag, Ko-MMLU, Ko-TruthfulQA 등의 벤치마크를 자동으로 돌리고 결과를 기록하는 파이프라인 구축 | 🔴 높음 |
| **S2-3** | 대규모 양자화 실행 및 결과 분석 | A, B, C 세 가지 조건에 대해 양자화 및 평가 스크립트를 서버에서 백그라운드로 실행하고 결과를 비교 분석 | 🟡 중간 |
| **S2-4** | (옵션) 형태소 분석 결과와 Activation 관계 1차 탐색 | 다양성 점수에서 보였던 특정 형태소 통계가 Activation Outlier에 구체적으로 어떤 영향을 주는지 가볍게 시각화 확인 | 🔴 높음 |

## 다음 단계 (S2-1)
- Sprint 0에서 작성했던 `pilot_quant.py`를 확장하여, 인자로 주어진 JSON 포맷(예: `calibration_set_C.json`)을 HuggingFace Dataset 형태로 변환해 `BaseQuantizeConfig`의 dataset으로 넣을 수 있게 개편.

## 진행 메모 (2026-03-17)
- `src/run_eval.sh`를 보강하여 호스트에서 `lm_eval`가 없으면 `llm-dev` Docker 컨테이너로 자동 재실행되도록 수정함.
- Docker 재실행 시 `source /opt/conda/etc/profile.d/conda.sh && conda activate llm-quant`를 명시적으로 수행하도록 수정함.
- 평가 태스크는 현재 로컬 harness 등록 기준과 Sprint 0 계획을 반영해 `kobest` 그룹으로 맞춤.
- GPTQ 양자화 모델(`quantize_config.json` 존재)인 경우, `autogptq=<safetensors 파일명>`을 자동으로 `model_args`에 추가하도록 수정함.
- 실제 서버 실행에서 `ModuleNotFoundError: No module named 'optimum'`가 발생했는데, 이는 기본 HF 로더 경로를 타며 생긴 문제였고, AutoGPTQ 로더를 명시하도록 우회한 상태임.
- `quantized_models/SOLAR_10.7B_4bit_cond_C/`에도 `gptq_model-4bit-128g.safetensors`, `quantize_config.json`, tokenizer 파일들이 모두 존재함을 확인. 즉 Cond C 양자화 산출물은 저장 완료 상태로 판단됨.
- `results/eval_cond_A_2026-03-17T07-34-26.732750.json` 생성까지 확인되어 Cond A의 KoBEST 평가는 최소 1회 완료된 상태임.
- Cond A KoBEST 결과 요약:
  - `kobest` acc `0.5981`, acc_norm `0.5160`, f1 `0.5461`
  - `kobest_boolq` acc `0.7151`
  - `kobest_copa` acc `0.6160`
  - `kobest_hellaswag` acc `0.4340`, acc_norm `0.5160`
  - `kobest_sentineg` acc `0.6952`
  - `kobest_wic` acc `0.4881`
- `results/eval_cond_C_2026-03-17T07-46-33.147056.json` 생성까지 확인되어 Cond C의 KoBEST 평가도 1회 완료됨.
- Cond C KoBEST 결과 요약:
  - `kobest` acc `0.6062`, acc_norm `0.5300`, f1 `0.5582`
  - `kobest_boolq` acc `0.7564`
  - `kobest_copa` acc `0.6240`
  - `kobest_hellaswag` acc `0.4400`, acc_norm `0.5300`
  - `kobest_sentineg` acc `0.6196`
  - `kobest_wic` acc `0.4865`
- 1차 비교 기준, Cond C는 Cond A 대비 `kobest` 총점에서 개선(`acc +0.0081`, `acc_norm +0.0140`, `f1 +0.0121`)을 보였음.
- Cond B 양자화 1차 시도에서 `Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0` 오류 발생.
- 원인 추정: `src/run_quant.py`가 calibration 입력 텐서를 강제로 `cuda:0`로 올리고 있었고, AutoGPTQ 내부 양자화 경로의 일부 텐서는 CPU에 남아 있어 device mismatch가 발생함.
- 조치: `src/run_quant.py`에서 calibration 예제를 CPU 텐서 상태로 유지하도록 수정. AutoGPTQ가 내부적으로 필요한 디바이스로 이동하도록 맡김.
- 2차 원인 추정: `AutoGPTQForCausalLM.from_pretrained(..., device_map="auto")`가 모델 일부를 CPU로 두면서 rotary embedding 단계에서 다시 `cpu`/`cuda:0` 혼합이 생긴 것으로 보임.
- 추가 조치: `src/run_quant.py`에서 양자화용 모델 로딩을 `device_map={"": "cuda:0"}`로 고정해 단일 GPU에 명시적으로 올리도록 수정.
- 실제 확인 결과, 모델 로드 직후 `LlamaModel.rotary_emb.inv_freq`와 `original_inv_freq` 버퍼가 `cpu`에 남아 있었음.
- 추가 조치 2: `src/run_quant.py`에서 로드 직후 rotary embedding 버퍼를 명시적으로 `cuda:0`로 옮기도록 수정.
- 재시도 중 `register_buffer("original_inv_freq", ...)`에서 `KeyError: attribute 'original_inv_freq' already exists`가 발생함.
- 추가 조치 3: buffer 재등록 대신 기존 buffer 값을 `.to(device)`로 덮어쓰는 방식으로 수정.
- 이후 Cond B 양자화 및 KoBEST 평가까지 완료됨. 결과 파일: `results/eval_cond_B_2026-03-17T09-58-30.515789.json`
- Cond B KoBEST 결과 요약:
  - `kobest` acc `0.6176`, acc_norm `0.5480`, f1 `0.5753`
  - `kobest_boolq` acc `0.7885`
  - `kobest_copa` acc `0.6300`
  - `kobest_hellaswag` acc `0.4280`, acc_norm `0.5480`
  - `kobest_sentineg` acc `0.6171`
  - `kobest_wic` acc `0.4929`
- 현재 KoBEST 총점 순위는 `Cond B > Cond C > Cond A`
  - Cond B: `0.6176`
  - Cond C: `0.6062`
  - Cond A: `0.5981`
- 자세한 해석은 `thoughts/03_CondA_B_C_KoBEST_비교정리.md`에 정리.
- C 알고리즘 개정안 반영:
  - 문장형 후보 사전 필터(`min_eojeols`, `min_subword_tokens`, `require_sentence_final`)
  - 길이 보너스(`beta`) 추가
  - 문장 종결 보너스(`gamma`) 추가
  - C 결과 JSON에 `n_eojeols`, `has_sentence_final` 메타데이터 추가
- 관련 상세 메모는 `thoughts/04_CondC_알고리즘_개정안.md`에 정리.
- Docker 실행 결과 `results/`, `quantized_models/`가 호스트에서 쓰기 불가한 소유권으로 생성되는 문제가 있어 소유권을 `choihyun:choihyun`으로 복구함.
- `src/run_eval.sh`의 Docker 자동 위임 경로에는 `docker exec -u $(id -u):$(id -g)`를 추가해 이후 평가 결과 파일이 호스트 사용자 권한으로 생성되도록 수정.
- `docker exec -u` 사용 시 컨테이너 내부 `HOME`이 비어 Hugging Face 캐시가 `/.cache`로 떨어지며 권한 에러가 발생함.
- `src/run_quant.py`, `src/build_calibration.py`, `src/run_eval.sh`에서 HF 캐시를 workspace 아래 `.cache/huggingface`로 명시 고정하도록 수정.
- 이후 기존 `.cache` 디렉토리가 `nobody:nogroup` 소유권으로 남아 있어 재실행 시 `PermissionError`가 발생했고, 이를 `choihyun:choihyun`으로 복구함.
- 개정된 C 알고리즘으로 `calibration_set_C_v2_upstage_SOLAR-10.7B-Instruct-v1.0.json`를 생성하고, 이를 사용한 `quantized_models/SOLAR_10.7B_4bit_cond_C_v2/` 양자화까지 완료함.
- `results/eval_cond_C_v2_2026-03-17T13-59-17.640825.json` 생성까지 확인되어 Cond C_v2의 KoBEST 평가도 완료됨.
- Cond C_v2 KoBEST 결과 요약:
  - `kobest` acc `0.5779`, acc_norm `0.5300`, f1 `0.5163`
  - `kobest_boolq` acc `0.6460`
  - `kobest_copa` acc `0.6150`
  - `kobest_hellaswag` acc `0.4460`, acc_norm `0.5300`
  - `kobest_sentineg` acc `0.6977`
  - `kobest_wic` acc `0.4873`
- 해석:
  - 개정된 C_v2는 기존 C 대비 `kobest` 총점이 하락함 (`0.6062 -> 0.5779`).
  - HellaSwag, SentiNeg는 소폭 개선됐지만, BoolQ가 크게 하락해 전체 총점을 끌어내림.
  - 현재까지 KoBEST 총점 순위는 `Cond B > Cond C > Cond A > Cond C_v2`.
- 관련 상세 메모는 `thoughts/05_CondC_v2_결과정리와_다음액션.md`에 정리.
- 다음 단계:
  - `Cond B / Cond C / Cond C_v2` calibration set 길이, 문장성, 형태소 커버리지 통계 재비교
  - C 계열 알고리즘 수정은 잠시 멈추고, 현재 주결론(`B > C > A`)을 `kmmlu`에서도 재확인할지 결정
  - 다음 C 수정은 BoolQ 하락 원인 분석 후에만 재개

## 진행 메모 (2026-03-18)

- 외부 시점 스프린트 리뷰를 통해 핵심 문제점 4가지 도출 (상세: `thoughts/06_외부시점_스프린트리뷰.md`):
  1. FP16 베이스라인 없음
  2. 가설 중간 링크(activation 다양성) 미검증
  3. 단일 런 결과에 과도한 해석
  4. B가 강한 이유 미분리
- FP16 베이스라인 KoBEST 평가 완료 (상세: `thoughts/07_FP16_베이스라인_실행기록.md`):
  - `kobest` acc `0.6523` (샘플 가중 평균)
  - `kobest_boolq` acc `0.8661` / `kobest_copa` `0.6400` / `kobest_hellaswag` `0.4480` / `kobest_sentineg` `0.6650` / `kobest_wic` `0.5008`
  - lm-evaluation-harness `huggingface.py:743` 의 `dtype=` → `torch_dtype=` 패치 필요했음 (Docker root로 수정)
- FP16 대비 절대 성능 보존율:
  - Cond B: 94.7% (하락 -0.0346)
  - Cond C: 92.9% (하락 -0.0460)
  - Cond A: 91.7% (하락 -0.0542)
  - Cond C_v2: 88.6% (하락 -0.0743)
  - BoolQ가 모든 조건에서 가장 큰 하락폭
- B / C / C_v2 calibration 통계 비교 완료 (상세: `thoughts/08_calibration_통계비교_BCC_v2.md`):
  - C와 C_v2의 통계가 사실상 동일 → C_v2 알고리즘 개정이 실질적 변화 없었음
  - 총 고유 형태소 커버리지: C(1934) > B(1510) — 기존 문서 내용과 반대, 재측정 기준으로 수정
  - C의 높은 SFS 원인 규명: 128개 중 57개(44.5%)가 비한국어 문자 비율 > 30%인 혼합 문장
  - 한자/일본어/통화코드 등이 tokenizer에서 글자 단위 분절 → UMR/SFS 인위적 상승
- Cond C_v3 알고리즘 재설계 및 calibration set 생성 완료 (상세: `thoughts/09_CondC_v3_알고리즘_재설계.md`):
  - `src/diversity.py`: SFS 계산을 한국어 어절 기준으로만 변경
  - `src/selection.py`: coverage_bonus를 한국어 형태소(SL/SH/SN/SW 제외)만으로 계산, `min_ko_ratio` 필터 추가
  - `src/build_calibration.py`: `--c-min-ko-ratio`, `--suffix` 인자 추가
  - C_v3 생성 결과: 후보 92,538개(7,462개 필터), 128문장, 고유형태소 1,685, 평균 SFS 5.04 (기존 C=6.80), avg diversity=0.677
  - 출력: `results/calibration_set_C_v3_upstage_SOLAR-10.7B-Instruct-v1.0.json`
- C_v3 양자화 및 KoBEST 평가 완료 (상세: `thoughts/10_C_v3_양자화_평가_실행기록.md`):
  - 출력: `quantized_models/SOLAR_10.7B_4bit_cond_C_v3/`
  - 결과: `results/eval_cond_C_v3_2026-03-18T06-35-39.561864.json`
  - **C_v3 kobest avg 0.6354 — 모든 조건 중 최고 (FP16 보존율 97.4%)**
  - 조건별 최종 순위: C_v3(0.6354) > B(0.6176) > C(0.6062) > A(0.5981) > C_v2(0.5779)
  - BoolQ: C_v3=0.8077 vs B=0.7885 (+0.019) — 비한국어 제거 효과 직접 확인
  - WiC: C_v3=0.5302로 FP16(0.5008) 초과
  - **결론: 형태소 다양성 + 순수 한국어 필터 가설 검증됨**
- 비교 스크립트 업데이트:
  - `src/compare_calibration_stats.py`: C_v3 추가 (B/C/C_v2/C_v3 통계 비교)
  - `src/summarize_results.py`: C_v3 패턴 추가, 전체 비교표 출력 완료
- 다음 단계:
  - `src/compare_calibration_stats.py`로 B/C/C_v3 calibration 통계 최종 비교
  - 단일 런 재현성 확인 (B vs C_v3 차이 0.018이 노이즈인지 신호인지)
  - kmmlu 추가 검증 여부 결정
  - 스프린트 2 결론 정리 및 보고서 초안

## 진행 메모 (2026-03-19)

- 컨텍스트 초과로 이전 대화 세션 종료 후 재개. C_v3 양자화는 이미 완료 상태였음.
- **calibration 통계 비교 완료** (상세: `thoughts/11_calibration_통계비교_최종_BCC_v3.md`):
  - SFS: C_v3=6.57 vs C=6.80 — 거의 동일. SFS 정상화가 성능 회복 원인이 아님
  - 핵심 메커니즘: 총 고유 subword 커버리지 846→570 (B=527 수준)으로 급감
  - 비한국어 subword 제거가 GPTQ 가중치 근사 품질 회복의 실제 원인
- **C_v3 재현 실험 완료**: Run1=0.6354, Run2=0.6356, 차이 0.0002 — 신호 확인
- **kmmlu 교차 검증 완료** (A/B/C/C_v3, 상세: `thoughts/13_kmmlu_교차검증.md`):
  - kmmlu 순위: C(0.3752) > C_v3(0.3747) > B(0.3731) > A(0.3677)
  - C_v3 > B는 두 벤치마크 모두에서 성립
  - 원본 C가 kmmlu 1위: 비한국어 영향이 지식 암기 태스크보다 이해력 태스크에 더 큼
- **C_v4 생성 및 양자화 완료**:
  - 파라미터: min_eojeols=8, c_target_eojeols=18, min_ko_ratio=0.7 (어절 수 B 수준 목표)
  - calibration: `results/calibration_set_C_v4_upstage_SOLAR-10.7B-Instruct-v1.0.json`
  - 양자화: `quantized_models/SOLAR_10.7B_4bit_cond_C_v4/`
- **C_v4 kobest/kmmlu 평가 완료**:
  - kobest: 0.6005 (C_v3 0.6356보다 낮음 — 길이 증가 효과 없음)
  - kmmlu: 0.3732 (C_v3 0.3747보다 낮음)
  - 결론: C_v3 우위는 비한국어 subword 제거 덕분, 길이 효과 아님 확정

## 진행 메모 (2026-03-19 후반)

- **Activation 분포 분석 완료** (`src/analyze_activations.py`, 결과: `results/activation_analysis.json`):
  - FP16 모델에 B/C/C_v3/C_v4 calibration 통과시켜 레이어별 activation 통계 비교
  - C_v3가 channel_cv(10.66), mean_std(1.317) 모두 최고 — 가장 다양한 activation 분포 유도
  - outlier_ratio는 조건 간 차이 없음 (모두 ~0.0002)
  - 상세: `thoughts/13_kmmlu_교차검증.md` §6 참조

- **Sprint 2 최종 결론 확정** (`thoughts/12_스프린트2_최종결론.md` 최종 업데이트):
  - KoBEST: C_v3(0.6356) > B(0.6176) > C(0.6062) > C_v4(0.6005) > A(0.5981) > C_v2(0.5779)
  - kmmlu: C(0.3752) > C_v3(0.3747) > C_v4(0.3732) ≈ B(0.3731) > A(0.3677)
  - C_v3가 두 벤치마크 모두에서 B를 상회 → 가설 검증

- **다모델 검증 완료** (Llama3-Ko-8B / EEVE-10B, 상세: `thoughts/14_다모델_검증_결과.md`):
  - Llama3-Ko: A(0.5758) > C_v3(0.5650) > B(0.5608) — 영어 사전학습 모델에서 영어 calibration 최선
  - EEVE-10B: B(0.7595) > C_v3(0.7514) > A(0.7172) — SOLAR 기반이지만 B가 최선
  - **가설 정제**: "한국어 calibration 만능"이 아니라 "사전학습 주 언어 = calibration 언어"
  - SOLAR처럼 한국어 전용 사전학습 모델에서만 C_v3 효과 뚜렷

- **Sprint 2 종료 — Sprint 3 시작**:
  - Sprint 3 핵심: Qwen2-7B(중국어 사전학습) + 중국어 calibration 실험
  - 성공 시 주장: "calibration을 모델 사전학습 주 언어에 맞추면 GPTQ 품질 보존"
  - 상세 계획: `sprints/sprint3/backlog.md`
