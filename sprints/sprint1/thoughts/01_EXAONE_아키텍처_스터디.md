# S1-1. 대상 모델(EXAONE 3.5) 아키텍처 및 로딩 스터디

## 목표
- `LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct` 등 EXAONE 모델의 아키텍처 특성 파악
- Hugging Face `transformers` 와 `auto-gptq` 패키지를 이용한 모델 로딩 및 양자화 수행 시 호환성(의존성) 문제 분석
- 공식 지원이 되지 않는 현재 상태에서, Custom Model 구조에 맞춰 양자화를 오프라인(또는 패치)으로 어떻게 진행할지 전략 채택

## 1. EXAONE 아키텍처 주요 특징
- 기존의 표준 Llama 아키텍처와 어떤 차이가 있는가? (예: RoPE 적용 방식, Attention 메커니즘 차이 등)
- 왜 `transformers <= 4.44`에서 `ImportError`가 발생하거나, 반대로 최신 버전에서 `auto-gptq` 연동에 구 버전 코드를 요구하는가?

## 2. 발생한 주요 이슈 (Sprint 0 파일럿 과정 회고)
- **증상 1**: 최신 버전을 설치하면 `auto-gptq` 내부에서 `RopeParameters` 등 존재하지 않는 모듈을 찾음.
- **증상 2**: `auto-gptq` 동작을 위해 `transformers`를 4.44.2로 내리면, EXAONE 3.5 구성 파일(`configuration_exaone.py`, `modeling_exaone.py`) 내 특성 로드 시 오류 혹은 구조적 미지원(`exaone isn't supported yet.`)이 발생함.
- **결론**: `AutoGPTQ` 내부 하드코딩된 Model Type 라우팅에 `exaone`이 등록되어 있지 않거나, 모듈 호환성이 맞지 않음.

## 3. 해결 방안 탐색 (스터디 과제)
1. **AutoGPTQ 소스 패치 / Custom Model 클래스 구현**: EXAONE의 구조를 GPTQ가 지원하는 표준 형태로 매핑하거나, `auto-gptq` 내부 딕셔너리에 임의 등록하는 방식 연구.
2. **Alternative 양자화 라이브러리 검토 (AWQ 등)**: AutoAWQ 등 타 라이브러리는 EXAONE 아키텍처를 보다 수월하게 지원하는가?
3. **업스테이지 모델 등 우선순위 교체**: 본래 논문(Pilot) 작성의 일정상 호환성 문제가 극복이 오래 걸릴 경우, 사용자가 앰배서더로 활동 중인 업스테이지 모델(Solar 등)이나 오픈 Llama 모델을 우선 실험체로 삼아 파이프라인(Sprint 1, 2)을 먼저 관철시키는 방안 점검.

## 4. 결론 ✅

### 결정사항
- **방안 3 채택**: 업스테이지(Solar 10.7B)와 오픈 Llama 3 한국어 모델을 우선 실험 대상으로 확정.
- EXAONE 3.5는 **3순위**로 보류. 양자화 파이프라인이 완성된 후 커스텀 패치로 편입 예정.

### 근거
1. EXAONE 3.5의 `auto-gptq` 미지원 문제를 해결하는 데 드는 시간 대비, 연구 핵심 가설("형태소 다양성 기반 calibration이 양자화 품질에 영향을 미치는가") 검증을 위한 파이프라인 구축이 더 급선무임.
2. Solar 10.7B와 Llama 3.1 Ko 8B 모두 `LlamaForCausalLM`으로 인식되어 AutoGPTQ와 **완벽 호환**.
3. EXAONE 4.0에서 GPTQ 공식 지원이 시작되었으므로, 향후 커뮤니티 패치가 EXAONE 3.5에도 적용될 가능성이 있음.

### 후속 조치
- → 상세 내용은 [02_대상모델_변경_업스테이지_Llama.md](02_대상모델_변경_업스테이지_Llama.md)에 정리 완료.
