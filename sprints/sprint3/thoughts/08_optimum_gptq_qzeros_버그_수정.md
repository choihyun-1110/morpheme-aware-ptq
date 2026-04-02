# optimum.gptq qzeros 포맷 버그 진단 및 수정

**작성일:** 2026-03-26

---

## 현상

Sprint 3에서 `optimum.gptq.GPTQQuantizer`로 저장한 모든 모델의 KoBEST 결과가 정확히 랜덤 기준선과 일치:
- boolq=0.4979, copa=0.517, hellaswag=0.248, sentineg=0.496, wic=0.512
- **서로 다른 calibration 조건의 모델이 완전히 동일한 점수**

영향 받은 모델: SOLAR g64 C_v3/A, EXAONE35 A/B/C_v3, EEVE C_v3_eeve

---

## 진단 과정

### 1단계: 포맷 격차 확인

| 속성 | Sprint 2 (auto_gptq) | Sprint 3 (optimum.gptq) |
|------|---------------------|------------------------|
| 저장 파일 | `gptq_model-4bit-128g.safetensors` | `model-00001-of-00002.safetensors` |
| quantize_config.json | 있음 | **없음** |
| gptqmodel 포맷 인식 | `FORMAT.GPTQ` (v1) | `gptq_v2` (v2로 오인식) |
| 평가 방법 | `autogptq=filename` | `dtype=float16` (HF 경로) |

### 2단계: NaN 원인 추적

Forward pass 중 activation 값 측정:
- layer 0: max_abs=23.6
- layer 1: max_abs=872
- layer 2: max_abs=2124
- layer 3: max_abs=3652
...
→ **지수적 증가** → float16 overflow (~layer 15) → logits NaN

### 3단계: 텐서 분석

```python
from safetensors import safe_open
# Sprint 2 qzeros: 0x77777777 (각 nibble = 7)
# Sprint 3 qzeros: 0x88888888 (각 nibble = 8)
```

**결정적 발견**: Sprint 2(auto_gptq)와 Sprint 3(optimum.gptq)의 qzeros 값이 1 nibble 차이.

---

## 근본 원인

### GPTQ 역양자화 공식 (gptqmodel/auto_gptq 기준)

```
dequantized = (qvalue - (qzeros_nibble + 1)) * scale
```

| 출처 | qzeros 저장 | 역양자화 |
|------|------------|---------|
| auto_gptq | zero_point - 1 = **7** | (q - (7+1)) * s = (q - 8) * s ✓ |
| optimum.gptq | zero_point = **8** | (q - (8+1)) * s = (q - 9) * s ✗ |

모든 가중치에 균일한 -scale 만큼의 편향 → 레이어마다 누적 → 지수 증가 → NaN

### 부가 원인: gptqmodel 포맷 오인식

- optimum.gptq는 `quantize_config.json`을 저장하지 않음 (HF 표준 `config.json`에만 포함)
- gptqmodel은 `quantize_config.json` 부재 시 → `gptq_v2` 포맷으로 자동 변환
- Sprint 2(auto_gptq)는 `quantize_config.json` 있음 → `FORMAT.GPTQ` (v1) 유지
- 하지만 실제 NaN의 **주원인은 qzeros 편향**이었음 (포맷 자체는 v2로 변환되어도 동작함)

---

## 수정 방법

### 즉시 적용: `patch_qzeros.py`

모든 safetensors 파일에서 `qzeros` 텐서를 `-0x11111111` 적용:
```python
for key in f.keys():
    t = f.get_tensor(key)
    if "qzeros" in key:
        t = t - 0x11111111  # 0x88888888 → 0x77777777
```

### 영구 수정: `run_quant_optimum.py`

저장 후 자동으로 qzeros 패치 + `quantize_config.json` 생성:
```python
# 저장 직후 qzeros 패치
for sf_path in glob(output_dir + "/model*.safetensors"):
    tensors = load_safetensors(sf_path)
    for k in tensors:
        if "qzeros" in k:
            tensors[k] -= 0x11111111
    save_file(tensors, sf_path)

# quantize_config.json 생성
json.dump({"bits": 4, "group_size": group_size, "desc_act": desc_act, ...}, ...)
```

---

## 검증

수정 후 SOLAR g64 C_v3 logits 확인:
```
NaN: False
top3 after "안녕하세요 오늘": ['은', '', '도']  ← 정상 한국어 조사/어미
```

Log-likelihood 테스트 (normalize 후 KoBEST-style):
```
서울은 대한민국의 수도이다 → 참: 11.777 > 거짓/length: ✓
```

---

## 수정된 파일

| 파일 | 변경 내용 |
|------|---------|
| `src/patch_qzeros.py` | 기존 모델 일괄 패치 스크립트 |
| `src/run_quant_optimum.py` | 저장 후 자동 qzeros 패치 + quantize_config.json 생성 |
| `src/smooth_scale.py` | OOM 방지: `weight.data = weight.data * s` → `weight.data.mul_(s)` (in-place) |
| 모든 기존 sprint3 모델 | `patch_qzeros.py`로 qzeros 패치 완료 |

---

## 현황 (2026-03-26 09:30 기준)

### 재평가 진행 중
- GPU 0: SOLAR C_v3 g64 KoBEST 재평가 중 (기대: ~0.63)
- GPU 1: EXAONE35 A KoBEST 재평가 중

### 이후 실행 예정
- SOLAR A g64 KoBEST 재평가
- SmoothScale + GPTQ (OOM 수정 후)
- EXAONE35 B/C_v3 KoBEST 재평가
- EEVE C_v3_eeve 양자화 + KoBEST (올바른 qzeros로)
- EXAONE35 FP16 베이스라인
- SOLAR desc_act=False 양자화 + KoBEST

---

## 교훈

`optimum.gptq`와 `auto_gptq`/`gptqmodel`의 qzeros 저장 규칙이 다름:
- **이 버그는 optimum.gptq로 저장한 모든 모델에 적용됨**
- Sprint 2 모델 (auto_gptq)은 영향 없음
- 향후 quantization 코드에서 백엔드 혼용 시 주의 필요
