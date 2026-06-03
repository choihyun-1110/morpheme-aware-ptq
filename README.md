# Morpheme-Aware PTQ Calibration for Korean LLMs

> **Research Question:** When quantizing Korean/multilingual LLMs with GPTQ 4-bit, does a morpheme-diversity-optimized calibration set better preserve model performance compared to random sampling?

## 📺 Presentation Video

[![Capstone Presentation](https://img.shields.io/badge/YouTube-Capstone_Presentation-red?logo=youtube)](https://youtu.be/REPLACE_WITH_VIDEO_ID)

> **[84팀] Morpheme-Aware Calibration for Post-Training Quantization of Korean LLMs**  
> Capstone Design, 2026

## Overview

This repository contains experiments on **Post-Training Quantization (PTQ) calibration data selection** for Korean and multilingual large language models.

The key insight: calibration data quality matters for GPTQ quantization, and **morpheme-diverse** calibration consistently outperforms random English or random Korean sampling — not just for Korean-dominant models, but across multiple model families.

## Key Results

### SOLAR-10.7B-Instruct-v1.0 (KoBEST)

| Condition | Calibration Data | KoBEST avg | FP16 Retention |
|-----------|-----------------|------------|----------------|
| FP16 (baseline) | — | 0.6523 | 100% |
| **C_v3 (ours)** | Korean morpheme-diverse | **0.6356** | **97.4%** |
| B | Korean random (NamuWiki) | 0.6176 | 94.7% |
| C | Korean morpheme-diverse (w/ non-Korean) | 0.6062 | 92.9% |
| A | English random (Wikitext-2) | 0.5981 | 91.7% |
| C_v2 | C + sentence filter | 0.5779 | 88.6% |

**C_v3 reduces quantization performance drop by 3.5× compared to English calibration (A).**

### Cross-Model Validation (5 Models)

| Model | Pretraining | Benchmark | FP16 | A (English) | B (Lang-aligned) | **Ours (MAC)** |
|-------|------------|-----------|------|-------------|------------------|----------------|
| SOLAR-10.7B | Korean | KoBEST | 0.6523 | 0.5981 (91.7%) | 0.6176 (94.7%) | **0.6356 (97.4%)** |
| EEVE-10.8B | Korean | KoBEST | 0.7759 | 0.7463 (96.2%) | 0.7498 (96.6%) | **0.7551 (97.3%)** |
| EXAONE-3.5-7.8B | Korean | KoBEST | 0.7437 | 0.6963 (93.6%) | 0.7244 (97.4%) | **0.7415 (99.7%)** |
| Qwen2-7B | Chinese | C-Eval | 0.8165 | 0.7630 (93.4%) | 0.7630 (93.4%) | **0.7868 (96.4%)** |
| Llama-3-8B | English | MMLU | 0.6241 | **0.5985 (95.9%)** | — | 0.5959 (95.5%) |

**Finding:** All 5 models favor pretraining-language-aligned calibration. MAC variant outperforms standard English GPTQ in 4 out of 5 models on reasoning tasks. (Llama-3-8B: C_en_v3 ≈ A due to Wikitext-2 ceiling effect.)

### SOLAR-10.7B Phase 2 Variants

| Config | KoBEST avg |
|--------|------------|
| C_v3 group_size=128 (baseline) | 0.6356 |
| **C_v3 group_size=64** | **0.6468** |
| A group_size=128 | 0.6174 |
| C_v3 desc_act=False | 0.6029 |

### vLLM Serving Benchmark (SOLAR-10.7B, TITAN RTX)

| | FP16 | GPTQ C_v3 |
|-|------|-----------|
| Model VRAM | ~20 GB | ~5.6 GB (**72% reduction**) |
| Load time | 7.85s | 38.01s |
| Throughput | 482.87 tok/s | **502.25 tok/s** (+4%) |
| Latency | 465.7 ms | **443.5 ms** (-5%) |

---

## Method

### Calibration Conditions

- **Condition A**: 128 sentences from Wikitext-2 (English random)
- **Condition B**: 128 sentences from NamuWiki (Korean random)
- **Condition C_v3**: 128 sentences selected via **Greedy Morpheme Diversity** algorithm
- **Condition C_zh**: 128 Chinese sentences (jieba + wikimedia/wikipedia, for Qwen2)

### Greedy Morpheme Diversity Selection (C_v3)

1. Extract 100K candidate sentences from NamuWiki
2. Filter: `min_eojeols ≥ 8`, `min_ko_ratio ≥ 0.7` (removes non-Korean mixed sentences)
3. Score each candidate:
   ```
   score = α × coverage_bonus(Korean morphemes only)
           + β × length_bonus
           + γ × sentence_final_bonus
   ```
4. Greedily select 128 sentences maximizing cumulative Korean morpheme coverage

### Why C_v3 Works

The original C failed because 44.5% of selected sentences contained non-Korean characters (CJK, Japanese, currency codes), inflating subword token counts artificially:

| | Cond B | Cond C | Cond C_v3 |
|-|--------|--------|-----------|
| Unique subword tokens | 527 | 846 | **570** |
| Avg SFS | 2.30 | 6.80 | 6.57 |

C_v3 removes non-Korean subword noise → subword coverage drops to B-level → GPTQ weight approximation quality recovers.

---

## Repository Structure

```
.
├── src/
│   ├── build_calibration.py      # Build calibration sets (A/B/C/C_v3)
│   ├── build_calibration_zh.py   # Build Chinese calibration set (C_zh, for Qwen2)
│   ├── diversity.py              # Diversity metrics (UMR, TTR, SFS)
│   ├── selection.py              # Greedy diversity selection
│   ├── preprocess.py             # NamuWiki preprocessing
│   ├── run_quant.py              # GPTQ quantization (AutoGPTQ)
│   ├── run_quant_optimum.py      # GPTQ quantization (optimum, newer models)
│   ├── patch_qzeros.py           # Fix qzeros bug (optimum GPTQ 0x88→0x77)
│   ├── patch_exaone.py           # EXAONE3.5 compatibility patch
│   ├── benchmark_serving.py      # vLLM serving benchmark (FP16 vs GPTQ)
│   ├── analyze_activations.py    # Activation distribution analysis
│   ├── visualize_activations.py  # Paper-ready activation figures
│   ├── compare_calibration_stats.py  # Calibration set statistics
│   ├── summarize_results.py      # KoBEST results comparison table
│   ├── summarize_kmmlu.py        # KMMLU results comparison table
│   └── run_eval.sh               # KoBEST / KMMLU evaluation runner
├── results/
│   ├── calibration_set_*.json    # Generated calibration sets
│   ├── activation_analysis.json  # Activation channel statistics
│   └── activation_paper_*.png    # Paper figures (ko/en)
└── pilot_quant.py                # Pilot quantization script
```

---

## Setup

```bash
# Inside Docker container (llm-dev), conda env (llm-quant)
pip install -r src/requirements.txt
```

## Usage

```bash
# Build calibration set (Condition C_v3)
python src/build_calibration.py \
  --condition C \
  --model upstage/SOLAR-10.7B-Instruct-v1.0 \
  --n-sentences 128 \
  --c-min-ko-ratio 0.7 \
  --suffix v3

# Quantize (AutoGPTQ, for SOLAR / Llama3-Ko)
python src/run_quant.py \
  --model upstage/SOLAR-10.7B-Instruct-v1.0 \
  --calib C_v3 \
  --calib-path results/calibration_set_C_v3_upstage_SOLAR-10.7B-Instruct-v1.0.json \
  --out-dir quantized_models/SOLAR_10.7B_4bit_cond_C_v3

# Quantize (optimum, for Qwen2 / EEVE / EXAONE)
python src/run_quant_optimum.py \
  --model /path/to/model \
  --calib-path results/calibration_set_C_v3_*.json \
  --label C_v3 \
  --out-dir quantized_models/model_cond_C_v3

# Evaluate (KoBEST)
bash src/run_eval.sh quantized_models/SOLAR_10.7B_4bit_cond_C_v3 cond_C_v3

# vLLM serving benchmark
python src/benchmark_serving.py \
  --model quantized_models/SOLAR_10.7B_4bit_cond_C_v3 \
  --label C_v3 \
  --quantized

# Summarize results
python src/summarize_results.py
```

---

## Model & Environment

- **Target models**: SOLAR-10.7B, EEVE-10B, Llama3-Ko-8B, EXAONE3.5-7.8B, Qwen2-7B
- **Quantization**: AutoGPTQ / optimum, 4-bit, group_size=128, desc_act=True
- **Evaluation**: [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), KoBEST / kmmlu / C-Eval
- **Morpheme analyzers**: [Kiwi](https://github.com/bab2min/Kiwi) (Korean), [jieba](https://github.com/fxsjy/jieba) (Chinese)
- **Serving**: [vLLM](https://github.com/vllm-project/vllm) 0.17.1
- **GPU**: NVIDIA TITAN RTX × 2 (24GB each, compute capability 7.5)
