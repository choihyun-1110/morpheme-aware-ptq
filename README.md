# Morpheme-Aware PTQ Calibration for Korean LLMs

> **Research Question:** When quantizing Korean/multilingual LLMs with GPTQ 4-bit, does a morpheme-diversity-optimized calibration set better preserve model performance compared to random sampling?

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

### Cross-Model Validation (Sprint 3)

| Model | Pretraining | Best Condition | KoBEST avg | Notes |
|-------|------------|----------------|------------|-------|
| SOLAR-10.7B | Korean | **C_v3** | 0.6356 | Morpheme-diverse Korean |
| EEVE-10B | Korean (SFT) | **B** | 0.7595 | Korean random |
| Llama3-Ko-8B | English | **A** | — | English random |
| EXAONE3.5-7.8B | Korean | **B ≈ C_v3** | 0.7196 / 0.7164 | B slightly best |
| Qwen2-7B | Chinese | **C_v3** | 0.6375 | Korean > Chinese > English |

**Finding:** 5 out of 5 models favor pretraining-language-aligned calibration. C_v3's morpheme diversity effect generalizes across languages (Qwen2 C-Eval: C_v3 0.7734 > A=C_zh 0.7630).

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
├── sprints/
│   ├── sprint_overview.md        # Project roadmap
│   ├── sprint0/                  # Environment setup & pilot
│   ├── sprint1/                  # Algorithm design
│   ├── sprint2/                  # SOLAR main experiments
│   │   └── thoughts/             # Analysis & review documents
│   ├── sprint3/                  # Cross-model validation + vLLM
│   │   └── thoughts/             # Qwen2/EXAONE/Phase2 analysis
│   └── sprint4/                  # Statistical testing & paper prep
└── pilot_quant.py                # Sprint 0 pilot script
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
