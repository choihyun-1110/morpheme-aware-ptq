# Morpheme-Aware PTQ Calibration for Korean LLMs

> **Research Question:** When quantizing Korean LLMs with GPTQ 4-bit, does a morpheme-diversity-optimized calibration set better preserve model performance compared to random sampling?

## Overview

This repository contains experiments on **Post-Training Quantization (PTQ) calibration data selection** for Korean large language models.

The key insight: calibration data quality matters for GPTQ quantization, and for Korean-dominant models, **language-aligned + morpheme-diverse** calibration outperforms random English or random Korean sampling.

## Key Results (SOLAR-10.7B-Instruct-v1.0, KoBEST)

| Condition | Calibration Data | KoBEST avg | FP16 Retention |
|-----------|-----------------|------------|----------------|
| FP16 (baseline) | — | 0.6523 | 100% |
| **C_v3 (ours)** | Korean morpheme-diverse | **0.6354** | **97.4%** |
| B | Korean random (NamuWiki) | 0.6176 | 94.7% |
| C | Korean morpheme-diverse (w/ non-Korean) | 0.6062 | 92.9% |
| A | English random (Wikitext-2) | 0.5981 | 91.7% |
| C_v2 | C + sentence filter | 0.5779 | 88.6% |

**C_v3 reduces quantization performance drop by 3.5× compared to English calibration (A).**

## Method

### Calibration Conditions

- **Condition A**: 128 sentences from Wikitext-2 (English random)
- **Condition B**: 128 sentences from NamuWiki (Korean random)
- **Condition C/C_v3**: 128 sentences selected via **Greedy Morpheme Diversity** algorithm

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

C_v3 removes non-Korean subword noise → subword coverage drops to B-level (570 vs 527) → GPTQ weight approximation quality recovers.

## Repository Structure

```
.
├── src/
│   ├── build_calibration.py     # Main CLI: build calibration sets (A/B/C)
│   ├── diversity.py             # Diversity metrics (UMR, TTR, SFS)
│   ├── selection.py             # Greedy diversity selection
│   ├── preprocess.py            # NamuWiki preprocessing
│   ├── run_quant.py             # GPTQ quantization pipeline
│   ├── run_eval.sh              # KoBEST / KMMLU evaluation
│   ├── compare_calibration_stats.py  # Calibration set statistics
│   ├── summarize_results.py     # KoBEST results comparison table
│   └── summarize_kmmlu.py       # KMMLU results comparison table
├── results/
│   ├── calibration_set_*.json   # Generated calibration sets
│   └── calibration_stats_comparison.json
├── sprints/
│   ├── sprint_overview.md       # Project roadmap
│   ├── sprint0/                 # Environment setup & pilot
│   ├── sprint1/                 # Algorithm design
│   └── sprint2/                 # Main experiments
│       └── thoughts/            # Analysis & review documents
└── pilot_quant.py               # Sprint 0 pilot script
```

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

# Quantize
python src/run_quant.py \
  --model upstage/SOLAR-10.7B-Instruct-v1.0 \
  --calib C_v3 \
  --calib-path results/calibration_set_C_v3_upstage_SOLAR-10.7B-Instruct-v1.0.json \
  --out-dir quantized_models/SOLAR_10.7B_4bit_cond_C_v3

# Evaluate (KoBEST)
bash src/run_eval.sh quantized_models/SOLAR_10.7B_4bit_cond_C_v3 cond_C_v3

# Summarize results
python src/summarize_results.py
```

## Model & Environment

- **Target model**: [upstage/SOLAR-10.7B-Instruct-v1.0](https://huggingface.co/upstage/SOLAR-10.7B-Instruct-v1.0)
- **Quantization**: AutoGPTQ, 4-bit, group_size=128, desc_act=True
- **Evaluation**: [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), KoBEST benchmark
- **Morpheme analyzer**: [Kiwi](https://github.com/bab2min/Kiwi)
- **GPU**: NVIDIA TITAN RTX × 2 (24GB each)
