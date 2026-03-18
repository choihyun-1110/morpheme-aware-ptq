"""
B / C / C_v2 calibration set 통계 비교 스크립트

Usage:
    python src/compare_calibration_stats.py
"""
import json
import os
import re
from pathlib import Path

import numpy as np

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).resolve().parent.parent
RESULTS_DIR = WORKSPACE / "results"

FILES = {
    "Cond B":    RESULTS_DIR / "calibration_set_B.json",
    "Cond C":    RESULTS_DIR / "calibration_set_C_upstage_SOLAR-10.7B-Instruct-v1.0.json",
    "Cond C_v2": RESULTS_DIR / "calibration_set_C_v2_upstage_SOLAR-10.7B-Instruct-v1.0.json",
    "Cond C_v3": RESULTS_DIR / "calibration_set_C_v3_upstage_SOLAR-10.7B-Instruct-v1.0.json",
}

MODEL_ID = "upstage/SOLAR-10.7B-Instruct-v1.0"
HF_HOME  = os.environ.get("HF_HOME", str(WORKSPACE / ".cache/huggingface"))
os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", HF_HOME + "/hub")
os.environ.setdefault("TRANSFORMERS_CACHE",    HF_HOME + "/transformers")

# ── 라이브러리 로드 ────────────────────────────────────────────────────────────
print("▶ Loading tokenizer …")
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)

print("▶ Loading Kiwi …")
from kiwipiepy import Kiwi
kiwi = Kiwi()

# 문장 종결 판정 (Kiwi EF/SF 태그 또는 문장부호)
SENT_FINAL_RE = re.compile(r"[.!?…]$")
SENT_FINAL_TAGS = {"EF", "SF"}

def has_sentence_final(text: str) -> bool:
    if SENT_FINAL_RE.search(text.strip()):
        return True
    tokens = kiwi.tokenize(text)
    if tokens:
        return tokens[-1].tag in SENT_FINAL_TAGS
    return False

# ── 분석 ───────────────────────────────────────────────────────────────────────
def analyze(name: str, path: Path) -> dict:
    print(f"\n▶ Analyzing {name} …")
    with open(path) as f:
        data = json.load(f)
    sentences = data["sentences"]

    n_eojeols_list        = []
    n_tokens_list         = []
    sent_final_list       = []
    n_morphemes_list      = []
    n_unique_morphemes_list = []
    umr_list              = []
    ttr_list              = []
    sfs_list              = []
    all_morphemes         = set()
    all_subword_tokens    = set()

    for i, s in enumerate(sentences):
        text = s["text"]

        # ── subword tokens ──────────────────────────────────────────────────
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        n_tok = len(token_ids)
        n_tokens_list.append(n_tok)
        all_subword_tokens.update(token_ids)

        # ── Kiwi morpheme analysis (B에는 메타데이터 없음) ──────────────────
        if "n_eojeols" in s:
            n_eo = s["n_eojeols"]
            sf   = s["has_sentence_final"]
        else:
            n_eo = len(text.split())
            sf   = has_sentence_final(text)

        n_eojeols_list.append(n_eo)
        sent_final_list.append(sf)

        if "n_morphemes" in s:
            nm   = s["n_morphemes"]
            nu   = s["n_unique_morphemes"]
            umr  = s["umr"]
            ttr  = s["ttr"]
            sfs  = s["sfs"]
            # morpheme 커버리지는 텍스트에서 직접 추출 (메타에 집합 없음)
            tokens_kiwi = kiwi.tokenize(text)
            morphs = [t.form for t in tokens_kiwi]
        else:
            tokens_kiwi = kiwi.tokenize(text)
            morphs = [t.form for t in tokens_kiwi]
            nm   = len(morphs)
            nu   = len(set(morphs))
            umr  = nu / nm if nm else 0.0
            ttr  = nu / nm if nm else 0.0   # TTR == UMR for short texts
            sfs  = nm / len(text.split()) if text.split() else 0.0

        n_morphemes_list.append(nm)
        n_unique_morphemes_list.append(nu)
        umr_list.append(umr)
        ttr_list.append(ttr)
        sfs_list.append(sfs)
        all_morphemes.update([t.form for t in kiwi.tokenize(text)])

        if (i + 1) % 32 == 0:
            print(f"  {i+1}/{len(sentences)} done")

    return {
        "name":                  name,
        "n_sentences":           len(sentences),
        "avg_eojeols":           float(np.mean(n_eojeols_list)),
        "median_eojeols":        float(np.median(n_eojeols_list)),
        "avg_tokens":            float(np.mean(n_tokens_list)),
        "median_tokens":         float(np.median(n_tokens_list)),
        "sent_final_ratio":      float(np.mean(sent_final_list)),
        "avg_morphemes":         float(np.mean(n_morphemes_list)),
        "avg_unique_morphemes":  float(np.mean(n_unique_morphemes_list)),
        "avg_umr":               float(np.mean(umr_list)),
        "avg_ttr":               float(np.mean(ttr_list)),
        "avg_sfs":               float(np.mean(sfs_list)),
        "total_unique_morphemes": len(all_morphemes),
        "total_unique_tokens":   len(all_subword_tokens),
    }

# ── 실행 ───────────────────────────────────────────────────────────────────────
stats = {}
for name, path in FILES.items():
    stats[name] = analyze(name, path)

# ── 출력 ───────────────────────────────────────────────────────────────────────
COLS = [
    ("avg_eojeols",           "평균 어절 수"),
    ("median_eojeols",        "중앙값 어절 수"),
    ("avg_tokens",            "평균 subword 토큰 수"),
    ("median_tokens",         "중앙값 subword 토큰 수"),
    ("sent_final_ratio",      "문장 종결형 비율"),
    ("avg_morphemes",         "평균 형태소 수"),
    ("avg_unique_morphemes",  "평균 고유 형태소 수"),
    ("avg_umr",               "평균 UMR"),
    ("avg_ttr",               "평균 TTR"),
    ("avg_sfs",               "평균 SFS"),
    ("total_unique_morphemes","총 고유 형태소 커버리지"),
    ("total_unique_tokens",   "총 고유 subword 커버리지"),
]

names = list(stats.keys())
print("\n" + "="*80)
print(f"{'지표':<28} | {'Cond B':>12} | {'Cond C':>12} | {'Cond C_v2':>12} | {'Cond C_v3':>12}")
print("-"*80)
for key, label in COLS:
    vals = [stats[n][key] for n in names]
    fmt  = ".0f" if key.startswith("total") or key in ("avg_morphemes","avg_unique_morphemes","avg_eojeols","median_eojeols","avg_tokens","median_tokens") else ".4f"
    row  = f"{label:<28} | " + " | ".join(f"{v:>12{fmt}}" for v in vals)
    print(row)
print("="*80)

# ── JSON 저장 ──────────────────────────────────────────────────────────────────
out_path = RESULTS_DIR / "calibration_stats_comparison.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)
print(f"\n✓ 결과 저장: {out_path}")
