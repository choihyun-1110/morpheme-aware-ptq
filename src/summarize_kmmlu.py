"""
kmmlu 결과 통합 비교표 출력

Usage:
    python src/summarize_kmmlu.py
"""
import json
import glob
from pathlib import Path

WORKSPACE   = Path(__file__).resolve().parent.parent
RESULTS_DIR = WORKSPACE / "results"

CONDS = {
    "Cond A":    "eval_kmmlu_cond_A*.json",
    "Cond B":    "eval_kmmlu_cond_B*.json",
    "Cond C":    "eval_kmmlu_cond_C_2*.json",
    "Cond C_v3": "eval_kmmlu_cond_C_v3*.json",
}

def load_result(pattern: str) -> dict | None:
    files = sorted(glob.glob(str(RESULTS_DIR / pattern)))
    if not files:
        return None
    with open(files[-1]) as f:
        data = json.load(f)
    return data.get("results", {})

def extract(results: dict, key: str) -> float | None:
    if results is None or key not in results:
        return None
    entry = results[key]
    for k in ("acc,none", "acc_norm,none", "acc"):
        if k in entry:
            return entry[k]
    return None

def kmmlu_avg(results) -> float | None:
    if results and "kmmlu" in results:
        v = extract(results, "kmmlu")
        if v is not None:
            return v
    # fallback: 서브태스크 평균
    vals = [v for k, v in results.items()
            if k.startswith("kmmlu_") and (v := extract(results, k)) is not None]
    return sum(vals) / len(vals) if vals else None

rows = {name: load_result(pat) for name, pat in CONDS.items()}
names = list(CONDS.keys())

print("\n" + "="*70)
print("KMMLU 결과 비교표")
print("="*70)
print(f"\n{'':20} | " + " | ".join(f"{n:>12}" for n in names))
print("-"*70)

line = f"{'kmmlu avg acc':20} | "
for n in names:
    v = kmmlu_avg(rows[n])
    line += f"{v:>12.4f} | " if v is not None else f"{'N/A':>12} | "
print(line)

# 서브태스크가 있으면 출력
if any(rows[n] for n in names):
    sample = next(r for r in rows.values() if r)
    subtasks = sorted(k for k in sample if k.startswith("kmmlu_"))
    if subtasks:
        print()
        for t in subtasks:
            label = t.replace("kmmlu_", "")[:20]
            line = f"{label:<20} | "
            for n in names:
                v = extract(rows[n], t)
                line += f"{v:>12.4f} | " if v is not None else f"{'N/A':>12} | "
            print(line)

print("\n" + "="*70)
