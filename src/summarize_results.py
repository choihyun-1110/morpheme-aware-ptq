"""
A / B / C / C_v2 / FP16 KoBEST 결과 통합 비교표 출력

Usage:
    python src/summarize_results.py
"""
import json
import glob
from pathlib import Path

WORKSPACE   = Path(__file__).resolve().parent.parent
RESULTS_DIR = WORKSPACE / "results"

# 조건명 → 결과 파일 패턴
CONDS = {
    "FP16":      "eval_fp16_baseline*.json",
    "Cond A":    "eval_cond_A*.json",
    "Cond B":    "eval_cond_B*.json",
    "Cond C":    "eval_cond_C_2*.json",
    "Cond C_v2": "eval_cond_C_v2*.json",
    "Cond C_v3": "eval_cond_C_v3*.json",
}

SUBTASKS = ["kobest_boolq", "kobest_copa", "kobest_hellaswag", "kobest_sentineg", "kobest_wic"]

def load_result(pattern: str) -> dict | None:
    files = sorted(glob.glob(str(RESULTS_DIR / pattern)))
    if not files:
        return None
    with open(files[-1]) as f:
        data = json.load(f)
    return data.get("results", {})

def extract(results: dict, key: str, metric: str) -> float | None:
    if results is None or key not in results:
        return None
    entry = results[key]
    # lm-eval v2 형식: {"acc,none": ..., "acc_norm,none": ...}
    for k in (f"{metric},none", metric):
        if k in entry:
            return entry[k]
    return None

# ── 데이터 로드 ────────────────────────────────────────────────────────────────
rows = {}
for name, pattern in CONDS.items():
    rows[name] = load_result(pattern)

# ── kobest 총점 ───────────────────────────────────────────────────────────────
def kobest_avg(results) -> float | None:
    # 그룹 레벨 점수 우선 사용 (샘플 수 가중 평균)
    if results and "kobest" in results:
        v = extract(results, "kobest", "acc")
        if v is not None:
            return v
    # fallback: 단순 평균
    vals = [extract(results, t, "acc") for t in SUBTASKS]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None

# ── FP16 기준 성능 보존율 ──────────────────────────────────────────────────────
fp16_scores = {}
if rows.get("FP16"):
    for t in SUBTASKS:
        v = extract(rows["FP16"], t, "acc")
        if v is not None:
            fp16_scores[t] = v

def retention(score, baseline):
    if score is None or baseline is None or baseline == 0:
        return None
    return score / baseline

# ── 출력 ───────────────────────────────────────────────────────────────────────
names = list(CONDS.keys())

print("\n" + "="*90)
print("KoBEST 결과 종합 비교표")
print("="*90)

# 총점
print(f"\n{'':20} | " + " | ".join(f"{n:>10}" for n in names))
print("-"*90)

line_avg = f"{'kobest avg acc':20} | "
for n in names:
    v = kobest_avg(rows[n])
    line_avg += f"{v:>10.4f} | " if v is not None else f"{'N/A':>10} | "
print(line_avg)

# 세부 태스크
print()
for t in SUBTASKS:
    label = t.replace("kobest_", "")
    line = f"{label:20} | "
    for n in names:
        v = extract(rows[n], t, "acc")
        line += f"{v:>10.4f} | " if v is not None else f"{'N/A':>10} | "
    print(line)

# FP16 대비 성능 보존율
if fp16_scores:
    print("\n" + "-"*90)
    print("FP16 대비 성능 보존율 (각 조건 acc / FP16 acc)")
    print("-"*90)
    line = f"{'kobest avg retention':20} | "
    for n in names:
        v = kobest_avg(rows[n])
        b = kobest_avg(rows["FP16"])
        r = retention(v, b)
        line += f"{r:>10.4f} | " if r is not None else f"{'N/A':>10} | "
    print(line)

    for t in SUBTASKS:
        label = t.replace("kobest_", "")
        line = f"{label:20} | "
        for n in names:
            v = extract(rows[n], t, "acc")
            b = fp16_scores.get(t)
            r = retention(v, b)
            line += f"{r:>10.4f} | " if r is not None else f"{'N/A':>10} | "
        print(line)

    # 절대 하락폭 (FP16 - quantized)
    print("\n" + "-"*90)
    print("FP16 대비 성능 하락폭 (FP16 acc - 조건 acc, 낮을수록 좋음)")
    print("-"*90)
    line = f"{'kobest avg drop':20} | "
    for n in names:
        v = kobest_avg(rows[n])
        b = kobest_avg(rows["FP16"])
        d = (b - v) if (v is not None and b is not None) else None
        line += f"{d:>10.4f} | " if d is not None else f"{'N/A':>10} | "
    print(line)

    for t in SUBTASKS:
        label = t.replace("kobest_", "")
        line = f"{label:20} | "
        for n in names:
            v = extract(rows[n], t, "acc")
            b = fp16_scores.get(t)
            d = (b - v) if (v is not None and b is not None) else None
            line += f"{d:>10.4f} | " if d is not None else f"{'N/A':>10} | "
        print(line)

print("\n" + "="*90)
