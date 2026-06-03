"""
전체 모델 × 조건 × 지표 종합 비교 테이블 + 패턴 시각화
목적: fig10 (KoBEST vs kmmlu 불일치) 의 근거를 한눈에 보여주기
"""
import os, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

OUT = "/home/choihyun/workspace/results/presentation_assets"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "NanumGothic", "axes.spines.top": False, "axes.spines.right": False})

R = "/home/choihyun/workspace/results"

# ── 헬퍼 ────────────────────────────────────────────────────────────────
def kobest_group(p):
    with open(p) as f: d = json.load(f)
    v = d["results"].get("kobest", {}).get("acc,none")
    return round(v, 4) if v else None

def kmmlu_group(p):
    with open(p) as f: d = json.load(f)
    v = d["results"].get("kmmlu", {}).get("acc,none")
    return round(v, 4) if v else None

def kmmlu_avg(p):
    with open(p) as f: d = json.load(f)
    r = d["results"]
    sc = [v.get("acc,none", 0) for k, v in r.items()
          if k.startswith("kmmlu_") and "acc,none" in v and not k.endswith("_continuation")]
    return round(sum(sc)/len(sc), 4) if sc else None

def ceval_group(p):
    with open(p) as f: d = json.load(f)
    v = d["results"].get("ceval-valid", {}).get("acc,none")
    return round(v, 4) if v else None

def mmlu_score(p):
    with open(p) as f: d = json.load(f)
    r = d["results"]
    if "mmlu" in r: return round(r["mmlu"].get("acc,none", 0), 4)
    sc = [v.get("acc,none", 0) for k, v in r.items()
          if k.startswith("mmlu_") and "acc,none" in v]
    return round(sum(sc)/len(sc), 4) if sc else None

def lat(pat):
    f = sorted(glob.glob(pat)); return f[-1] if f else None

# ── 실제 데이터 수집 ────────────────────────────────────────────────────
data = {
    # (model, metric_label, fp16, A, B_or_lang_aligned, C_v3_variant, variant_label)
    # KoBEST 행
    "SOLAR KoBEST": {
        "fp16": 0.6523, "A": 0.5981, "B": 0.6176, "C": 0.6356,
        "C_label": "C_v3", "metric": "KoBEST", "type": "reasoning"
    },
    "EEVE KoBEST": {
        "fp16": 0.7759,
        "A":    kobest_group(lat(f"{R}/eval_kobest_eeve_10b_A_rerun*.json")),
        "B":    kobest_group(lat(f"{R}/eval_kobest_eeve_10b_B_rerun*.json")),
        "C":    kobest_group(lat(f"{R}/eval_kobest_eeve_10b_C_v3_eeve_rerun*.json")),
        "C_label": "C_v3_eeve", "metric": "KoBEST", "type": "reasoning"
    },
    "EXAONE KoBEST": {
        "fp16": 0.7437,
        "A":    kobest_group(lat(f"{R}/eval_kobest_exaone35_7b_A_2026-03-26T09*.json")),
        "B":    kobest_group(lat(f"{R}/eval_kobest_exaone35_7b_B_2026-03-26T09*.json")),
        "C":    kobest_group(lat(f"{R}/eval_kobest_exaone35_7b_C_v3_exaone*.json")),
        "C_label": "C_v3_exaone", "metric": "KoBEST", "type": "reasoning"
    },
    "Llama3-Ko KoBEST": {
        "fp16": kobest_group(lat(f"{R}/eval_kobest_llama3_ko_8b_fp16*.json")),
        "A":    kobest_group(lat(f"{R}/eval_kobest_llama3_ko_8b_A*.json")),
        "B":    kobest_group(lat(f"{R}/eval_kobest_llama3_ko_8b_B*.json")),
        "C":    kobest_group(lat(f"{R}/eval_kobest_llama3_ko_8b_C_en_v3*.json")),
        "C_label": "C_en_v3", "metric": "KoBEST", "type": "reasoning"
    },
    "Qwen2 C-Eval": {
        "fp16": ceval_group(lat(f"{R}/eval_ceval_qwen2_fp16*.json")),
        "A":    ceval_group(lat(f"{R}/eval_ceval_qwen2_gptq_A*.json")),
        "B":    ceval_group(lat(f"{R}/eval_ceval_qwen2_gptq_C_zh_2*.json")),
        "C":    ceval_group(lat(f"{R}/eval_ceval_qwen2_7b_C_zh_v3*.json")),
        "C_label": "C_zh_v3", "metric": "C-Eval", "type": "reasoning"
    },
    # kmmlu 행
    "SOLAR kmmlu": {
        "fp16": None,
        "A":    kmmlu_group(lat(f"{R}/eval_kmmlu_solar_cond_A*.json")),
        "B":    kmmlu_group(lat(f"{R}/eval_kmmlu_solar_cond_B*.json")),
        "C":    kmmlu_group(lat(f"{R}/eval_kmmlu_solar_cond_C_v3*.json")),
        "C_label": "C_v3", "metric": "kmmlu", "type": "knowledge"
    },
    "EEVE kmmlu": {
        "fp16": None,
        "A":    kmmlu_avg(lat(f"{R}/eval_kmmlu_eeve_10b_A*.json")),
        "B":    kmmlu_avg(lat(f"{R}/eval_kmmlu_eeve_10b_B*.json")),
        "C":    kmmlu_avg(lat(f"{R}/eval_kmmlu_eeve_10b_C_v3_eeve*.json")),
        "C_label": "C_v3_eeve", "metric": "kmmlu", "type": "knowledge"
    },
    "EXAONE kmmlu": {
        "fp16": None,
        "A":    kmmlu_avg(lat(f"{R}/eval_kmmlu_exaone35_7b_A*.json")),
        "B":    kmmlu_avg(lat(f"{R}/eval_kmmlu_exaone35_7b_B*.json")),
        "C":    kmmlu_avg(lat(f"{R}/eval_kmmlu_exaone35_7b_C_v3_exaone*.json")),
        "C_label": "C_v3_exaone", "metric": "kmmlu", "type": "knowledge"
    },
}

# ════════════════════════════════════════════════════════════════════════
# Figure: 상단 히트맵 테이블 + 하단 패턴 요약
# ════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 14), facecolor="white")
gs = GridSpec(2, 2, figure=fig, height_ratios=[1.8, 1],
              hspace=0.48, wspace=0.32)

# ── 패널 1: 전체 수치 히트맵 테이블 ────────────────────────────────────
ax_table = fig.add_subplot(gs[0, :])
ax_table.axis("off")

C_A   = "#E74C3C"
C_B   = "#F39C12"
C_CV3 = "#27AE60"
C_FP  = "#2C3E50"

rows = list(data.keys())
cols = ["FP16", "A\n(English\nRandom)", "B / Lang-\naligned\nRandom", "Ours\n(C_v3\nVariant)"]
n_rows, n_cols = len(rows), len(cols)

cell_w, cell_h = 2.5, 0.82
x0, y0 = 0.5, 0.5
pad = 0.06

# 배경 구분 (KoBEST 영역 vs kmmlu 영역)
ax_table.set_xlim(0, x0 + n_cols * cell_w + 2.5)
ax_table.set_ylim(0, y0 + n_rows * cell_h + 1.2)

# 섹션 배경
from matplotlib.patches import FancyBboxPatch as FBP
kobest_rows = [i for i, k in enumerate(rows) if data[k]["type"] == "reasoning"]
kmmlu_rows  = [i for i, k in enumerate(rows) if data[k]["type"] == "knowledge"]

for ir in kobest_rows:
    y = y0 + (n_rows - 1 - ir) * cell_h
    bg = FBP((x0 - 0.1, y - pad), n_cols * cell_w + 0.2, cell_h - pad*2,
             boxstyle="square,pad=0", facecolor="#F0F9F4", edgecolor="none")
    ax_table.add_patch(bg)
for ir in kmmlu_rows:
    y = y0 + (n_rows - 1 - ir) * cell_h
    bg = FBP((x0 - 0.1, y - pad), n_cols * cell_w + 0.2, cell_h - pad*2,
             boxstyle="square,pad=0", facecolor="#FEF9E7", edgecolor="none")
    ax_table.add_patch(bg)

# 구분선
divider_y = y0 + kmmlu_rows[-1] * cell_h + cell_h - 0.05
ax_table.axhline(divider_y + cell_h*0.5 + 0.3, color="#BDC3C7", lw=2, linestyle="--",
                 xmin=0.03, xmax=0.97)

# 컬럼 헤더
col_colors = [C_FP, C_A, C_B, C_CV3]
col_xs = [x0 + (j + 0.5) * cell_w for j in range(n_cols)]
for j, (label, cx, cc) in enumerate(zip(cols, col_xs, col_colors)):
    ax_table.text(cx, y0 + n_rows * cell_h + 0.65, label,
                  ha="center", va="center", fontsize=11, fontweight="bold", color=cc)

# 행 레이블 + 데이터
for i, (row_key, row_data) in enumerate(data.items()):
    ir = n_rows - 1 - i
    y_center = y0 + ir * cell_h + cell_h / 2

    # 행 레이블 (모델명 + 지표)
    model_part = row_key.replace(" KoBEST", "").replace(" kmmlu", "").replace(" C-Eval", "")
    metric_part = row_data["metric"]
    ax_table.text(x0 - 0.15, y_center + 0.12, model_part,
                  ha="right", va="center", fontsize=11, fontweight="bold", color=C_FP)
    metric_color = C_CV3 if row_data["type"] == "reasoning" else "#E67E22"
    ax_table.text(x0 - 0.15, y_center - 0.18, f"[{metric_part}]",
                  ha="right", va="center", fontsize=9, color=metric_color)

    vals = [row_data["fp16"], row_data["A"], row_data["B"], row_data["C"]]
    fp16_val = row_data["fp16"]

    # 각 조건 중 최고값 찾기 (FP16 제외, 실제값 기준)
    gptq_vals = [v for v in vals[1:] if v is not None]
    best_v = max(gptq_vals) if gptq_vals else None

    for j, (val, cx) in enumerate(zip(vals, col_xs)):
        if val is None:
            ax_table.text(cx, y_center, "—", ha="center", va="center",
                          fontsize=10, color="#BDC3C7")
            continue

        # 셀 배경 색상
        is_best = (j > 0 and val == best_v)
        cell_bg = "#EAFAF1" if is_best else "white"
        cell_edge = C_CV3 if is_best else "#E8EAED"
        cell_lw = 2 if is_best else 0.5

        rect = FBP((cx - cell_w/2 + pad, y_center - cell_h/2 + pad*1.5),
                   cell_w - pad*2, cell_h - pad*3,
                   boxstyle="round,pad=0.05",
                   facecolor=cell_bg, edgecolor=cell_edge, lw=cell_lw, zorder=2)
        ax_table.add_patch(rect)

        # 수치
        ax_table.text(cx, y_center + 0.08, f"{val:.4f}",
                      ha="center", va="center", fontsize=11.5,
                      fontweight="bold" if is_best else "normal",
                      color=C_CV3 if is_best else C_FP, zorder=3)

        # 보존율 (FP16 있는 경우)
        if fp16_val and j > 0:
            ret = val / fp16_val * 100
            ax_table.text(cx, y_center - 0.22, f"({ret:.1f}%)",
                          ha="center", va="center", fontsize=8.5,
                          color=C_CV3 if is_best else "#95A5A6", zorder=3)

        # 최고값 별표
        if is_best:
            ax_table.text(cx + cell_w/2 - 0.25, y_center + 0.15, "★",
                          ha="center", va="center", fontsize=9, color=C_CV3, zorder=4)

# 섹션 라벨
ax_table.text(0.15, y0 + len(kobest_rows) * cell_h + cell_h/2 - 0.1,
              "Reasoning\n& NLU\n(KoBEST\n/ C-Eval)",
              ha="center", va="center", fontsize=8.5, color=C_CV3, fontweight="bold",
              rotation=90)
ax_table.text(0.15, y0 + len(kmmlu_rows) / 2 * cell_h,
              "Knowledge\nRecall\n(kmmlu)",
              ha="center", va="center", fontsize=8.5, color="#E67E22", fontweight="bold",
              rotation=90)

ax_table.set_title("All Models × All Conditions × Both Metrics — Complete Comparison\n"
                   "★ = Best GPTQ condition per row   ( ) = retention vs FP16",
                   fontsize=12.5, fontweight="bold", color=C_FP, pad=10)

# ── 패널 2: KoBEST 기준 C vs B 승패 ────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_facecolor("white")

reasoning_models = [k.replace(" KoBEST","").replace(" C-Eval","")
                    for k in rows if data[k]["type"] == "reasoning"]
c_wins_r = [data[k]["C"] - data[k]["B"] for k in rows if data[k]["type"] == "reasoning"]
bar_colors_r = [C_CV3 if v >= 0 else C_B for v in c_wins_r]

bars = ax2.barh(reasoning_models, [v*100 for v in c_wins_r],
                color=bar_colors_r, edgecolor="white", lw=1.2)
ax2.axvline(0, color="#2C3E50", lw=1.5, linestyle="-")

for bar, v in zip(bars, c_wins_r):
    x_pos = v*100 + (0.03 if v >= 0 else -0.03)
    ax2.text(x_pos, bar.get_y() + bar.get_height()/2,
             f"{v*100:+.2f}pp", va="center",
             ha="left" if v >= 0 else "right",
             fontsize=8.5, fontweight="bold",
             color=C_CV3 if v >= 0 else C_B)

ax2.set_xlabel("C_v3 variant − B  (pp)", fontsize=10)
ax2.set_title("Reasoning Tasks\n(KoBEST / C-Eval)\nC_v3 vs B", fontsize=11,
              fontweight="bold", color=C_FP)
ax2.grid(axis="x", alpha=0.3, linestyle="--")

# ── 패널 3: kmmlu 기준 C vs B 승패 ────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_facecolor("white")

knowledge_models = [k.replace(" kmmlu","") for k in rows if data[k]["type"] == "knowledge"]
c_wins_k = [data[k]["C"] - data[k]["B"] for k in rows if data[k]["type"] == "knowledge"]
bar_colors_k = [C_CV3 if v >= 0 else C_B for v in c_wins_k]

bars3 = ax3.barh(knowledge_models, [v*100 for v in c_wins_k],
                 color=bar_colors_k, edgecolor="white", lw=1.2)
ax3.axvline(0, color="#2C3E50", lw=1.5)

for bar, v in zip(bars3, c_wins_k):
    x_pos = v*100 + (0.002 if v >= 0 else -0.002)
    ax3.text(x_pos, bar.get_y() + bar.get_height()/2,
             f"{v*100:+.2f}pp", va="center",
             ha="left" if v >= 0 else "right",
             fontsize=8.5, fontweight="bold",
             color=C_CV3 if v >= 0 else C_B)

ax3.set_xlabel("C_v3 variant − B  (pp)", fontsize=10)
ax3.set_title("Knowledge Recall Tasks\n(kmmlu)\nC_v3 vs B", fontsize=11,
              fontweight="bold", color=C_FP)
ax3.grid(axis="x", alpha=0.3, linestyle="--")

# 범례
legend_els = [
    mpatches.Patch(color=C_CV3, label="C_v3 variant wins"),
    mpatches.Patch(color=C_B,   label="B (random) wins"),
]
fig.legend(handles=legend_els, loc="lower center", ncol=2,
           fontsize=10, bbox_to_anchor=(0.5, -0.02),
           framealpha=0.9)

fig.suptitle("Morpheme Diversity (C_v3) wins on Reasoning  |  Domain Diversity (B) wins on Knowledge Recall\n"
             "→ This asymmetry motivates the KoBEST vs kmmlu analysis (fig10)",
             fontsize=13, fontweight="bold", color=C_FP, y=1.01)

path = f"{OUT}/fig_full_comparison_table.png"
fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"saved: {path}  ({os.path.getsize(path)//1024} KB)")
