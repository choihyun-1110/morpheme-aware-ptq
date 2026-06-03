"""
캡스톤 발표 자료 플레이스홀더 6개 생성 스크립트
결과물: results/presentation_assets/ 에 300dpi PNG 저장
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.gridspec import GridSpec

OUT = "/home/choihyun/workspace/results/presentation_assets"
os.makedirs(OUT, exist_ok=True)

# ── 공통 스타일 ──────────────────────────────────────────────────────────────
PLT_STYLE = {
    "font.family": "NanumGothic",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
}
plt.rcParams.update(PLT_STYLE)

C_FP16  = "#2C3E50"   # 진한 네이비 (FP16 기준)
C_A     = "#E74C3C"   # 빨강 (표준 GPTQ)
C_B     = "#F39C12"   # 주황 (랜덤 한국어)
C_CV3   = "#27AE60"   # 초록 (우리 방법)
C_LIGHT = "#ECF0F1"   # 배경
BG      = "white"

def save(fig, name):
    path = f"{OUT}/{name}"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  saved: {path}")

# ════════════════════════════════════════════════════════════════════════════
# Fig 1 — Korean Morpheme Diagram (Slide 5)
# ════════════════════════════════════════════════════════════════════════════
def fig1_morpheme_diagram():
    fig, ax = plt.subplots(figsize=(14, 7), facecolor=BG)
    ax.set_xlim(0, 14); ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_facecolor(BG)

    # ── 컬럼 헤더 ──
    headers = ["Verb Root", "Morpheme\nAttachments", "Surface Forms", "Tokens", "Activations"]
    xs = [1.2, 3.5, 6.5, 9.5, 12.2]
    for x, h in zip(xs, headers):
        ax.text(x, 6.5, h, ha="center", va="center", fontsize=12, fontweight="bold",
                color="#2C3E50",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#D5E8F4", edgecolor="#2980B9", lw=1.5))

    # ── 동사 어간 ──
    ax.text(xs[0], 3.5, "가\n(ga-)", ha="center", va="center", fontsize=22, fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#2C3E50", edgecolor="#1A252F", lw=2))

    # ── 형태소 요소들 ──
    morphemes = [
        ("-ㄴ다 (Declarative)", 5.5),
        ("-았- (Past)", 4.5),
        ("-겠- (Prospective)", 3.5),
        ("-시- (Honorific)", 2.5),
        ("-버리- (Completive)", 1.5),
    ]
    for label, y in morphemes:
        ax.text(xs[1], y, label, ha="center", va="center", fontsize=9.5,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEF9E7", edgecolor="#F39C12", lw=1.2))
        ax.annotate("", xy=(xs[1]-0.6, y), xytext=(xs[0]+0.45, 3.5),
                    arrowprops=dict(arrowstyle="-|>", color="#95A5A6", lw=0.8,
                                   connectionstyle="arc3,rad=0.0"))

    # ── 표면형 ──
    surfaces = ["가다", "갔다", "가겠다", "가셨습니다", "가버렸다"]
    colors_s  = ["#1ABC9C","#3498DB","#9B59B6","#E67E22","#E74C3C"]
    for i, (sf, cs) in enumerate(zip(surfaces, colors_s)):
        y = 5.2 - i * 0.95
        ax.text(xs[2], y, sf, ha="center", va="center", fontsize=11, fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round,pad=0.35", facecolor=cs, edgecolor="white", lw=1))
        ax.annotate("", xy=(xs[2]-0.55, y), xytext=(xs[1]+0.55, 5.5-i*0.95),
                    arrowprops=dict(arrowstyle="-|>", color="#BDC3C7", lw=0.7,
                                   connectionstyle="arc3,rad=0.0"))

    # ── 토큰 시퀀스 ──
    token_rows = [
        ["가", "다"],
        ["갔", "다"],
        ["가", "겠", "다"],
        ["가", "셨", "습니다"],
        ["가", "버", "렸", "다"],
    ]
    tok_colors = ["#1ABC9C","#3498DB","#9B59B6","#E67E22","#E74C3C"]
    for i, (toks, tc) in enumerate(zip(token_rows, tok_colors)):
        y = 5.2 - i * 0.95
        x_start = xs[3] - (len(toks)-1)*0.32
        for j, t in enumerate(toks):
            tx = x_start + j*0.64
            ax.text(tx, y, t, ha="center", va="center", fontsize=8.5, fontweight="bold",
                    color="white",
                    bbox=dict(boxstyle="square,pad=0.25", facecolor=tc, alpha=0.8))
        ax.annotate("", xy=(xs[3]-0.7, y), xytext=(xs[2]+0.45, y),
                    arrowprops=dict(arrowstyle="-|>", color="#BDC3C7", lw=0.7))

    # ── Activations (히트맵 느낌) ──
    act_labels = ["h₁", "h₂", "h₃", "h₄", "h₅"]
    act_vals_list = [
        [0.9, 0.1, 0.2, 0.3],
        [0.3, 0.8, 0.1, 0.2],
        [0.5, 0.2, 0.9, 0.1],
        [0.2, 0.5, 0.3, 0.9],
        [0.8, 0.3, 0.7, 0.2],
    ]
    act_colors = ["#1ABC9C","#3498DB","#9B59B6","#E67E22","#E74C3C"]
    for i, (vals, ac) in enumerate(zip(act_vals_list, act_colors)):
        y = 5.2 - i * 0.95
        x_start = xs[4] - 0.45
        for j, v in enumerate(vals):
            rect = FancyBboxPatch((x_start + j*0.32 - 0.14, y-0.22), 0.28, 0.44,
                                  boxstyle="square,pad=0.0",
                                  facecolor=matplotlib.colormaps["RdYlGn"](v),
                                  edgecolor="white", lw=0.5)
            ax.add_patch(rect)
        ax.annotate("", xy=(xs[4]-0.65, y), xytext=(xs[3]+0.7, y),
                    arrowprops=dict(arrowstyle="-|>", color="#BDC3C7", lw=0.7))

    ax.text(xs[4], 0.6, "Different activation\npatterns per surface form",
            ha="center", va="center", fontsize=8.5, style="italic", color="#7F8C8D")

    ax.set_title("Korean Agglutination → Morphological Diversity → Diverse Activations",
                 fontsize=13, fontweight="bold", pad=10, color="#2C3E50")
    save(fig, "fig1_korean_morpheme_diagram.png")

# ════════════════════════════════════════════════════════════════════════════
# Fig 2 — MAC Pipeline (Slide 6)
# ════════════════════════════════════════════════════════════════════════════
def fig2_mac_pipeline():
    fig, ax = plt.subplots(figsize=(15, 5), facecolor=BG)
    ax.set_xlim(0, 15); ax.set_ylim(0, 5)
    ax.axis("off"); ax.set_facecolor(BG)

    stages = [
        ("NamuWiki\nCorpus", "~5M\nsentences", "#8E44AD", "white"),
        ("Korean\nQuality Filter", "ko_ratio ≥ 0.7\nmin 5 eojeols", "#2980B9", "white"),
        ("Kiwipiepy\nMorpheme Analysis", "(lemma, POS)\npairs", "#16A085", "white"),
        ("Greedy MAC\nSelection", "max new\n(lemma, POS)", "#D35400", "white"),
        ("Calibration Set\nC_v3.json", "N = 128\nsentences", "#27AE60", "white"),
        ("GPTQ\n4-bit", "optimum.gptq\ng=128", "#2C3E50", "white"),
    ]
    xs = [1.2, 3.5, 5.8, 8.1, 10.4, 12.7]
    w, h_box = 1.9, 1.6

    for i, ((title, sub, fc, tc), x) in enumerate(zip(stages, xs)):
        # 박스
        rect = FancyBboxPatch((x - w/2, 1.7), w, h_box,
                              boxstyle="round,pad=0.15",
                              facecolor=fc, edgecolor="white", lw=2)
        ax.add_patch(rect)
        ax.text(x, 2.5 + 0.15, title, ha="center", va="center",
                fontsize=10.5, fontweight="bold", color=tc)
        ax.text(x, 1.95, sub, ha="center", va="center",
                fontsize=8, color=tc, alpha=0.88)

        # 화살표
        if i < len(stages) - 1:
            ax.annotate("", xy=(xs[i+1] - w/2 - 0.05, 2.5),
                        xytext=(x + w/2 + 0.05, 2.5),
                        arrowprops=dict(arrowstyle="-|>", color="#7F8C8D", lw=2.5))

    # 하단 설명
    notes = [
        (xs[1], "Removes non-Korean\n& code-mixed text"),
        (xs[2], "Extracts morphological\nfeatures per sentence"),
        (xs[3], "Iteratively selects\nmax coverage gain"),
        (xs[4], "128 diverse Korean\nsentences"),
    ]
    for x, note in notes:
        ax.text(x, 1.3, note, ha="center", va="center",
                fontsize=7.5, color="#7F8C8D", style="italic")

    # 성능 배지
    ax.text(xs[4], 4.2, "✓ 97.4% FP16 preserved", ha="center",
            fontsize=10, fontweight="bold", color=C_CV3,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#EAFAF1", edgecolor=C_CV3, lw=1.5))

    ax.set_title("MAC (Morpheme-Aware Calibration) Pipeline", fontsize=14,
                 fontweight="bold", pad=8, color="#2C3E50")
    save(fig, "fig2_mac_pipeline.png")

# ════════════════════════════════════════════════════════════════════════════
# Fig 3 — SOLAR Bar Chart + Effect Decomposition (Slide 7)
# ════════════════════════════════════════════════════════════════════════════
def fig3_solar_bar():
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
    ax.set_facecolor(BG)

    labels = ["A\n(Standard GPTQ\nEnglish Random)", "B\n(Korean Random)", "C_v3\n(MAC — Ours)"]
    values = [0.5981, 0.6176, 0.6356]
    colors = [C_A, C_B, C_CV3]
    fp16   = 0.6523

    bars = ax.bar(labels, values, color=colors, width=0.52, edgecolor="white",
                  linewidth=1.5, zorder=3)

    # FP16 점선
    ax.axhline(fp16, color=C_FP16, linestyle="--", linewidth=2, zorder=4, label=f"FP16 baseline ({fp16:.4f})")
    ax.text(2.42, fp16 + 0.002, f"FP16 = {fp16}", fontsize=9.5, color=C_FP16, va="bottom", fontweight="bold")

    # 수치 라벨
    retentions = [91.7, 94.7, 97.4]
    for bar, val, ret in zip(bars, values, retentions):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.002,
                f"{val:.4f}\n({ret:.1f}%)", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="#2C3E50")

    # ── 효과 분해 화살표 ──
    # B - A = +2.0pp (Language alignment)
    x0, x1 = 0, 1
    mid_y = (values[0] + values[1]) / 2 + 0.012
    ax.annotate("", xy=(x1 - 0.26, values[1]), xytext=(x0 + 0.26, values[0]),
                arrowprops=dict(arrowstyle="<->", color="#2980B9", lw=2.5))
    ax.text(0.5, mid_y + 0.003, "Language\nAlignment\n+2.0 pp", ha="center",
            fontsize=9, color="#2980B9", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#EBF5FB", edgecolor="#2980B9", lw=1))

    # C_v3 - B = +1.8pp (Diversity)
    x0, x1 = 1, 2
    mid_y = (values[1] + values[2]) / 2 + 0.012
    ax.annotate("", xy=(x1 - 0.26, values[2]), xytext=(x0 + 0.26, values[1]),
                arrowprops=dict(arrowstyle="<->", color=C_CV3, lw=2.5))
    ax.text(1.5, mid_y + 0.003, "Morpheme\nDiversity\n+1.8 pp", ha="center",
            fontsize=9, color=C_CV3, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#EAFAF1", edgecolor=C_CV3, lw=1))

    ax.set_ylim(0.56, 0.685)
    ax.set_ylabel("KoBEST Average Accuracy", fontsize=12, labelpad=8)
    ax.set_title("SOLAR-10.7B GPTQ 4-bit — KoBEST Performance & Effect Decomposition",
                 fontsize=13, fontweight="bold", pad=10)
    ax.legend(fontsize=10, loc="lower right")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    save(fig, "fig3_solar_bar_chart.png")

# ════════════════════════════════════════════════════════════════════════════
# Fig 4 — Layer Sensitivity Heatmap (Slide 8-1)
# ════════════════════════════════════════════════════════════════════════════
def fig4_sensitivity_heatmap():
    with open("/home/choihyun/workspace/results/sensitivity_score.json") as f:
        s = json.load(f)

    n_layers = max(int(k) for k in s["per_layer"].keys()) + 1
    scores = np.array([s["per_layer"][str(i)]["score"] for i in range(n_layers)])
    high_layers = set(s["summary"]["high_sensitivity_layers"])
    threshold = s["summary"]["threshold_q75"]

    fig, ax = plt.subplots(figsize=(14, 4.5), facecolor=BG)
    ax.set_facecolor(BG)

    # 배경 highlight (고감도 레이어)
    for i in range(n_layers):
        if i in high_layers:
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.18, color="#E74C3C", zorder=1)

    # 막대 색상: 고감도=빨강, 일반=회색
    bar_colors = [C_A if i in high_layers else "#BDC3C7" for i in range(n_layers)]
    ax.bar(range(n_layers), scores, color=bar_colors, width=0.75, zorder=2, edgecolor="white", lw=0.3)

    # threshold 선
    ax.axhline(threshold, color="#E74C3C", linestyle="--", linewidth=1.8, zorder=3,
               label=f"Q75 threshold ({threshold:.3f})")

    # 구간 브라켓 표시
    hl = sorted(high_layers)
    hl_start, hl_end = min(hl), max(hl)
    ax.annotate("", xy=(hl_end + 0.5, -0.045), xytext=(hl_start - 0.5, -0.045),
                arrowprops=dict(arrowstyle="<->", color=C_A, lw=2),
                annotation_clip=False)
    ax.text((hl_start + hl_end) / 2, -0.068, f"High-Sensitivity Zone\nL{hl_start}-L{hl_end}  ({len(high_layers)} layers)",
            ha="center", va="top", fontsize=9.5, color=C_A, fontweight="bold",
            transform=ax.transData, clip_on=False)

    # top-5 레이어 별표
    for li in s["summary"]["top5_layers"]:
        ax.text(li, scores[li] + 0.012, "★", ha="center", va="bottom",
                fontsize=9, color=C_A)

    ax.set_xlabel("Transformer Layer Index", fontsize=11, labelpad=6)
    ax.set_ylabel("Activation Sensitivity Score", fontsize=11, labelpad=6)
    ax.set_title("Layer-wise Activation Sensitivity Score  (A vs C_v3 Calibration on FP16 Model)",
                 fontsize=12, fontweight="bold", pad=8)
    ax.set_xlim(-1, n_layers)
    ax.set_ylim(-0.01, scores.max() * 1.18)
    ax.set_xticks([0, 8, 16, 24, 32, 40, n_layers-1])
    ax.legend(fontsize=9.5, loc="upper left")

    # 범례 패치
    hp = mpatches.Patch(color=C_A, alpha=0.5, label=f"High sensitivity ({len(high_layers)} layers)")
    lp = mpatches.Patch(color="#BDC3C7", label="Normal sensitivity")
    ax.legend(handles=[hp, lp] + ax.get_legend_handles_labels()[0], fontsize=9, loc="upper left")

    save(fig, "fig4_sensitivity_heatmap.png")

# ════════════════════════════════════════════════════════════════════════════
# Fig 5 — Activation Distortion Line Chart (Slide 8-2)
# ════════════════════════════════════════════════════════════════════════════
def fig5_distortion_line():
    with open("/home/choihyun/workspace/results/model_activation_comparison.json") as f:
        d = json.load(f)

    per_layer_A   = d["per_layer"]["GPTQ-A"]
    per_layer_Cv3 = d["per_layer"]["GPTQ-C_v3"]
    n_layers = max(int(k) for k in per_layer_A.keys()) + 1

    dist_A   = [per_layer_A[str(i)]["distortion"] for i in range(n_layers)]
    dist_Cv3 = [per_layer_Cv3[str(i)]["distortion"] for i in range(n_layers)]

    fig, ax = plt.subplots(figsize=(13, 5.5), facecolor=BG)
    ax.set_facecolor(BG)

    xs = np.arange(n_layers)
    ax.fill_between(xs, dist_A, alpha=0.15, color=C_A)
    ax.fill_between(xs, dist_Cv3, alpha=0.15, color=C_CV3)
    ax.plot(xs, dist_A,   color=C_A,   linewidth=2,   label="GPTQ-A  (Standard, English Random)", zorder=3)
    ax.plot(xs, dist_Cv3, color=C_CV3, linewidth=2.2, label="GPTQ-C_v3  (MAC — Ours)",           zorder=4)
    ax.axhline(0, color=C_FP16, linestyle="--", linewidth=1.8, label="FP16 reference (distortion = 0)", zorder=2)

    # 고감도 구간 강조
    for li in [23, 24, 25, 26, 27, 28, 29, 30, 34, 35, 36, 37]:
        if li < n_layers:
            ax.axvspan(li - 0.5, li + 0.5, alpha=0.12, color="#F39C12", zorder=1)

    # 최대 차이 지점 annotation
    diffs = [a - c for a, c in zip(dist_A, dist_Cv3)]
    max_diff_layer = int(np.argmax(diffs))
    ax.annotate(
        f"Max gap at L{max_diff_layer}\nA={dist_A[max_diff_layer]:.4f}\nC_v3={dist_Cv3[max_diff_layer]:.4f}",
        xy=(max_diff_layer, dist_A[max_diff_layer]),
        xytext=(max_diff_layer + 3, dist_A[max_diff_layer] + 0.005),
        fontsize=8.5, color=C_A,
        arrowprops=dict(arrowstyle="->", color=C_A, lw=1.2),
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#FDEDEC", edgecolor=C_A, lw=1)
    )

    # 요약 텍스트
    imp = d["summary"]["improvement_pct"]
    ax.text(0.97, 0.94, f"C_v3 reduces distortion by\n{imp:.1f}% vs GPTQ-A",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=11, fontweight="bold", color=C_CV3,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#EAFAF1", edgecolor=C_CV3, lw=1.5))

    # 고감도 구간 범례
    span_patch = mpatches.Patch(color="#F39C12", alpha=0.25, label="High-sensitivity zone (L23–L37)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [span_patch], labels=labels + ["High-sensitivity zone (L23–L37)"],
              fontsize=9.5, loc="upper left")

    ax.set_xlabel("Transformer Layer Index", fontsize=11, labelpad=6)
    ax.set_ylabel("Activation Distortion  (‖GPTQ − FP16‖_F / ‖FP16‖_F)", fontsize=10, labelpad=6)
    ax.set_title("Layer-wise Activation Distortion: Korean Input on GPTQ-A vs GPTQ-C_v3",
                 fontsize=12, fontweight="bold", pad=8)
    ax.set_xlim(-1, n_layers)
    ax.set_ylim(-0.002, max(dist_A) * 1.22)
    ax.set_xticks([0, 8, 16, 24, 32, 40, n_layers-1])

    save(fig, "fig5_distortion_line_chart.png")

# ════════════════════════════════════════════════════════════════════════════
# Fig 6 — Cross-Model Grouped Bar Chart (Slide 9)
# ════════════════════════════════════════════════════════════════════════════
def fig6_crossmodel_bar():
    models = ["SOLAR\n10.7B", "EEVE\n10.8B", "EXAONE35\n7.8B", "Qwen2\n7B\n(C-Eval)", "Llama3-Ko\n8B"]
    # [A, B/language-aligned-random, C_v3-variant] 보존율(%)
    data = {
        "A (Standard GPTQ)":       [91.7, 96.2, 93.6, 93.5, 97.6],
        "B (Language-aligned)":    [94.7, 96.7, 97.4, 93.5, 95.1],
        "Ours (MAC C_v3 variant)": [97.4, 97.3, 99.7, 96.4, 98.4],
    }
    colors = [C_A, C_B, C_CV3]

    n_models = len(models)
    n_groups = len(data)
    bar_w = 0.24
    x = np.arange(n_models)

    fig, ax = plt.subplots(figsize=(13, 6.5), facecolor=BG)
    ax.set_facecolor(BG)

    offsets = [-bar_w, 0, bar_w]
    for (label, vals), offset, color in zip(data.items(), offsets, colors):
        bars = ax.bar(x + offset, vals, width=bar_w, label=label,
                      color=color, edgecolor="white", linewidth=1.2, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.2,
                    f"{v:.1f}%", ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold", color="#2C3E50")

    # 100% 점선
    ax.axhline(100, color=C_FP16, linestyle="--", linewidth=1.5,
               label="FP16 baseline (100%)", zorder=2)

    # 모든 모델에서 우리 방법이 최고임을 강조하는 별표
    best_vals = data["Ours (MAC C_v3 variant)"]
    for i, v in enumerate(best_vals):
        ax.text(i + bar_w, v + 1.0, "★", ha="center", va="bottom",
                fontsize=12, color=C_CV3, zorder=5)

    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel("KoBEST / C-Eval Retention  (%  of FP16)", fontsize=11, labelpad=8)
    ax.set_ylim(88, 103)
    ax.set_title("Cross-Model Generalization — MAC Outperforms Standard GPTQ on All 5 Models",
                 fontsize=12.5, fontweight="bold", pad=10)
    ax.legend(fontsize=10, loc="lower right",
              bbox_to_anchor=(1.0, 0.0), framealpha=0.9)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g%%"))
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # ★ 범례 주석
    ax.text(0.01, 0.96, "★ Best GPTQ condition per model",
            transform=ax.transAxes, fontsize=8.5, color=C_CV3, va="top")

    save(fig, "fig6_crossmodel_grouped_bar.png")

# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating presentation figures...")
    fig1_morpheme_diagram();  print("  [1/6] Korean morpheme diagram")
    fig2_mac_pipeline();      print("  [2/6] MAC pipeline")
    fig3_solar_bar();         print("  [3/6] SOLAR bar chart")
    fig4_sensitivity_heatmap(); print("  [4/6] Sensitivity heatmap")
    fig5_distortion_line();   print("  [5/6] Distortion line chart")
    fig6_crossmodel_bar();    print("  [6/6] Cross-model grouped bar")
    print(f"\nAll figures saved to: {OUT}")
    import os
    for f in sorted(os.listdir(OUT)):
        size = os.path.getsize(f"{OUT}/{f}") // 1024
        print(f"  {f}  ({size} KB)")
