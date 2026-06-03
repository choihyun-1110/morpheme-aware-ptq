"""
발표 자료 추가 figure 생성 (빠진 4개)
- fig7: 영어 calibration 문제 개념도
- fig8: PPL 결과 막대
- fig9: Greedy 알고리즘 step-by-step
- fig10: KoBEST vs kmmlu 2x2 matrix
- fig11: 결론 요약
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

OUT = "/home/choihyun/workspace/results/presentation_assets"
os.makedirs(OUT, exist_ok=True)

PLT_STYLE = {
    "font.family": "NanumGothic",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
}
plt.rcParams.update(PLT_STYLE)

C_FP16 = "#2C3E50"
C_A    = "#E74C3C"
C_B    = "#F39C12"
C_CV3  = "#27AE60"
BG     = "white"

def save(fig, name):
    path = f"{OUT}/{name}"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  saved: {path}")


# ════════════════════════════════════════════════════════════════════════
# Fig 7 — 영어 calibration 문제 개념도 (Slide 2~3)
# ════════════════════════════════════════════════════════════════════════
def fig7_problem_concept():
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG)
    fig.suptitle("Why English Calibration Fails for Korean LLMs",
                 fontsize=15, fontweight="bold", color=C_FP16, y=1.01)

    for ax in axes:
        ax.set_xlim(0, 10); ax.set_ylim(0, 10)
        ax.axis("off"); ax.set_facecolor(BG)

    # ── 왼쪽: 영어 calibration (A 조건) ──
    ax = axes[0]
    ax.set_title("Standard GPTQ  (Condition A — English)", fontsize=12,
                 fontweight="bold", color=C_A, pad=8)

    # Step 1: 영어 문장
    box1 = FancyBboxPatch((0.5, 8.2), 9, 1.1, boxstyle="round,pad=0.15",
                           facecolor="#FDEDEC", edgecolor=C_A, lw=1.5)
    ax.add_patch(box1)
    ax.text(5, 8.77, '"The government announced new economic policies."',
            ha="center", va="center", fontsize=9, style="italic", color="#2C3E50")
    ax.text(5, 8.35, "English Wikitext-2  (Calibration Input)", ha="center",
            va="center", fontsize=8.5, color=C_A, fontweight="bold")

    # 화살표
    ax.annotate("", xy=(5, 7.5), xytext=(5, 8.2),
                arrowprops=dict(arrowstyle="-|>", color="#7F8C8D", lw=2))

    # Step 2: 모델 내부 (뉴런)
    ax.text(5, 7.25, "FP16 Model (SOLAR-10.7B)", ha="center", va="center",
            fontsize=10, fontweight="bold", color=C_FP16)
    neurons = [
        (2.0, 6.2, "General\nLayer", 0.9, "#3498DB"),
        (3.8, 6.2, "English\nSyntax", 0.85, "#3498DB"),
        (5.6, 6.2, "Korean\nMorpheme\n(dormant)", 0.15, "#BDC3C7"),
        (7.4, 6.2, "Korean\nParticle\n(dormant)", 0.1, "#BDC3C7"),
    ]
    for nx, ny, label, alpha, color in neurons:
        circle = plt.Circle((nx, ny), 0.75, color=color, alpha=alpha, zorder=3)
        ax.add_patch(circle)
        ax.text(nx, ny, label, ha="center", va="center", fontsize=7.5,
                color="white" if alpha > 0.3 else "#7F8C8D", fontweight="bold")

    ax.text(5.0, 4.95, "❌  Korean neurons DORMANT", ha="center",
            fontsize=9, color=C_A, fontweight="bold")

    ax.annotate("", xy=(5, 4.3), xytext=(5, 4.9),
                arrowprops=dict(arrowstyle="-|>", color="#7F8C8D", lw=2))

    # Step 3: Hessian
    hess_box = FancyBboxPatch((1.5, 3.0), 7, 1.1, boxstyle="round,pad=0.12",
                               facecolor="#FDEDEC", edgecolor=C_A, lw=1.5)
    ax.add_patch(hess_box)
    ax.text(5, 3.62, "H = 2XX^T  →  Korean channels UNDERWEIGHTED",
            ha="center", va="center", fontsize=9, color=C_A, fontweight="bold")
    ax.text(5, 3.18, "Hessian underestimates Korean weight importance",
            ha="center", va="center", fontsize=8, color="#7F8C8D")

    ax.annotate("", xy=(5, 2.3), xytext=(5, 3.0),
                arrowprops=dict(arrowstyle="-|>", color="#7F8C8D", lw=2))

    # Step 4: 결과
    res_box = FancyBboxPatch((0.5, 1.0), 9, 1.1, boxstyle="round,pad=0.12",
                              facecolor="#FDEDEC", edgecolor=C_A, lw=2)
    ax.add_patch(res_box)
    ax.text(5, 1.62, "KoBEST: 0.5981  (91.7% of FP16)",
            ha="center", va="center", fontsize=11, fontweight="bold", color=C_A)
    ax.text(5, 1.18, "Wrong weight rounding → Korean accuracy drops",
            ha="center", va="center", fontsize=8, color="#7F8C8D")

    # ── 오른쪽: MAC (C_v3 조건) ──
    ax = axes[1]
    ax.set_title("MAC — Morpheme-Aware Calibration  (C_v3)", fontsize=12,
                 fontweight="bold", color=C_CV3, pad=8)

    box1r = FancyBboxPatch((0.5, 8.2), 9, 1.1, boxstyle="round,pad=0.15",
                            facecolor="#EAFAF1", edgecolor=C_CV3, lw=1.5)
    ax.add_patch(box1r)
    ax.text(5, 8.77, '"갔다, 가겠다, 가셨습니다, 가버렸다 …"  (128 diverse sentences)',
            ha="center", va="center", fontsize=9, style="italic", color="#2C3E50")
    ax.text(5, 8.35, "Greedy Morpheme-Diverse Korean  (Calibration Input)", ha="center",
            va="center", fontsize=8.5, color=C_CV3, fontweight="bold")

    ax.annotate("", xy=(5, 7.5), xytext=(5, 8.2),
                arrowprops=dict(arrowstyle="-|>", color="#7F8C8D", lw=2))

    ax.text(5, 7.25, "FP16 Model (SOLAR-10.7B)", ha="center", va="center",
            fontsize=10, fontweight="bold", color=C_FP16)
    neurons_r = [
        (2.0, 6.2, "General\nLayer", 0.9, "#3498DB"),
        (3.8, 6.2, "English\nSyntax", 0.6, "#3498DB"),
        (5.6, 6.2, "Korean\nMorpheme\n(ACTIVE)", 0.9, C_CV3),
        (7.4, 6.2, "Korean\nParticle\n(ACTIVE)", 0.9, C_CV3),
    ]
    for nx, ny, label, alpha, color in neurons_r:
        circle = plt.Circle((nx, ny), 0.75, color=color, alpha=alpha, zorder=3)
        ax.add_patch(circle)
        ax.text(nx, ny, label, ha="center", va="center", fontsize=7.5,
                color="white", fontweight="bold")

    ax.text(5.0, 4.95, "✓  Korean neurons ACTIVE", ha="center",
            fontsize=9, color=C_CV3, fontweight="bold")

    ax.annotate("", xy=(5, 4.3), xytext=(5, 4.9),
                arrowprops=dict(arrowstyle="-|>", color="#7F8C8D", lw=2))

    hess_box_r = FancyBboxPatch((1.5, 3.0), 7, 1.1, boxstyle="round,pad=0.12",
                                 facecolor="#EAFAF1", edgecolor=C_CV3, lw=1.5)
    ax.add_patch(hess_box_r)
    ax.text(5, 3.62, "H = 2XX^T  →  Korean channels PROPERLY WEIGHTED",
            ha="center", va="center", fontsize=9, color=C_CV3, fontweight="bold")
    ax.text(5, 3.18, "Hessian correctly captures Korean weight importance",
            ha="center", va="center", fontsize=8, color="#7F8C8D")

    ax.annotate("", xy=(5, 2.3), xytext=(5, 3.0),
                arrowprops=dict(arrowstyle="-|>", color="#7F8C8D", lw=2))

    res_box_r = FancyBboxPatch((0.5, 1.0), 9, 1.1, boxstyle="round,pad=0.12",
                                facecolor="#EAFAF1", edgecolor=C_CV3, lw=2)
    ax.add_patch(res_box_r)
    ax.text(5, 1.62, "KoBEST: 0.6356  (97.4% of FP16)",
            ha="center", va="center", fontsize=11, fontweight="bold", color=C_CV3)
    ax.text(5, 1.18, "Correct rounding → Korean accuracy preserved",
            ha="center", va="center", fontsize=8, color="#7F8C8D")

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    save(fig, "fig7_problem_concept.png")


# ════════════════════════════════════════════════════════════════════════
# Fig 8 — Korean PPL 막대 (Slide 7 메커니즘)
# ════════════════════════════════════════════════════════════════════════
def fig8_ppl_bar():
    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=BG)
    ax.set_facecolor(BG)

    labels = ["FP16\n(Reference)", "A\n(Standard GPTQ)", "C_v3\n(MAC — Ours)"]
    ppls   = [19.34, 21.66, 20.08]
    colors = [C_FP16, C_A, C_CV3]
    increases = ["+0%", "+12.0%", "+3.8%"]

    bars = ax.bar(labels, ppls, color=colors, width=0.45,
                  edgecolor="white", linewidth=1.5, zorder=3)

    for bar, ppl, inc in zip(bars, ppls, increases):
        ax.text(bar.get_x() + bar.get_width()/2, ppl + 0.08,
                f"PPL = {ppl}\n({inc})", ha="center", va="bottom",
                fontsize=10.5, fontweight="bold", color="#2C3E50")

    # 이중 화살표: A vs C_v3 차이
    ax.annotate("", xy=(2 - 0.22, ppls[2]), xytext=(1 + 0.22, ppls[1]),
                arrowprops=dict(arrowstyle="<->", color="#8E44AD", lw=2.2))
    ax.text(1.5, (ppls[1]+ppls[2])/2 + 0.15, "C_v3 reduces\nPPL distortion\nby 68%",
            ha="center", va="bottom", fontsize=9, fontweight="bold", color="#8E44AD",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#F5EEF8", edgecolor="#8E44AD", lw=1))

    ax.set_ylim(18.0, 23.5)
    ax.set_ylabel("Korean Perplexity (PPL)  ↓ lower is better", fontsize=11, labelpad=8)
    ax.set_title("Korean PPL — SOLAR-10.7B GPTQ 4-bit\n(Namuwiki held-out 500 sentences)",
                 fontsize=12, fontweight="bold", pad=10)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # 참고: PPL 낮을수록 좋다 주석
    ax.text(0.01, 0.96, "Lower PPL = better language modeling quality",
            transform=ax.transAxes, fontsize=8.5, color="#7F8C8D", va="top", style="italic")

    save(fig, "fig8_ppl_bar.png")


# ════════════════════════════════════════════════════════════════════════
# Fig 9 — Greedy 알고리즘 Step-by-Step (Slide 5)
# ════════════════════════════════════════════════════════════════════════
def fig9_greedy_steps():
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.5), facecolor=BG)
    fig.suptitle("Greedy Morpheme Coverage Selection — Step by Step",
                 fontsize=14, fontweight="bold", color=C_FP16, y=1.02)

    steps = [
        {
            "title": "Step 1  (coverage = ∅)",
            "sentences": [
                ("S1: 그는 책을 읽으며 커피를 마셨다",
                 ["그(NP)", "책(NNG)", "읽다(VV)", "으며(EC)", "커피(NNG)", "마시다(VV)", "었(EP)", "다(EF)"],
                 8, True),
                ("S2: 나는 오늘 학교에 갔다",
                 ["나(NP)", "오늘(NNG)", "학교(NNG)", "가다(VV)", "었(EP)", "다(EF)"],
                 6, False),
                ("S3: 어제 비가 많이 왔어요",
                 ["어제(NNG)", "비(NNG)", "많이(MAG)", "오다(VV)", "아요(EF)"],
                 5, False),
            ],
            "selected": "S1  →  8 new pairs",
            "note": "S1 선택: 8개 새 형태소 쌍\n(그, 책, 읽다, 으며, 커피, 마시다, 었, 다)"
        },
        {
            "title": "Step 2  (coverage = {그, 책, 읽다, 으며, 커피, 마시다, 었, 다})",
            "sentences": [
                ("S2: 나는 오늘 학교에 갔다",
                 ["나(NP)★", "오늘(NNG)★", "학교(NNG)★", "가다(VV)★", "었 ✓", "다 ✓"],
                 4, False),
                ("S3: 어제 비가 많이 왔어요",
                 ["어제(NNG)★", "비(NNG)★", "많이(MAG)★", "오다(VV)★", "아요(EF)★"],
                 5, True),
                ("S4: 나는 오늘 시장에 갔다",
                 ["나(NP)★", "오늘 ✓", "시장(NNG)★", "가다 ✓", "었 ✓", "다 ✓"],
                 2, False),
            ],
            "selected": "S3  →  5 new pairs",
            "note": "S3 선택: 5개 새 쌍\n(어제, 비, 많이, 오다, 아요)\nS4는 2개뿐 — 비효율"
        },
        {
            "title": "Step 3  (coverage += {어제, 비, 많이, 오다, 아요})",
            "sentences": [
                ("S2: 나는 오늘 학교에 갔다",
                 ["나(NP)★", "오늘(NNG)★", "학교(NNG)★", "가다(VV)★", "었 ✓", "다 ✓"],
                 4, True),
                ("S4: 나는 오늘 시장에 갔다",
                 ["나 ✓", "오늘 ✓", "시장(NNG)★", "가다 ✓", "었 ✓", "다 ✓"],
                 1, False),
                ("S5: 경제성장률이 3분기 하락했다",
                 ["경제(NNG)★", "성장률(NNG)★", "분기(NNG)★", "하락(VV)★", "었 ✓", "다 ✓"],
                 4, False),
            ],
            "selected": "S2 or S5  →  4 new pairs each",
            "note": "S4는 1개뿐 → 탈락\nS2 선택 (tie-break: 인덱스 순)"
        },
    ]

    for ax, step in zip(axes, steps):
        ax.set_xlim(0, 10); ax.set_ylim(0, 10)
        ax.axis("off"); ax.set_facecolor(BG)
        ax.set_title(step["title"], fontsize=9.5, fontweight="bold",
                     color=C_FP16, pad=6, wrap=True)

        y = 9.2
        for sent_text, morphemes, new_count, selected in step["sentences"]:
            bg = "#EAFAF1" if selected else "#FDFEFE"
            edge = C_CV3 if selected else "#D5D8DC"
            lw = 2 if selected else 1

            rect = FancyBboxPatch((0.3, y-1.45), 9.4, 1.35,
                                  boxstyle="round,pad=0.1",
                                  facecolor=bg, edgecolor=edge, lw=lw, zorder=2)
            ax.add_patch(rect)

            ax.text(0.7, y - 0.35, sent_text, fontsize=8, va="center",
                    color=C_CV3 if selected else C_FP16, fontweight="bold" if selected else "normal")

            morpheme_str = "  ".join(morphemes[:5]) + ("  …" if len(morphemes) > 5 else "")
            ax.text(0.7, y - 0.78, morpheme_str, fontsize=6.8, va="center",
                    color="#5D6D7E")

            color_badge = C_CV3 if selected else (C_A if new_count <= 2 else C_B)
            ax.text(9.3, y - 0.75, f"+{new_count}", ha="right", va="center",
                    fontsize=11, fontweight="bold", color=color_badge)

            if selected:
                ax.text(9.3, y - 0.3, "✓ SELECTED", ha="right", va="center",
                        fontsize=7.5, fontweight="bold", color=C_CV3)
            y -= 1.65

        # 선택 결과 배지
        ax.text(5, 0.85, step["selected"], ha="center", va="center",
                fontsize=9.5, fontweight="bold", color=C_CV3,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#EAFAF1", edgecolor=C_CV3, lw=1.5))
        ax.text(5, 0.3, step["note"], ha="center", va="center",
                fontsize=7.5, color="#7F8C8D", style="italic")

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    save(fig, "fig9_greedy_steps.png")


# ════════════════════════════════════════════════════════════════════════
# Fig 10 — KoBEST vs kmmlu 불일치 2x2 매트릭스 (추가 실험)
# ════════════════════════════════════════════════════════════════════════
def fig10_task_matrix():
    fig, ax = plt.subplots(figsize=(11, 7), facecolor=BG)
    ax.set_xlim(0, 11); ax.set_ylim(0, 8)
    ax.axis("off"); ax.set_facecolor(BG)

    ax.set_title("Why KoBEST and kmmlu Disagree — Two Types of Diversity",
                 fontsize=13, fontweight="bold", color=C_FP16, pad=10)

    # 축 라벨
    ax.text(5.5, 7.3, "Calibration Type", ha="center", fontsize=11,
            fontweight="bold", color=C_FP16)
    ax.text(0.5, 3.8, "Evaluation\nTask", ha="center", fontsize=11,
            fontweight="bold", color=C_FP16, rotation=90, va="center")

    # 열 헤더
    ax.text(3.2, 6.8, "Morpheme Diversity\n(C_v3, C_v3_eeve…)", ha="center",
            fontsize=10, fontweight="bold", color=C_CV3,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#EAFAF1", edgecolor=C_CV3, lw=1.5))
    ax.text(7.8, 6.8, "Domain Diversity\n(B — random Korean)", ha="center",
            fontsize=10, fontweight="bold", color=C_B,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEF9E7", edgecolor=C_B, lw=1.5))

    # 행 헤더
    ax.text(1.2, 5.2, "KoBEST\n(Reasoning\n& NLU)", ha="center", fontsize=10,
            fontweight="bold", color="#2980B9",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#EBF5FB", edgecolor="#2980B9", lw=1.5))
    ax.text(1.2, 2.5, "kmmlu\n(Knowledge\nRecall)", ha="center", fontsize=10,
            fontweight="bold", color="#8E44AD",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5EEF8", edgecolor="#8E44AD", lw=1.5))

    # 셀 1: KoBEST × Morpheme Diversity → BEST
    cell1 = FancyBboxPatch((2.0, 4.3), 2.5, 1.7, boxstyle="round,pad=0.12",
                            facecolor="#EAFAF1", edgecolor=C_CV3, lw=2.5)
    ax.add_patch(cell1)
    ax.text(3.25, 5.4, "★  BEST", ha="center", fontsize=12, fontweight="bold", color=C_CV3)
    ax.text(3.25, 5.0, "SOLAR: 0.6356\nEEVE: 0.7551 / EXAONE: 0.7415",
            ha="center", fontsize=8, color="#2C3E50")
    ax.text(3.25, 4.55, "Diverse morphemes → rich H\n→ reasoning preserved",
            ha="center", fontsize=7.5, color="#7F8C8D", style="italic")

    # 셀 2: KoBEST × Domain Diversity → OK
    cell2 = FancyBboxPatch((6.6, 4.3), 2.5, 1.7, boxstyle="round,pad=0.12",
                            facecolor="#FEF9E7", edgecolor=C_B, lw=1.5)
    ax.add_patch(cell2)
    ax.text(7.85, 5.4, "OK", ha="center", fontsize=12, fontweight="bold", color=C_B)
    ax.text(7.85, 5.0, "SOLAR: 0.6176\nEEVE: 0.7498 / EXAONE: 0.7244",
            ha="center", fontsize=8, color="#2C3E50")
    ax.text(7.85, 4.55, "Random Korean → language OK\nbut less structured",
            ha="center", fontsize=7.5, color="#7F8C8D", style="italic")

    # 셀 3: kmmlu × Morpheme Diversity → OK
    cell3 = FancyBboxPatch((2.0, 1.5), 2.5, 1.7, boxstyle="round,pad=0.12",
                            facecolor="#EAFAF1", edgecolor=C_CV3, lw=1.5)
    ax.add_patch(cell3)
    ax.text(3.25, 2.6, "OK", ha="center", fontsize=12, fontweight="bold", color=C_CV3)
    ax.text(3.25, 2.2, "SOLAR: 0.3750\nEEVE: 0.4051 / EXAONE: 0.4298",
            ha="center", fontsize=8, color="#2C3E50")
    ax.text(3.25, 1.72, "Morpheme ≠ domain diversity\nknowledge recall limited",
            ha="center", fontsize=7.5, color="#7F8C8D", style="italic")

    # 셀 4: kmmlu × Domain Diversity → BEST for kmmlu
    cell4 = FancyBboxPatch((6.6, 1.5), 2.5, 1.7, boxstyle="round,pad=0.12",
                            facecolor="#FEF9E7", edgecolor="#E67E22", lw=2.5)
    ax.add_patch(cell4)
    ax.text(7.85, 2.6, "★  BEST for kmmlu", ha="center", fontsize=11,
            fontweight="bold", color="#E67E22")
    ax.text(7.85, 2.2, "SOLAR: 0.3731\nEEVE: 0.4126 / EXAONE: 0.4345(A)",
            ha="center", fontsize=8, color="#2C3E50")
    ax.text(7.85, 1.72, "Domain variety → knowledge\nrecall channels activated",
            ha="center", fontsize=7.5, color="#7F8C8D", style="italic")

    # 구분선
    ax.plot([2.0, 9.1], [4.05, 4.05], color="#D5D8DC", lw=1.5, linestyle="--")
    ax.plot([5.9, 5.9], [1.35, 6.15], color="#D5D8DC", lw=1.5, linestyle="--")

    # 하단 인사이트
    ax.text(5.5, 0.7, "→  Morpheme diversity ≠ Domain diversity\n"
                      "    Different calibration types optimize different cognitive abilities",
            ha="center", va="center", fontsize=9.5, color=C_FP16,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#EBF5FB", edgecolor=C_FP16, lw=1.2))

    save(fig, "fig10_kobest_kmmlu_matrix.png")


# ════════════════════════════════════════════════════════════════════════
# Fig 11 — 결론 요약 (마지막 슬라이드)
# ════════════════════════════════════════════════════════════════════════
def fig11_conclusion():
    fig, ax = plt.subplots(figsize=(14, 6), facecolor=BG)
    ax.set_xlim(0, 14); ax.set_ylim(0, 6)
    ax.axis("off"); ax.set_facecolor(BG)
    ax.set_title("MAC — Summary of Contributions", fontsize=15,
                 fontweight="bold", color=C_FP16, pad=10)

    contributions = [
        {
            "icon": "①",
            "title": "Morpheme-Aware\nCalibration Algorithm",
            "body": "Greedy (lemma, POS) coverage selection\nLanguage-agnostic — works for KO / ZH / EN\nN=128, plug-in replacement for Wikitext-2",
            "result": "SOLAR: 97.4% vs 91.7%\n(+5.7pp over standard GPTQ)",
            "color": C_CV3,
            "light": "#EAFAF1",
            "x": 1.0,
        },
        {
            "icon": "②",
            "title": "Causal Mechanism\nAnalysis",
            "body": "Activation Sensitivity Score (L23-L37)\nModel Activation Distortion: -86.1%\nKorean PPL distortion: -68%",
            "result": "3 independent metrics\nall support C_v3",
            "color": "#2980B9",
            "light": "#EBF5FB",
            "x": 5.3,
        },
        {
            "icon": "③",
            "title": "Cross-Language\nGeneralization",
            "body": "5 models × 2 benchmarks\nPretraining lang = calibration lang principle\nKO / ZH / EN all validated",
            "result": "All 5 models:\nMAC > Standard GPTQ",
            "color": "#8E44AD",
            "light": "#F5EEF8",
            "x": 9.6,
        },
    ]

    for c in contributions:
        w = 3.8
        # 메인 박스
        rect = FancyBboxPatch((c["x"], 0.7), w, 4.6, boxstyle="round,pad=0.15",
                               facecolor=c["light"], edgecolor=c["color"], lw=2)
        ax.add_patch(rect)

        # 아이콘
        ax.text(c["x"] + w/2, 4.95, c["icon"], ha="center", va="center",
                fontsize=22, fontweight="bold", color=c["color"])

        # 제목
        ax.text(c["x"] + w/2, 4.2, c["title"], ha="center", va="center",
                fontsize=10.5, fontweight="bold", color=c["color"])

        # 구분선
        ax.plot([c["x"]+0.2, c["x"]+w-0.2], [3.85, 3.85],
                color=c["color"], lw=1, alpha=0.4)

        # 본문
        ax.text(c["x"] + w/2, 2.85, c["body"], ha="center", va="center",
                fontsize=8.5, color="#2C3E50", linespacing=1.5)

        # 결과 배지
        res_rect = FancyBboxPatch((c["x"]+0.2, 0.85), w-0.4, 1.3,
                                   boxstyle="round,pad=0.1",
                                   facecolor=c["color"], edgecolor="white", lw=0)
        ax.add_patch(res_rect)
        ax.text(c["x"] + w/2, 1.5, c["result"], ha="center", va="center",
                fontsize=9, fontweight="bold", color="white", linespacing=1.4)

    save(fig, "fig11_conclusion_summary.png")


if __name__ == "__main__":
    import matplotlib
    print("Generating additional presentation figures...")
    fig7_problem_concept();  print("  [7] Problem concept diagram")
    fig8_ppl_bar();          print("  [8] PPL bar chart")
    fig9_greedy_steps();     print("  [9] Greedy algorithm steps")
    fig10_task_matrix();     print("  [10] KoBEST vs kmmlu matrix")
    fig11_conclusion();      print("  [11] Conclusion summary")

    print(f"\nAll figures in: {OUT}")
    for f in sorted(os.listdir(OUT)):
        kb = os.path.getsize(f"{OUT}/{f}") // 1024
        print(f"  {f}  ({kb} KB)")
