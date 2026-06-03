"""
A vs C_v3 Activation 분포 비교 시각화
목적: C_v3가 더 넓고 고르게 퍼진 activation 분포를 만든다는 것을 직접 보여주기
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

OUT = "/home/choihyun/workspace/results/presentation_assets"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "NanumGothic",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

C_A   = "#E74C3C"
C_CV3 = "#27AE60"
C_FP16 = "#2C3E50"
BG = "white"

# ── 데이터 로드 ─────────────────────────────────────────────────────────
with open("/home/choihyun/workspace/results/activation_analysis.json") as f:
    act = json.load(f)
with open("/home/choihyun/workspace/results/sensitivity_score.json") as f:
    sens = json.load(f)

n_layers = max(int(k) for k in act["A"]["per_layer"]) + 1

# 레이어별 channel_cv (FP16 모델에 각 calibration 통과시켰을 때)
cv_A   = [act["A"]["per_layer"][str(i)]["channel_cv"] for i in range(n_layers)]
cv_Cv3 = [act["C_v3"]["per_layer"][str(i)]["channel_cv"] for i in range(n_layers)]
std_A   = [act["A"]["per_layer"][str(i)]["std"] for i in range(n_layers)]
std_Cv3 = [act["C_v3"]["per_layer"][str(i)]["std"] for i in range(n_layers)]

high_layers = set(sens["summary"]["high_sensitivity_layers"])
xs = np.arange(n_layers)

# ════════════════════════════════════════════════════════════════════════
# Fig A: 메인 비교 — Channel CV per layer (A vs C_v3)
# ════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(15, 10), facecolor=BG)
gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

# ── 패널 1: Channel CV 꺾은선 ────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
ax1.set_facecolor(BG)

# 고감도 구간 배경
for li in high_layers:
    ax1.axvspan(li - 0.5, li + 0.5, alpha=0.1, color=C_CV3, zorder=1)

ax1.fill_between(xs, cv_A,   alpha=0.15, color=C_A,   zorder=2)
ax1.fill_between(xs, cv_Cv3, alpha=0.15, color=C_CV3, zorder=2)
ax1.plot(xs, cv_A,   color=C_A,   linewidth=2.2, label="A  — Standard GPTQ (English Random)", zorder=3)
ax1.plot(xs, cv_Cv3, color=C_CV3, linewidth=2.5, label="C_v3  — MAC (Korean Morpheme Diversity)", zorder=4)

# 차이 영역 음영
ax1.fill_between(xs, cv_A, cv_Cv3,
                 where=[c > a for a, c in zip(cv_A, cv_Cv3)],
                 alpha=0.2, color=C_CV3, label="C_v3 > A  (wider distribution)")

# 대표 레이어 annotation (L35 — top sensitivity)
peak = 35
ax1.annotate(
    f"L{peak}\nA={cv_A[peak]:.1f}\nC_v3={cv_Cv3[peak]:.1f}\n(+{(cv_Cv3[peak]-cv_A[peak])/cv_A[peak]*100:.0f}%)",
    xy=(peak, cv_Cv3[peak]),
    xytext=(peak + 3, cv_Cv3[peak] + 1.5),
    fontsize=8.5, color=C_CV3,
    arrowprops=dict(arrowstyle="->", color=C_CV3, lw=1.3),
    bbox=dict(boxstyle="round,pad=0.25", facecolor="#EAFAF1", edgecolor=C_CV3, lw=1)
)

# 고감도 구간 브라켓
hl = sorted(high_layers)
ax1.annotate("", xy=(max(hl)+0.5, -1.2), xytext=(min(hl)-0.5, -1.2),
             arrowprops=dict(arrowstyle="<->", color="#E67E22", lw=2),
             annotation_clip=False)
ax1.text((min(hl)+max(hl))/2, -2.5,
         f"High-sensitivity zone  L{min(hl)}-L{max(hl)}",
         ha="center", fontsize=9, color="#E67E22", fontweight="bold",
         transform=ax1.transData, clip_on=False)

ax1.set_xlim(-1, n_layers)
ax1.set_ylim(-0.5, max(cv_Cv3)*1.15)
ax1.set_xlabel("Transformer Layer Index", fontsize=11)
ax1.set_ylabel("Channel CV\n(채널별 activation 변동계수)", fontsize=11)
ax1.set_title("Activation Distribution Width per Layer  —  A vs C_v3 Calibration (FP16 Model)",
              fontsize=13, fontweight="bold", color=C_FP16)
ax1.legend(fontsize=10, loc="upper right")
ax1.grid(axis="y", alpha=0.25, linestyle="--")
ax1.set_xticks([0, 8, 16, 24, 32, 40, 47])

# ── 패널 2: 존별 평균 channel_cv 막대 ───────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_facecolor(BG)

zones = {
    "L0–L22\n(early)":   list(range(0, 23)),
    "L23–L37\n(mid-high)": list(range(23, 38)),
    "L38–L47\n(late)":   list(range(38, 48)),
}
zone_names = list(zones.keys())
avg_A   = [np.mean([cv_A[i]   for i in idx]) for idx in zones.values()]
avg_Cv3 = [np.mean([cv_Cv3[i] for i in idx]) for idx in zones.values()]

x = np.arange(len(zone_names))
bw = 0.35
bars_a   = ax2.bar(x - bw/2, avg_A,   width=bw, color=C_A,   label="A",    edgecolor="white", lw=1.2)
bars_cv3 = ax2.bar(x + bw/2, avg_Cv3, width=bw, color=C_CV3, label="C_v3", edgecolor="white", lw=1.2)

for ba, bc, za, zc in zip(bars_a, bars_cv3, avg_A, avg_Cv3):
    ax2.text(ba.get_x()+ba.get_width()/2, za+0.15, f"{za:.1f}",
             ha="center", va="bottom", fontsize=8.5, color=C_A, fontweight="bold")
    ax2.text(bc.get_x()+bc.get_width()/2, zc+0.15, f"{zc:.1f}",
             ha="center", va="bottom", fontsize=8.5, color=C_CV3, fontweight="bold")

ax2.set_xticks(x); ax2.set_xticklabels(zone_names, fontsize=9.5)
ax2.set_ylabel("Mean Channel CV", fontsize=10)
ax2.set_title("Zone-wise Average\nActivation Width", fontsize=11, fontweight="bold")
ax2.legend(fontsize=9)
ax2.grid(axis="y", alpha=0.25, linestyle="--")
ax2.set_axisbelow(True)

# 핵심 구간 강조 박스
ax2.patches[3].set_edgecolor("#E67E22"); ax2.patches[3].set_linewidth(2.5)
ax2.patches[4].set_edgecolor("#E67E22"); ax2.patches[4].set_linewidth(2.5)

# ── 패널 3: 비율(C_v3/A) 꺾은선 ─────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_facecolor(BG)

ratio = [c/a if a > 0 else 1 for a, c in zip(cv_A, cv_Cv3)]
ax3.fill_between(xs, 1, ratio,
                 where=[r > 1 for r in ratio],
                 alpha=0.25, color=C_CV3)
ax3.plot(xs, ratio, color=C_CV3, linewidth=2, label="Channel CV ratio (C_v3 / A)")
ax3.axhline(1.0, color=C_A, linestyle="--", linewidth=1.8,
            label="Ratio = 1.0  (no difference)")

for li in high_layers:
    ax3.axvspan(li - 0.5, li + 0.5, alpha=0.1, color="#E67E22")

# 평균 ratio annotation
mean_ratio_mid = np.mean([ratio[i] for i in range(23, 38)])
ax3.text(30, mean_ratio_mid + 0.05, f"avg ×{mean_ratio_mid:.2f}\nin L23–L37",
         ha="center", fontsize=9, color=C_CV3, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.25", facecolor="#EAFAF1", edgecolor=C_CV3, lw=1))

ax3.set_xlim(-1, n_layers)
ax3.set_ylim(0.7, max(ratio)*1.12)
ax3.set_xlabel("Transformer Layer Index", fontsize=10)
ax3.set_ylabel("Channel CV ratio  (C_v3 / A)", fontsize=10)
ax3.set_title("C_v3 vs A  Distribution Width Ratio\n(>1 = C_v3 is wider)", fontsize=11, fontweight="bold")
ax3.legend(fontsize=9, loc="upper left")
ax3.grid(alpha=0.25, linestyle="--")
ax3.set_xticks([0, 16, 24, 32, 40, 47])

fig.suptitle("C_v3 Calibration Produces Wider, More Diverse Activation Distributions\n"
             "→ GPTQ Hessian Better Estimates Korean Weight Importance",
             fontsize=13.5, fontweight="bold", color=C_FP16, y=1.01)

path = f"{OUT}/fig4_activation_distribution_comparison.png"
fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"saved: {path}")
print(f"size: {os.path.getsize(path)//1024} KB")
