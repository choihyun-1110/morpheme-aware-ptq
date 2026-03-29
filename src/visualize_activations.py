"""
Activation 분포 시각화 대시보드
- 조건: B / C / C_v3 / C_v4 (SOLAR-10.7B calibration 비교)
- 지표: channel_cv, std, outlier_ratio, entropy (레이어별)
- 출력: results/activation_dashboard.png, results/activation_delta.png
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.font_manager as _fm
import seaborn as sns

# ─── 한국어 폰트 등록 ────────────────────────────────────────────────────────
_NANUM_FONTS = [
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
    "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
]
for _f in _NANUM_FONTS:
    if os.path.exists(_f):
        _fm.fontManager.addfont(_f)
# NanumGothic 우선, 없으면 다른 Nanum 계열
_kr_font = next(
    (f.name for f in _fm.fontManager.ttflist if f.name == "NanumGothic"),
    next((f.name for f in _fm.fontManager.ttflist if "Nanum" in f.name), None)
)

def _apply_korean_font():
    """seaborn set_style 후 한국어 폰트 재적용"""
    if _kr_font:
        plt.rcParams["font.family"] = _kr_font
    plt.rcParams["axes.unicode_minus"] = False

_apply_korean_font()

# ─── 설정 ─────────────────────────────────────────────────────────────────────
DATA_PATH   = "/home/choihyun/workspace/results/activation_analysis.json"
OUT_DIR     = "/home/choihyun/workspace/results"

COND_COLORS = {
    "B":    "#4e79a7",   # 파랑 — 한국어 랜덤
    "C":    "#f28e2b",   # 주황 — 원본 C
    "C_v3": "#e15759",   # 빨강 — 형태소 다양성 (핵심)
    "C_v4": "#76b7b2",   # 청록 — C_v4
}
COND_LABELS = {
    "B":    "B (한국어 랜덤)",
    "C":    "C (원본)",
    "C_v3": "C_v3 (형태소 다양성) ★",
    "C_v4": "C_v4 (어절 확장)",
}

METRIC_META = {
    "channel_cv":    ("Channel CV (채널간 변동계수)", "높을수록 채널 불균형 → GPTQ에 도전적"),
    "std":           ("Activation Std (표준편차)",    "높을수록 outlier 위험 증가"),
    "outlier_ratio": ("Outlier Ratio (|x|>3σ 비율)", "높을수록 quantization 오차 큼"),
    "entropy":       ("Entropy (활성 다양성)",        "높을수록 다양한 activation 패턴"),
}

# 레이어 구역 표시용
LAYER_ZONES = [
    (0,  3,  "#eaf4fb", "Embed"),
    (4,  11, "#fef9e7", "Early"),
    (12, 18, "#fdedec", "Core-A"),
    (19, 29, "#eafaf1", "Mid"),
    (30, 36, "#fdf2f8", "Core-B"),
    (37, 44, "#fdfefe", "Late"),
    (45, 47, "#f0f3f4", "Output"),
]


def load_data(path: str):
    with open(path) as f:
        raw = json.load(f)
    conds = list(raw.keys())
    n_layers = len(raw[conds[0]]["per_layer"])
    layers = list(range(n_layers))

    # {metric: {cond: [layer0, layer1, ...]}}
    data = {m: {c: [] for c in conds} for m in METRIC_META}
    for cond in conds:
        for layer_idx in layers:
            layer_data = raw[cond]["per_layer"][str(layer_idx)]
            for metric in METRIC_META:
                data[metric][cond].append(layer_data[metric])

    summaries = {c: raw[c]["summary"] for c in conds}
    return conds, layers, data, summaries


def add_zone_bg(ax, zones, alpha=0.08):
    for (lo, hi, color, label) in zones:
        ax.axvspan(lo - 0.5, hi + 0.5, color=color, alpha=alpha, zorder=0)
    # 구역 레이블을 axes 좌표계 (ylim 무관)로 표시
    for (lo, hi, color, label) in zones:
        mid = (lo + hi) / 2
        ax.text(mid, 0, label, ha="center", va="bottom",
                fontsize=5.5, color="#888888", fontweight="bold",
                transform=ax.get_xaxis_transform(), clip_on=True)


def make_dashboard(conds, layers, data, summaries):
    """메인 대시보드: 4개 지표 × 레이어별 라인 플롯"""
    sns.set_style("whitegrid")
    _apply_korean_font()
    fig = plt.figure(figsize=(18, 13))
    fig.patch.set_facecolor("#fafafa")
    fig.suptitle(
        "SOLAR-10.7B-Instruct: Calibration 조건별 Activation 분포 비교\n"
        "( B=한국어랜덤  C=원본  C_v3=형태소다양성★  C_v4=어절확장 )",
        fontsize=14, fontweight="bold", y=0.98, color="#222222"
    )

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.30,
                           top=0.92, bottom=0.09, left=0.07, right=0.97)

    metrics_list = list(METRIC_META.keys())

    axes = []
    for i, metric in enumerate(metrics_list):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        axes.append((ax, metric))

    for ax, metric in axes:
        title, desc = METRIC_META[metric]
        ys_all = []
        for cond in conds:
            y = np.array(data[metric][cond])
            lw = 2.2 if cond == "C_v3" else 1.2
            zo = 4   if cond == "C_v3" else 2
            alpha = 1.0 if cond == "C_v3" else 0.72
            ax.plot(layers, y,
                    color=COND_COLORS[cond],
                    label=COND_LABELS[cond],
                    linewidth=lw, zorder=zo, alpha=alpha)
            ys_all.extend(y.tolist())

        # y축: 전체 범위의 상하 10% 여백만 주되, 최솟값 근방 10% 이내로 줌인
        y_min, y_max = min(ys_all), max(ys_all)
        y_span = y_max - y_min if y_max != y_min else 1.0
        ax.set_ylim(y_min - y_span * 0.08, y_max + y_span * 0.12)

        # 구역 배경
        ax.set_xlim(-0.5, max(layers) + 0.5)
        add_zone_bg(ax, LAYER_ZONES)

        ax.set_title(f"{title}", fontsize=10, fontweight="bold", pad=4)
        ax.set_xlabel("Layer", fontsize=8, color="#555")
        ax.set_ylabel(metric.replace("_", " "), fontsize=8, color="#555")
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", alpha=0.4, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)

        # C_v3 최고 레이어 강조
        y_v3 = np.array(data[metric]["C_v3"])
        peak_layer = int(np.argmax(y_v3))
        ax.axvline(peak_layer, color="#e15759", linewidth=0.8, linestyle=":", alpha=0.7, zorder=3)
        ax.annotate(f"L{peak_layer}", xy=(peak_layer, y_v3[peak_layer]),
                    xytext=(peak_layer + 1.2, y_v3[peak_layer]),
                    fontsize=7, color="#e15759",
                    arrowprops=dict(arrowstyle="-", color="#e15759", lw=0.8))

    # 범례 (마지막 subplot 아래)
    handles = [mpatches.Patch(color=COND_COLORS[c], label=COND_LABELS[c]) for c in conds]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               fontsize=9, frameon=True, fancybox=True,
               bbox_to_anchor=(0.5, 0.01))

    # 5번째 서브플롯: Summary bar chart
    ax_bar = fig.add_subplot(gs[2, :])
    _make_summary_bars(ax_bar, conds, summaries)

    out = os.path.join(OUT_DIR, "activation_dashboard.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[저장] {out}")


def _make_summary_bars(ax, conds, summaries):
    """Summary 지표 그룹 막대그래프"""
    metrics_s = list(METRIC_META.keys())
    n_metrics = len(metrics_s)
    n_conds   = len(conds)
    bar_w = 0.18
    group_gap = 1.0

    positions = np.arange(n_metrics) * (n_conds * bar_w + group_gap)

    for i, cond in enumerate(conds):
        xs = positions + i * bar_w
        ys = []
        for m in metrics_s:
            key_map = {
                "channel_cv":    "mean_channel_cv",
                "std":           "mean_std",
                "outlier_ratio": "mean_outlier_ratio",
                "entropy":       "mean_entropy",
            }
            ys.append(summaries[cond][key_map[m]])

        # 정규화: 각 지표별 최대값으로 나눔 (비교용)
        ys_raw = ys
        bars = ax.bar(xs, ys_raw,
                      width=bar_w,
                      color=COND_COLORS[cond],
                      label=COND_LABELS[cond],
                      alpha=0.85,
                      edgecolor="white", linewidth=0.5)

        for bar, val in zip(bars, ys_raw):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.015,
                    f"{val:.3f}" if val < 10 else f"{val:.2f}",
                    ha="center", va="bottom", fontsize=6.5, color="#333")

    tick_centers = positions + bar_w * (n_conds - 1) / 2
    ax.set_xticks(tick_centers)
    ax.set_xticklabels([METRIC_META[m][0] for m in metrics_s], fontsize=9)
    ax.set_title("평균 Activation 지표 비교 (전체 레이어 평균)", fontsize=10, fontweight="bold")
    ax.set_ylabel("값 (원 단위)", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.35, linewidth=0.5)


def make_delta_heatmap(conds, layers, data):
    """C_v3 vs B 차이를 레이어 × 지표 히트맵으로"""
    metrics_list = list(METRIC_META.keys())
    ref_cond = "B"
    target_cond = "C_v3"

    # delta[metric][layer] = C_v3 - B
    delta_matrix = np.zeros((len(metrics_list), len(layers)))
    for mi, metric in enumerate(metrics_list):
        for li, layer in enumerate(layers):
            ref_val = data[metric][ref_cond][layer]
            tgt_val = data[metric][target_cond][layer]
            # 정규화: ref 대비 % 변화
            if ref_val != 0:
                delta_matrix[mi, li] = (tgt_val - ref_val) / abs(ref_val) * 100
            else:
                delta_matrix[mi, li] = 0.0

    _apply_korean_font()
    fig, axes = plt.subplots(1, 2, figsize=(20, 5),
                             gridspec_kw={"width_ratios": [4, 1]})
    fig.patch.set_facecolor("#fafafa")
    fig.suptitle(
        f"C_v3 vs B: Activation 지표 변화율 히트맵 (레이어 × 지표)\n"
        f"양수(빨강) = C_v3가 높음, 음수(파랑) = B가 높음",
        fontsize=12, fontweight="bold", y=1.01
    )

    # ── 히트맵 ──
    ax = axes[0]
    cmap = LinearSegmentedColormap.from_list(
        "div", ["#2166ac", "#f7f7f7", "#d6604d"], N=256
    )
    vmax = np.percentile(np.abs(delta_matrix), 95)
    im = ax.imshow(delta_matrix, aspect="auto", cmap=cmap,
                   vmin=-vmax, vmax=vmax, interpolation="nearest")

    ax.set_yticks(range(len(metrics_list)))
    ax.set_yticklabels([METRIC_META[m][0] for m in metrics_list], fontsize=9)
    ax.set_xlabel("Layer", fontsize=9)

    # 구역 수직선
    zone_bounds = []
    for (lo, hi, color, label) in LAYER_ZONES:
        if lo > 0:
            ax.axvline(lo - 0.5, color="#aaa", linewidth=0.6, linestyle="--")
        mid = (lo + hi) / 2
        ax.text(mid, -0.7, label, ha="center", va="top", fontsize=6.5,
                color="#666", transform=ax.get_xaxis_transform())

    # x축 눈금 간격
    tick_step = 4
    ax.set_xticks(range(0, len(layers), tick_step))
    ax.set_xticklabels(range(0, len(layers), tick_step), fontsize=7)

    plt.colorbar(im, ax=ax, label="변화율 (%)", shrink=0.85)
    ax.set_title("레이어별 C_v3 - B 변화율 (%)", fontsize=10, fontweight="bold")

    # ── 레이어 구역별 평균 막대 ──
    ax2 = axes[1]
    zone_names = []
    zone_means = []   # shape: (n_zones, n_metrics)
    for (lo, hi, color, label) in LAYER_ZONES:
        zone_names.append(label)
        cols = delta_matrix[:, lo:hi+1]
        zone_means.append(cols.mean(axis=1))

    zone_means = np.array(zone_means)   # (n_zones, n_metrics)

    # 각 구역별 평균 (모든 지표 합산 대신 채널 CV만 강조)
    cv_idx = metrics_list.index("channel_cv")
    cv_zone = zone_means[:, cv_idx]

    colors_zone = ["#d6604d" if v > 0 else "#2166ac" for v in cv_zone]
    bars = ax2.barh(zone_names, cv_zone, color=colors_zone, alpha=0.8, edgecolor="white")
    ax2.axvline(0, color="#333", linewidth=0.8)
    for bar, val in zip(bars, cv_zone):
        ax2.text(val + (0.3 if val >= 0 else -0.3),
                 bar.get_y() + bar.get_height() / 2,
                 f"{val:+.1f}%", va="center", fontsize=8,
                 ha="left" if val >= 0 else "right", color="#333")
    ax2.set_xlabel("Channel CV 변화율 (%)", fontsize=8)
    ax2.set_title("구역별\nChannel CV\nC_v3 우위", fontsize=9, fontweight="bold")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.tick_params(labelsize=8)
    ax2.grid(axis="x", alpha=0.4)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "activation_delta.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[저장] {out}")


def make_layer_detail(conds, layers, data):
    """채널 CV 레이어별 + 핵심 레이어 확대 조합"""
    _apply_korean_font()
    fig, axes = plt.subplots(2, 1, figsize=(16, 10),
                             gridspec_kw={"height_ratios": [2, 1]}, facecolor="#fafafa")
    fig.suptitle(
        "Channel CV (채널간 변동계수): GPTQ Hessian 추정 난이도 지표",
        fontsize=13, fontweight="bold", y=1.00
    )

    # ── 위: 전체 레이어 ──
    ax = axes[0]
    metric = "channel_cv"
    y_v3 = np.array(data[metric]["C_v3"])

    for cond in conds:
        y = np.array(data[metric][cond])
        lw = 2.5 if cond == "C_v3" else 1.3
        zo = 4   if cond == "C_v3" else 2
        al = 1.0 if cond == "C_v3" else 0.65
        ax.plot(layers, y, color=COND_COLORS[cond], label=COND_LABELS[cond],
                linewidth=lw, zorder=zo, alpha=al)

    # 채워진 영역: C_v3 vs B 차이
    y_b   = np.array(data[metric]["B"])
    ax.fill_between(layers, y_b, y_v3,
                    where=(y_v3 >= y_b),
                    alpha=0.15, color="#e15759", label="C_v3 > B (영역)")
    ax.fill_between(layers, y_b, y_v3,
                    where=(y_v3 < y_b),
                    alpha=0.12, color="#4e79a7", label="B > C_v3 (영역)")

    add_zone_bg(ax, LAYER_ZONES, alpha=0.07)
    ax.set_xlim(-0.5, max(layers) + 0.5)
    ax.set_title("전체 레이어: Channel CV", fontsize=10, fontweight="bold")
    ax.set_xlabel("Layer Index", fontsize=9)
    ax.set_ylabel("Channel CV", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, ncol=3, loc="upper right", framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.4, linewidth=0.5)

    # 핵심 레이어 어노테이션
    key_layers = [12, 15, 18, 31, 33]
    for kl in key_layers:
        ax.annotate(f"L{kl}\n★",
                    xy=(kl, y_v3[kl]),
                    xytext=(kl, y_v3[kl] * 1.07),
                    ha="center", fontsize=7, color="#c0392b",
                    arrowprops=dict(arrowstyle="->", color="#c0392b", lw=0.8))

    # ── 아래: Layer 10~20 확대 ──
    ax2 = axes[1]
    focus_layers = list(range(10, 22))
    for cond in conds:
        y = np.array(data[metric][cond])[10:22]
        lw = 2.5 if cond == "C_v3" else 1.3
        zo = 4   if cond == "C_v3" else 2
        al = 1.0 if cond == "C_v3" else 0.65
        ax2.plot(focus_layers, y, color=COND_COLORS[cond],
                 linewidth=lw, zorder=zo, alpha=al, marker="o", markersize=5)
        # 값 레이블
        if cond == "C_v3":
            for li, (fl, yv) in enumerate(zip(focus_layers, y)):
                ax2.text(fl, yv + 0.15, f"{yv:.1f}", ha="center", fontsize=6.5, color="#c0392b")

    y_b_focus   = np.array(data[metric]["B"])[10:22]
    y_v3_focus  = np.array(data[metric]["C_v3"])[10:22]
    ax2.fill_between(focus_layers, y_b_focus, y_v3_focus,
                     where=(y_v3_focus >= y_b_focus),
                     alpha=0.2, color="#e15759")

    ax2.set_xlim(9.5, 21.5)
    ax2.set_xticks(focus_layers)
    ax2.set_title("Layer 10-21 확대: C_v3가 가장 높은 channel_cv (GPTQ에 가장 도전적인 구간)",
                  fontsize=9, fontweight="bold")
    ax2.set_xlabel("Layer Index", fontsize=9)
    ax2.set_ylabel("Channel CV", fontsize=9)
    ax2.tick_params(labelsize=8)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="y", alpha=0.4, linewidth=0.5)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "activation_channel_cv.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[저장] {out}")


def make_radar(conds, summaries):
    """B 대비 상대 변화율(%) 레이더 + KoBEST 성능 연결 콤보 차트"""
    metrics_r = ["mean_channel_cv", "mean_std", "mean_outlier_ratio", "mean_entropy"]
    labels_r  = ["Channel CV\n(채널 불균형)", "Activation Std\n(표준편차)",
                 "Outlier Ratio\n(이상값 비율)", "Entropy\n(활성 다양성)"]
    # KoBEST 결과 (Sprint 2 실측)
    kobest = {"B": 0.6176, "C": 0.6062, "C_v3": 0.6356, "C_v4": 0.6005}
    ref = "B"

    _apply_korean_font()
    fig = plt.figure(figsize=(16, 7), facecolor="#fafafa")
    fig.suptitle(
        "Calibration 조건: Activation 지표 (B 대비 변화율) ↔ KoBEST 성능",
        fontsize=13, fontweight="bold", y=1.01
    )
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35, left=0.05, right=0.95)

    # ── 왼쪽: 레이더 (B 대비 상대 변화율%) ──
    ax_r = fig.add_subplot(gs[0], polar=True)

    N = len(metrics_r)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    # B 제외 나머지 3개만 상대 변화율로 표시
    plot_conds = [c for c in conds if c != ref]

    for cond in plot_conds:
        # B 대비 % 변화 (양수=B보다 높음, 음수=B보다 낮음)
        vals = []
        for m in metrics_r:
            ref_v = summaries[ref][m]
            tgt_v = summaries[cond][m]
            pct = (tgt_v - ref_v) / abs(ref_v) * 100 if ref_v != 0 else 0.0
            vals.append(pct)

        # 스케일: -15%~+15% 를 0~1로 매핑 (중앙=0%, 외곽=+15%)
        scaled = [(v + 15) / 30 for v in vals]
        scaled += scaled[:1]

        lw = 2.8 if cond == "C_v3" else 1.6
        al = 1.0 if cond == "C_v3" else 0.75
        ax_r.plot(angles, scaled, color=COND_COLORS[cond],
                  linewidth=lw, label=COND_LABELS[cond], alpha=al)
        ax_r.fill(angles, scaled, color=COND_COLORS[cond],
                  alpha=0.12 if cond == "C_v3" else 0.05)

    # B 기준선 (0% = 0.5)
    base = [0.5] * (N + 1)
    ax_r.plot(angles, base, color=COND_COLORS[ref], linewidth=2.0,
              linestyle="--", label=f"{COND_LABELS[ref]} (기준)", alpha=0.9)

    ax_r.set_xticks(angles[:-1])
    ax_r.set_xticklabels(labels_r, fontsize=9.5)
    # y축: 0.0=−15%, 0.5=0%, 1.0=+15%
    ax_r.set_ylim(0, 1)
    ytick_vals = [0.0, 0.25, 0.5, 0.75, 1.0]
    ax_r.set_yticks(ytick_vals)
    ax_r.set_yticklabels([f"{int((v-0.5)*30):+d}%" for v in ytick_vals], fontsize=7.5)
    ax_r.set_title("B 대비 Activation 지표 변화율(%)\n(외곽=B보다 강함, 중앙=동일)",
                   fontsize=10, fontweight="bold", pad=18)
    ax_r.legend(loc="lower left", bbox_to_anchor=(-0.15, -0.18), fontsize=8.5,
                frameon=True, fancybox=True, ncol=2)
    ax_r.grid(alpha=0.35)

    # ── 오른쪽: KoBEST 성능 막대 + 지표 산점도 ──
    ax2 = fig.add_subplot(gs[1])

    bar_x = np.arange(len(conds))
    bar_vals = [kobest[c] for c in conds]
    bar_colors = [COND_COLORS[c] for c in conds]
    bars = ax2.bar(bar_x, bar_vals, color=bar_colors, width=0.55,
                   alpha=0.85, edgecolor="white", linewidth=1.5, zorder=3)

    # 값 레이블
    for bar, val, cond in zip(bars, bar_vals, conds):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 val + 0.002,
                 f"{val:.4f}",
                 ha="center", va="bottom", fontsize=10, fontweight="bold",
                 color=COND_COLORS[cond])

    # FP16 기준선 (0.6523)
    ax2.axhline(0.6523, color="#555", linewidth=1.2, linestyle="--", zorder=2)
    ax2.text(len(conds) - 0.45, 0.6523 + 0.001, "FP16 = 0.6523",
             fontsize=8.5, color="#555", ha="right")

    # 랜덤 기준선 (0.5)
    ax2.axhline(0.5, color="#aaa", linewidth=0.8, linestyle=":", zorder=1)
    ax2.text(len(conds) - 0.45, 0.501, "랜덤 기준", fontsize=7.5, color="#aaa", ha="right")

    ax2.set_xticks(bar_x)
    ax2.set_xticklabels([COND_LABELS[c] for c in conds], fontsize=8.5, rotation=10)
    ax2.set_ylim(0.55, 0.67)
    ax2.set_ylabel("KoBEST 정확도 (agg)", fontsize=10)
    ax2.set_title("KoBEST 성능 비교\n(SOLAR-10.7B GPTQ 4bit)", fontsize=10, fontweight="bold")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="y", alpha=0.4, linewidth=0.5, zorder=0)
    ax2.tick_params(labelsize=8)

    # Channel CV 값을 점으로 오버레이 (보조 y축)
    ax2b = ax2.twinx()
    cv_vals = [summaries[c]["mean_channel_cv"] for c in conds]
    ax2b.scatter(bar_x, cv_vals, color=[COND_COLORS[c] for c in conds],
                 s=100, zorder=5, marker="D", edgecolors="white", linewidths=1.0)
    for xi, (cv, cond) in enumerate(zip(cv_vals, conds)):
        ax2b.text(xi + 0.18, cv + 0.05, f"CV={cv:.2f}",
                  fontsize=7.5, color=COND_COLORS[cond], va="bottom")
    ax2b.set_ylabel("평균 Channel CV (◆)", fontsize=8.5, color="#666")
    ax2b.tick_params(labelsize=7.5, colors="#666")
    ax2b.spines[["top"]].set_visible(False)
    ax2b.set_ylim(9, 12)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "activation_radar.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[저장] {out}")


def main():
    print("데이터 로딩 중...")
    conds, layers, data, summaries = load_data(DATA_PATH)
    print(f"  조건: {conds}, 레이어 수: {len(layers)}")

    print("\n[1/4] 메인 대시보드 (4지표 × 레이어 + Summary 막대)...")
    make_dashboard(conds, layers, data, summaries)

    print("[2/4] Delta 히트맵 (C_v3 vs B)...")
    make_delta_heatmap(conds, layers, data)

    print("[3/4] Channel CV 상세 (전체 + L10-21 확대)...")
    make_layer_detail(conds, layers, data)

    print("[4/4] Summary 레이더 차트...")
    make_radar(conds, summaries)

    print("\n완료! 생성된 파일:")
    for fname in ["activation_dashboard.png", "activation_delta.png",
                  "activation_channel_cv.png", "activation_radar.png"]:
        path = os.path.join(OUT_DIR, fname)
        size = os.path.getsize(path) // 1024
        print(f"  {path}  ({size} KB)")


if __name__ == "__main__":
    main()
