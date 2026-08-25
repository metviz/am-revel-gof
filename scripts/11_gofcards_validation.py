"""
Script 11 — GoFCards External Validation Figure
================================================
Generates Figure7_GoFCards_Validation.png

Panels:
  A  Scatter AM vs REVEL for 27 GoFCards GOF variants (ABCC8/KCNJ11)
     overlaid on Hopkins GOF variants — shows independent replication
  B  Detection rates (% ≥ Supporting / Moderate / Strong) for AM vs REVEL
     in GoFCards high-confidence GOF (Pscore≥3) — bar chart
  C  AM score distribution (violin) for GoFCards GOF vs Hopkins GOF/LoF
     with threshold lines — shows same blind-spot pattern
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
from scipy import stats

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

GOFCARDS_CSV = os.path.join(DATA_DIR, "gofcards", "gofcards_validation_final.csv")
HOPKINS_CSV = os.path.join(DATA_DIR, "am_revel_scores_combined.csv")
FIG_PATH = os.path.join(FIGURES_DIR, "Figure7_GoFCards_Validation.png")

# ── ACMG thresholds ──────────────────────────────────────────────────────────
THRESH = {
    "AM": {"Supporting": 0.564, "Moderate": 0.740, "Strong": 0.890},
    "REVEL": {"Supporting": 0.644, "Moderate": 0.773, "Strong": 0.932},
}

# ── Colour palette ───────────────────────────────────────────────────────────
C = {
    "LoF": "#2166AC",
    "GoF": "#D6604D",
    "GoF_GC": "#8B2500",  # GoFCards GOF (darker orange-red)
    "ABCC8": "#4DAF4A",
    "KCNJ11": "#FF7F00",
    "thresh": "#888888",
}

# ═══════════════════════════════════════════════════════════════════════════════
# Load data
# ═══════════════════════════════════════════════════════════════════════════════
gof = pd.read_csv(GOFCARDS_CSV)
hop = pd.read_csv(HOPKINS_CSV)

hop_gof = hop[hop["mechanism"] == "GoF"].copy()
hop_lof = hop[hop["mechanism"] == "LoF"].copy()
gof_hc = gof[gof["pscore"] >= 3].copy()  # high-confidence subset

print(f"GoFCards GOF: {len(gof)} total, {len(gof_hc)} high-confidence (Pscore≥3)")
print(f"Hopkins GoF: {len(hop_gof)}, LoF: {len(hop_lof)}")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure layout
# ═══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 5.5))
gs = fig.add_gridspec(1, 3, wspace=0.38, left=0.06, right=0.97, top=0.91, bottom=0.14)
ax_a = fig.add_subplot(gs[0])
ax_b = fig.add_subplot(gs[1])
ax_c = fig.add_subplot(gs[2])

# ═══════════════════════════════════════════════════════════════════════════════
# Panel A — Scatter: Hopkins GoF/LoF + GoFCards GoF
# ═══════════════════════════════════════════════════════════════════════════════
ax = ax_a

# Background shading
ax.axhspan(THRESH["REVEL"]["Strong"], 1.05, alpha=0.06, color=C["GoF"], zorder=0)
ax.axvspan(THRESH["AM"]["Strong"], 1.05, alpha=0.06, color=C["GoF"], zorder=0)

# Threshold lines
for tool, axis in [("AM", "x"), ("REVEL", "y")]:
    for level, lw, ls in [
        ("Supporting", 0.8, "--"),
        ("Moderate", 0.8, ":"),
        ("Strong", 1.2, "-"),
    ]:
        val = THRESH[tool][level]
        fn = ax.axvline if axis == "x" else ax.axhline
        fn(val, color=C["thresh"], lw=lw, ls=ls, alpha=0.65, zorder=1)

# Diagonal
ax.plot([0, 1], [0, 1], color="#aaaaaa", lw=0.8, ls="--", zorder=1)

# Hopkins LoF (background)
ax.scatter(
    hop_lof["am_score"],
    hop_lof["revel_score"],
    c=C["LoF"],
    s=22,
    alpha=0.35,
    zorder=3,
    marker="o",
    edgecolors="none",
    label="Hopkins LoF",
)

# Hopkins GoF
ax.scatter(
    hop_gof["am_score"],
    hop_gof["revel_score"],
    c=C["GoF"],
    s=22,
    alpha=0.35,
    zorder=3,
    marker="s",
    edgecolors="none",
    label="Hopkins GoF",
)

# GoFCards — ABCC8 and KCNJ11 with different markers
for gene, marker, ec in [("ABCC8", "D", "#2d0000"), ("KCNJ11", "*", "#2d0000")]:
    sub = gof[gof["gene"] == gene]
    ms = 70 if marker == "*" else 48
    ax.scatter(
        sub["am_score"],
        sub["revel_score"],
        c=C["GoF_GC"],
        s=ms,
        alpha=0.9,
        zorder=6,
        marker=marker,
        edgecolors=ec,
        linewidths=0.6,
        label=f"GoFCards {gene} GoF",
    )

# Label the two high-confidence KCNJ11 blind-spot variants
for _, row in gof[
    (gof["gene"] == "KCNJ11")
    & (gof["am_score"] < THRESH["AM"]["Supporting"])
    & (gof["revel_score"] >= THRESH["REVEL"]["Supporting"])
].iterrows():
    prot = row["hgvs_p"].replace("p.", "")
    ax.annotate(
        prot,
        (row["am_score"], row["revel_score"]),
        xytext=(-28, 6),
        textcoords="offset points",
        fontsize=7,
        color="#8B2500",
        arrowprops=dict(arrowstyle="->", color="#8B2500", lw=0.7),
    )

ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.set_xlabel("AlphaMissense score", fontsize=9)
ax.set_ylabel("REVEL score", fontsize=9)
ax.set_title(
    "A   GoFCards validation on Hopkins background",
    fontsize=9.5,
    loc="left",
    fontweight="bold",
)
ax.tick_params(labelsize=8)
ax.legend(fontsize=7, loc="upper left", framealpha=0.85, markerscale=1.2)

# Quadrant blind-spot annotation
ax.text(
    0.18,
    0.96,
    f"AM blind spot\n5/27 GoFCards GoF\n(REVEL≥Supp, AM<Supp)",
    ha="center",
    va="top",
    fontsize=7,
    color=C["GoF_GC"],
    transform=ax.transAxes,
    bbox=dict(boxstyle="round,pad=0.3", fc="#fff5f0", ec="#cc4400", alpha=0.9),
)

# ═══════════════════════════════════════════════════════════════════════════════
# Panel B — Detection rates: GoFCards high-confidence GoF (Pscore≥3)
# ═══════════════════════════════════════════════════════════════════════════════
ax = ax_b

levels = ["Supporting", "Moderate", "Strong"]
x = np.arange(len(levels))
wd = 0.32
n_hc = len(gof_hc)
n_all = len(gof)

am_rates_hc = [100 * (gof_hc["am_score"] >= THRESH["AM"][l]).mean() for l in levels]
rv_rates_hc = [
    100 * (gof_hc["revel_score"] >= THRESH["REVEL"][l]).mean() for l in levels
]
am_rates_all = [100 * (gof["am_score"] >= THRESH["AM"][l]).mean() for l in levels]
rv_rates_all = [100 * (gof["revel_score"] >= THRESH["REVEL"][l]).mean() for l in levels]

# High-confidence bars (solid)
ax.bar(x - wd / 2, am_rates_hc, wd, color="#4878CF", label="AM (Pscore≥3)", zorder=3)
ax.bar(x + wd / 2, rv_rates_hc, wd, color="#D65F5F", label="REVEL (Pscore≥3)", zorder=3)

# All variants (outline)
ax.bar(
    x - wd / 2,
    am_rates_all,
    wd,
    color="none",
    edgecolor="#4878CF",
    lw=1.5,
    linestyle="--",
    label="AM (all)",
    zorder=4,
)
ax.bar(
    x + wd / 2,
    rv_rates_all,
    wd,
    color="none",
    edgecolor="#D65F5F",
    lw=1.5,
    linestyle="--",
    label="REVEL (all)",
    zorder=4,
)

# Value labels
for xi, (av, rv) in enumerate(zip(am_rates_hc, rv_rates_hc)):
    ax.text(
        xi - wd / 2,
        av + 1.5,
        f"{av:.0f}%",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#4878CF",
        fontweight="bold",
    )
    ax.text(
        xi + wd / 2,
        rv + 1.5,
        f"{rv:.0f}%",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#D65F5F",
        fontweight="bold",
    )

# Gap arrows for Supporting
gap = rv_rates_hc[0] - am_rates_hc[0]
ax.annotate(
    "",
    xy=(0 + wd / 2, rv_rates_hc[0] + 12),
    xytext=(0 - wd / 2, am_rates_hc[0] + 12),
    arrowprops=dict(arrowstyle="<->", color="#555", lw=1.2),
)
ax.text(
    0,
    rv_rates_hc[0] + 15,
    f"Δ{gap:.0f} pp",
    ha="center",
    va="bottom",
    fontsize=7.5,
    color="#555",
)

ax.set_xticks(x)
ax.set_xticklabels(levels, fontsize=8.5)
ax.set_ylabel("GoF variants detected (%)", fontsize=9)
ax.set_ylim(0, 105)
ax.set_title(
    f"B   Detection rates — GoFCards GOF\n(solid: Pscore≥3, n={n_hc}; dashed: all, n={n_all})",
    fontsize=9.5,
    loc="left",
    fontweight="bold",
)
ax.tick_params(labelsize=8)
ax.legend(fontsize=7, ncol=2, loc="upper right", framealpha=0.85)
ax.yaxis.grid(True, alpha=0.3, zorder=0)
ax.set_axisbelow(True)

# ═══════════════════════════════════════════════════════════════════════════════
# Panel C — AM score violin: GoFCards GoF vs Hopkins GoF vs Hopkins LoF
# ═══════════════════════════════════════════════════════════════════════════════
ax = ax_c

groups = [
    ("Hopkins\nLoF", hop_lof["am_score"].dropna().values, C["LoF"]),
    ("Hopkins\nGoF", hop_gof["am_score"].dropna().values, C["GoF"]),
    ("GoFCards\nGoF", gof["am_score"].dropna().values, C["GoF_GC"]),
]

for i, (label, data, color) in enumerate(groups):
    parts = ax.violinplot(
        [data], positions=[i], showmedians=True, showextrema=True, widths=0.7
    )
    parts["bodies"][0].set_facecolor(color)
    parts["bodies"][0].set_alpha(0.55)
    parts["bodies"][0].set_edgecolor(color)
    for pc in ["cmedians", "cmins", "cmaxes", "cbars"]:
        if pc in parts:
            parts[pc].set_color(color)
            parts[pc].set_linewidth(1.2)

    # Jitter points
    jitter = np.random.RandomState(42).uniform(-0.12, 0.12, len(data))
    ax.scatter(i + jitter, data, s=12, c=color, alpha=0.55, zorder=5, edgecolors="none")

    # % ≥ Strong annotation inside plot
    pct_strong = 100 * (data >= THRESH["AM"]["Strong"]).mean()
    ax.text(
        i,
        0.12,
        f"{pct_strong:.0f}%\n≥Strong",
        ha="center",
        va="bottom",
        fontsize=8,
        color=color,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
    )

# AM Strong threshold line
ax.axhline(THRESH["AM"]["Strong"], color="#888", lw=1.2, ls="-", alpha=0.7, zorder=2)
ax.text(
    2.48,
    THRESH["AM"]["Strong"] + 0.01,
    "Strong\n(0.890)",
    va="bottom",
    ha="right",
    fontsize=7,
    color="#888",
)

ax.axhline(
    THRESH["AM"]["Supporting"], color="#888", lw=0.8, ls="--", alpha=0.55, zorder=2
)
ax.text(
    2.48,
    THRESH["AM"]["Supporting"] + 0.01,
    "Supp.\n(0.564)",
    va="bottom",
    ha="right",
    fontsize=7,
    color="#888",
)

ax.set_xticks([0, 1, 2])
ax.set_xticklabels([g[0] for g in groups], fontsize=8.5)
ax.set_ylabel("AlphaMissense score", fontsize=9)
ax.set_ylim(-0.02, 1.08)
ax.set_xlim(-0.55, 2.7)
ax.set_title(
    "C   AM score distribution — GoFCards replicates\n    Hopkins GoF blind spot",
    fontsize=9.5,
    loc="left",
    fontweight="bold",
)
ax.tick_params(labelsize=8)
ax.yaxis.grid(True, alpha=0.25, zorder=0)
ax.set_axisbelow(True)

# ═══════════════════════════════════════════════════════════════════════════════
# Footer stats
# ═══════════════════════════════════════════════════════════════════════════════
gof_am_mean = gof["am_score"].mean()
gof_rv_mean = gof["revel_score"].mean()
hop_gof_am = hop_gof["am_score"].mean()
hop_gof_rv = hop_gof["revel_score"].mean()

fig.text(
    0.5,
    0.01,
    (
        f"GoFCards GOF (n=27): AM mean={gof_am_mean:.3f}, REVEL mean={gof_rv_mean:.3f}   |   "
        f"Hopkins GoF (n={len(hop_gof)}): AM mean={hop_gof_am:.3f}, REVEL mean={hop_gof_rv:.3f}   |   "
        f"All GoFCards variants independently curated (PMID: GoFCards 2025)"
    ),
    ha="center",
    va="bottom",
    fontsize=7,
    color="#555",
)

fig.suptitle(
    "Independent Validation: GoFCards Curated GOF Variants (ABCC8/KCNJ11)",
    fontsize=11,
    fontweight="bold",
    y=0.99,
)

plt.savefig(FIG_PATH, dpi=200, bbox_inches="tight")
plt.savefig(FIG_PATH.replace(".png", ".pdf"), bbox_inches="tight")
plt.close()
print(f"Saved: {FIG_PATH}")

# ─── Summary stats ────────────────────────────────────────────────────────────
print("\n=== GoFCards Validation Summary ===")
print(
    f"Total GoF variants: {len(gof)} (ABCC8={len(gof[gof['gene'] == 'ABCC8'])}, KCNJ11={len(gof[gof['gene'] == 'KCNJ11'])})"
)
print(f"High-confidence (Pscore≥3): {len(gof_hc)}")
print(
    f"\nAM blind spot (REVEL≥Supp, AM<Supp): {len(gof[(gof['revel_score'] >= 0.644) & (gof['am_score'] < 0.564)])}/27"
)
print(f"AM benign on confirmed GoF: {len(gof[gof['am_score'] < 0.564])}/27")
print(f"\nHigh-confidence detection gap at Supporting:")
print(f"  AM:    {100 * (gof_hc['am_score'] >= 0.564).mean():.1f}%")
print(f"  REVEL: {100 * (gof_hc['revel_score'] >= 0.644).mean():.1f}%")
print(
    f"  Gap:   {100 * ((gof_hc['revel_score'] >= 0.644).mean() - (gof_hc['am_score'] >= 0.564).mean()):.1f} pp"
)
