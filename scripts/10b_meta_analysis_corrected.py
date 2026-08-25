"""
10b — Corrected meta-analysis.

Two fixes vs 10_meta_analysis.py. Originals are NOT overwritten; all outputs
carry a _corrected suffix.

FIX 1 — Remove the "ProteinGym v1.0" stratum.
    Script 09's ProteinGym download failed and silently fell back to ClinVar GCK.
    The stratum labelled "ProteinGym v1.0" (155 LoF / 31 GoF) is byte-identical to
    the ClinVar GCK subset. It is not an independent dataset and is dropped.

FIX 2 — Correct the standard error of the LoF-GoF gap.
    The original computed SE from Wilson CI HALF-WIDTHS added in quadrature.
    A Wilson half-width is already ~1.96*SE, so the original SE was ~1.96x too
    large; every study CI was ~2x too wide, and tau^2 / I^2 were distorted.
    Here SE is the textbook SE of a difference in two independent proportions:
        SE = sqrt( p1(1-p1)/n1 + p2(1-p2)/n2 )  * 100
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")

DROP_DATASETS = ["ProteinGym"]  # FIX 1
C = {"AM": "#C0392B", "REVEL": "#2C7BB6"}
GENE_C = {"ABCC8": "#1a9641", "KCNJ11": "#d7191c", "GCK": "#2c7bb6", "All": "#555555"}


# ---------------------------------------------------------------- statistics
def gap_se(lof_k, lof_n, gof_k, gof_n):
    """Textbook SE of a difference of two independent proportions, in pp."""
    p1, p2 = lof_k / lof_n, gof_k / gof_n
    var = p1 * (1 - p1) / lof_n + p2 * (1 - p2) / gof_n
    return float(np.sqrt(var) * 100)


def dl_meta(y, se):
    """DerSimonian-Laird random-effects pooling."""
    y, v = np.asarray(y, float), np.asarray(se, float) ** 2
    w = 1.0 / v
    theta_fe = (w * y).sum() / w.sum()
    Q = float((w * (y - theta_fe) ** 2).sum())
    df = len(y) - 1
    c = w.sum() - (w**2).sum() / w.sum()
    tau2 = max(0.0, (Q - df) / c) if c > 0 else 0.0
    I2 = max(0.0, (Q - df) / Q * 100) if Q > 0 else 0.0
    wr = 1.0 / (v + tau2)
    theta = (wr * y).sum() / wr.sum()
    se_re = float(np.sqrt(1.0 / wr.sum()))
    p_het = float(1 - chi2.cdf(Q, df)) if df > 0 else 1.0
    return dict(
        k=len(y),
        pooled_gap=float(theta),
        se=se_re,
        ci_lo=theta - 1.96 * se_re,
        ci_hi=theta + 1.96 * se_re,
        tau2=tau2,
        Q=Q,
        I2=I2,
        p_heterogeneity=p_het,
        weights=(wr / wr.sum()),
    )


# ---------------------------------------------------------------- load & fix
eff = pd.read_csv(os.path.join(RES, "meta_analysis_effects.csv"))

mask_drop = eff["dataset"].str.contains("|".join(DROP_DATASETS), case=False, na=False)
print(f"Dropping {int(mask_drop.sum())} contaminated stratum row(s):")
for _, r in eff[mask_drop].iterrows():
    print(
        f"   - {r['dataset']} / {r['gene']} / {r['tool']} "
        f"(LoF {r['lof_n']}, GoF {r['gof_n']}, gap {r['gap']:+.1f} pp)"
    )
eff = eff[~mask_drop].copy()

# FIX 2 — recompute SE and per-study CI from the raw counts
eff["se"] = [gap_se(r.lof_k, r.lof_n, r.gof_k, r.gof_n) for r in eff.itertuples()]
eff["ci_lo"] = eff["gap"] - 1.96 * eff["se"]
eff["ci_hi"] = eff["gap"] + 1.96 * eff["se"]

eff.to_csv(os.path.join(RES, "meta_analysis_effects_corrected.csv"), index=False)

# ---------------------------------------------------------------- pool
rows = []
for tool in ["AM", "REVEL"]:
    d = eff[(eff.tool == tool) & (eff.gene != "All")]
    subgroups = {
        "All": d,
        "Clinical": d[d.data_type == "Clinical"],
        "Functional": d[d.data_type != "Clinical"],
        "ABCC8": d[d.gene == "ABCC8"],
        "KCNJ11": d[d.gene == "KCNJ11"],
        "GCK": d[d.gene == "GCK"],
    }
    for name, sub in subgroups.items():
        if len(sub) < 1:
            continue
        m = dl_meta(sub.gap.values, sub.se.values)
        m.pop("weights")
        rows.append(dict(tool=tool, subgroup=name, **m))

summary = pd.DataFrame(rows)
summary.to_csv(os.path.join(RES, "meta_analysis_summary_corrected.csv"), index=False)

print("\n" + "=" * 78)
print("CORRECTED POOLED ESTIMATES  (ProteinGym removed; proper SE)")
print("=" * 78)
for _, r in summary[summary.subgroup == "All"].iterrows():
    print(
        f"  {r.tool:<6} k={int(r.k)}  pooled {r.pooled_gap:+.1f} pp  "
        f"95% CI {r.ci_lo:+.1f} to {r.ci_hi:+.1f}  "
        f"I2={r.I2:.0f}%  tau2={r.tau2:.1f}  p_het={r.p_heterogeneity:.3f}"
    )

am = summary[(summary.tool == "AM") & (summary.subgroup == "All")].iloc[0]
rv = summary[(summary.tool == "REVEL") & (summary.subgroup == "All")].iloc[0]
print(f"\n  AM / REVEL ratio: {am.pooled_gap / rv.pooled_gap:.2f}x")
print("\nClinical-only subgroup:")
for _, r in summary[summary.subgroup == "Clinical"].iterrows():
    print(
        f"  {r.tool:<6} k={int(r.k)}  {r.pooled_gap:+.1f} pp "
        f"({r.ci_lo:+.1f} to {r.ci_hi:+.1f})"
    )

# ---------------------------------------------------------------- forest plot
# figsize aspect must match the poster's fig_block ar=1.425 (w/h) so it drops in cleanly
fig, axes = plt.subplots(1, 2, figsize=(20, 20 / 1.425), sharex=True)
for ax, tool in zip(axes, ["AM", "REVEL"]):
    d = eff[(eff.tool == tool) & (eff.gene != "All")].reset_index(drop=True)
    m = dl_meta(d.gap.values, d.se.values)
    w = m["weights"]

    ys = np.arange(len(d))[::-1]
    for i, (_, r) in enumerate(d.iterrows()):
        y = ys[i]
        ax.plot([r.ci_lo, r.ci_hi], [y, y], color="#444", lw=1.6, zorder=2)
        ax.plot([r.ci_lo, r.ci_lo], [y - 0.12, y + 0.12], color="#444", lw=1.6)
        ax.plot([r.ci_hi, r.ci_hi], [y - 0.12, y + 0.12], color="#444", lw=1.6)
        ax.scatter(
            r.gap,
            y,
            s=60 + 900 * w[i],
            marker="s",
            color=GENE_C.get(r.gene, "#555"),
            edgecolor="black",
            lw=0.8,
            zorder=3,
        )
        ax.text(-30, y, f"{r.dataset} · {r.gene}", ha="left", va="center", fontsize=10)
        ax.text(
            86,
            y,
            f"{r.gap:+.1f} ({r.ci_lo:+.1f}, {r.ci_hi:+.1f})   {w[i] * 100:.0f}%",
            ha="right",
            va="center",
            fontsize=9,
            family="monospace",
        )

    yd = -1.4
    ax.add_patch(
        plt.Polygon(
            [
                (m["pooled_gap"], yd + 0.32),
                (m["ci_hi"], yd),
                (m["pooled_gap"], yd - 0.32),
                (m["ci_lo"], yd),
            ],
            closed=True,
            facecolor=C[tool],
            edgecolor="black",
            lw=1.2,
            zorder=4,
        )
    )
    ax.text(
        -30,
        yd,
        "Pooled (random effects)",
        ha="left",
        va="center",
        fontsize=11,
        fontweight="bold",
    )
    ax.text(
        86,
        yd,
        f"{m['pooled_gap']:+.1f} ({m['ci_lo']:+.1f}, {m['ci_hi']:+.1f})",
        ha="right",
        va="center",
        fontsize=10,
        fontweight="bold",
        family="monospace",
    )

    ax.axvline(0, color="#999", lw=1.2, ls="--", zorder=1)
    ax.set_xlim(-32, 88)
    ax.set_ylim(yd - 1.0, len(d) - 0.3)
    ax.set_yticks([])
    ax.set_xlabel("LoF − GoF sensitivity gap at Strong threshold (pp)", fontsize=11)
    thr = "≥ 0.890" if tool == "AM" else "≥ 0.932"
    name = "AlphaMissense" if tool == "AM" else "REVEL"
    ax.set_title(
        f"{name} (Strong {thr})\n"
        f"k={m['k']}  pooled {m['pooled_gap']:+.1f} pp  I²={m['I2']:.0f}%",
        fontsize=13,
        fontweight="bold",
        color=C[tool],
    )
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

fig.suptitle(
    "Meta-analysis of the LoF−GoF sensitivity gap (corrected: "
    "ProteinGym stratum removed, proper SE of a difference in proportions)",
    fontsize=14,
    fontweight="bold",
    y=0.99,
)
fig.tight_layout(rect=[0, 0, 1, 0.95])
for ext in ("png", "pdf"):
    fig.savefig(
        os.path.join(FIG, f"Figure6_MetaAnalysis_ForestPlot_corrected.{ext}"),
        dpi=200,
        bbox_inches="tight",
    )
plt.close(fig)

print("\nWrote:")
print("  results/meta_analysis_effects_corrected.csv")
print("  results/meta_analysis_summary_corrected.csv")
print("  figures/Figure6_MetaAnalysis_ForestPlot_corrected.png/.pdf")
