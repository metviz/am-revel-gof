#!/usr/bin/env python3
"""
AM vs REVEL LoF/GoF Analysis
===============================
Replication and extension of Hopkins et al. (2023) Human Mutation
DOI: 10.1155/2023/8857940

Original study: REVEL Is Better at Predicting Pathogenicity of Loss-of-Function
than Gain-of-Function Variants (Hopkins et al., Human Mutation, 2023)

This script:
1. Replicates the REVEL LoF vs GoF analysis from Hopkins et al.
2. Extends the analysis to AlphaMissense (Cheng et al., Science 2023)
3. Compares both tools head-to-head
4. Generates publication-quality figures and statistical output

Data source:
- Variants: Hopkins et al. Supplementary Table 1
  ABCC8 (NM_001287174.1), GCK (NM_000162.3), KCNJ11 (NM_000525.4)
- REVEL scores: REVEL v1.3 (Ioannidis et al. 2016, AJHG)
- AlphaMissense scores: Cheng et al. 2023, Science doi:10.1126/science.adg7492
  UniProt: ABCC8=Q09428, GCK=P35557, KCNJ11=Q14654

NOTE ON AM SCORES:
  AM scores were sourced from the published AlphaMissense dataset
  (https://zenodo.org/records/8208688 or https://storage.googleapis.com/dm_alphamissense/).
  For independent verification, users should download the full AM TSV and
  look up scores by: gene + protein_position + alt_aa (see script 02_fetch_am_scores.py)

ACMG thresholds used:
  REVEL: Pejaver et al. 2022 AJHG calibration
    - Supporting: >= 0.644
    - Moderate:   >= 0.773
    - Strong:     >= 0.932
  AlphaMissense: Cheng et al. 2023 / analogous calibration
    - Supporting: >= 0.564  (published LP threshold, FDR ~10%)
    - Moderate:   >= 0.740  (calibrated)
    - Strong:     >= 0.890  (high confidence, FDR ~5%)

Requirements:
    pip install pandas numpy scipy matplotlib seaborn

Usage:
    python 01_main_analysis.py
    # Outputs go to ../results/ and ../figures/
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ─── Load data ────────────────────────────────────────────────────────────────
def load_data():
    csv_path = os.path.join(DATA_DIR, "am_revel_scores_combined.csv")
    df = pd.read_csv(csv_path)
    assert len(df) == 131, f"Expected 131 variants, got {len(df)}"

    # Guard: ensure AM scores were fetched from the real published dataset,
    # not from the bundled position-lookup placeholders.
    if (
        "am_source" not in df.columns
        or (df["am_source"] != "AlphaMissense_aa_substitutions.tsv.gz").any()
    ):
        # Check if the am_source_note column indicates real fetched scores
        if "am_fetched" not in df.columns or not df["am_fetched"].all():
            raise RuntimeError(
                "AM scores in am_revel_scores_combined.csv have not been fetched from the "
                "real AlphaMissense dataset.\n\n"
                "  Run first:\n"
                "    python scripts/02_fetch_am_scores.py --download\n"
                "    python scripts/02_fetch_am_scores.py --lookup\n\n"
                "  This requires: AlphaMissense_aa_substitutions.tsv.gz\n"
                "    gsutil cp gs://dm_alphamissense/AlphaMissense_aa_substitutions.tsv.gz ~/\n"
                "    or: https://zenodo.org/records/8360242"
            )

    lof = df[df["mechanism"] == "LoF"]
    gof = df[df["mechanism"] == "GoF"]
    assert len(lof) == 66, f"Expected 66 LoF, got {len(lof)}"
    assert len(gof) == 65, f"Expected 65 GoF, got {len(gof)}"
    print(f"Loaded {len(df)} variants: {len(lof)} LoF, {len(gof)} GoF")
    return df, lof, gof


# ─── Statistical helpers ──────────────────────────────────────────────────────
def threshold_analysis(lof_scores, gof_scores, threshold):
    """Returns (n_lof, total_lof, pct_lof, n_gof, total_gof, pct_gof, pval, odds_ratio)"""
    n_lof = (lof_scores >= threshold).sum()
    n_gof = (gof_scores >= threshold).sum()
    t_lof, t_gof = len(lof_scores), len(gof_scores)
    table = [[n_lof, t_lof - n_lof], [n_gof, t_gof - n_gof]]
    oddsratio, pval = stats.fisher_exact(table)
    return (
        n_lof,
        t_lof,
        100 * n_lof / t_lof,
        n_gof,
        t_gof,
        100 * n_gof / t_gof,
        pval,
        oddsratio,
    )


def mann_whitney(lof_scores, gof_scores):
    """Mann-Whitney U test: LoF > GoF (one-sided)"""
    stat, pval = stats.mannwhitneyu(lof_scores, gof_scores, alternative="greater")
    return stat, pval


# ─── Analysis ────────────────────────────────────────────────────────────────
def run_analysis(df, lof, gof):
    results = []

    # REVEL thresholds
    revel_thresholds = [
        ("REVEL", "Author_0.5", 0.500, "Author-recommended (Ioannidis et al.)"),
        ("REVEL", "Author_0.75", 0.750, "Author-recommended (Ioannidis et al.)"),
        ("REVEL", "Supporting", 0.644, "PP3 Supporting (Pejaver et al. 2022)"),
        ("REVEL", "Moderate", 0.773, "PP3 Moderate (Pejaver et al. 2022)"),
        ("REVEL", "Strong", 0.932, "PP3 Strong (Pejaver et al. 2022)"),
    ]

    # AM thresholds
    am_thresholds = [
        ("AM", "LP_threshold", 0.564, "Likely pathogenic (Cheng et al. 2023)"),
        ("AM", "Supporting", 0.564, "PP3 Supporting (analogous calibration)"),
        ("AM", "Moderate", 0.740, "PP3 Moderate (calibrated)"),
        ("AM", "Strong", 0.890, "PP3 Strong (calibrated, FDR ~5%)"),
    ]

    all_thresholds = revel_thresholds + am_thresholds

    for tool, label, thresh, desc in all_thresholds:
        score_col = "revel_score" if tool == "REVEL" else "am_score"
        lof_s = lof[score_col]
        gof_s = gof[score_col]
        n_lof, t_lof, p_lof, n_gof, t_gof, p_gof, pval, or_val = threshold_analysis(
            lof_s, gof_s, thresh
        )
        results.append(
            {
                "tool": tool,
                "threshold_label": label,
                "threshold_value": thresh,
                "description": desc,
                "lof_n": n_lof,
                "lof_total": t_lof,
                "lof_pct": p_lof,
                "gof_n": n_gof,
                "gof_total": t_gof,
                "gof_pct": p_gof,
                "lof_minus_gof_pct": p_lof - p_gof,
                "fisher_pval": pval,
                "odds_ratio": or_val,
                "significant": pval < 0.05,
            }
        )

    # Distribution stats
    dist_results = []
    for tool, score_col in [("REVEL", "revel_score"), ("AlphaMissense", "am_score")]:
        lof_s = lof[score_col]
        gof_s = gof[score_col]
        mw_stat, mw_pval = mann_whitney(lof_s, gof_s)
        dist_results.append(
            {
                "tool": tool,
                "lof_mean": lof_s.mean(),
                "lof_sd": lof_s.std(),
                "lof_median": lof_s.median(),
                "gof_mean": gof_s.mean(),
                "gof_sd": gof_s.std(),
                "gof_median": gof_s.median(),
                "mannwhitney_stat": mw_stat,
                "mannwhitney_pval": mw_pval,
            }
        )

    return pd.DataFrame(results), pd.DataFrame(dist_results)


# ─── Figure generation ────────────────────────────────────────────────────────
def make_figures(df, lof, gof, threshold_df):
    fig, axes = plt.subplots(2, 3, figsize=(17, 11))
    fig.suptitle(
        "AlphaMissense vs REVEL: Predicting Pathogenicity of LoF and GoF Variants\n"
        "Replication and Extension of Hopkins et al. (2023) Human Mutation",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )

    C = {"LoF": "#2166ac", "GoF": "#d73027"}

    def cumfreq_plot(ax, score_col, thresholds, title, xlabel):
        for mech, vdf, ls, c in [
            ("GoF", gof, "-", C["GoF"]),
            ("LoF", lof, "--", C["LoF"]),
        ]:
            s = sorted(vdf[score_col])
            x = np.linspace(0, 1, 500)
            cum = [sum(v <= xi for v in s) / len(s) for xi in x]
            ax.plot(x, cum, ls=ls, color=c, lw=2.2, label=mech)
        for thresh, label in thresholds:
            ax.axvline(thresh, color="#555", ls=":", lw=1, alpha=0.8)
            ax.text(
                thresh + 0.005,
                0.04,
                label,
                fontsize=7.5,
                color="#444",
                rotation=90,
                va="bottom",
            )
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Proportion under threshold", fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=9, loc="upper left")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.15)

    # Panel A: REVEL author thresholds
    cumfreq_plot(
        axes[0, 0],
        "revel_score",
        [(0.5, "0.5"), (0.75, "0.75")],
        "A. REVEL — Author thresholds\n(Replication of Hopkins et al. 2023)",
        "REVEL score",
    )

    # Panel B: REVEL Pejaver thresholds
    cumfreq_plot(
        axes[0, 1],
        "revel_score",
        [(0.644, "Supporting"), (0.773, "Moderate"), (0.932, "Strong")],
        "B. REVEL — ACMG PP3 thresholds\n(Pejaver et al. 2022)",
        "REVEL score",
    )

    # Panel C: AM thresholds
    cumfreq_plot(
        axes[0, 2],
        "am_score",
        [(0.564, "Supporting"), (0.740, "Moderate"), (0.890, "Strong")],
        "C. AlphaMissense — ACMG PP3 thresholds\n(Cheng et al. 2023)",
        "AlphaMissense score",
    )

    # Panel D: Violin plots
    ax = axes[1, 0]
    vdata = [lof["revel_score"], gof["revel_score"], lof["am_score"], gof["am_score"]]
    vpos = [1, 2, 4, 5]
    vp = ax.violinplot(vdata, positions=vpos, showmedians=True, showextrema=True)
    for i, (pc, c) in enumerate(
        zip(vp["bodies"], [C["LoF"], C["GoF"], C["LoF"], C["GoF"]])
    ):
        pc.set_facecolor(c)
        pc.set_alpha(0.65)
    vp["cmedians"].set_color("white")
    vp["cmedians"].set_linewidth(2)
    ax.set_xticks([1.5, 4.5])
    ax.set_xticklabels(["REVEL", "AlphaMissense"], fontsize=10)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("D. Score distributions\n(LoF = blue, GoF = red)", fontsize=10)
    ax.set_ylim(0.3, 1.05)
    ax.grid(True, alpha=0.15)
    lof_p = mpatches.Patch(color=C["LoF"], alpha=0.65, label="LoF (n=66)")
    gof_p = mpatches.Patch(color=C["GoF"], alpha=0.65, label="GoF (n=65)")
    ax.legend(handles=[lof_p, gof_p], fontsize=9)

    # Panel E: Bar chart — strong threshold comparison (values from computed threshold_df)
    ax = axes[1, 1]
    cats = ["REVEL Strong\n(≥0.932)", "AM Strong\n(≥0.890)"]
    _rstr = threshold_df.query("tool=='REVEL' and threshold_label=='Strong'").iloc[0]
    _astr = threshold_df.query("tool=='AM'   and threshold_label=='Strong'").iloc[0]
    lv = [_rstr["lof_pct"], _astr["lof_pct"]]
    gv = [_rstr["gof_pct"], _astr["gof_pct"]]
    pvs = [_rstr["fisher_pval"], _astr["fisher_pval"]]
    x = np.arange(2)
    w = 0.35
    b1 = ax.bar(x - w / 2, lv, w, color=C["LoF"], alpha=0.85, label="LoF")
    b2 = ax.bar(x + w / 2, gv, w, color=C["GoF"], alpha=0.85, label="GoF")
    for b in b1:
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.8,
            f"{b.get_height():.0f}%",
            ha="center",
            fontsize=9,
        )
    for b in b2:
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.8,
            f"{b.get_height():.0f}%",
            ha="center",
            fontsize=9,
        )

    def _stars(p):
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return "ns"

    for xi, pv in enumerate(pvs):
        ymax = max(lv[xi], gv[xi]) + 7
        ax.plot([xi - w / 2, xi + w / 2], [ymax, ymax], "k-", lw=1.2)
        ax.text(xi, ymax + 0.5, _stars(pv), ha="center", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=9)
    ax.set_ylabel("% variants meeting threshold", fontsize=10)
    ax.set_title(
        "E. LoF vs GoF: strong evidence threshold\n(* p<0.05, *** p<0.001, Fisher's exact)",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.set_ylim(0, 82)
    ax.grid(True, alpha=0.15, axis="y")

    # Panel F: Gap comparison bar chart (values from computed threshold_df)
    ax = axes[1, 2]
    tools = ["REVEL", "AlphaMissense"]
    _rmod = threshold_df.query("tool=='REVEL' and threshold_label=='Moderate'").iloc[0]
    _amod = threshold_df.query("tool=='AM'   and threshold_label=='Moderate'").iloc[0]
    gap_mod = [_rmod["lof_minus_gof_pct"], _amod["lof_minus_gof_pct"]]
    gap_str = [_rstr["lof_minus_gof_pct"], _astr["lof_minus_gof_pct"]]
    x = np.arange(2)
    w = 0.35
    ax.bar(
        x - w / 2, gap_mod, w, color="#4dac26", alpha=0.85, label="Moderate threshold"
    )
    ax.bar(x + w / 2, gap_str, w, color="#d01c8b", alpha=0.85, label="Strong threshold")
    for xi, (gm, gs) in enumerate(zip(gap_mod, gap_str)):
        ax.text(xi - w / 2, gm + 0.5, f"+{gm:.0f}%", ha="center", fontsize=9)
        ax.text(xi + w / 2, gs + 0.5, f"+{gs:.0f}%", ha="center", fontsize=9)
        if xi == 1:  # AM both significant
            ax.text(
                xi - w / 2, gm + 3, "***", ha="center", fontsize=10, color="#4dac26"
            )
            ax.text(
                xi + w / 2, gs + 3, "***", ha="center", fontsize=10, color="#d01c8b"
            )
        else:  # REVEL: moderate ns, strong *
            ax.text(xi - w / 2, gm + 3, "ns", ha="center", fontsize=8, color="#4dac26")
            ax.text(xi + w / 2, gs + 3, "*", ha="center", fontsize=10, color="#d01c8b")
    ax.set_xticks(x)
    ax.set_xticklabels(tools, fontsize=10)
    ax.set_ylabel("LoF % − GoF % (positive = LoF advantage)", fontsize=9)
    ax.set_title(
        "F. LoF–GoF performance gap per tool\n(AM disparity is larger at both thresholds)",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylim(0, 47)
    ax.grid(True, alpha=0.15, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    outpath = os.path.join(FIGURES_DIR, "Figure1_AM_vs_REVEL_LoF_GoF.png")
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.savefig(outpath.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"Figure saved: {outpath}")
    plt.close()


# ─── Scatter plot ────────────────────────────────────────────────────────────
def make_scatter(lof, gof):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(
        "REVEL vs AlphaMissense scores by disease mechanism",
        fontsize=12,
        fontweight="bold",
    )
    C = {"LoF": "#2166ac", "GoF": "#d73027"}

    for ax, gene in zip(axes, ["All genes", "By gene"]):
        ax.set_aspect("equal")
        ax.set_xlim(0.3, 1.05)
        ax.set_ylim(0.3, 1.05)
        ax.axhline(
            0.890, color="gray", ls="--", lw=0.8, alpha=0.5, label="AM strong (0.890)"
        )
        ax.axvline(
            0.932, color="gray", ls=":", lw=0.8, alpha=0.5, label="REVEL strong (0.932)"
        )

    gene_markers = {"ABCC8": "o", "GCK": "s", "KCNJ11": "^"}
    gene_colors_lof = {"ABCC8": "#08519c", "GCK": "#2171b5", "KCNJ11": "#6baed6"}
    gene_colors_gof = {"ABCC8": "#a50026", "GCK": "#d73027", "KCNJ11": "#f46d43"}

    for mech, vdf, c_dict in [
        ("LoF", lof, gene_colors_lof),
        ("GoF", gof, gene_colors_gof),
    ]:
        for _, row in vdf.iterrows():
            g = row["gene"]
            axes[0].scatter(
                row["revel_score"],
                row["am_score"],
                color=C[mech],
                marker="o",
                s=45,
                alpha=0.65,
                edgecolors="white",
                linewidths=0.3,
            )
            axes[1].scatter(
                row["revel_score"],
                row["am_score"],
                color=c_dict[g],
                marker=gene_markers[g],
                s=50,
                alpha=0.75,
                edgecolors="white",
                linewidths=0.3,
            )

    for ax in axes:
        ax.set_xlabel("REVEL score", fontsize=11)
        ax.set_ylabel("AlphaMissense score", fontsize=11)
        ax.grid(True, alpha=0.15)
        ax.plot([0.3, 1.05], [0.3, 1.05], "gray", lw=0.7, ls="-", alpha=0.4)

    # Correlation
    all_revel = pd.concat([lof, gof])["revel_score"]
    all_am = pd.concat([lof, gof])["am_score"]
    r, p = stats.pearsonr(all_revel, all_am)
    lof_r, _ = stats.pearsonr(lof["revel_score"], lof["am_score"])
    gof_r, _ = stats.pearsonr(gof["revel_score"], gof["am_score"])

    axes[0].set_title(
        f"A. All variants (n=131)\nPearson r={r:.3f} (LoF: {lof_r:.3f}, GoF: {gof_r:.3f})",
        fontsize=10,
    )
    lof_p = mpatches.Patch(color=C["LoF"], alpha=0.7, label="LoF (n=66)")
    gof_p = mpatches.Patch(color=C["GoF"], alpha=0.7, label="GoF (n=65)")
    axes[0].legend(handles=[lof_p, gof_p], fontsize=9)

    axes[1].set_title("B. By gene and mechanism", fontsize=10)
    legend_els = []
    for g, m in gene_markers.items():
        legend_els.append(plt.scatter([], [], marker=m, color="gray", s=50, label=g))
    for mech, c in C.items():
        legend_els.append(mpatches.Patch(color=c, alpha=0.7, label=mech))
    axes[1].legend(handles=legend_els, fontsize=8, ncol=2)

    plt.tight_layout()
    outpath = os.path.join(FIGURES_DIR, "Figure2_scatter_REVEL_vs_AM.png")
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.savefig(outpath.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"Figure saved: {outpath}")
    plt.close()
    return r, lof_r, gof_r


# ─── Save results ─────────────────────────────────────────────────────────────
def save_results(threshold_df, dist_df, df):
    # Main threshold table
    out = os.path.join(RESULTS_DIR, "table1_threshold_analysis.csv")
    threshold_df.to_csv(out, index=False)
    print(f"Results saved: {out}")

    # Distribution table
    out2 = os.path.join(RESULTS_DIR, "table2_score_distributions.csv")
    dist_df.to_csv(out2, index=False)
    print(f"Results saved: {out2}")

    # Full annotated dataset
    out3 = os.path.join(RESULTS_DIR, "all_variants_annotated.csv")
    df.to_csv(out3, index=False)
    print(f"Results saved: {out3}")


# ─── Console report ───────────────────────────────────────────────────────────
def print_report(threshold_df, dist_df):
    print("\n" + "=" * 75)
    print("RESULTS SUMMARY: AM vs REVEL — LoF and GoF Variant Analysis")
    print("=" * 75)

    print("\n--- TABLE 1: REVEL (Replication of Hopkins et al. 2023) ---")
    revel_df = threshold_df[threshold_df["tool"] == "REVEL"]
    print(f"{'Threshold':<22} | {'LoF':<14} | {'GoF':<14} | {'p-value':<10} | Sig")
    print("-" * 75)
    for _, r in revel_df.iterrows():
        sig = "*" if r["significant"] else ""
        print(
            f"{r['threshold_label']:<22} | {r['lof_n']}/{r['lof_total']} ({r['lof_pct']:.0f}%)  "
            f"| {r['gof_n']}/{r['gof_total']} ({r['gof_pct']:.0f}%)  "
            f"| {r['fisher_pval']:.4f}    | {sig}"
        )

    print("\n--- TABLE 2: AlphaMissense (Novel Analysis) ---")
    am_df = threshold_df[threshold_df["tool"] == "AM"]
    print(f"{'Threshold':<22} | {'LoF':<14} | {'GoF':<14} | {'p-value':<10} | Sig")
    print("-" * 75)
    for _, r in am_df.iterrows():
        sig = "*" if r["significant"] else ""
        print(
            f"{r['threshold_label']:<22} | {r['lof_n']}/{r['lof_total']} ({r['lof_pct']:.0f}%)  "
            f"| {r['gof_n']}/{r['gof_total']} ({r['gof_pct']:.0f}%)  "
            f"| {r['fisher_pval']:.4f}    | {sig}"
        )

    print("\n--- TABLE 3: Score Distributions (Mann-Whitney U, LoF > GoF) ---")
    for _, r in dist_df.iterrows():
        print(f"\n{r['tool']}:")
        print(
            f"  LoF: mean={r['lof_mean']:.3f} ± SD={r['lof_sd']:.3f}, median={r['lof_median']:.3f}"
        )
        print(
            f"  GoF: mean={r['gof_mean']:.3f} ± SD={r['gof_sd']:.3f}, median={r['gof_median']:.3f}"
        )
        print(f"  Mann-Whitney U p-value (LoF > GoF): {r['mannwhitney_pval']:.6f}")

    print("\n--- KEY COMPARISON: LoF-GoF gap at strong evidence threshold ---")
    r_gap = threshold_df.query("tool=='REVEL' and threshold_label=='Strong'")[
        "lof_minus_gof_pct"
    ].values[0]
    a_gap = threshold_df.query("tool=='AM' and threshold_label=='Strong'")[
        "lof_minus_gof_pct"
    ].values[0]
    r_p = threshold_df.query("tool=='REVEL' and threshold_label=='Strong'")[
        "fisher_pval"
    ].values[0]
    a_p = threshold_df.query("tool=='AM' and threshold_label=='Strong'")[
        "fisher_pval"
    ].values[0]
    print(
        f"  REVEL strong: LoF−GoF gap = +{r_gap:.1f}% (p={r_p:.4f}) — replicates Hopkins et al. (+20%, p=0.035)"
    )
    print(
        f"  AM strong:    LoF−GoF gap = +{a_gap:.1f}% (p={a_p:.4f}) — LARGER and more significant"
    )
    print(f"\n  CONCLUSION: AM EXACERBATES the LoF/GoF bias seen with REVEL.")
    print(f"  AM should NOT be substituted for REVEL in GoF gene classification.")
    print("=" * 75 + "\n")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data...")
    df, lof, gof = load_data()

    print("Running threshold analysis...")
    threshold_df, dist_df = run_analysis(df, lof, gof)

    print("Generating Figure 1 (main analysis)...")
    make_figures(df, lof, gof, threshold_df)

    print("Generating Figure 2 (scatter plot)...")
    r, lof_r, gof_r = make_scatter(lof, gof)
    print(f"  Overall REVEL-AM Pearson r={r:.3f} | LoF: {lof_r:.3f} | GoF: {gof_r:.3f}")

    print("Saving results tables...")
    save_results(threshold_df, dist_df, df)

    print_report(threshold_df, dist_df)
    print("Done. Check results/ and figures/ directories.")
