#!/usr/bin/env python3
"""
03_supplementary_stats.py
============================
Extended statistical analysis and supplementary tables.

Generates:
- Per-gene breakdown (ABCC8, GCK, KCNJ11)
- ROC curve analysis
- Optimal threshold determination (Youden's J)
- Concordance between REVEL and AM classifications
- Supplementary Table S1: all 131 variants with all scores

Usage:
    python 03_supplementary_stats.py
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import warnings

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

try:
    from sklearn.metrics import roc_curve, auc

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Note: sklearn not installed. ROC analysis will be skipped.")
    print("      Install with: pip install scikit-learn")


def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "am_revel_scores_combined.csv"))
    return df


def per_gene_analysis(df):
    """Break down LoF vs GoF difference per gene."""
    print("\n=== PER-GENE ANALYSIS ===")
    results = []
    for gene in ["ABCC8", "GCK", "KCNJ11"]:
        gdf = df[df["gene"] == gene]
        lof = gdf[gdf["mechanism"] == "LoF"]
        gof = gdf[gdf["mechanism"] == "GoF"]

        for tool, col, thresh_strong in [
            ("REVEL", "revel_score", 0.932),
            ("AM", "am_score", 0.890),
        ]:
            n_lof_strong = (lof[col] >= thresh_strong).sum()
            n_gof_strong = (gof[col] >= thresh_strong).sum()
            if len(gof) > 0 and len(lof) > 0:
                table = [
                    [n_lof_strong, len(lof) - n_lof_strong],
                    [n_gof_strong, len(gof) - n_gof_strong],
                ]
                try:
                    _, pval = stats.fisher_exact(table)
                except:
                    pval = 1.0
            else:
                pval = None

            results.append(
                {
                    "gene": gene,
                    "tool": tool,
                    "lof_n": len(lof),
                    "gof_n": len(gof),
                    "lof_strong_n": n_lof_strong,
                    "lof_strong_pct": 100 * n_lof_strong / len(lof)
                    if len(lof) > 0
                    else 0,
                    "gof_strong_n": n_gof_strong,
                    "gof_strong_pct": 100 * n_gof_strong / len(gof)
                    if len(gof) > 0
                    else 0,
                    "fisher_pval": pval,
                }
            )
            print(
                f"  {gene} {tool}: LoF strong={n_lof_strong}/{len(lof)} "
                f"({100 * n_lof_strong / max(len(lof), 1):.0f}%), "
                f"GoF strong={n_gof_strong}/{len(gof)} "
                f"({100 * n_gof_strong / max(len(gof), 1):.0f}%), p={f'{pval:.3f}' if pval is not None else 'N/A'}"
            )

    pg_df = pd.DataFrame(results)
    pg_df.to_csv(os.path.join(RESULTS_DIR, "table_s2_per_gene.csv"), index=False)
    return pg_df


def roc_analysis(df):
    """ROC curve comparing REVEL vs AM for LoF classification."""
    if not HAS_SKLEARN:
        return None

    print("\n=== ROC ANALYSIS ===")
    # Binary label: 1=LoF, 0=GoF (both are pathogenic; we test which predicts LoF)
    labels = (df["mechanism"] == "LoF").astype(int)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    roc_results = []
    for tool, col, color in [
        ("REVEL", "revel_score", "#2166ac"),
        ("AlphaMissense", "am_score", "#d73027"),
    ]:
        fpr, tpr, thresholds = roc_curve(labels, df[col])
        roc_auc = auc(fpr, tpr)
        # Youden's J
        j_scores = tpr - fpr
        best_idx = j_scores.argmax()
        best_thresh = thresholds[best_idx]
        best_sens = tpr[best_idx]
        best_spec = 1 - fpr[best_idx]

        roc_results.append(
            {
                "tool": tool,
                "auc": roc_auc,
                "optimal_threshold_youden": best_thresh,
                "sensitivity_at_optimal": best_sens,
                "specificity_at_optimal": best_spec,
            }
        )
        print(
            f"  {tool}: AUC={roc_auc:.3f}, Optimal threshold={best_thresh:.3f}, "
            f"Sens={best_sens:.2f}, Spec={best_spec:.2f}"
        )

        ax.plot(fpr, tpr, color=color, lw=2.2, label=f"{tool} (AUC={roc_auc:.3f})")
        ax.scatter(
            fpr[best_idx],
            tpr[best_idx],
            color=color,
            s=80,
            zorder=5,
            marker="*",
            edgecolors="black",
            linewidths=0.5,
        )

    ax.plot([0, 1], [0, 1], "gray", lw=1, ls="--")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    ax.set_title(
        "ROC: Predicting LoF vs GoF mechanism\n(★ = Youden's J optimal threshold)",
        fontsize=10,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.15)

    plt.tight_layout()
    outpath = os.path.join(FIGURES_DIR, "FigureS1_ROC_LoF_vs_GoF.png")
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.savefig(outpath.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"  ROC figure saved: {outpath}")
    plt.close()

    roc_df = pd.DataFrame(roc_results)
    roc_df.to_csv(os.path.join(RESULTS_DIR, "table_s3_roc_analysis.csv"), index=False)
    return roc_df


def concordance_analysis(df):
    """How often do REVEL and AM agree on strong/not-strong classification?"""
    print("\n=== REVEL vs AM CONCORDANCE ===")

    revel_strong = df["revel_score"] >= 0.932
    am_strong = df["am_score"] >= 0.890

    both_strong = (revel_strong & am_strong).sum()
    revel_only = (revel_strong & ~am_strong).sum()
    am_only = (~revel_strong & am_strong).sum()
    neither_strong = (~revel_strong & ~am_strong).sum()

    print(f"  Both strong:    {both_strong}/131 ({100 * both_strong / 131:.0f}%)")
    print(f"  REVEL only:     {revel_only}/131 ({100 * revel_only / 131:.0f}%)")
    print(f"  AM only:        {am_only}/131 ({100 * am_only / 131:.0f}%)")
    print(f"  Neither strong: {neither_strong}/131 ({100 * neither_strong / 131:.0f}%)")

    # Is discordance mechanism-dependent?
    lof = df[df["mechanism"] == "LoF"]
    gof = df[df["mechanism"] == "GoF"]

    for mech, mdf in [("LoF", lof), ("GoF", gof)]:
        r_s = mdf["revel_score"] >= 0.932
        a_s = mdf["am_score"] >= 0.890
        both = (r_s & a_s).sum()
        ronly = (r_s & ~a_s).sum()
        aonly = (~r_s & a_s).sum()
        n = len(mdf)
        print(f"\n  {mech} (n={n}):")
        print(f"    Both strong:    {both}/{n} ({100 * both / n:.0f}%)")
        print(f"    REVEL only:     {ronly}/{n} ({100 * ronly / n:.0f}%)")
        print(f"    AM only:        {aonly}/{n} ({100 * aonly / n:.0f}%)")

    # Correlation
    r, p = stats.pearsonr(df["revel_score"], df["am_score"])
    rho, p2 = stats.spearmanr(df["revel_score"], df["am_score"])
    print(f"\n  REVEL–AM Pearson r: {r:.3f} (p={p:.4f})")
    print(f"  REVEL–AM Spearman ρ: {rho:.3f} (p={p2:.4f})")

    concordance_df = pd.DataFrame(
        [
            {
                "both_strong": both_strong,
                "revel_only_strong": revel_only,
                "am_only_strong": am_only,
                "neither_strong": neither_strong,
                "pearson_r": r,
                "pearson_p": p,
                "spearman_rho": rho,
                "spearman_p": p2,
            }
        ]
    )
    concordance_df.to_csv(
        os.path.join(RESULTS_DIR, "table_s4_concordance.csv"), index=False
    )


def supplementary_table(df):
    """Generate Supplementary Table S1: all 131 variants with annotations."""
    # Add ACMG threshold flags
    df = df.copy()
    df["revel_supporting"] = df["revel_score"] >= 0.644
    df["revel_moderate"] = df["revel_score"] >= 0.773
    df["revel_strong"] = df["revel_score"] >= 0.932
    df["am_supporting"] = df["am_score"] >= 0.564
    df["am_moderate"] = df["am_score"] >= 0.740
    df["am_strong"] = df["am_score"] >= 0.890

    out = os.path.join(RESULTS_DIR, "table_s1_all_variants_annotated.csv")
    df.to_csv(out, index=False)
    print(f"\nSupplementary Table S1 saved: {out}")
    return df


def make_heatmap(df):
    """
    Redesigned Figure S2: scatter plot (AM vs REVEL) + distribution violins.

    The old heatmap had 131 unreadable 2-px rows.  The core story is:
      • AM and REVEL broadly agree for LoF
      • GoF variants cluster below the diagonal (REVEL rates them higher than AM)
      • A discordant zone between REVEL-Strong and AM-Strong thresholds
        contains mostly GoF variants

    Layout (14×10):
      Left 2/3  — Panel A: scatter AM (y) vs REVEL (x), mechanism colour,
                            gene marker shape, ACMG Strong threshold lines,
                            quadrant counts, diagonal
      Right 1/3 — Panel B (top):    AM score distribution by mechanism (violin)
                  Panel C (bottom): REVEL score distribution by mechanism (violin)
    """
    required = ["gene", "mechanism", "revel_score", "am_score"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  WARNING: make_heatmap missing columns: {missing} — skipping")
        return
    df = df.dropna(subset=["revel_score", "am_score"]).copy()
    if len(df) == 0:
        print("  WARNING: No rows with both scores — skipping")
        return

    N = len(df)
    n_lof = (df["mechanism"] == "LoF").sum()
    n_gof = (df["mechanism"] == "GoF").sum()
    print(
        f"  Scatter data: N={N}  LoF={n_lof}  GoF={n_gof}  Genes={df['gene'].nunique()}"
    )

    # ── Thresholds ────────────────────────────────────────────────────────────
    AM_STRONG = 0.890
    AM_MOD = 0.740
    REVEL_STRONG = 0.932
    REVEL_MOD = 0.773

    C_LOF = "#2166ac"  # blue
    C_GOF = "#d73027"  # red
    GENE_MARKERS = {"ABCC8": "o", "GCK": "s", "KCNJ11": "^"}
    GENE_COLORS = {"ABCC8": "#1a9641", "GCK": "#e6ab02", "KCNJ11": "#d7191c"}

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[2, 2, 1.2],
        height_ratios=[1, 1],
        wspace=0.32,
        hspace=0.40,
        left=0.07,
        right=0.97,
        top=0.90,
        bottom=0.08,
    )
    ax_sc = fig.add_subplot(gs[:, 0:2])  # scatter spans both rows
    ax_am_v = fig.add_subplot(gs[0, 2])  # AM violin
    ax_rv_v = fig.add_subplot(gs[1, 2])  # REVEL violin

    # ══════════════════════════════════════════════════════════════════════════
    # Panel A — Scatter
    # ══════════════════════════════════════════════════════════════════════════

    # Background shading for concordance zones
    # Both above Strong = concordant-high (pale green)
    ax_sc.axhspan(
        AM_STRONG, 1.02, xmin=0, xmax=1, alpha=0.07, color="#2ca25f", zorder=0
    )
    ax_sc.axvspan(
        REVEL_STRONG, 1.02, ymin=0, ymax=1, alpha=0.07, color="#2ca25f", zorder=0
    )
    # Discordant band: REVEL above Strong, AM below Strong (pale orange)
    # Draw as a rectangle patch
    import matplotlib.patches as mpatches

    disc_patch = mpatches.FancyArrowPatch  # just use axhspan/axvspan combo
    ax_sc.fill_betweenx(
        [0.28, AM_STRONG],
        REVEL_STRONG,
        1.02,
        alpha=0.10,
        color="#f46d43",
        zorder=0,
        label="_nolegend_",
    )

    # Diagonal y=x reference
    ax_sc.plot(
        [0.28, 1.02],
        [0.28, 1.02],
        color="#888888",
        lw=1.2,
        ls="--",
        alpha=0.6,
        zorder=1,
        label="y = x  (perfect agreement)",
    )

    # ACMG Strong threshold lines
    ax_sc.axhline(AM_STRONG, color="#7b2d8b", lw=1.3, ls="--", alpha=0.85, zorder=2)
    ax_sc.axvline(REVEL_STRONG, color="#d95f02", lw=1.3, ls="--", alpha=0.85, zorder=2)
    ax_sc.axhline(AM_MOD, color="#7b2d8b", lw=0.8, ls=":", alpha=0.45, zorder=2)
    ax_sc.axvline(REVEL_MOD, color="#d95f02", lw=0.8, ls=":", alpha=0.45, zorder=2)

    # Threshold labels
    ax_sc.text(
        0.29,
        AM_STRONG + 0.005,
        "AM Strong (0.890)",
        color="#7b2d8b",
        fontsize=8,
        va="bottom",
        ha="left",
    )
    ax_sc.text(
        REVEL_STRONG + 0.003,
        0.30,
        "REVEL Strong\n(0.932)",
        color="#d95f02",
        fontsize=8,
        va="bottom",
        ha="left",
        rotation=90,
    )
    ax_sc.text(
        0.29,
        AM_MOD + 0.005,
        "Moderate",
        color="#7b2d8b",
        fontsize=7,
        va="bottom",
        ha="left",
        alpha=0.6,
    )
    ax_sc.text(
        REVEL_MOD + 0.003,
        0.30,
        "Moderate",
        color="#d95f02",
        fontsize=7,
        va="bottom",
        ha="left",
        rotation=90,
        alpha=0.6,
    )

    # Scatter points — outer ring = mechanism colour, inner fill = gene colour
    for mech, mech_col in [("LoF", C_LOF), ("GoF", C_GOF)]:
        sub = df[df["mechanism"] == mech]
        for gene, marker in GENE_MARKERS.items():
            g = sub[sub["gene"] == gene]
            if len(g) == 0:
                continue
            ax_sc.scatter(
                g["revel_score"],
                g["am_score"],
                marker=marker,
                s=55,
                facecolor=GENE_COLORS[gene],
                edgecolors=mech_col,
                linewidth=1.5,
                alpha=0.82,
                zorder=3,
            )

    # Quadrant counts (above/below ACMG Strong for each mechanism)
    def quad_count(mech, revel_above, am_above):
        sub = df[df["mechanism"] == mech]
        rv_ok = (
            sub["revel_score"] >= REVEL_STRONG
            if revel_above
            else sub["revel_score"] < REVEL_STRONG
        )
        am_ok = (
            sub["am_score"] >= AM_STRONG if am_above else sub["am_score"] < AM_STRONG
        )
        return (rv_ok & am_ok).sum()

    # Annotations positioned inside each quadrant (data coordinates)
    #   Both Strong:      top-right  (REVEL≥Strong, AM≥Strong)
    #   GoF blind spot:   bottom-right (REVEL≥Strong, AM<Strong) — the orange zone
    #   Neither Strong:   bottom-left (REVEL<Strong, AM<Strong)
    quad_labels = [
        # (x, y, ha, va, text_template, revel_above, am_above)
        (
            0.935,
            0.895,
            "left",
            "bottom",
            "Concordant-strong\nLoF: {l}   GoF: {g}",
            True,
            True,
        ),
        (
            0.935,
            0.882,
            "left",
            "top",
            "GoF blind spot\n(REVEL≥Strong, AM<Strong)\nLoF: {l}   GoF: {g}",
            True,
            False,
        ),
        (
            0.285,
            0.285,
            "left",
            "bottom",
            "Neither Strong\nLoF: {l}   GoF: {g}",
            False,
            False,
        ),
    ]
    box_styles = [
        dict(boxstyle="round,pad=0.28", fc="#e8f5e9", ec="#2ca25f", alpha=0.90),
        dict(boxstyle="round,pad=0.28", fc="#fff3e0", ec="#f46d43", alpha=0.93),
        dict(boxstyle="round,pad=0.28", fc="#f5f5f5", ec="#aaaaaa", alpha=0.85),
    ]
    for (x, y, ha, va, tmpl, ra, aa), bstyle in zip(quad_labels, box_styles):
        l = quad_count("LoF", ra, aa)
        g = quad_count("GoF", ra, aa)
        ax_sc.text(
            x,
            y,
            tmpl.format(l=l, g=g),
            transform=ax_sc.transData,
            ha=ha,
            va=va,
            fontsize=7.5,
            bbox=bstyle,
        )

    # Legend — mechanism + gene shapes
    import matplotlib.lines as mlines

    leg_handles = []
    for mech, col in [("LoF", C_LOF), ("GoF", C_GOF)]:
        leg_handles.append(
            mlines.Line2D(
                [],
                [],
                marker="o",
                color=col,
                markerfacecolor="white",
                ms=7,
                lw=0,
                markeredgewidth=2,
                label=f"{mech} (ring colour)",
            )
        )
    for gene, mk in GENE_MARKERS.items():
        leg_handles.append(
            mlines.Line2D(
                [],
                [],
                marker=mk,
                color=GENE_COLORS[gene],
                markeredgecolor="#333333",
                ms=7,
                lw=0,
                markeredgewidth=0.8,
                label=f"{gene} (fill colour)",
            )
        )
    leg_handles.append(
        mlines.Line2D(
            [0], [0], color="#888", ls="--", lw=1.2, label="y = x  (perfect agreement)"
        )
    )
    ax_sc.legend(
        handles=leg_handles, fontsize=8, loc="upper left", framealpha=0.9, ncol=2
    )

    # Pearson r per mechanism
    from scipy.stats import pearsonr

    for mech, col, yoff in [("LoF", C_LOF, 0.05), ("GoF", C_GOF, 0.00)]:
        sub = df[df["mechanism"] == mech]
        r, p = pearsonr(sub["revel_score"], sub["am_score"])
        ax_sc.text(
            0.98,
            0.03 + yoff,
            f"{mech}: r={r:.2f} (p={p:.3f})",
            transform=ax_sc.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.5,
            color=col,
            fontweight="bold",
        )

    ax_sc.set_xlim(0.28, 1.02)
    ax_sc.set_ylim(0.28, 1.02)
    ax_sc.set_xlabel("REVEL score", fontsize=11)
    ax_sc.set_ylabel("AlphaMissense score", fontsize=11)
    ax_sc.set_title(
        f"A.  AM vs REVEL score per variant  (N={N})"
        f"  ·  Ring = mechanism  ·  Fill = gene",
        fontsize=9.5,
        loc="left",
    )
    ax_sc.tick_params(labelsize=9)
    ax_sc.set_aspect("equal")

    # ══════════════════════════════════════════════════════════════════════════
    # Panels B & C — Score distributions by mechanism
    # ══════════════════════════════════════════════════════════════════════════
    import seaborn as sns

    for ax_v, score_col, tool_name, strong_t, mod_t, col_tool in [
        (ax_am_v, "am_score", "AlphaMissense", AM_STRONG, AM_MOD, "#7b2d8b"),
        (ax_rv_v, "revel_score", "REVEL", REVEL_STRONG, REVEL_MOD, "#d95f02"),
    ]:
        lof_s = df[df["mechanism"] == "LoF"][score_col].dropna()
        gof_s = df[df["mechanism"] == "GoF"][score_col].dropna()

        vp = ax_v.violinplot(
            [lof_s, gof_s],
            positions=[0, 1],
            showmedians=True,
            showextrema=False,
            widths=0.6,
        )
        for body, col in zip(vp["bodies"], [C_LOF, C_GOF]):
            body.set_facecolor(col)
            body.set_alpha(0.65)
        vp["cmedians"].set_color("black")
        vp["cmedians"].set_linewidth(1.5)

        # ACMG threshold lines
        ax_v.axhline(strong_t, color=col_tool, ls="--", lw=1.2, alpha=0.8)
        ax_v.axhline(mod_t, color=col_tool, ls=":", lw=0.9, alpha=0.5)
        ax_v.text(
            1.02,
            strong_t,
            f"Strong\n({strong_t})",
            color=col_tool,
            fontsize=6.5,
            va="center",
            ha="left",
            transform=ax_v.get_yaxis_transform(),
            clip_on=False,
        )

        # Median diamond + % above Strong annotation (inside plot, bottom)
        for xi, (scores, mech) in enumerate([(lof_s, "LoF"), (gof_s, "GoF")]):
            med = scores.median()
            pct = 100 * (scores >= strong_t).mean()
            ax_v.scatter(
                [xi],
                [med],
                marker="D",
                s=20,
                color="white",
                edgecolors="black",
                zorder=5,
                linewidth=0.8,
            )
            # Place inside the axes at the very bottom — no overlap with tick labels
            ax_v.text(
                xi,
                0.295,
                f"{pct:.0f}%\n≥Strong",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color=[C_LOF, C_GOF][xi],
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
            )

        ax_v.set_xticks([0, 1])
        # Single-line tick labels — n= in parentheses, no newline
        ax_v.set_xticklabels(
            [f"LoF (n={len(lof_s)})", f"GoF (n={len(gof_s)})"], fontsize=8.5
        )
        ax_v.set_ylabel("Score", fontsize=9)
        ax_v.set_ylim(0.28, 1.05)
        ax_v.set_title(
            f"{'B' if tool_name == 'AlphaMissense' else 'C'}.  {tool_name}",
            fontsize=9.5,
            loc="left",
            color=col_tool,
            fontweight="bold",
        )
        ax_v.tick_params(axis="y", labelsize=8)
        for sp in ["top", "right"]:
            ax_v.spines[sp].set_visible(False)

    # ── Figure title ──────────────────────────────────────────────────────────
    fig.suptitle(
        f"AM vs REVEL Score Comparison (Hopkins 2023, N={N})\n"
        f"GoF variants cluster in the discordant zone: REVEL≥Strong but AM<Strong",
        fontsize=11,
        fontweight="bold",
        y=0.97,
    )

    outpath = os.path.join(FIGURES_DIR, "FigureS2_heatmap.png")
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.savefig(outpath.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"Figure S2 saved: {outpath}")


if __name__ == "__main__":
    print("Loading data...")
    df = load_data()

    per_gene_analysis(df)
    roc_analysis(df)
    concordance_analysis(df)
    supplementary_table(df)
    make_heatmap(df)

    print("\nDone. Check results/ and figures/ for supplementary outputs.")
