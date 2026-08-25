"""
10c — De-duplicated meta-analysis.

One fix vs 10b_meta_analysis_corrected.py. Originals are NOT overwritten; all
outputs carry a _dedup suffix.

FIX 3 — Remove variants shared between the Hopkins and De Franco strata.
    100 of the 103 Hopkins ABCC8/KCNJ11 variants (97%: ABCC8 45/47, KCNJ11 55/56)
    also appear in the 547-variant De Franco cohort. Pooling Hopkins ABCC8 with
    De Franco ABCC8 as independent strata therefore double-counts almost the whole
    Hopkins KATP arm, violating the independence assumption of both Cochran's Q and
    the DerSimonian-Laird variance, and making the pooled CI anticonservative.

    Here the De Franco strata are restricted to the 447 variants not present in
    Hopkins (data/de_franco/de_franco_novel_only.csv), so no variant enters the
    pool twice. The Hopkins strata are left whole.

    Note this does not de-duplicate Hopkins GCK (n=28) against the GCK DMS
    (n=6,016), which necessarily contains them. That stratum pair is only pooled
    in the per-gene GCK estimate and in the all-strata pool, both of which are
    reported as secondary; the primary estimate is clinical-only.

PRIMARY ESTIMATE. Following review, the headline pooled estimate is the
clinical-only, de-duplicated pool (k=5: Hopkins x 3 genes, De Franco-novel x 2
genes). The GCK DMS uses an abundance-derived label that is not the same
construct as the clinical GoF/LoF labels, so the all-strata pool that includes it
is reported as a secondary sensitivity analysis, not as the headline.
"""

import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
DATA = os.path.join(ROOT, "data")

STRONG = {"AM": 0.890, "REVEL": 0.932}
SCORE_COL = {"AM": "am_score", "REVEL": "revel_score"}
C = {"AM": "#C0392B", "REVEL": "#2C7BB6"}


def strip_p(s):
    r"""Remove an HGVS 'p.' prefix and surrounding parentheses -- and nothing else.

    The obvious one-liner, re.sub(r"[p\.\(\)]", "", s), deletes EVERY 'p'
    in the string, so Asp209Glu becomes As209Glu and Trp becomes Tr. That silently
    mangles every variant at an Asp or Trp residue.
    """
    s = str(s).strip()
    s = re.sub(r"^p\.", "", s)
    return s.strip("()").strip()


# ---------------------------------------------------------------- statistics
def gap_se(lof_k, lof_n, gof_k, gof_n):
    """Textbook SE of a difference of two independent proportions, in pp."""
    p1, p2 = lof_k / lof_n, gof_k / gof_n
    var = p1 * (1 - p1) / lof_n + p2 * (1 - p2) / gof_n
    return float(np.sqrt(var) * 100)


def dl_meta(y, se):
    """DerSimonian-Laird random-effects pooling. Returns dict."""
    y, v = np.asarray(y, float), np.asarray(se, float) ** 2
    w = 1.0 / v
    theta_fe = (w * y).sum() / w.sum()
    Q = float((w * (y - theta_fe) ** 2).sum())
    df = len(y) - 1
    c = w.sum() - (w**2).sum() / w.sum()
    tau2 = max(0.0, (Q - df) / c) if c > 0 else 0.0
    I2 = max(0.0, (Q - df) / Q * 100) if Q > 0 else 0.0
    wr = 1.0 / (v + tau2)
    theta = float((wr * y).sum() / wr.sum())
    se_re = float(np.sqrt(1.0 / wr.sum()))
    p_het = float(1 - chi2.cdf(Q, df)) if df > 0 else 1.0
    # Riley/IntHout prediction interval: where a new stratum would be expected to
    # fall. Undefined for tau2=0 and uselessly wide below k=4 (t crit blows up on
    # 1-2 df), so it is only reported for the two main pools.
    if df >= 3 and tau2 > 0:
        from scipy.stats import t as tdist

        crit = tdist.ppf(0.975, df)
        half = crit * np.sqrt(tau2 + se_re**2)
        pi = (theta - half, theta + half)
    else:
        pi = (np.nan, np.nan)
    return dict(
        k=len(y),
        pooled_gap=theta,
        se=se_re,
        ci_lo=theta - 1.96 * se_re,
        ci_hi=theta + 1.96 * se_re,
        tau2=tau2,
        Q=Q,
        I2=I2,
        p_heterogeneity=p_het,
        pi_lo=pi[0],
        pi_hi=pi[1],
        weights=list(np.asarray(1.0 / (v + tau2)) / (1.0 / (v + tau2)).sum() * 100),
    )


# ---------------------------------------------------------------- input data
def norm_hgvs(s):
    """'p.(Ser3Cys)' and 'p.Ser3Cys' -> 'Ser3Cys'."""
    return strip_p(s)


def build_strata():
    """Return the de-duplicated stratum table, one row per (dataset, gene, tool)."""
    hop = pd.read_csv(os.path.join(RES, "all_variants_annotated.csv"))
    dfr = pd.read_csv(os.path.join(DATA, "de_franco", "de_franco_processed.csv"))

    hop_keys = set(zip(hop.gene, hop.protein.map(norm_hgvs)))
    dfr["_dedup_key"] = list(zip(dfr.gene, dfr.hgvs_p.map(norm_hgvs)))
    shared = dfr._dedup_key.isin(hop_keys)
    novel = dfr[~shared].copy()

    print(
        f"  overlap: {int(shared.sum())} of {len(hop[hop.gene.isin(['ABCC8', 'KCNJ11'])])} "
        f"Hopkins KATP variants also in De Franco "
        f"({int(shared.sum()) / 103 * 100:.0f}%); "
        f"De Franco retained = {len(novel)}"
    )
    for g in ("ABCC8", "KCNJ11"):
        hn = (hop.gene == g).sum()
        sn = int((shared & (dfr.gene == g)).sum())
        print(f"    {g}: Hopkins {hn}, shared {sn} ({sn / hn * 100:.0f}%)")

    rows = []
    for tool in ("AM", "REVEL"):
        col, cut = SCORE_COL[tool], STRONG[tool]
        for label, src, genes, dtype in (
            ("Hopkins 2023", hop, ("KCNJ11", "ABCC8", "GCK"), "Clinical"),
            ("De Franco 2020 (novel)", novel, ("KCNJ11", "ABCC8"), "Clinical"),
        ):
            for g in genes:
                sub = src[(src.gene == g)].dropna(subset=[col])
                lof = sub[sub.mechanism == "LoF"]
                gof = sub[sub.mechanism == "GoF"]
                if len(lof) < 2 or len(gof) < 2:
                    continue
                lk, gk = int((lof[col] >= cut).sum()), int((gof[col] >= cut).sum())
                gap = (lk / len(lof) - gk / len(gof)) * 100
                rows.append(
                    dict(
                        dataset=label,
                        gene=g,
                        tool=tool,
                        lof_n=len(lof),
                        gof_n=len(gof),
                        lof_k=lk,
                        gof_k=gk,
                        lof_pct=lk / len(lof) * 100,
                        gof_pct=gk / len(gof) * 100,
                        gap=gap,
                        se=gap_se(lk, len(lof), gk, len(gof)),
                        data_type=dtype,
                    )
                )

    # GCK DMS stratum: carried over unchanged from 10b (no Hopkins de-duplication
    # is possible here without discarding the DMS, and it is secondary anyway)
    prev = pd.read_csv(os.path.join(RES, "meta_analysis_effects_corrected.csv"))
    dms = prev[prev.dataset.str.startswith("GCK DMS")]
    for _, r in dms.iterrows():
        rows.append(
            dict(
                dataset=r.dataset,
                gene=r.gene,
                tool=r.tool,
                lof_n=int(r.lof_n),
                gof_n=int(r.gof_n),
                lof_k=int(r.lof_k),
                gof_k=int(r.gof_k),
                lof_pct=r.lof_pct,
                gof_pct=r.gof_pct,
                gap=r.gap,
                se=r.se,
                data_type="DMS Functional",
            )
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------- pooling
SUBGROUPS = [
    ("Clinical", lambda d: d.data_type == "Clinical"),  # PRIMARY
    ("All", lambda d: d.index == d.index),
    ("Functional", lambda d: d.data_type == "DMS Functional"),
    ("ABCC8", lambda d: d.gene == "ABCC8"),
    ("KCNJ11", lambda d: d.gene == "KCNJ11"),
    ("GCK", lambda d: d.gene == "GCK"),
]


def pool(eff):
    out = []
    for tool in ("AM", "REVEL"):
        t = eff[eff.tool == tool]
        for name, sel in SUBGROUPS:
            sub = t[sel(t)]
            if len(sub) == 0:
                continue
            m = dl_meta(sub.gap.values, sub.se.values)
            m.pop("weights")
            out.append(dict(tool=tool, subgroup=name, **m))
    return pd.DataFrame(out)


# ---------------------------------------------------------------- forest plot
def forest(eff, summ):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), sharex=True)
    for ax, tool in zip(axes, ("AM", "REVEL")):
        e = eff[eff.tool == tool].reset_index(drop=True)
        rows = [
            (f"{r.dataset} · {r.gene}", r.gap, r.se, r.data_type)
            for r in e.itertuples()
        ]
        ys = list(range(len(rows)))[::-1]
        for y, (lab, g, s, dt) in zip(ys, rows):
            lo, hi = g - 1.96 * s, g + 1.96 * s
            ax.plot([lo, hi], [y, y], color=C[tool], lw=1.4, zorder=2)
            ax.scatter([g], [y], s=90, color=C[tool], marker="s", zorder=3)
            ax.text(-124, y, lab, va="center", ha="left", fontsize=8.2)
            ax.text(
                112,
                y,
                f"{g:5.1f} [{lo:5.1f}, {hi:5.1f}]",
                va="center",
                ha="right",
                fontsize=8,
                family="monospace",
            )

        y = -1.4
        for name, style in (
            ("Clinical", dict(fc=C[tool], label="Clinical-only (primary)")),
            ("All", dict(fc="white", label="All strata (secondary)")),
        ):
            s = summ[(summ.tool == tool) & (summ.subgroup == name)]
            if s.empty:
                continue
            s = s.iloc[0]
            lo, hi = s.ci_lo, s.ci_hi
            ax.add_patch(
                plt.Polygon(
                    [
                        [lo, y],
                        [s.pooled_gap, y + 0.34],
                        [hi, y],
                        [s.pooled_gap, y - 0.34],
                    ],
                    facecolor=style["fc"],
                    edgecolor=C[tool],
                    lw=1.5,
                    zorder=4,
                )
            )
            ax.text(
                -124,
                y,
                f"{style['label']}  (k={int(s.k)}, I²={s.I2:.0f}%)",
                va="center",
                ha="left",
                fontsize=8.4,
                fontweight="bold",
            )
            ax.text(
                112,
                y,
                f"{s.pooled_gap:5.1f} [{lo:5.1f}, {hi:5.1f}]",
                va="center",
                ha="right",
                fontsize=8,
                family="monospace",
                fontweight="bold",
            )
            y -= 1.5

        ax.axvline(0, color="#888", lw=1, ls="--", zorder=1)
        ax.set_xlim(-126, 114)
        ax.set_xticks([-40, -20, 0, 20, 40, 60])
        ax.set_ylim(y - 0.4, len(rows) - 0.3)
        ax.set_yticks([])
        ax.set_xlabel("LoF − GoF sensitivity gap at ACMG Strong (pp)", fontsize=9.5)
        ax.set_title(tool, fontsize=12, fontweight="bold", color=C[tool])
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)

    fig.suptitle(
        "LoF−GoF sensitivity gap, de-duplicated strata (De Franco restricted to the "
        "447 variants not in Hopkins)",
        fontsize=10.5,
        y=0.985,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("png", "pdf"):
        fig.savefig(
            os.path.join(FIG, f"Figure6_MetaAnalysis_ForestPlot_dedup.{ext}"),
            dpi=200,
            bbox_inches="tight",
        )
    plt.close(fig)
    print("  wrote Figure6_MetaAnalysis_ForestPlot_dedup.png/.pdf")


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    print("10c — de-duplicated meta-analysis")
    eff = build_strata()
    summ = pool(eff)

    eff.to_csv(os.path.join(RES, "meta_analysis_effects_dedup.csv"), index=False)
    summ.to_csv(os.path.join(RES, "meta_analysis_summary_dedup.csv"), index=False)
    print("  wrote meta_analysis_{effects,summary}_dedup.csv")

    forest(eff, summ)

    print("\n  pooled estimates (primary = Clinical):")
    for _, r in summ.iterrows():
        pi = "" if np.isnan(r.pi_lo) else f"  PI {r.pi_lo:.1f} to {r.pi_hi:.1f}"
        print(
            f"    {r.tool:5s} {r.subgroup:10s} k={int(r.k)}  "
            f"{r.pooled_gap:5.1f} pp  CI {r.ci_lo:5.1f}–{r.ci_hi:5.1f}  "
            f"I²={r.I2:3.0f}%  p-het={r.p_heterogeneity:.3f}{pi}"
        )

    # DMS share of weight in the all-strata AM pool — the manuscript claims
    # this stratum "carries the predominant weight"; check it.
    am = eff[eff.tool == "AM"]
    m = dl_meta(am.gap.values, am.se.values)
    w = dict(zip(am.dataset + " " + am.gene, m["weights"]))
    dms_w = sum(v for k, v in w.items() if k.startswith("GCK DMS"))
    print(
        f"\n  GCK DMS share of AM all-strata weight: {dms_w:.1f}%  "
        f"(tau2={m['tau2']:.1f}); manuscript says 'predominant'"
    )
