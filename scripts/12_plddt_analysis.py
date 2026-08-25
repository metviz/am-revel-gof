"""
12_plddt_analysis.py
Explore AlphaFold pLDDT scores and positional mean AM scores at variant
positions as proxies for structural disorder and AM's site-level prior.
Tests whether the GOF-LOF AM score gap could be confounded by GOF
variants falling in lower-confidence (more disordered) structural regions,
or at sites where AM assigns a globally lower pathogenicity prior.

Input:  results/all_variants_annotated.csv  (Hopkins cohort, n=131)
        results/tier4_positional_prior.csv   (leave-one-out site priors, script 16)
Output: figures/FigureS6_pLDDT_analysis.png / .pdf
        results/plddt_annotated_variants.csv

Analyses:
  A  pLDDT distributions at variant positions: GOF vs LOF (per gene + pooled)
  B  AM score vs pLDDT scatter coloured by mechanism — Spearman r per mechanism
  C  pLDDT-matched residual: does the GOF-LOF AM gap persist after pLDDT control?
  D  Leave-one-out positional mean AM score at GOF vs LOF positions
  E  Variant AM score vs leave-one-out positional mean AM, by mechanism
"""

import re
import requests
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from collections import defaultdict

# ── Paths ─────────────────────────────────────────────────────────────────────
VARIANTS_CSV = "results/all_variants_annotated.csv"
OUT_CSV = "results/plddt_annotated_variants.csv"
FIG_PNG = "figures/FigureS6_pLDDT_analysis.png"
FIG_PDF = "figures/FigureS6_pLDDT_analysis.pdf"

# ── UniProt → AlphaFold accession map ─────────────────────────────────────────
UNIPROT_IDS = {
    "ABCC8": "Q09428",
    "KCNJ11": "Q14654",
    "GCK": "P35557",
}

# ── Colours (consistent with manuscript figures) ──────────────────────────────
COL_LOF = "#2166ac"  # blue
COL_GOF = "#d6604d"  # red-orange
GENE_COLOURS = {"ABCC8": "#4dac26", "KCNJ11": "#7b3294", "GCK": "#d01c8b"}


# ─── 1a. Fetch AlphaFold pLDDT ────────────────────────────────────────────────


def fetch_plddt(uniprot: str) -> dict[int, float]:
    """
    Fetch per-residue pLDDT from the AlphaFold confidence JSON endpoint.
    Returns {residue_number: pLDDT_score}.
    Also returns confidenceCategory per residue (D/H/M/L) as a side effect
    stored in the global plddt_cat_db.
    """
    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot}-F1-confidence_v6.json"
    print(f"  Fetching {url}")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    data = r.json()

    residues = data["residueNumber"]
    scores = data["confidenceScore"]
    cats = data.get("confidenceCategory", [None] * len(residues))

    plddt = {}
    plddt_cat = {}
    for res, score, cat in zip(residues, scores, cats):
        plddt[res] = float(score) if score is not None else 0.0
        plddt_cat[res] = cat
    plddt_cat_db[uniprot] = plddt_cat
    print(f"    {len(plddt)} residues parsed")
    return plddt


print("Fetching AlphaFold pLDDT scores...")
plddt_db: dict[str, dict[int, float]] = {}
plddt_cat_db: dict[str, dict[int, str]] = {}  # populated by fetch_plddt
for gene, uid in UNIPROT_IDS.items():
    plddt_db[gene] = fetch_plddt(uid)


# ─── 1b. Leave-one-out positional AM prior (from script 16) ───────────────────
# The AlphaFold aa-substitutions endpoint has no file for KCNJ11 (Q14654), so
# fetching from it silently dropped the gene that supplies both of the paper's
# discordant exemplar variants. It also returned the mean over ALL 19
# substitutions at a position, which includes the observed variant and so
# inflates the variant-versus-prior correlation of panel E by construction.
# 16_positional_and_dynamics.py computes the prior LEAVE-ONE-OUT from the local
# all-substitutions cache, covering all three genes and all 131 variants, and
# resolves the ABCC8 isoform offset. Panels D and E read that file, so the
# figure and the manuscript text report the same quantity.
PRIOR_CSV = "results/tier4_positional_prior.csv"


def load_loo_prior() -> pd.DataFrame:
    pri = pd.read_csv(PRIOR_CSV)
    missing = pri.prior_loo.isna().sum()
    print(f"\nLeave-one-out positional prior: {len(pri) - missing}/{len(pri)} "
          f"variants across {pri.gene.nunique()} genes ({PRIOR_CSV})")
    return pri[["gene", "protein", "prior_loo"]]


# ─── 2. Load variants and annotate pLDDT + positional mean AM ─────────────────

df = pd.read_csv(VARIANTS_CSV)
df["pos"] = df["protein"].str.extract(r"p\.\w+?(\d+)\w+").astype(int)

df["plddt"] = df.apply(
    lambda r: plddt_db.get(r["gene"], {}).get(r["pos"], np.nan), axis=1
)
# confidenceCategory: D=very high(>90), H/M=high(70-90), L=low(50-70), VL=very low(<50)
uid_map = {g: u for g, u in UNIPROT_IDS.items()}
df["plddt_cat"] = df.apply(
    lambda r: plddt_cat_db.get(uid_map[r["gene"]], {}).get(r["pos"], None), axis=1
)
df = df.merge(load_loo_prior(), on=["gene", "protein"], how="left")
df = df.rename(columns={"prior_loo": "pos_mean_am"})

missing_plddt = df["plddt"].isna().sum()
missing_pos = df["pos_mean_am"].isna().sum()
if missing_plddt:
    print(f"WARNING: pLDDT missing for {missing_plddt} variants")
if missing_pos:
    print(f"WARNING: positional mean AM missing for {missing_pos} variants")

df.to_csv(OUT_CSV, index=False)
print(f"Annotated variants saved: {OUT_CSV}")
print(df.groupby(["gene", "mechanism"])["plddt"].describe().round(1).to_string())


# ─── 3. Statistical tests ──────────────────────────────────────────────────────

lof = df[df["mechanism"] == "LoF"]
gof = df[df["mechanism"] == "GoF"]

# 3a. GOF vs LOF pLDDT — pooled
u_stat, p_mw = stats.mannwhitneyu(
    lof["plddt"].dropna(), gof["plddt"].dropna(), alternative="two-sided"
)
print(
    f"\nPooled pLDDT  LOF median={lof['plddt'].median():.1f}  "
    f"GOF median={gof['plddt'].median():.1f}  "
    f"Mann-Whitney U={u_stat:.0f}  p={p_mw:.3f}"
)

# 3b. Spearman AM ~ pLDDT within each mechanism
for mech, sub in [("LoF", lof), ("GoF", gof)]:
    r, p = stats.spearmanr(
        sub["plddt"].dropna(), sub.loc[sub["plddt"].notna(), "am_score"]
    )
    print(f"  Spearman AM~pLDDT  {mech}: r={r:.3f}  p={p:.3f}  n={len(sub)}")

# 3c. Per-gene pLDDT GOF vs LOF
print("\nPer-gene Mann-Whitney (pLDDT GOF vs LOF):")
for gene in ["ABCC8", "KCNJ11", "GCK"]:
    g = df[df["gene"] == gene]
    l_g = g[g["mechanism"] == "LoF"]["plddt"].dropna()
    go_g = g[g["mechanism"] == "GoF"]["plddt"].dropna()
    if len(l_g) > 1 and len(go_g) > 1:
        u, p = stats.mannwhitneyu(l_g, go_g, alternative="two-sided")
        print(
            f"  {gene}  LOF med={l_g.median():.1f}  GOF med={go_g.median():.1f}  p={p:.3f}"
        )

# 3d. Partial: does pLDDT explain AM score gap?
# Regress AM on pLDDT; take residual; compare GOF vs LOF residuals
from numpy.polynomial import polynomial as P

valid = df[["am_score", "plddt", "mechanism"]].dropna()
slope, intercept = np.polyfit(valid["plddt"], valid["am_score"], 1)
valid = valid.copy()
valid["am_resid"] = valid["am_score"] - (slope * valid["plddt"] + intercept)
r_lof = valid[valid["mechanism"] == "LoF"]["am_resid"]
r_gof = valid[valid["mechanism"] == "GoF"]["am_resid"]
u2, p2 = stats.mannwhitneyu(r_lof, r_gof, alternative="two-sided")
print(
    f"\npLDDT-adjusted AM residuals  LOF med={r_lof.median():.3f}  "
    f"GOF med={r_gof.median():.3f}  Mann-Whitney p={p2:.4f}"
)

# 3e. Positional mean AM: GOF vs LOF
valid_pos = df[["am_score", "pos_mean_am", "mechanism"]].dropna()
lof_pos = valid_pos[valid_pos["mechanism"] == "LoF"]["pos_mean_am"]
gof_pos = valid_pos[valid_pos["mechanism"] == "GoF"]["pos_mean_am"]
u3, p3 = stats.mannwhitneyu(lof_pos, gof_pos, alternative="two-sided")
print(
    f"\nLeave-one-out site prior  LOF median={lof_pos.median():.3f}  "
    f"GOF median={gof_pos.median():.3f}  Mann-Whitney p={p3:.4f}"
)

# 3f. Spearman: variant AM ~ positional mean AM within each mechanism
print("Spearman variant AM ~ leave-one-out site prior:")
for mech, sub in [
    ("LoF", valid_pos[valid_pos["mechanism"] == "LoF"]),
    ("GoF", valid_pos[valid_pos["mechanism"] == "GoF"]),
]:
    r_val, p_val = stats.spearmanr(sub["pos_mean_am"], sub["am_score"])
    print(f"  {mech}: r={r_val:.3f}  p={p_val:.3f}  n={len(sub)}")


# ─── 4. Figure ────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(
    2, 3, wspace=0.38, hspace=0.48, left=0.07, right=0.97, top=0.93, bottom=0.08
)

rng = np.random.default_rng(42)

# ── Panel A: pLDDT distributions by mechanism, split by gene ──────────────────
ax_a = fig.add_subplot(gs[0, 0])

genes = ["ABCC8", "KCNJ11", "GCK"]
x_pos = np.arange(len(genes))
w = 0.28
offsets = {"LoF": -w / 2, "GoF": +w / 2}
mechs = {"LoF": (COL_LOF, "LOF"), "GoF": (COL_GOF, "GOF")}

handles_a = []
for mech, (col, label) in mechs.items():
    data_gene = [
        df[(df["gene"] == g) & (df["mechanism"] == mech)]["plddt"].dropna()
        for g in genes
    ]
    ax_a.boxplot(
        data_gene,
        positions=x_pos + offsets[mech],
        widths=w * 0.85,
        patch_artist=True,
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(color=col),
        capprops=dict(color=col),
        flierprops=dict(
            marker="o", markerfacecolor=col, markersize=3, alpha=0.5, linestyle="none"
        ),
        boxprops=dict(facecolor=col, alpha=0.75, edgecolor=col),
    )
    for i, series in enumerate(data_gene):
        jx = x_pos[i] + offsets[mech] + rng.uniform(-w * 0.3, w * 0.3, len(series))
        ax_a.scatter(jx, series, color=col, alpha=0.35, s=12, zorder=3)
    handles_a.append(plt.Rectangle((0, 0), 1, 1, fc=col, alpha=0.75, label=label))

ax_a.axhline(70, color="gray", lw=0.8, ls="--", alpha=0.6)
ax_a.axhline(50, color="gray", lw=0.8, ls=":", alpha=0.6)
ax_a.text(2.85, 71.5, "pLDDT=70", fontsize=6.5, color="gray", ha="right", va="bottom")
ax_a.text(2.85, 51.5, "pLDDT=50", fontsize=6.5, color="gray", ha="right", va="bottom")
ax_a.set_xticks(x_pos)
ax_a.set_xticklabels(genes, fontsize=9)
ax_a.set_ylabel("pLDDT at variant position", fontsize=9)
ax_a.set_ylim(0, 110)
ax_a.set_title(
    "A  pLDDT by gene and mechanism", fontsize=9, fontweight="bold", loc="left"
)
ax_a.legend(handles=handles_a, fontsize=8, frameon=False, loc="lower right")
p_txt = f"Pooled p={p_mw:.2f}" if p_mw >= 0.001 else "Pooled p<0.001"
ax_a.text(
    0.5,
    0.97,
    p_txt,
    transform=ax_a.transAxes,
    ha="center",
    va="top",
    fontsize=8,
    color="black",
)

# ── Panel B: AM score vs pLDDT scatter by mechanism ───────────────────────────
ax_b = fig.add_subplot(gs[0, 1])

for mech, col in [("LoF", COL_LOF), ("GoF", COL_GOF)]:
    sub = df[df["mechanism"] == mech].dropna(subset=["plddt", "am_score"])
    ax_b.scatter(
        sub["plddt"], sub["am_score"], color=col, alpha=0.55, s=22, label=mech, zorder=3
    )
    m_, b_ = np.polyfit(sub["plddt"], sub["am_score"], 1)
    xs = np.linspace(sub["plddt"].min(), sub["plddt"].max(), 100)
    ax_b.plot(xs, m_ * xs + b_, color=col, lw=1.2, alpha=0.7)
    r_val, p_val = stats.spearmanr(sub["plddt"], sub["am_score"])
    p_str = f"{p_val:.3f}" if p_val >= 0.001 else "<0.001"
    ax_b.text(
        0.03 if mech == "LoF" else 0.55,
        0.97 if mech == "LoF" else 0.91,
        f"{mech}: r={r_val:.2f}, p={p_str}",
        transform=ax_b.transAxes,
        fontsize=7.5,
        color=col,
        va="top",
    )

ax_b.axhline(0.890, color="gray", lw=0.8, ls="--", alpha=0.6)
ax_b.text(101, 0.895, "Strong", fontsize=6.5, color="gray", va="bottom")
ax_b.axvline(70, color="gray", lw=0.8, ls="--", alpha=0.6)
ax_b.axvline(50, color="gray", lw=0.8, ls=":", alpha=0.6)
ax_b.set_xlabel("pLDDT at variant position", fontsize=9)
ax_b.set_ylabel("AM pathogenicity score", fontsize=9)
ax_b.set_xlim(0, 103)
ax_b.set_ylim(-0.02, 1.05)
ax_b.set_title(
    "B  AM score vs pLDDT by mechanism", fontsize=9, fontweight="bold", loc="left"
)
ax_b.legend(fontsize=8, frameon=False, loc="lower right")

# ── Panel C: pLDDT-adjusted AM residuals GOF vs LOF ───────────────────────────
ax_c = fig.add_subplot(gs[0, 2])

resid_lof = valid[valid["mechanism"] == "LoF"]["am_resid"]
resid_gof = valid[valid["mechanism"] == "GoF"]["am_resid"]

for i, (series, col, lbl) in enumerate(
    [(resid_lof, COL_LOF, "LOF"), (resid_gof, COL_GOF, "GOF")]
):
    ax_c.boxplot(
        [series],
        positions=[i],
        widths=0.45,
        patch_artist=True,
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(color=col),
        capprops=dict(color=col),
        flierprops=dict(
            marker="o", markerfacecolor=col, markersize=3, alpha=0.5, linestyle="none"
        ),
        boxprops=dict(facecolor=col, alpha=0.75, edgecolor=col),
    )
    jx = i + rng.uniform(-0.15, 0.15, len(series))
    ax_c.scatter(jx, series, color=col, alpha=0.4, s=14, zorder=3)

ax_c.axhline(0, color="black", lw=0.8, ls="-", alpha=0.4)
y_max = max(resid_lof.max(), resid_gof.max()) + 0.03
ax_c.plot(
    [0, 0, 1, 1], [y_max, y_max + 0.02, y_max + 0.02, y_max], color="black", lw=1.0
)
p2_str = f"p={p2:.4f}" if p2 >= 0.0001 else "p<0.0001"
ax_c.text(0.5, y_max + 0.03, p2_str, ha="center", va="bottom", fontsize=8)
ax_c.set_xticks([0, 1])
ax_c.set_xticklabels(["LOF", "GOF"], fontsize=9)
ax_c.set_ylabel("AM score residual (pLDDT-adjusted)", fontsize=9)
ax_c.set_title(
    "C  AM gap after pLDDT adjustment", fontsize=9, fontweight="bold", loc="left"
)
ax_c.set_xlim(-0.6, 1.6)

# ── Panel D: Positional mean AM score by mechanism ────────────────────────────
ax_d = fig.add_subplot(gs[1, 0])

pos_lof = valid_pos[valid_pos["mechanism"] == "LoF"]["pos_mean_am"]
pos_gof = valid_pos[valid_pos["mechanism"] == "GoF"]["pos_mean_am"]

for i, (series, col, lbl) in enumerate(
    [(pos_lof, COL_LOF, "LOF"), (pos_gof, COL_GOF, "GOF")]
):
    ax_d.boxplot(
        [series],
        positions=[i],
        widths=0.45,
        patch_artist=True,
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(color=col),
        capprops=dict(color=col),
        flierprops=dict(
            marker="o", markerfacecolor=col, markersize=3, alpha=0.5, linestyle="none"
        ),
        boxprops=dict(facecolor=col, alpha=0.75, edgecolor=col),
    )
    jx = i + rng.uniform(-0.15, 0.15, len(series))
    ax_d.scatter(jx, series, color=col, alpha=0.4, s=14, zorder=3)

ax_d.axhline(0.890, color="gray", lw=0.8, ls="--", alpha=0.6)
ax_d.text(1.55, 0.895, "Strong", fontsize=6.5, color="gray", va="bottom", ha="right")

# the prior is a mean of AM scores and tops out near 1.0, so the significance
# bracket has to be given headroom explicitly or it lands on the axis limit
y_max_d = max(pos_lof.max(), pos_gof.max()) + 0.02
ax_d.plot(
    [0, 0, 1, 1],
    [y_max_d, y_max_d + 0.02, y_max_d + 0.02, y_max_d],
    color="black",
    lw=1.0,
)
p3_str = f"p={p3:.4f}" if p3 >= 0.0001 else "p<0.0001"
ax_d.text(0.5, y_max_d + 0.03, p3_str, ha="center", va="bottom", fontsize=8)
ax_d.set_xticks([0, 1])
ax_d.set_xticklabels(["LOF", "GOF"], fontsize=9)
ax_d.set_ylabel("Leave-one-out positional\nmean AM score", fontsize=9)
ax_d.set_title(
    "D  Leave-one-out site prior by mechanism", fontsize=9, fontweight="bold",
    loc="left"
)
ax_d.set_xlim(-0.6, 1.6)
ax_d.set_ylim(-0.02, y_max_d + 0.12)

# ── Panel E: Variant AM score vs positional mean AM ───────────────────────────
ax_e = fig.add_subplot(gs[1, 1:])

for mech, col in [("LoF", COL_LOF), ("GoF", COL_GOF)]:
    sub = valid_pos[valid_pos["mechanism"] == mech]
    ax_e.scatter(
        sub["pos_mean_am"],
        sub["am_score"],
        color=col,
        alpha=0.55,
        s=22,
        label=mech,
        zorder=3,
    )
    m_, b_ = np.polyfit(sub["pos_mean_am"], sub["am_score"], 1)
    xs = np.linspace(sub["pos_mean_am"].min(), sub["pos_mean_am"].max(), 100)
    ax_e.plot(xs, m_ * xs + b_, color=col, lw=1.2, alpha=0.7)
    r_val, p_val = stats.spearmanr(sub["pos_mean_am"], sub["am_score"])
    p_str = f"{p_val:.3f}" if p_val >= 0.001 else "<0.001"
    ax_e.text(
        0.03 if mech == "LoF" else 0.55,
        0.97 if mech == "LoF" else 0.91,
        f"{mech}: r={r_val:.2f}, p={p_str}",
        transform=ax_e.transAxes,
        fontsize=7.5,
        color=col,
        va="top",
    )

# 1:1 reference line
lim_lo = min(valid_pos["pos_mean_am"].min(), valid_pos["am_score"].min()) - 0.02
lim_hi = max(valid_pos["pos_mean_am"].max(), valid_pos["am_score"].max()) + 0.02
ax_e.plot(
    [lim_lo, lim_hi],
    [lim_lo, lim_hi],
    color="gray",
    lw=0.8,
    ls="--",
    alpha=0.5,
    zorder=1,
)
ax_e.axhline(0.890, color="gray", lw=0.8, ls="--", alpha=0.6)
ax_e.text(
    lim_hi - 0.01, 0.895, "Strong", fontsize=6.5, color="gray", va="bottom", ha="right"
)
ax_e.set_xlabel("Leave-one-out positional mean AM score", fontsize=9)
ax_e.set_ylabel("Variant AM pathogenicity score", fontsize=9)
ax_e.set_xlim(lim_lo, lim_hi)
ax_e.set_ylim(-0.02, 1.05)
ax_e.set_title(
    "E  Variant AM vs leave-one-out site prior", fontsize=9, fontweight="bold",
    loc="left"
)
ax_e.legend(fontsize=8, frameon=False, loc="lower right")

# ── Footer annotation ──────────────────────────────────────────────────────────
fig.text(
    0.5,
    0.01,
    "pLDDT from AlphaFold v6 confidence JSON (EBI). "
    "Site prior: mean AM across the other 18 substitutions at the position, observed variant excluded "
    "(local all-substitutions cache; scripts/16_positional_and_dynamics.py). "
    "Dashed lines: pLDDT=70 (flexible), 50 (disordered), AM Strong threshold (0.890). "
    "Panel C: AM residuals after linear regression on pLDDT.",
    ha="center",
    va="bottom",
    fontsize=6.5,
    color="gray",
)

fig.savefig(FIG_PNG, dpi=180, bbox_inches="tight")
fig.savefig(FIG_PDF, bbox_inches="tight")
print(f"\nFigure saved: {FIG_PNG}")
print(f"Figure saved: {FIG_PDF}")
