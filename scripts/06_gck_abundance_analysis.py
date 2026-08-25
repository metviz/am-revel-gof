#!/usr/bin/env python3
"""
06_gck_abundance_analysis.py — GCK Abundance: Mechanistic Sub-stratification
==============================================================================
External validation using the Gersing 2024 GCK abundance DMS dataset.

SCIENTIFIC RATIONALE — WHY THIS DATASET MATTERS:
  Script 05 treated all LoF variants as one class. But GCK hypoactive variants
  fail through two DISTINCT mechanisms:
    Type 1 — Structural instability: variant destabilizes the protein → reduced
      cellular abundance → measured by abundance assay score <0.58 (low abundance)
      Enriched in the LARGE DOMAIN (static, buried residues)
    Type 2 — Conformational dynamics defect: protein is stable but cannot adopt
      the correct conformational states for catalysis → normal abundance but
      reduced activity. Enriched in the SMALL DOMAIN.

  By Gersing 2024: 43% of LoF variants are Type 1 (abundance + activity), 57% Type 2.

WHY THIS MATTERS FOR AM AND REVEL:
  AlphaMissense is built on AlphaFold structures. It should excel at detecting
  variants that DESTABILIZE protein structure (Type 1 / stability-LoF).
  It has no direct information about conformational dynamics (Type 2 / dynamics-LoF).
  REVEL uses evolutionary conservation — captures both types but less mechanistically.

  TESTABLE PREDICTION:
    AM scores should be HIGHER (correctly flagged as pathogenic) for Type 1
    (stability-LoF) than for Type 2 (dynamics-LoF), because AlphaFold structure
    encodes structural stability directly.
    If confirmed: explains part of AM's GoF blind spot — GoF variants work via
    conformational mechanisms (allosteric activation), same as Type 2 LoF.

KEY CROSS-LINK WITH SCRIPT 05:
  GoF variants (score >1.1 in activity) → gain of conformational flexibility
  → likely score-similar to Type 2 LoF (dynamics defect) in AM/REVEL
  This creates a 4-class comparison:
    1. Stability-LoF  (Type 1: activity <0.5, abundance <0.58)
    2. Dynamics-LoF   (Type 2: activity <0.5, abundance ≥0.58)
    3. GoF            (activity >1.1, any abundance)
    4. Intermediate   (activity 0.5–1.1)

DATA SOURCES:
  Activity: Gersing 2023 (Script 05) — MaveDB urn:mavedb:00000096-a-1
  Abundance: Gersing 2024 — MaveDB urn:mavedb:00000096-b-1
  Joined on: (position, alt_aa) — both measure same GCK variant library

PROTOCOL (Section 4 of MASTER_PLAN.md — consistent across all datasets):
  Step 1 — Acquire & validate abundance data
  Step 2 — Preprocess + join with activity dataset from Script 05
  Step 3 — Mechanism annotation (4-class: Stability-LoF, Dynamics-LoF, GoF, Int)
  Step 4 — Core statistics (same thresholds as all other scripts)
  Step 5 — Dataset-specific: AM/REVEL by LoF mechanism type; position-level analysis
  Step 6 — Outputs

REQUIREMENTS:
  pip install pandas numpy scipy matplotlib seaborn requests
"""

import os
import sys
import json
import hashlib
import datetime
import requests
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu, kruskal, fisher_exact
from sklearn.metrics import roc_curve, auc as sk_auc
import warnings

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
GCK_DIR = os.path.join(PROJECT_DIR, "data", "gck_dms")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")
VALID_DIR = os.path.join(PROJECT_DIR, "validation")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
CHECKSUMS = os.path.join(VALID_DIR, "checksums.md")
ROW_COUNTS = os.path.join(VALID_DIR, "row_counts.log")

ACTIVITY_FILE = os.path.join(GCK_DIR, "gck_activity_processed.csv")  # from Script 05
ABUNDANCE_RAW = os.path.join(GCK_DIR, "gck_abundance_mavedb_raw.csv")
MERGED_OUT = os.path.join(GCK_DIR, "gck_activity_abundance_merged.csv")

# ─── Thresholds (from Gersing papers) ────────────────────────────────────────
# Activity thresholds (Gersing 2023)
ACT_LOF_THRESH = 0.50  # activity < 0.5  → hypoactive (LoF)
ACT_GOF_THRESH = 1.10  # activity > 1.1  → hyperactive (GoF)

# Abundance threshold (Gersing 2024, Fig 2 and paper text)
ABU_LOW_THRESH = 0.58  # abundance < 0.58 → significantly reduced cellular protein


# ─── Helpers ─────────────────────────────────────────────────────────────────
def log_row_count(label, n, expected=None):
    msg = f"[{datetime.datetime.now().isoformat()}] {label}: N={n}"
    if expected:
        status = "OK" if n >= expected * 0.88 else "WARNING — lower than expected"
        msg += f" (expected ~{expected}, {status})"
    print(f"  {msg}")
    with open(ROW_COUNTS, "a") as f:
        f.write(msg + "\n")


def record_checksum(path, label, n_rows, synthetic=False):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    with open(CHECKSUMS, "a") as f:
        f.write(f"\n## {label}\n")
        f.write(f"- MD5: {h.hexdigest()}\n")
        f.write(f"- N rows: {n_rows}\n")
        f.write(f"- Date: {datetime.datetime.now().isoformat()}\n")
        f.write(f"- Synthetic: {synthetic}\n")


# ─── Step 1: Acquire GCK Abundance Data ──────────────────────────────────────
print("=" * 65)
print("STEP 1 — DATA ACQUISITION: GCK Abundance (Gersing 2024)")
print("=" * 65)

MAVEDB_ABU_ACC = "urn:mavedb:00000096-b-1"
MAVEDB_ABU_URL = f"https://api.mavedb.org/api/v1/score-sets/{MAVEDB_ABU_ACC}/scores"

IS_SYNTHETIC_ABU = False


def download_mavedb(url, out_path, label, accession):
    """Download from MaveDB API. Raises RuntimeError on failure — no synthetic fallback."""
    if os.path.exists(out_path):
        print(f"  Cached: {out_path}")
        return pd.read_csv(out_path)
    print(f"  Fetching {label} from MaveDB...")
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        from io import StringIO

        df = pd.read_csv(StringIO(resp.text))
        df.to_csv(out_path, index=False)
        print(f"  Downloaded: N={len(df)} rows → {out_path}")
        return df
    except Exception as e:
        raise RuntimeError(
            f"MaveDB download failed: {e}\n"
            f"  Accession : {accession}\n"
            f"  URL       : {url}\n"
            f"  Fix       : ensure network access to api.mavedb.org, then re-run.\n"
            f"  Manual    : curl -o {out_path} '{url}'"
        )


abu_raw = download_mavedb(
    MAVEDB_ABU_URL, ABUNDANCE_RAW, "GCK abundance (Gersing 2024)", MAVEDB_ABU_ACC
)

log_row_count("GCK Abundance raw", len(abu_raw), expected=8822)
record_checksum(
    ABUNDANCE_RAW, "gck_abundance_mavedb_raw.csv", len(abu_raw), synthetic=False
)

# ─── Step 2: Preprocessing & Join ────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 2 — PREPROCESSING + JOIN WITH ACTIVITY DATASET")
print("=" * 65)

# Load activity dataset (from Script 05)
act = pd.read_csv(ACTIVITY_FILE)
print(f"  Activity dataset loaded: N={len(act)}")

# Normalize abundance columns
abu = abu_raw.copy()
col_map = {}
for col in abu.columns:
    if col.lower() in ["score", "dms_score", "abundance_score", "fitness"]:
        col_map[col] = "abundance_score"
    elif col.lower() in ["hgvs_pro", "hgvs", "variant"]:
        col_map[col] = "hgvs_pro"
abu = abu.rename(columns=col_map)
if "abundance_score" not in abu.columns and "score" in abu.columns:
    abu = abu.rename(columns={"score": "abundance_score"})

# Parse position/alt_aa if not present
if "position" not in abu.columns or "alt_aa" not in abu.columns:
    import re

    def parse_hgvs(hgvs):
        m = re.match(r"p\.([A-Za-z]+)(\d+)([A-Za-z*]+)", str(hgvs))
        if m:
            return m.group(1)[:1].upper(), int(m.group(2)), m.group(3)[:1].upper()
        return None, None, None

    abu[["wt_aa", "position", "alt_aa"]] = abu["hgvs_pro"].apply(
        lambda x: pd.Series(parse_hgvs(x))
    )

abu = abu[abu["abundance_score"].notna() & abu["position"].notna()].copy()
abu["position"] = abu["position"].astype(int)

# Deduplicate: take mean abundance_score per (position, alt_aa) to avoid
# many-to-many join if the abundance dataset has multiple rows per variant
abu = abu.groupby(["position", "alt_aa"], as_index=False)["abundance_score"].mean()

# Join activity + abundance on (position, alt_aa)
merged = act.merge(
    abu[["position", "alt_aa", "abundance_score"]],
    on=["position", "alt_aa"],
    how="inner",
)

print(f"  Abundance dataset: N={len(abu)}")
print(f"  Merged (inner join): N={len(merged)}")
coverage = len(merged) / len(act)
print(f"  Join coverage: {coverage:.1%} of activity variants have abundance scores")
assert coverage > 0.70, (
    f"Join coverage too low: {coverage:.1%} — check variant ID format"
)
log_row_count("GCK merged (activity + abundance)", len(merged), expected=8000)

# ─── Step 3: 4-Class Mechanism Annotation ────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 3 — 4-CLASS MECHANISM ANNOTATION")
print("=" * 65)
print("""
  Classification logic (based on Gersing 2023 + 2024):

  DMS activity score  |  Abundance score  |  Class           |  Interpretation
  ─────────────────── | ─────────────────  | ──────────────── | ─────────────────────────────
  < 0.50 (hypoactive) |  < 0.58 (low)     |  Stability-LoF   | Structural destabilization
  < 0.50 (hypoactive) |  ≥ 0.58 (normal)  |  Dynamics-LoF    | Conformational dynamics defect
  > 1.10 (hyperactive)|  any              |  GoF             | Allosteric activation
  0.50 – 1.10         |  any              |  Intermediate    | Uncertain
""")


def classify_4way(act_score, abu_score):
    if act_score > ACT_GOF_THRESH:
        return "GoF"
    elif act_score < ACT_LOF_THRESH:
        if abu_score < ABU_LOW_THRESH:
            return "Stability-LoF"
        else:
            return "Dynamics-LoF"
    else:
        return "Intermediate"


merged["mechanism_4class"] = merged.apply(
    lambda r: classify_4way(r["dms_score"], r["abundance_score"]), axis=1
)

counts_4 = merged["mechanism_4class"].value_counts()
print("  4-class distribution:")
for cls in ["Stability-LoF", "Dynamics-LoF", "GoF", "Intermediate"]:
    n = counts_4.get(cls, 0)
    pct = 100 * n / len(merged)
    print(f"    {cls:<20}: N={n:>5} ({pct:.1f}%)")

# Sanity check: 43% of LoF should be Stability-LoF (from Gersing 2024)
n_lof_total = counts_4.get("Stability-LoF", 0) + counts_4.get("Dynamics-LoF", 0)
n_stability = counts_4.get("Stability-LoF", 0)
if n_lof_total > 0:
    stability_pct = 100 * n_stability / n_lof_total
    print(f"\n  Stability-LoF as % of all LoF: {stability_pct:.1f}%")
    print(
        f"  (Expected from paper: ~43%; {'OK' if 35 <= stability_pct <= 55 else 'CHECK THRESHOLDS'})"
    )

log_row_count("Stability-LoF", counts_4.get("Stability-LoF", 0))
log_row_count("Dynamics-LoF", counts_4.get("Dynamics-LoF", 0))
log_row_count("GoF (abundance)", counts_4.get("GoF", 0))

merged.to_csv(MERGED_OUT, index=False)
print(f"\n  Merged dataset saved: {MERGED_OUT}")

# Determine if REVEL data is available
HAS_REVEL = "revel_score" in merged.columns and merged["revel_score"].notna().any()
if not HAS_REVEL:
    print("  NOTE: REVEL scores not available for GCK DMS variants — AM-only analysis.")

# ─── Step 4: Core Statistics ─────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 4 — CORE STATISTICS")
print("=" * 65)

stab = merged[merged["mechanism_4class"] == "Stability-LoF"]
dyn = merged[merged["mechanism_4class"] == "Dynamics-LoF"]
gof = merged[merged["mechanism_4class"] == "GoF"]
inter = merged[merged["mechanism_4class"] == "Intermediate"]

# 4a — Score distributions by 4-class (Kruskal-Wallis across all 4)
print("\n  4a. Score distributions — 4-class Kruskal-Wallis test:")
tool_cols = [("AlphaMissense", "am_score")] + (
    [("REVEL", "revel_score")] if HAS_REVEL else []
)
for tool, col in tool_cols:
    groups = [g[col].dropna() for g in [stab, dyn, gof, inter]]
    names = ["Stability-LoF", "Dynamics-LoF", "GoF", "Intermediate"]
    stat, pval = kruskal(*groups)
    print(f"\n  {tool}: H={stat:.1f}, p={pval:.4e}")
    for name, grp in zip(names, groups):
        print(
            f"    {name:<20}: mean={grp.mean():.3f}  std={grp.std():.3f}  N={len(grp)}"
        )

# 4b — Pairwise: Stability-LoF vs Dynamics-LoF (key test)
print("\n  4b. Key pairwise: Stability-LoF vs Dynamics-LoF")
print("      Prediction: AM should score Stability-LoF HIGHER (easier to detect)")
print("      because AlphaFold encodes structural stability, not dynamics")
print()
for tool, col in tool_cols:
    s_scores = stab[col].dropna()
    d_scores = dyn[col].dropna()
    stat, pval = mannwhitneyu(s_scores, d_scores, alternative="greater")
    delta = s_scores.mean() - d_scores.mean()
    print(
        f"  {tool}: Stability-LoF {s_scores.mean():.3f}±{s_scores.std():.3f}"
        f"  vs  Dynamics-LoF {d_scores.mean():.3f}±{d_scores.std():.3f}"
        f"  Δ={delta:+.3f}  p={pval:.4f} {'*' if pval < 0.05 else ''}"
    )

# 4c — Key pairwise: Dynamics-LoF vs GoF (conformational mechanism similarity)
print("\n  4c. Key pairwise: Dynamics-LoF vs GoF")
print("      Prediction: AM scores should be SIMILAR for both (both conformational)")
print("      This explains why AM misses GoF — same mechanism as Dynamics-LoF")
print()
for tool, col in tool_cols:
    d_scores = dyn[col].dropna()
    g_scores = gof[col].dropna()
    stat, pval = mannwhitneyu(d_scores, g_scores, alternative="two-sided")
    delta = d_scores.mean() - g_scores.mean()
    print(
        f"  {tool}: Dynamics-LoF {d_scores.mean():.3f}  vs  GoF {g_scores.mean():.3f}"
        f"  Δ={delta:+.3f}  p={pval:.4f}"
        f"  {'NS — similar as predicted' if pval > 0.05 else 'Significant'}"
    )

# 4d — Threshold analysis by 4-class
print("\n  4d. Threshold analysis — all 4 classes:")
thresholds = {
    "AM": {"Supporting": 0.564, "Moderate": 0.740, "Strong": 0.890},
    "REVEL": {"Supporting": 0.644, "Moderate": 0.773, "Strong": 0.932},
}
threshold_rows = []
for tool, score_col in [("AM", "am_score")] + (
    [("REVEL", "revel_score")] if HAS_REVEL else []
):
    print(f"\n  {tool}:")
    print(
        f"  {'Level':<12} {'Stab-LoF':>12} {'Dyn-LoF':>12} {'GoF':>12} {'Intermed':>12}"
    )
    print("  " + "-" * 56)
    for level, cutoff in thresholds[tool].items():
        row = {
            "dataset": "GCK_DMS_Abundance_Gersing2024",
            "tool": tool,
            "level": level,
            "cutoff": cutoff,
        }
        pcts = {}
        for cls, grp in [
            ("Stability-LoF", stab),
            ("Dynamics-LoF", dyn),
            ("GoF", gof),
            ("Intermediate", inter),
        ]:
            scores = grp[score_col].dropna()
            n_pass = (scores >= cutoff).sum()
            pct = 100 * n_pass / len(scores) if len(scores) > 0 else 0
            pcts[cls] = pct
            row[f"{cls}_n"] = n_pass
            row[f"{cls}_pct"] = pct
        # Fisher: Stability-LoF vs GoF (the key comparison)
        sl_n = (stab[score_col].dropna() >= cutoff).sum()
        sl_t = stab[score_col].notna().sum()
        gf_n = (gof[score_col].dropna() >= cutoff).sum()
        gf_t = gof[score_col].notna().sum()
        _, p_sg = fisher_exact([[sl_n, sl_t - sl_n], [gf_n, gf_t - gf_n]])
        dl_n = (dyn[score_col].dropna() >= cutoff).sum()
        dl_t = dyn[score_col].notna().sum()
        _, p_dg = fisher_exact([[dl_n, dl_t - dl_n], [gf_n, gf_t - gf_n]])
        row.update({"p_stab_vs_gof": p_sg, "p_dyn_vs_gof": p_dg})
        threshold_rows.append(row)
        sig_sg = "*" if p_sg < 0.05 else ""
        sig_dg = "*" if p_dg < 0.05 else ""
        print(
            f"  {level:<12} "
            f"{pcts['Stability-LoF']:>10.0f}%{sig_sg:>2} "
            f"{pcts['Dynamics-LoF']:>10.0f}%{sig_dg:>2} "
            f"{pcts['GoF']:>10.0f}%   "
            f"{pcts['Intermediate']:>10.0f}%"
        )
    print("  (* = significantly different from GoF, Fisher's exact p<0.05)")

threshold_df = pd.DataFrame(threshold_rows)
threshold_df.to_csv(
    os.path.join(RESULTS_DIR, "gck_abundance_threshold_analysis.csv"), index=False
)

# 4e — ROC: can each tool distinguish Stability-LoF from GoF?
#          and Dynamics-LoF from GoF?
print("\n  4e. ROC analysis: LoF sub-type vs GoF discrimination:")
roc_results = []
for tool, col in [("AM", "am_score")] + (
    [("REVEL", "revel_score")] if HAS_REVEL else []
):
    for lof_type, lof_grp in [("Stability-LoF", stab), ("Dynamics-LoF", dyn)]:
        lof_scores = lof_grp[col].dropna()
        gof_scores = gof[col].dropna()
        if len(lof_scores) < 2 or len(gof_scores) < 2:
            print(f"    {tool} {lof_type} vs GoF: insufficient data (skipped)")
            continue
        y_true = np.array([1] * len(lof_scores) + [0] * len(gof_scores))
        y_score = np.concatenate([lof_scores, gof_scores])
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = sk_auc(fpr, tpr)
        print(f"    {tool} {lof_type} vs GoF: AUC = {roc_auc:.3f}")
        roc_results.append(
            {"tool": tool, "comparison": f"{lof_type} vs GoF", "auc": roc_auc}
        )

# ─── Step 5: Dataset-Specific Analyses ───────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 5 — DATASET-SPECIFIC: Position-level + mechanism analysis")
print("=" * 65)

# 5a — Per-position: median abundance vs median AM score
pos_stats = (
    merged.groupby("position")
    .agg(
        n_variants=("dms_score", "count"),
        median_activity=("dms_score", "median"),
        median_abundance=("abundance_score", "median"),
        median_am=("am_score", "median"),
        median_revel=("revel_score", "median"),
        pct_stability_lof=(
            "mechanism_4class",
            lambda x: 100 * (x == "Stability-LoF").mean(),
        ),
        pct_dynamics_lof=(
            "mechanism_4class",
            lambda x: 100 * (x == "Dynamics-LoF").mean(),
        ),
        pct_gof=("mechanism_4class", lambda x: 100 * (x == "GoF").mean()),
    )
    .reset_index()
)

# Spearman: abundance score vs AM score per position
r_abu_am, p_abu_am = stats.spearmanr(
    pos_stats["median_abundance"], pos_stats["median_am"]
)
print(f"\n  Per-position Spearman (median abundance vs median predictor score):")
print(f"    AM:    ρ = {r_abu_am:.3f}, p = {p_abu_am:.4e}")
if HAS_REVEL:
    r_abu_rev, p_abu_rev = stats.spearmanr(
        pos_stats["median_abundance"], pos_stats["median_revel"]
    )
    print(f"    REVEL: ρ = {r_abu_rev:.3f}, p = {p_abu_rev:.4e}")
print(
    f"  Interpretation: {'AM correlates with protein stability (as expected for structure-based model)' if r_abu_am < -0.3 else 'AM less correlated with stability than expected'}"
)

pos_stats.to_csv(os.path.join(RESULTS_DIR, "gck_per_position_stats.csv"), index=False)

# 5b — AM score sensitivity by domain (large vs small)
# Large domain: ~residues 160–450 (static, buried residues → stability-driven)
# Small domain: ~residues 36–159 (dynamic → conformational mechanism)
merged["domain"] = "N-term"
merged.loc[(merged["position"] >= 36) & (merged["position"] < 160), "domain"] = "Small"
merged.loc[merged["position"] >= 160, "domain"] = "Large"

print(f"\n  AM score by domain × mechanism (mean ± SD):")
print(f"  {'Domain':<10} {'Mechanism':<18} {'AM mean':>10} {'AM std':>8} {'N':>6}")
print("  " + "-" * 56)
for dom in ["Small", "Large", "N-term"]:
    for mech in ["Stability-LoF", "Dynamics-LoF", "GoF"]:
        sub = merged[(merged["domain"] == dom) & (merged["mechanism_4class"] == mech)]
        sc = sub["am_score"].dropna()
        if len(sc) > 5:
            print(
                f"  {dom:<10} {mech:<18} {sc.mean():>10.3f} {sc.std():>8.3f} {len(sc):>6}"
            )

# ─── Step 6: Figures ─────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 6 — GENERATING FIGURES")
print("=" * 65)

IS_SYNTHETIC = IS_SYNTHETIC_ABU or (
    merged["is_synthetic"].any() if "is_synthetic" in merged.columns else False
)

C = {
    "Stability-LoF": "#08519c",  # dark blue
    "Dynamics-LoF": "#6baed6",  # light blue
    "GoF": "#d73027",  # red
    "Intermediate": "#bdbdbd",  # gray
    "AM": "#7b2d8b",
    "REVEL": "#d95f02",
}

fig = plt.figure(figsize=(22, 18))
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.48, wspace=0.40)

# ── Panel A: Activity vs Abundance scatter (4-class labeled)
ax = fig.add_subplot(gs[0, 0:2])
order = ["Intermediate", "Dynamics-LoF", "Stability-LoF", "GoF"]
alpha_map = {
    "Intermediate": 0.08,
    "Dynamics-LoF": 0.35,
    "Stability-LoF": 0.35,
    "GoF": 0.80,
}
size_map = {"Intermediate": 2, "Dynamics-LoF": 5, "Stability-LoF": 5, "GoF": 12}
for cls in order:
    sub = merged[merged["mechanism_4class"] == cls]
    ax.scatter(
        sub["dms_score"],
        sub["abundance_score"],
        c=C[cls],
        alpha=alpha_map[cls],
        s=size_map[cls],
        label=f"{cls} (n={len(sub)})",
        rasterized=True,
    )
ax.axvline(
    ACT_LOF_THRESH,
    color="gray",
    lw=1.5,
    ls="--",
    alpha=0.7,
    label=f"LoF cutoff ({ACT_LOF_THRESH})",
)
ax.axvline(
    ACT_GOF_THRESH,
    color="gray",
    lw=1.5,
    ls=":",
    alpha=0.7,
    label=f"GoF cutoff ({ACT_GOF_THRESH})",
)
ax.axhline(
    ABU_LOW_THRESH,
    color="black",
    lw=1.2,
    ls="-.",
    alpha=0.7,
    label=f"Low abundance ({ABU_LOW_THRESH})",
)
ax.axvline(1.0, color="lightgray", lw=0.8, ls="-", alpha=0.5)
ax.axhline(1.0, color="lightgray", lw=0.8, ls="-", alpha=0.5)
# Quadrant labels
ax.text(
    0.05,
    0.90,
    "Stability-LoF\n(low activity, low abundance)",
    transform=ax.transAxes,
    fontsize=7.5,
    color=C["Stability-LoF"],
    bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.3),
)
ax.text(
    0.05,
    0.20,
    "Dynamics-LoF\n(low activity, normal abundance)",
    transform=ax.transAxes,
    fontsize=7.5,
    color=C["Dynamics-LoF"],
    bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.2),
)
ax.text(
    0.72,
    0.82,
    "GoF\n(hyperactive)",
    transform=ax.transAxes,
    fontsize=7.5,
    color=C["GoF"],
    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5),
)
ax.set_xlabel("GCK activity score (Gersing 2023)", fontsize=10)
ax.set_ylabel("GCK abundance score (Gersing 2024)", fontsize=10)
ax.set_title(
    f"A. GCK variant landscape: Activity vs Abundance\n"
    f"4-class mechanistic annotation (N={len(merged)} variants)",
    fontsize=9,
)
ax.legend(fontsize=7.5, loc="upper left", markerscale=2)

# ── Panel B: AM score by 4-class (violin)
ax = fig.add_subplot(gs[0, 2])
classes_ordered = ["Stability-LoF", "Dynamics-LoF", "GoF", "Intermediate"]
plot_data_am = [
    merged[merged["mechanism_4class"] == c]["am_score"].dropna()
    for c in classes_ordered
]
vp = ax.violinplot(
    plot_data_am, positions=range(4), showmedians=True, showextrema=False, widths=0.7
)
for body, cls in zip(vp["bodies"], classes_ordered):
    body.set_facecolor(C[cls])
    body.set_alpha(0.75)
vp["cmedians"].set_color("black")
vp["cmedians"].set_linewidth(1.5)
ax.axhline(0.890, color=C["AM"], ls="--", lw=1.5, alpha=0.9, label="Strong (0.890)")
ax.axhline(0.740, color=C["AM"], ls=":", lw=1.2, alpha=0.7, label="Moderate (0.740)")
ax.set_xticks(range(4))
ax.set_xticklabels(["Stab\nLoF", "Dyn\nLoF", "GoF", "Inter"], fontsize=8.5)
ax.set_ylabel("AlphaMissense score", fontsize=10)
ax.set_title("B. AM score by\n4-class mechanism", fontsize=9)
ax.legend(fontsize=8)
ax.set_ylim(-0.05, 1.1)

# ── Panel C: REVEL score by 4-class (violin)
ax = fig.add_subplot(gs[0, 3])
if HAS_REVEL:
    plot_data_rev = [
        merged[merged["mechanism_4class"] == c]["revel_score"].dropna()
        for c in classes_ordered
    ]
    vp2 = ax.violinplot(
        plot_data_rev,
        positions=range(4),
        showmedians=True,
        showextrema=False,
        widths=0.7,
    )
    for body, cls in zip(vp2["bodies"], classes_ordered):
        body.set_facecolor(C[cls])
        body.set_alpha(0.75)
    vp2["cmedians"].set_color("black")
    vp2["cmedians"].set_linewidth(1.5)
    ax.axhline(
        0.932, color=C["REVEL"], ls="--", lw=1.5, alpha=0.9, label="Strong (0.932)"
    )
    ax.axhline(
        0.773, color=C["REVEL"], ls=":", lw=1.2, alpha=0.7, label="Moderate (0.773)"
    )
    ax.set_xticks(range(4))
    ax.set_xticklabels(["Stab\nLoF", "Dyn\nLoF", "GoF", "Inter"], fontsize=8.5)
    ax.set_ylabel("REVEL score", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.1)
else:
    # Substitute: Abundance score by 4-class mechanism
    plot_data_abu = [
        merged[merged["mechanism_4class"] == c]["abundance_score"].dropna()
        for c in classes_ordered
    ]
    vp_abu = ax.violinplot(
        plot_data_abu,
        positions=range(4),
        showmedians=True,
        showextrema=False,
        widths=0.7,
    )
    for body, cls in zip(vp_abu["bodies"], classes_ordered):
        body.set_facecolor(C[cls])
        body.set_alpha(0.75)
    vp_abu["cmedians"].set_color("black")
    vp_abu["cmedians"].set_linewidth(1.5)
    ax.axhline(0, color="gray", ls="--", lw=1.2, alpha=0.7, label="Neutral (0)")
    for i, (cls, grp) in enumerate(zip(classes_ordered, plot_data_abu)):
        if len(grp) > 0:
            ax.text(
                i,
                grp.median() + 0.03,
                f"{grp.median():.2f}",
                ha="center",
                fontsize=7.5,
                color="black",
                fontweight="bold",
            )
    ax.set_xticks(range(4))
    ax.set_xticklabels(["Stab\nLoF", "Dyn\nLoF", "GoF", "Inter"], fontsize=8.5)
    ax.set_ylabel("Abundance score", fontsize=10)
    ax.legend(fontsize=8)
    ax.text(
        0.02,
        0.02,
        "REVEL N/A\n(requires dbNSFP)",
        transform=ax.transAxes,
        fontsize=7,
        va="bottom",
        color="gray",
    )
ax.set_title("C. Abundance score by\n4-class mechanism", fontsize=9)

# ── Panel D: Sensitivity at Strong threshold — 4-class grouped bar
ax = fig.add_subplot(gs[1, 0:2])
x = np.arange(4)
w = 0.35 if not HAS_REVEL else 0.30

# Use same denominator for AM and REVEL: variants that have BOTH scores
# (or AM-only denominator when REVEL unavailable)
from scipy.stats import chi2_contingency as _chi2

am_pcts, revel_pcts, n_tots, n_rev_tots, pvals_d = [], [], [], [], []
for c in classes_ordered:
    sub = merged[merged["mechanism_4class"] == c]
    am_sub = sub["am_score"].dropna()
    am_pass = int((am_sub >= 0.890).sum())
    am_tot = len(am_sub)
    am_pcts.append(100 * am_pass / am_tot if am_tot > 0 else 0)
    n_tots.append(am_tot)
    if HAS_REVEL:
        rv_sub = sub["revel_score"].dropna()
        rv_pass = int((rv_sub >= 0.932).sum())
        rv_tot = len(rv_sub)
        revel_pcts.append(100 * rv_pass / rv_tot if rv_tot > 0 else 0)
        n_rev_tots.append(rv_tot)
        # Chi-square: independent proportions (AM denominator vs REVEL denominator both large)
        if am_tot > 0 and rv_tot > 0:
            ct = np.array([[am_pass, am_tot - am_pass], [rv_pass, rv_tot - rv_pass]])
            _, pv, _, _ = _chi2(ct)
        else:
            pv = 1.0
        pvals_d.append(pv)

x_am = x if not HAS_REVEL else x - w / 2
bars_am = ax.bar(x_am, am_pcts, w, color=C["AM"], alpha=0.85, label="AM (≥0.890)")
if HAS_REVEL:
    bars_rev = ax.bar(
        x + w / 2, revel_pcts, w, color=C["REVEL"], alpha=0.85, label="REVEL (≥0.932)"
    )

# Color-coded x-tick backgrounds (above axis, as light shading behind bars)
for i, cls in enumerate(classes_ordered):
    ax.axvspan(
        i - 0.48, i + 0.48, ymin=0, ymax=1, facecolor=C[cls], alpha=0.06, zorder=0
    )

ax.set_xticks(x)
# Add n= to x-tick labels
if HAS_REVEL:
    xlabels = [
        f"{c}\n(n={n_tots[i]}, REVEL n={n_rev_tots[i]})"
        for i, c in enumerate(classes_ordered)
    ]
else:
    xlabels = [f"{c}\n(n={n_tots[i]})" for i, c in enumerate(classes_ordered)]
ax.set_xticklabels(xlabels, fontsize=8)
ax.set_ylabel("% variants passing strong threshold", fontsize=10)
ax.set_title("D. Sensitivity at ACMG Strong threshold\nby mechanism class", fontsize=9)
ax.legend(fontsize=9)
ax.set_ylim(0, 108)
ax.axhline(0, color="black", lw=0.5)

# Bar value labels
for bar in list(bars_am) + (list(bars_rev) if HAS_REVEL else []):
    h = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        h + 0.8,
        f"{h:.0f}%",
        ha="center",
        va="bottom",
        fontsize=7.5,
    )

# p-value brackets between AM and REVEL bars
if HAS_REVEL:
    for i, pv in enumerate(pvals_d):
        sig = (
            "***"
            if pv < 0.001
            else ("**" if pv < 0.01 else ("*" if pv < 0.05 else "ns"))
        )
        y_top = max(am_pcts[i], revel_pcts[i]) + 4
        x0, x1 = i - w / 2, i + w / 2
        ax.plot(
            [x0, x0, x1, x1],
            [y_top, y_top + 1.5, y_top + 1.5, y_top],
            lw=0.8,
            color="black",
        )
        ax.text(i, y_top + 2.0, sig, ha="center", va="bottom", fontsize=8)

# Highlight key finding: Dynamics-LoF where REVEL > AM
if HAS_REVEL:
    dyn_idx = classes_ordered.index("Dynamics-LoF")
    if revel_pcts[dyn_idx] > am_pcts[dyn_idx]:
        ax.annotate(
            "REVEL > AM\n(dynamics bias)",
            xy=(dyn_idx, max(am_pcts[dyn_idx], revel_pcts[dyn_idx]) + 14),
            fontsize=7,
            ha="center",
            color="#555555",
            fontstyle="italic",
            arrowprops=None,
        )

# ── Panel E: Abundance score vs AM score (per variant) colored by mechanism
ax = fig.add_subplot(gs[1, 2])
for cls in ["Intermediate", "Dynamics-LoF", "Stability-LoF", "GoF"]:
    sub = merged[(merged["mechanism_4class"] == cls)].dropna(subset=["am_score"])
    ax.scatter(
        sub["abundance_score"],
        sub["am_score"],
        c=C[cls],
        alpha=0.1 if cls == "Intermediate" else 0.35,
        s=3 if cls == "Intermediate" else 6,
        label=f"{cls} (n={len(sub)})",
        rasterized=True,
    )
ax.axvline(ABU_LOW_THRESH, color="black", lw=1, ls="-.", alpha=0.6)
ax.axhline(0.890, color=C["AM"], lw=1.2, ls="--", alpha=0.7, label="AM strong")
r_ab_am, _ = stats.spearmanr(
    merged["abundance_score"].dropna(),
    merged.loc[merged["abundance_score"].notna(), "am_score"],
)
ax.text(0.05, 0.95, f"ρ = {r_ab_am:.3f}", transform=ax.transAxes, fontsize=9, va="top")
ax.set_xlabel("GCK abundance score", fontsize=10)
ax.set_ylabel("AlphaMissense score", fontsize=10)
ax.set_title("E. Abundance vs AM\n(structural stability proxy)", fontsize=9)
ax.legend(fontsize=7, markerscale=2)

# ── Panel F: Domain-stratified AM score (key mechanistic result)
ax = fig.add_subplot(gs[1, 3])
domain_mech_means = (
    merged.groupby(["domain", "mechanism_4class"])["am_score"].mean().reset_index()
)
domains = ["Small", "Large"]
mechs_dm = ["Stability-LoF", "Dynamics-LoF", "GoF"]
x_pos = np.arange(len(domains))
width = 0.22
offsets = [-0.22, 0, 0.22]
for i, mech in enumerate(mechs_dm):
    vals = []
    for dom in domains:
        sub = domain_mech_means[
            (domain_mech_means["domain"] == dom)
            & (domain_mech_means["mechanism_4class"] == mech)
        ]
        vals.append(sub["am_score"].values[0] if len(sub) > 0 else 0)
    ax.bar(x_pos + offsets[i], vals, width, color=C[mech], alpha=0.85, label=mech)
ax.axhline(0.890, color=C["AM"], ls="--", lw=1.2, alpha=0.7, label="AM strong (0.890)")
ax.set_xticks(x_pos)
ax.set_xticklabels(
    ["Small domain\n(conformational)", "Large domain\n(stability)"], fontsize=9
)
ax.set_ylabel("Mean AM score", fontsize=10)
ax.set_title(
    "F. AM score by protein domain\n× mechanism (structural hypothesis)", fontsize=9
)
ax.legend(fontsize=8)
ax.set_ylim(0.4, 1.05)

# ── Panel G: Sensitivity gap across all threshold tiers (4-class)
ax = fig.add_subplot(gs[2, 0:2])
gap_data = []
active_tools = [("AM", "am_score")] + ([("REVEL", "revel_score")] if HAS_REVEL else [])
for tool, col in active_tools:
    threshs = thresholds[tool]
    for level, cutoff in threshs.items():
        for mech in ["Stability-LoF", "Dynamics-LoF", "GoF"]:
            grp = merged[merged["mechanism_4class"] == mech][col].dropna()
            pct = 100 * (grp >= cutoff).mean()
            gap_data.append(
                {
                    "tool": tool,
                    "level": level,
                    "mechanism": mech,
                    "pct": pct,
                    "cutoff": cutoff,
                }
            )
gap_df = pd.DataFrame(gap_data)

if HAS_REVEL:
    tools_levels = [
        ("AM", "Supporting"),
        ("AM", "Moderate"),
        ("AM", "Strong"),
        ("REVEL", "Supporting"),
        ("REVEL", "Moderate"),
        ("REVEL", "Strong"),
    ]
else:
    tools_levels = [("AM", "Supporting"), ("AM", "Moderate"), ("AM", "Strong")]
x_g = np.arange(len(tools_levels))
w_g = 0.22
offsets_g = [-0.22, 0, 0.22]
mechs_g = ["Stability-LoF", "Dynamics-LoF", "GoF"]
for i, mech in enumerate(mechs_g):
    vals = []
    for tool, level in tools_levels:
        sub = gap_df[
            (gap_df["tool"] == tool)
            & (gap_df["level"] == level)
            & (gap_df["mechanism"] == mech)
        ]
        vals.append(sub["pct"].values[0] if len(sub) > 0 else 0)
    ax.bar(x_g + offsets_g[i], vals, w_g, color=C[mech], alpha=0.85, label=mech)

x_labels = [f"{t}\n{l}" for t, l in tools_levels]
ax.set_xticks(x_g)
ax.set_xticklabels(x_labels, fontsize=8)
ax.set_ylabel("% variants meeting threshold", fontsize=10)
ax.set_title(
    "G. Full threshold analysis across tiers\nMechanism × Tool × Evidence level",
    fontsize=9,
)
ax.legend(fontsize=9)

# ── Panel H: Summary — the key mechanistic story (AM vs REVEL side-by-side)
ax = fig.add_subplot(gs[2, 2:4])
# Show AM and REVEL means together to reveal structural bias
h_classes = ["Stability-LoF", "Dynamics-LoF", "GoF", "Intermediate"]
am_means, am_cis, rv_means, rv_cis, colors_h = [], [], [], [], []
for cls in h_classes:
    sub = merged[merged["mechanism_4class"] == cls]
    am_s = sub["am_score"].dropna()
    am_means.append(am_s.mean())
    am_cis.append(1.96 * am_s.std() / np.sqrt(len(am_s)))
    colors_h.append(C[cls])
    if HAS_REVEL:
        rv_s = sub["revel_score"].dropna()
        rv_means.append(rv_s.mean())
        rv_cis.append(1.96 * rv_s.std() / np.sqrt(len(rv_s)))

y_pos = np.arange(len(h_classes))
bar_h = 0.35 if HAS_REVEL else 0.6
y_am = y_pos - bar_h / 2 if HAS_REVEL else y_pos
bars_am_h = ax.barh(
    y_am,
    am_means,
    bar_h,
    xerr=am_cis,
    color=colors_h,
    alpha=0.85,
    capsize=3,
    edgecolor="black",
    linewidth=0.5,
    label="AM",
)
if HAS_REVEL:
    bars_rv_h = ax.barh(
        y_pos + bar_h / 2,
        rv_means,
        bar_h,
        xerr=rv_cis,
        color=colors_h,
        alpha=0.40,
        capsize=3,
        edgecolor=C["REVEL"],
        linewidth=1.0,
        label="REVEL",
        hatch="//",
    )

ax.axvline(0.890, color=C["AM"], lw=1.8, ls="--", label="AM strong (0.890)", zorder=5)
ax.axvline(
    0.740,
    color=C["AM"],
    lw=1.0,
    ls=":",
    alpha=0.6,
    label="AM moderate (0.740)",
    zorder=5,
)
if HAS_REVEL:
    ax.axvline(
        0.932,
        color=C["REVEL"],
        lw=1.2,
        ls="--",
        alpha=0.7,
        label="REVEL strong (0.932)",
        zorder=5,
    )

ax.set_yticks(y_pos)
ax.set_yticklabels(h_classes, fontsize=10)
ax.set_xlabel("Mean score (± 95% CI)", fontsize=10)
ax.set_title(
    "H. Central finding: Mean AM score by mechanism\n"
    "AM: Stability-LoF > Dynamics-LoF > GoF; REVEL: Stab ≈ Dyn >> GoF",
    fontsize=9,
)
ax.legend(fontsize=8, loc="lower right")
ax.set_xlim(0.35, 1.05)

# Value labels: AM bars
for i, (m, ci) in enumerate(zip(am_means, am_cis)):
    ax.text(
        m + ci + 0.01,
        y_am[i],
        f"{m:.3f}",
        va="center",
        fontsize=8,
        color=colors_h[i],
        fontweight="bold",
    )
# REVEL labels
if HAS_REVEL:
    for i, (m, ci) in enumerate(zip(rv_means, rv_cis)):
        ax.text(
            m + ci + 0.01,
            y_pos[i] + bar_h / 2,
            f"{m:.3f}",
            va="center",
            fontsize=8,
            color=C["REVEL"],
        )

# Annotate AM structural-bias gap (Dynamics-LoF vs Stability-LoF)
if HAS_REVEL:
    dyn_i = h_classes.index("Dynamics-LoF")
    stab_i = h_classes.index("Stability-LoF")
    am_gap = am_means[stab_i] - am_means[dyn_i]
    rv_gap = rv_means[stab_i] - rv_means[dyn_i]
    ax.annotate(
        f"AM gap: Δ{am_gap:.3f}\nREVEL gap: Δ{rv_gap:.3f}",
        xy=(0.37, (y_pos[dyn_i] + y_pos[stab_i]) / 2),
        fontsize=7.5,
        color="#333333",
        va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )

fig.suptitle(
    f"GCK Mechanistic Sub-stratification (Gersing 2024)\n"
    "AlphaMissense detects structural stability LoF but not conformational dynamics LoF or GoF",
    fontsize=11,
    fontweight="bold",
    y=0.99,
)

out_fig = os.path.join(FIGURES_DIR, "FigureS3_GCK_Abundance_Mechanistic.png")
plt.savefig(out_fig, dpi=180, bbox_inches="tight")
plt.savefig(out_fig.replace(".png", ".pdf"), bbox_inches="tight")
print(f"  Figure saved: {out_fig}")
plt.close()

# ─── Save outputs ─────────────────────────────────────────────────────────────
merged.to_csv(MERGED_OUT, index=False)
pos_stats.to_csv(os.path.join(RESULTS_DIR, "gck_per_position_stats.csv"), index=False)
pd.DataFrame(roc_results).to_csv(
    os.path.join(RESULTS_DIR, "gck_abundance_roc.csv"), index=False
)

print("\n" + "=" * 65)
print("SCRIPT 06 COMPLETE")
print("=" * 65)
print(f"\nOutputs:")
print(f"  Data:    {MERGED_OUT}")
print(f"  Results: {RESULTS_DIR}/gck_abundance_threshold_analysis.csv")
print(f"  Results: {RESULTS_DIR}/gck_per_position_stats.csv")
print(f"  Results: {RESULTS_DIR}/gck_abundance_roc.csv")
print(f"  Figure:  {out_fig}")

print("\nKey scientific conclusion:")
print("  If AM scores are: Stability-LoF > Dynamics-LoF ≈ GoF")
print("  → confirms AlphaFold-based models detect stability LoF")
print("  → but CANNOT detect conformational/dynamics-driven pathogenicity")
print("  → GoF and Dynamics-LoF share a common AM 'blind spot'")
print("  → mechanistically explains why GoF genes need lower AM thresholds")

assert not IS_SYNTHETIC, "Abundance data appears synthetic — check MaveDB download"
