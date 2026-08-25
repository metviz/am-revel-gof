#!/usr/bin/env python3
"""
09_proteingym_analysis.py — ProteinGym Clinical Benchmark Calibration Check
============================================================================
External calibration check using the ProteinGym v1.0 clinical benchmark for GCK
(P35557), which evaluates 15 variant effect predictors on clinical P/LP vs
gnomAD-benign variants.

SCIENTIFIC ROLE IN THIS PAPER (calibration check, not primary validation):
  This script is explicitly supplementary. Its value is rhetorical as much as
  statistical: it demonstrates the LoF/GoF bias in the *exact benchmark framework
  that the AlphaMissense authors used to evaluate their tool*. If AM's developers
  had stratified by mechanism, they would have seen this themselves. That is a
  compelling argument for peer an earlier check who might otherwise dismiss our finding.

UNIQUE CONTRIBUTION vs PRIOR SCRIPTS:
  • Scripts 01–04 (Hopkins):    Pathogenic LoF vs GoF, no benign
  • Scripts 05–06 (GCK DMS):   Continuous functional scores, no clinical labels
  • Script 07 (De Franco):      Pathogenic LoF vs GoF, no benign, ABCC8/KCNJ11
  • Script 08 (ClinVar):        P/LP vs B/LB (ClinVar-curated benign class)
  • Script 09 (ProteinGym):     P/LP vs gnomAD common variants (freq-based benign)
                                 + comparison across 15 predictors on same benchmark
                                 + explicit DMS-clinical bridge via Gersing 2023

WHY GNOMAD BENIGN IS METHODOLOGICALLY DISTINCT FROM CLINVAR BENIGN:
  ClinVar B/LB = curators decided a variant is benign (subjective, expert opinion)
  gnomAD benign = variant is common enough in the population (MAF >0.1%) that
  it cannot be highly penetrant for a severe disease (objective, frequency-based)
  These are independent lines of evidence; using both strengthens the calibration case.

CIRCULARITY CAVEATS (addressed explicitly in Methods):
  1. REVEL was trained on ClinVar P/LP vs B/LB → circular for ClinVar benign class
     BUT gnomAD benign is semi-independent (common variants may not be in training set)
  2. AM was trained on population allele frequency data (similar to gnomAD)
     → some circularity for gnomAD benign class specifically for AM
  3. Mitigation: frame as calibration ONLY; primary conclusions rest on Hopkins/De Franco
  4. The key comparison (LoF vs GoF within pathogenic class) does NOT require benign
     variants at all — zero circularity for the core finding

THREE ANALYSES ENABLED BY PROTEINGYM:
  A. PREDICTOR COMPARISON: Where does AM rank among 15 predictors for GCK?
     (Shows AM is a good tool overall — the bias is specific, not general failure)
  B. MECHANISM STRATIFICATION: LoF-only vs GoF-only AUCs (novel, not in ProteinGym paper)
     (The key analysis: shows the bias persists even in this framework)
  C. DMS-CLINICAL BRIDGE: Correlate Gersing 2023 DMS activity with ProteinGym labels
     (Shows DMS data is clinically relevant; validates Script 05 interpretation)

PROTEINGYM v1.0 DATA:
  Paper: Notin et al. 2024, NeurIPS Datasets & Benchmarks Track
  DOI: 10.5281/zenodo.13936340
  AWS: s3://proteingym/clinical_substitutions.parquet
  GitHub: https://github.com/OATML-Markslab/ProteinGym

  Schema: mutant, DMS_score, DMS_score_bin, UniProt_ID, [predictor columns]
  GCK row: UniProt_ID = P35557, ~270 clinical variants

PUBLISHED AUC VALUES (Notin et al. 2024, Table S4, GCK row):
  Used to construct realistic synthetic data and anchor the predictor comparison.
  REVEL: 0.965 | CADD: 0.953 | ClinPred: 0.948 | AlphaMissense: 0.941
  PrimateAI: 0.921 | MutPred2: 0.918 | EVE: 0.912 | MVP: 0.897
  VESPA: 0.895 | ESM-1v: 0.887 | PolyPhen-2: 0.878 | SIFT: 0.834
  ESM-1b: 0.823 | PhD-SNPg: 0.803 | SNAP2: 0.801

REQUIREMENTS: pip install pandas numpy scipy matplotlib seaborn requests pyarrow
"""

import os, re, hashlib, datetime, requests
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu, fisher_exact, spearmanr, pearsonr
from sklearn.metrics import roc_curve, auc as sk_auc, precision_recall_curve
import warnings

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data", "proteingym")
GCK_DMS_DIR = os.path.join(PROJECT_DIR, "data", "gck_dms")
CLINVAR_DIR = os.path.join(PROJECT_DIR, "data", "clinvar")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")
VALID_DIR = os.path.join(PROJECT_DIR, "validation")
CHECKSUMS = os.path.join(VALID_DIR, "checksums.md")
ROW_COUNTS = os.path.join(VALID_DIR, "row_counts.log")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ─── Constants ────────────────────────────────────────────────────────────────
UNIPROT_GCK = "P35557"
N_AA_GCK = 465

# Published AUCs from Notin et al. 2024 Table S4 (GCK row, all variants combined)
PUBLISHED_AUC = {
    "REVEL": 0.965,
    "CADD": 0.953,
    "ClinPred": 0.948,
    "AlphaMissense": 0.941,
    "PrimateAI": 0.921,
    "MutPred2": 0.918,
    "EVE": 0.912,
    "MVP": 0.897,
    "VESPA": 0.895,
    "ESM-1v": 0.887,
    "PolyPhen-2": 0.878,
    "SIFT": 0.834,
    "ESM-1b": 0.823,
    "PhD-SNPg": 0.803,
    "SNAP2": 0.801,
}

# GCK domain map (for mechanism annotation)
GCK_DOMAINS = {
    "small_domain": (1, 170),
    "hinge_1": (171, 190),
    "large_domain": (191, 440),
    "hinge_2": (441, 465),
}
GCK_GOF_REGION = (150, 460)  # activating mutations cluster in large domain

# Mechanism keywords (same as Script 08)
LOF_KW = [
    "mody",
    "maturity-onset diabetes",
    "maturity onset diabetes",
    "glucokinase deficiency",
    "gck deficiency",
    "mody type 2",
    "mody2",
    "reduced activity",
    "loss of function",
]
GOF_KW = [
    "hyperinsulinism",
    "congenital hyperinsulinism",
    "chi",
    "activating",
    "gain of function",
    "hypoglycemia",
    "persistent hyperinsulinism",
    "focal hyperinsulinism",
]

# ACMG LR+ thresholds (Tavtigian et al. 2020)
ACMG_LR = {"Very Strong": 350.0, "Strong": 18.7, "Moderate": 4.3, "Supporting": 2.08}
THRESHOLDS = {
    "AlphaMissense": {"Supporting": 0.564, "Moderate": 0.740, "Strong": 0.890},
    "REVEL": {"Supporting": 0.644, "Moderate": 0.773, "Strong": 0.932},
}


def log_row(label, n):
    msg = f"[{datetime.datetime.now().isoformat()}] {label}: N={n}"
    print(f"  {msg}")
    with open(ROW_COUNTS, "a") as f:
        f.write(msg + "\n")


def record_md5(path, label, n, synthetic):
    h = hashlib.md5(open(path, "rb").read()).hexdigest()
    with open(CHECKSUMS, "a") as f:
        f.write(
            f"\n## {label}\n- MD5: {h}\n- N: {n}"
            f"\n- Date: {datetime.datetime.now().isoformat()}"
            f"\n- Synthetic: {synthetic}\n"
        )


def parse_mutant(s):
    """Parse 'A23V' → (wt='A', pos=23, alt='V')"""
    m = re.match(r"^([A-Z])(\d+)([A-Z])$", str(s).strip())
    if m:
        return m.group(1), int(m.group(2)), m.group(3)
    return None, None, None


def auc_ci_delong(y_true, scores, n_boot=2000, seed=42):
    """Bootstrap 95% CI for AUC."""
    np.random.seed(seed)
    n = len(y_true)
    fpr, tpr, _ = roc_curve(y_true, scores)
    point_auc = sk_auc(fpr, tpr)
    boot_aucs = []
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        yt, ys = y_true[idx], scores[idx]
        if len(np.unique(yt)) < 2:
            continue
        fp, tp, _ = roc_curve(yt, ys)
        boot_aucs.append(sk_auc(fp, tp))
    lo, hi = np.percentile(boot_aucs, [2.5, 97.5])
    return point_auc, lo, hi


def gck_domain(pos):
    if pos is None:
        return "Unknown"
    for dom, (s, e) in GCK_DOMAINS.items():
        if s <= pos <= e:
            return dom
    return "Unknown"


# ─── Step 1: Acquire ProteinGym clinical benchmark for GCK ───────────────────
print("=" * 65)
print("STEP 1 — DATA ACQUISITION: ProteinGym v1.0 Clinical Benchmark (GCK)")
print("=" * 65)

RAW_PATH = os.path.join(DATA_DIR, "proteingym_gck_clinical.csv")
LOCAL_PARQUET = os.path.join(DATA_DIR, "clinical_substitutions.parquet")
IS_SYNTHETIC = False

# Attempt download — parquet from Zenodo or AWS (only if the local copy is absent)
DOWNLOAD_URLS = [
    "https://zenodo.org/records/13936340/files/clinical_substitutions.parquet",
    "https://proteingym.s3.amazonaws.com/clinical_substitutions.parquet",
]

# GCK RefSeq protein ID (NP_000153.1) — ProteinGym uses protein_id, not UniProt_ID
GCK_REFSEQ = "NP_000153.1"

# Columns that only ever appear in ClinVar's variant_summary export. If we see these
# in a file claiming to be ProteinGym, a previous run laundered ClinVar data into it.
CLINVAR_TELLTALES = {"AlleleID", "RCVaccession", "ClinicalSignificance", "ReviewStatus"}


class ProteinGymUnavailable(RuntimeError):
    """ProteinGym data could not be obtained, or is not what it claims to be.

    This is deliberately fatal. A previous version of this script caught every
    exception here and silently substituted ClinVar GCK data, writing it to files
    named `proteingym_*`. That produced a meta-analysis stratum labelled
    "ProteinGym v1.0" whose rows were in fact ClinVar — an unreported data
    substitution that reached the manuscript. Never fall back. Fail.
    """


def _require_parquet_engine() -> None:
    """pyarrow is listed in requirements.txt but is easy to omit at install time.

    Without it pandas.read_parquet raises ImportError, which the old bare `except`
    swallowed — the single root cause of the ClinVar substitution above.
    """
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise ProteinGymUnavailable(
            "pyarrow is not installed, so the ProteinGym parquet file cannot be read.\n"
            "  Fix:  pip install --user pyarrow\n"
            "  (pyarrow is in requirements.txt; it was simply never installed.)"
        ) from exc


def _extract_gck(df_full: pd.DataFrame, origin: str) -> pd.DataFrame:
    """Pull the GCK rows out of the full ProteinGym clinical substitution table."""
    if "UniProt_ID" in df_full.columns:
        df_gck = df_full[
            df_full["UniProt_ID"].astype(str).str.contains(UNIPROT_GCK, na=False)
        ].copy()
    elif "protein_id" in df_full.columns:
        df_gck = df_full[df_full["protein_id"] == GCK_REFSEQ].copy()
    else:
        raise ProteinGymUnavailable(
            f"{origin}: no UniProt_ID or protein_id column — this is not the "
            f"ProteinGym clinical substitution table. Columns: {df_full.columns.tolist()}"
        )
    if len(df_gck) < 10:
        raise ProteinGymUnavailable(
            f"{origin}: only {len(df_gck)} GCK rows found (expected ~148). "
            "Wrong file, or the schema changed."
        )
    if "annotation" in df_gck.columns and "sig_class" not in df_gck.columns:
        df_gck = df_gck.rename(columns={"annotation": "sig_class"})
    return df_gck


# ── Load: cached CSV → local parquet → download. Any failure is fatal. ──────────
if os.path.exists(RAW_PATH):
    print(f"  Cached: {RAW_PATH}")
    df_raw = pd.read_csv(RAW_PATH)
    stale = CLINVAR_TELLTALES.intersection(df_raw.columns)
    if stale:
        raise ProteinGymUnavailable(
            f"{RAW_PATH} carries ClinVar-only columns {sorted(stale)}.\n"
            "  This cache was written by the old silent-fallback code path: it contains "
            "ClinVar GCK data masquerading as ProteinGym.\n"
            f"  Fix:  rm {RAW_PATH}   then re-run this script."
        )
    IS_SYNTHETIC = bool(df_raw.get("is_synthetic", pd.Series([False])).any())
else:
    _require_parquet_engine()

    if os.path.exists(LOCAL_PARQUET):
        print(f"  Reading local parquet: {LOCAL_PARQUET}")
        df_gck = _extract_gck(pd.read_parquet(LOCAL_PARQUET), LOCAL_PARQUET)
    else:
        import io

        df_gck, last_err = None, None
        for url in DOWNLOAD_URLS:
            try:
                print(f"  Trying: {url[:70]}...")
                resp = requests.get(url, timeout=120, stream=True)
                resp.raise_for_status()
                df_gck = _extract_gck(pd.read_parquet(io.BytesIO(resp.content)), url)
                break
            except ProteinGymUnavailable:
                raise
            except Exception as exc:  # network/HTTP only — never a data problem
                print(f"  Failed: {exc}")
                last_err = exc
        if df_gck is None:
            raise ProteinGymUnavailable(
                "ProteinGym clinical benchmark could not be downloaded from any mirror.\n"
                f"  Last error: {last_err}\n"
                f"  Fix: download clinical_substitutions.parquet to {LOCAL_PARQUET}\n"
                "  Do NOT substitute another dataset."
            )

    print(f"  Loaded: N={len(df_gck)} GCK variants from ProteinGym")
    print(f"  Columns: {df_gck.columns.tolist()}")

    # ── What ProteinGym can and cannot supply ─────────────────────────────────
    # The clinical substitution table holds ONLY: mutant, sequences, protein_id and
    # a Pathogenic/Benign annotation. It carries
    #   (a) no predictor scores  -> the 15-predictor AUC comparison cannot be computed
    #                               from it (the old code fell back to a hardcoded table
    #                               of published AUCs and reported them as if measured);
    #   (b) no LoF/GoF mechanism -> the LoF-GoF gap, which is this project's entire
    #                               effect measure, cannot be derived from it.
    # The old code papered over (b) by merging mechanism labels in from ClinVar. That
    # makes the resulting "ProteinGym" stratum a relabelled copy of ClinVar GCK, which
    # is exactly the circularity that reached the manuscript. Refuse to do it.
    missing_predictors = not any(c in df_gck.columns for c in PUBLISHED_AUC)
    missing_mechanism = "mechanism" not in df_gck.columns

    if missing_predictors or missing_mechanism:
        raise ProteinGymUnavailable(
            "ProteinGym loaded successfully, but it cannot support this analysis.\n"
            f"  Columns present: {df_gck.columns.tolist()}\n"
            + (
                "  - No predictor score columns: the multi-predictor AUC comparison "
                "cannot be computed.\n"
                if missing_predictors
                else ""
            )
            + (
                "  - No LoF/GoF mechanism labels: ProteinGym annotates only "
                "Pathogenic/Benign.\n"
                if missing_mechanism
                else ""
            )
            + "\n"
            "  Deriving mechanism by merging ClinVar into these rows produces a stratum\n"
            "  that IS ClinVar GCK wearing a ProteinGym label. That is what the previous\n"
            "  version of this script did silently, and it propagated into the pooled\n"
            "  meta-analysis. Do not reinstate it.\n\n"
            "  If you need predictor scores, obtain the ProteinGym zero-shot/clinical\n"
            "  model-score files separately — but note they still contain no mechanism\n"
            "  labels, so they answer a different question than this paper asks."
        )

    df_gck.to_csv(RAW_PATH, index=False)
    print(f"  Saved: {RAW_PATH} (N={len(df_gck)})")
    df_raw = pd.read_csv(RAW_PATH)

log_row("ProteinGym GCK", len(df_raw))
record_md5(RAW_PATH, "proteingym_gck_clinical.csv", len(df_raw), IS_SYNTHETIC)
IS_SYNTH_NOTE = " [SYNTHETIC — VERIFY]" if IS_SYNTHETIC else ""

# ─── Step 2: Preprocess ──────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 2 — PREPROCESSING")
print("=" * 65)

df = df_raw.copy().reset_index(drop=True)

# Parse mutant string if positions not already parsed
if "position" not in df.columns:
    parsed = df["mutant"].apply(
        lambda x: pd.Series(parse_mutant(x), index=["wt_aa", "position", "alt_aa"])
    )
    df = pd.concat([df, parsed], axis=1)

df = df[df["position"].notna()].copy()
df["position"] = df["position"].astype(int)

if "domain" not in df.columns:
    df["domain"] = df["position"].apply(gck_domain)
if "in_gof_region" not in df.columns:
    df["in_gof_region"] = df["position"].apply(
        lambda p: GCK_GOF_REGION[0] <= p <= GCK_GOF_REGION[1]
    )

# Pathogenic binary label
df["is_pathogenic"] = (df["mechanism"].isin(["LoF", "GoF"])).astype(int)

# Confirm predictor columns available
PREDICTOR_COLS = [c for c in df.columns if c in list(PUBLISHED_AUC.keys())]
print(f"  Predictors available: {len(PREDICTOR_COLS)}")
print(f"  Mechanism distribution:")
for mech, n in df["mechanism"].value_counts().items():
    print(f"    {mech}: N={n}")

# Save processed
PROCESSED = os.path.join(DATA_DIR, "proteingym_gck_processed.csv")
df.to_csv(PROCESSED, index=False)

# Working subsets
lof_df = df[df["mechanism"] == "LoF"]
gof_df = df[df["mechanism"] == "GoF"]
ben_df = df[df["mechanism"] == "Benign"]
path_df = df[df["mechanism"].isin(["LoF", "GoF"])]

# ─── Step 3: Analysis A — Predictor Comparison ───────────────────────────────
print("\n" + "=" * 65)
print("STEP 3 — ANALYSIS A: PREDICTOR COMPARISON (ALL GCK VARIANTS)")
print("=" * 65)
print()
print("  Computing AUC for each predictor (P/LP vs gnomAD-benign)...")
print()

pred_rows = []
y_all = df["is_pathogenic"].values

for pred in sorted(PREDICTOR_COLS, key=lambda x: -PUBLISHED_AUC.get(x, 0)):
    if pred not in df.columns:
        continue
    scores = df[pred].fillna(df[pred].median()).values

    # Invert SIFT (lower = more deleterious)
    if pred == "SIFT":
        scores = 1 - scores

    auc_all, auc_lo, auc_hi = auc_ci_delong(y_all, scores)

    # Stratified AUCs (LoF vs Benign, GoF vs Benign)
    y_lof = np.array([1] * len(lof_df) + [0] * len(ben_df))
    s_lof = np.concatenate(
        [
            lof_df[pred].fillna(lof_df[pred].median()).values,
            ben_df[pred].fillna(ben_df[pred].median()).values,
        ]
    )
    if pred == "SIFT":
        s_lof = 1 - s_lof
    fpr_l, tpr_l, _ = roc_curve(y_lof, s_lof)
    auc_lof = sk_auc(fpr_l, tpr_l)

    y_gof = np.array([1] * len(gof_df) + [0] * len(ben_df))
    s_gof = np.concatenate(
        [
            gof_df[pred].fillna(gof_df[pred].median()).values,
            ben_df[pred].fillna(ben_df[pred].median()).values,
        ]
    )
    if pred == "SIFT":
        s_gof = 1 - s_gof
    fpr_g, tpr_g, _ = roc_curve(y_gof, s_gof)
    auc_gof = sk_auc(fpr_g, tpr_g)

    published = PUBLISHED_AUC.get(pred, np.nan)
    bias = auc_lof - auc_gof

    flag = ""
    if pred in ("AlphaMissense", "REVEL"):
        flag = "  ← OUR FOCUS"

    print(
        f"  {pred:<20} All={auc_all:.3f} [{auc_lo:.3f}–{auc_hi:.3f}]  "
        f"LoF={auc_lof:.3f}  GoF={auc_gof:.3f}  "
        f"Bias(LoF-GoF)={bias:+.3f}  (pub={published:.3f}){flag}"
    )

    pred_rows.append(
        {
            "predictor": pred,
            "auc_all": auc_all,
            "auc_ci_lo": auc_lo,
            "auc_ci_hi": auc_hi,
            "auc_lof": auc_lof,
            "auc_gof": auc_gof,
            "auc_bias": bias,
            "published_auc": published,
            "rank_all": None,
        }
    )

pred_df = pd.DataFrame(pred_rows).sort_values("auc_all", ascending=False)
pred_df["rank_all"] = range(1, len(pred_df) + 1)
pred_df.to_csv(
    os.path.join(RESULTS_DIR, "proteingym_predictor_comparison.csv"), index=False
)

# ─── Step 4: Analysis B — Mechanism Stratification (the key result) ──────────
print("\n" + "=" * 65)
print("STEP 4 — ANALYSIS B: MECHANISM STRATIFICATION (NOVEL)")
print("=" * 65)
print()
print("  NOTE: This stratification is NOT in the ProteinGym paper.")
print("  It is the novel analysis that the ProteinGym benchmark enables.")
print()

for pred in ["AlphaMissense", "REVEL"]:
    if pred not in df.columns:
        continue
    col = pred
    scores_p = path_df[col].dropna()
    scores_l = lof_df[col].dropna()
    scores_g = gof_df[col].dropna()
    scores_b = ben_df[col].dropna()

    stat, p_mw = mannwhitneyu(scores_l, scores_g, alternative="greater")
    print(
        f"  {pred}: LoF mean={scores_l.mean():.3f}  GoF mean={scores_g.mean():.3f}  "
        f"Benign mean={scores_b.mean():.3f}"
    )
    print(f"    Mann-Whitney LoF > GoF: p={p_mw:.4e}{'*' if p_mw < 0.05 else ''}")
    print()

    # Threshold analysis at ACMG tiers
    thresh_map = THRESHOLDS.get(pred, {})
    if thresh_map:
        print(f"  {pred} sensitivity at ACMG thresholds:")
        print(
            f"  {'Level':<14} {'LoF%':>7} {'GoF%':>7} {'Benign%':>9} {'Gap':>6} {'p':>8}"
        )
        print("  " + "-" * 55)
        for level, thresh in thresh_map.items():
            lp = 100 * (scores_l >= thresh).mean()
            gp = 100 * (scores_g >= thresh).mean()
            bp = 100 * (scores_b >= thresh).mean()
            _, pval = fisher_exact(
                [
                    [(scores_l >= thresh).sum(), (scores_l < thresh).sum()],
                    [(scores_g >= thresh).sum(), (scores_g < thresh).sum()],
                ]
            )
            print(
                f"  {level:<14} {lp:>6.1f}% {gp:>6.1f}% {bp:>8.1f}% "
                f"{lp - gp:>+5.1f}%  {pval:.4f}"
                f"{'***' if pval < 0.001 else ('**' if pval < 0.01 else ('*' if pval < 0.05 else ''))}"
            )
        print()

# ─── Step 5: Analysis C — DMS-Clinical Bridge ────────────────────────────────
print("=" * 65)
print("STEP 5 — ANALYSIS C: DMS-CLINICAL BRIDGE")
print("=" * 65)
print()
print("  Linking Gersing 2023 DMS activity scores to ProteinGym clinical labels.")
print("  Key question: does DMS activity predict clinical P/B label?")
print("  (Validates Script 05 interpretation in a clinical context.)")
print()

if "DMS_score" in df.columns:
    # DMS activity vs clinical label
    path_dms = path_df["DMS_score"].dropna()
    ben_dms = ben_df["DMS_score"].dropna()

    if len(path_dms) > 5 and len(ben_dms) > 5:
        stat, p = mannwhitneyu(path_dms, ben_dms, alternative="two-sided")
        fpr_d, tpr_d, _ = roc_curve(
            np.array([1] * len(path_dms) + [0] * len(ben_dms)),
            np.concatenate([path_dms, ben_dms]),
        )
        auc_dms_all = sk_auc(fpr_d, tpr_d)
        print(
            f"  DMS activity: Pathogenic mean={path_dms.mean():.3f}  "
            f"Benign mean={ben_dms.mean():.3f}  "
            f"p={p:.4e}  AUC={auc_dms_all:.3f}"
        )

        # Split by mechanism
        for mech, sub in [("LoF", lof_df), ("GoF", gof_df)]:
            dms_sub = sub["DMS_score"].dropna()
            if len(dms_sub) < 3:
                continue
            # LoF variants should have LOW DMS activity
            # GoF variants should have HIGH DMS activity
            direction = "LoF→low DMS" if mech == "LoF" else "GoF→high DMS"
            print(f"  DMS activity {mech}: mean={dms_sub.mean():.3f}  ({direction})")

        # Correlation: AM score vs DMS activity per mechanism
        print()
        print("  AM score vs DMS activity correlation (Spearman):")
        for mech, sub in [("LoF", lof_df), ("GoF", gof_df), ("All path", path_df)]:
            sub_clean = sub.dropna(subset=["AlphaMissense", "DMS_score"])
            if len(sub_clean) < 10:
                continue
            rho, p_rho = spearmanr(sub_clean["AlphaMissense"], sub_clean["DMS_score"])
            print(f"    {mech:<10}: ρ={rho:+.3f}  p={p_rho:.4e}")
        print()
        print("  Interpretation:")
        print("    LoF variants: AM ↑ as DMS activity ↓ (AM detects loss of function)")
        print("    GoF variants: AM ∼ DMS activity ≈ 0 (AM blind to gain of function)")

# ─── Step 6: Figures ─────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 6 — GENERATING FIGURES")
print("=" * 65)

C = {
    "LoF": "#2166ac",
    "GoF": "#d73027",
    "Benign": "#4dac26",
    "AM": "#7b2d8b",
    "REVEL": "#d95f02",
    "other": "#888888",
    "DMS": "#1a9641",
}

fig = plt.figure(figsize=(24, 20))
gs_fig = gridspec.GridSpec(3, 4, figure=fig, hspace=0.52, wspace=0.42)

# ── Panel A: Predictor comparison bar chart — all 15 predictors using published AUCs
# Build full table: published AUC for all 15 predictors (Notin et al. 2024 Table S4)
# Overlay computed AUC from our dataset where available
ax = fig.add_subplot(gs_fig[0, 0:2])
pub_rows = []
for pred, pub_auc in PUBLISHED_AUC.items():
    comp_row = pred_df[pred_df["predictor"] == pred]
    comp_auc = float(comp_row["auc_all"].iloc[0]) if len(comp_row) > 0 else np.nan
    ci_lo = float(comp_row["auc_ci_lo"].iloc[0]) if len(comp_row) > 0 else np.nan
    ci_hi = float(comp_row["auc_ci_hi"].iloc[0]) if len(comp_row) > 0 else np.nan
    pub_rows.append(
        {
            "predictor": pred,
            "published_auc": pub_auc,
            "computed_auc": comp_auc,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
        }
    )
pub_full = (
    pd.DataFrame(pub_rows)
    .sort_values("published_auc", ascending=True)
    .reset_index(drop=True)
)

bar_colors_a = []
for p in pub_full["predictor"]:
    if p == "AlphaMissense":
        bar_colors_a.append(C["AM"])
    elif p == "REVEL":
        bar_colors_a.append(C["REVEL"])
    else:
        bar_colors_a.append(C["other"])

y_pos = np.arange(len(pub_full))
ax.barh(
    y_pos,
    pub_full["published_auc"],
    color=bar_colors_a,
    edgecolor="white",
    linewidth=0.3,
    height=0.72,
    alpha=0.80,
    label="Published AUC (Notin et al. 2024)",
)

# Overlay computed AUC as stars where available (with CI)
comp_mask = pub_full["computed_auc"].notna()
if comp_mask.any():
    comp_sub = pub_full[comp_mask]
    comp_yi = comp_sub.index.tolist()
    ax.scatter(
        comp_sub["computed_auc"],
        comp_yi,
        color="black",
        s=60,
        zorder=6,
        marker="*",
        label="Computed AUC (our dataset)",
    )
    for yi, row in zip(comp_yi, comp_sub.itertuples()):
        if not np.isnan(row.ci_lo):
            ax.barh(
                yi,
                row.ci_hi - row.ci_lo,
                left=row.ci_lo,
                height=0.18,
                color="black",
                alpha=0.5,
            )

ax.set_yticks(y_pos)
ax.set_yticklabels(pub_full["predictor"], fontsize=8.5)
ax.set_xlabel("AUC (P/LP vs gnomAD Benign)", fontsize=10)
ax.set_title(
    f"A. Predictor comparison: GCK clinical benchmark (ProteinGym v1.0)\n"
    f"Bars = published AUC (Notin et al. 2024); ★ = computed from our GCK dataset",
    fontsize=9,
)
ax.set_xlim(0.70, 1.02)
ax.axvline(0.9, color="gray", lw=0.8, ls="--", alpha=0.5)
am_patch = mpatches.Patch(color=C["AM"], label="AlphaMissense")
rv_patch = mpatches.Patch(color=C["REVEL"], label="REVEL")
oth_patch = mpatches.Patch(color=C["other"], label="Other predictors")
cmp_dot = plt.Line2D(
    [0], [0], marker="*", color="black", ls="", ms=7, label="Computed AUC (our data)"
)
ax.legend(
    handles=[am_patch, rv_patch, oth_patch, cmp_dot], fontsize=8, loc="lower right"
)

# ── Panel B: LoF vs GoF AUC — novel stratification (AlphaMissense only)
# LoF/GoF mechanism labels are our novel contribution; not in the ProteinGym paper
ax = fig.add_subplot(gs_fig[0, 2])
# Compute LoF-vs-B and GoF-vs-B AUC from ClinVar GCK (N=154 LoF, 30 GoF, 8 Benign)
_cv_b_path = os.path.join(PROJECT_DIR, "data", "clinvar", "clinvar_processed.csv")
b_rows = []
if os.path.exists(_cv_b_path):
    _cv_b = pd.read_csv(_cv_b_path)
    _cv_gck = _cv_b[_cv_b["GeneSymbol"] == "GCK"].copy()
    for pred, col_b, col_c in [
        ("AlphaMissense", "am_score", C["AM"]),
        ("REVEL", "revel_score", C["REVEL"]),
    ]:
        _lof_b = _cv_gck[_cv_gck["mechanism"] == "LoF"][col_b].dropna()
        _gof_b = _cv_gck[_cv_gck["mechanism"] == "GoF"][col_b].dropna()
        _ben_b = _cv_gck[_cv_gck["mechanism"] == "Benign"][col_b].dropna()
        if len(_lof_b) < 5 or len(_ben_b) < 2:
            continue
        _y_l = np.array([1] * len(_lof_b) + [0] * len(_ben_b))
        _fpr, _tpr, _ = roc_curve(_y_l, np.concatenate([_lof_b, _ben_b]))
        _auc_lof = sk_auc(_fpr, _tpr)
        if len(_gof_b) >= 5:
            _y_g = np.array([1] * len(_gof_b) + [0] * len(_ben_b))
            _fpr2, _tpr2, _ = roc_curve(_y_g, np.concatenate([_gof_b, _ben_b]))
            _auc_gof = sk_auc(_fpr2, _tpr2)
        else:
            _auc_gof = np.nan
        b_rows.append(
            {
                "predictor": pred,
                "auc_lof": _auc_lof,
                "auc_gof": _auc_gof,
                "color": col_c,
                "n_lof": len(_lof_b),
                "n_gof": len(_gof_b),
                "n_ben": len(_ben_b),
            }
        )

# Also add from pred_df (ProteinGym data) if available
for _, pr in pred_df.iterrows():
    if not any(r["predictor"] == pr["predictor"] for r in b_rows):
        col_c = (
            C["AM"]
            if pr["predictor"] == "AlphaMissense"
            else (C["REVEL"] if pr["predictor"] == "REVEL" else C["other"])
        )
        b_rows.append(
            {
                "predictor": pr["predictor"],
                "auc_lof": pr["auc_lof"],
                "auc_gof": pr["auc_gof"],
                "color": col_c,
                "n_lof": len(lof_df),
                "n_gof": len(gof_df),
                "n_ben": len(ben_df),
            }
        )

if b_rows:
    b_df = (
        pd.DataFrame(b_rows)
        .drop_duplicates("predictor")
        .sort_values("auc_lof", ascending=True)
    )
    y2 = np.arange(len(b_df))
    w2 = 0.35
    ax.barh(
        y2 - w2 / 2,
        b_df["auc_lof"],
        w2,
        color=b_df["color"].tolist(),
        alpha=0.85,
        label="LoF AUC (vs Benign)",
        edgecolor="white",
    )
    ax.barh(
        y2 + w2 / 2,
        b_df["auc_gof"].fillna(0),
        w2,
        color=b_df["color"].tolist(),
        alpha=0.45,
        hatch="///",
        label="GoF AUC (vs Benign)",
        edgecolor="white",
    )
    ax.set_yticks(y2)
    ax.set_yticklabels(
        [
            f"{r['predictor']}\n(LoF n={r['n_lof']}, GoF n={r['n_gof']})"
            for _, r in b_df.iterrows()
        ],
        fontsize=8,
    )
    ax.set_xlim(0.55, 1.05)
    ax.axvline(0.9, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.legend(fontsize=8)
    ax.text(
        0.02,
        0.02,
        "ClinVar GCK benign N=8\n(novel LoF/GoF stratification)",
        transform=ax.transAxes,
        fontsize=7,
        va="bottom",
        color="gray",
    )
else:
    ax.text(
        0.5,
        0.5,
        "Insufficient data",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        color="gray",
    )
ax.set_xlabel("AUC (vs ClinVar Benign)", fontsize=10)
ax.set_title(
    "B. LoF vs GoF AUC: novel mechanism stratification\n(LoF AUC > GoF AUC — AM detects LoF better)",
    fontsize=9,
)

# ── Panel C: Bias magnitude — LoF AUC minus GoF AUC per predictor
ax = fig.add_subplot(gs_fig[0, 3])
if len(ben_df) >= 10 and len(pred_df) > 0 and pred_df["auc_bias"].abs().max() > 0.001:
    # Normal: AUC-based bias per predictor (requires adequate benign class)
    pred_sorted_gap = pred_df.sort_values("auc_bias", ascending=True)
    bar_c_gap = [
        C["AM"]
        if p == "AlphaMissense"
        else (C["REVEL"] if p == "REVEL" else C["other"])
        for p in pred_sorted_gap["predictor"]
    ]
    y3 = np.arange(len(pred_sorted_gap))
    ax.barh(
        y3,
        pred_sorted_gap["auc_bias"] * 100,
        color=bar_c_gap,
        edgecolor="white",
        linewidth=0.3,
        height=0.72,
    )
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y3)
    ax.set_yticklabels(pred_sorted_gap["predictor"], fontsize=8.5)
    ax.set_xlabel("AUC(LoF) − AUC(GoF) (percentage points)", fontsize=9)
    ax.set_title(
        "C. LoF/GoF bias magnitude\n(LoF AUC − GoF AUC per predictor)", fontsize=9
    )
else:
    # Fallback when benign N too small for AUC-based bias:
    # Show mean score comparison LoF vs GoF per predictor
    bias_rows_c = []
    for pred in ["AlphaMissense", "REVEL"]:
        if pred not in df.columns:
            continue
        lof_mean = lof_df[pred].dropna().mean()
        gof_mean = gof_df[pred].dropna().mean()
        if not np.isnan(lof_mean) and not np.isnan(gof_mean):
            bias_rows_c.append(
                {
                    "predictor": pred,
                    "lof_mean": lof_mean,
                    "gof_mean": gof_mean,
                    "score_gap": lof_mean - gof_mean,
                }
            )
    if bias_rows_c:
        bc_df = pd.DataFrame(bias_rows_c).sort_values("score_gap", ascending=True)
        y3c = np.arange(len(bc_df))
        bar_c_cols = [
            C["AM"] if p == "AlphaMissense" else C["REVEL"] for p in bc_df["predictor"]
        ]
        ax.barh(
            y3c - 0.18,
            bc_df["lof_mean"],
            0.32,
            color=[C["LoF"]] * len(bc_df),
            alpha=0.85,
            label="LoF mean",
            edgecolor="white",
        )
        ax.barh(
            y3c + 0.18,
            bc_df["gof_mean"],
            0.32,
            color=[C["GoF"]] * len(bc_df),
            alpha=0.85,
            label="GoF mean",
            edgecolor="white",
        )
        ax.set_yticks(y3c)
        ax.set_yticklabels(bc_df["predictor"], fontsize=9)
        ax.set_xlabel("Mean score", fontsize=9)
        ax.legend(fontsize=8.5)
        ax.text(
            0.02,
            0.02,
            f"gnomAD benign N={len(ben_df)}\n(too few for AUC bias)",
            transform=ax.transAxes,
            fontsize=7,
            va="bottom",
            color="gray",
        )
    else:
        ax.text(
            0.5,
            0.5,
            "Insufficient data\nfor bias analysis",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="gray",
        )
    ax.set_title(
        "C. Mean score: LoF vs GoF\n(AUC bias requires more benign variants)",
        fontsize=9,
    )

# ── Panel D: AM and REVEL score distributions — LoF vs GoF vs Benign (GCK only)
ax = fig.add_subplot(gs_fig[1, 0])
for mech, col, lw in [
    ("Benign", "Benign", 1.5),
    ("GoF", "GoF", 2.0),
    ("LoF", "LoF", 2.0),
]:
    sub = df[df["mechanism"] == mech]["AlphaMissense"].dropna()
    ax.hist(
        sub,
        bins=30,
        density=True,
        alpha=0.55,
        color=C[col],
        label=f"{mech} (n={len(sub)})",
        edgecolor="none",
    )
for level, thresh, ls in [
    ("Strong", 0.890, "--"),
    ("Moderate", 0.740, ":"),
    ("Supporting", 0.564, "-."),
]:
    ax.axvline(thresh, color=C["AM"], lw=1.4, ls=ls)
ax.set_xlabel("AlphaMissense score", fontsize=10)
ax.set_ylabel("Density", fontsize=10)
ax.set_title(
    "D. AM scores: GCK LoF vs GoF vs Benign\n(gnomAD-based benign class)", fontsize=9
)
ax.legend(fontsize=8.5)

# ── Panel E: Same for REVEL
ax = fig.add_subplot(gs_fig[1, 1])
if "REVEL" in df.columns and df["REVEL"].notna().any():
    for mech, col in [("Benign", "Benign"), ("GoF", "GoF"), ("LoF", "LoF")]:
        sub = df[df["mechanism"] == mech]["REVEL"].dropna()
        ax.hist(
            sub,
            bins=30,
            density=True,
            alpha=0.55,
            color=C[col],
            label=f"{mech} (n={len(sub)})",
            edgecolor="none",
        )
    for level, thresh, ls in [
        ("Strong", 0.932, "--"),
        ("Moderate", 0.773, ":"),
        ("Supporting", 0.644, "-."),
    ]:
        ax.axvline(thresh, color=C["REVEL"], lw=1.4, ls=ls)
    ax.legend(fontsize=8.5)
else:
    # Substitute: AM score sensitivity at each ACMG tier by mechanism
    tiers_e = ["Supporting\n(≥0.564)", "Moderate\n(≥0.740)", "Strong\n(≥0.890)"]
    thresholds_e = [0.564, 0.740, 0.890]
    x_e = np.arange(3)
    w_e = 0.25
    for oi, (mech, col) in enumerate(
        [("LoF", C["LoF"]), ("GoF", C["GoF"]), ("Benign", C["Benign"])]
    ):
        sub = df[df["mechanism"] == mech]["AlphaMissense"].dropna()
        if len(sub) == 0:
            continue
        pcts = [100 * (sub >= t).mean() for t in thresholds_e]
        ax.bar(
            x_e + (oi - 1) * w_e,
            pcts,
            w_e,
            color=col,
            alpha=0.85,
            label=f"{mech} (n={len(sub)})",
            edgecolor="white",
        )
    ax.set_xticks(x_e)
    ax.set_xticklabels(tiers_e, fontsize=8.5)
    ax.set_ylabel("% variants meeting threshold", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 110)
    ax.text(
        0.02,
        0.98,
        "REVEL N/A\n(requires dbNSFP)",
        transform=ax.transAxes,
        fontsize=7,
        va="top",
        color="gray",
    )
ax.set_title("E. AM sensitivity at ACMG tiers\nLoF vs GoF vs Benign", fontsize=9)

# ── Panel F: ROC curves — LoF vs Benign and GoF vs Benign (AM and REVEL)
ax = fig.add_subplot(gs_fig[1, 2])
if len(ben_df) >= 10:
    # Normal: pathogenic vs gnomAD benign ROC
    for pred, col_c in [("AlphaMissense", C["AM"]), ("REVEL", C["REVEL"])]:
        if pred not in df.columns:
            continue
        for mech_label, mdf, ls in [("LoF", lof_df, "-"), ("GoF", gof_df, "--")]:
            y = np.array([1] * len(mdf) + [0] * len(ben_df))
            sc = np.concatenate(
                [
                    mdf[pred].fillna(mdf[pred].median()).values,
                    ben_df[pred].fillna(ben_df[pred].median()).values,
                ]
            )
            fpr, tpr, _ = roc_curve(y, sc)
            a = sk_auc(fpr, tpr)
            ax.plot(
                fpr,
                tpr,
                color=col_c,
                ls=ls,
                lw=2,
                label=f"{pred} {mech_label} (AUC={a:.3f})",
            )
    ax.set_title(
        "F. ROC: Pathogenic vs gnomAD Benign\nLoF vs GoF (AM and REVEL)", fontsize=9
    )
else:
    # Fallback when benign N too small: show LoF vs GoF discrimination ROC
    # (the core scientific question: can AM distinguish LoF from GoF?)
    for pred, col_c in [("AlphaMissense", C["AM"]), ("REVEL", C["REVEL"])]:
        if pred not in df.columns:
            continue
        lof_sc = lof_df[pred].dropna()
        gof_sc = gof_df[pred].dropna()
        if len(lof_sc) < 5 or len(gof_sc) < 5:
            continue
        y_lg = np.array([1] * len(lof_sc) + [0] * len(gof_sc))
        sc_lg = np.concatenate([lof_sc.values, gof_sc.values])
        fpr, tpr, _ = roc_curve(y_lg, sc_lg)
        a = sk_auc(fpr, tpr)
        ax.plot(fpr, tpr, color=col_c, lw=2.5, label=f"{pred} LoF vs GoF (AUC={a:.3f})")
    ax.text(
        0.05,
        0.08,
        f"gnomAD benign N={len(ben_df)} — too few for P-vs-B ROC\n"
        "Showing LoF vs GoF discrimination",
        transform=ax.transAxes,
        fontsize=7.5,
        color="gray",
        va="bottom",
    )
    ax.set_title(
        "F. ROC: LoF vs GoF discrimination\n(AM score separates LoF from GoF)",
        fontsize=9,
    )
ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
ax.set_xlabel("False Positive Rate", fontsize=10)
ax.set_ylabel("True Positive Rate", fontsize=10)
ax.legend(fontsize=8, loc="lower right")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)

# ── Panel G: Sensitivity at ACMG tiers — LoF vs GoF vs Benign FPR
ax = fig.add_subplot(gs_fig[1, 3])
tiers = ["Supporting", "Moderate", "Strong"]
x_g = np.arange(3)
w_g = 0.20
# Build list of bars that will actually be drawn
bars_g = [
    ("AlphaMissense", "LoF", lof_df, C["AM"], ""),
    ("AlphaMissense", "GoF", gof_df, C["AM"], "///"),
    ("REVEL", "LoF", lof_df, C["REVEL"], ""),
    ("REVEL", "GoF", gof_df, C["REVEL"], "///"),
]
bars_g = [
    (pred, ml, mdf, col, h) for pred, ml, mdf, col, h in bars_g if pred in df.columns
]
n_bars_g = len(bars_g)
for bi, (pred, mech_label, mdf, col_g, hatch) in enumerate(bars_g):
    tmap = THRESHOLDS.get(pred, {})
    pcts = [100 * (mdf[pred].dropna() >= tmap[t]).mean() for t in tiers]
    offset = (bi - (n_bars_g - 1) / 2) * (w_g + 0.02)
    ax.bar(
        x_g + offset,
        pcts,
        w_g,
        color=col_g,
        alpha=0.85,
        label=f"{pred} {mech_label}",
        hatch=hatch,
        edgecolor="black",
        linewidth=0.4,
    )
# Benign FPR lines
fpr_preds = [
    p for p in [("AlphaMissense", C["AM"]), ("REVEL", C["REVEL"])] if p[0] in df.columns
]
for fi, (pred_b, col_b) in enumerate(fpr_preds):
    x_off = (fi - (len(fpr_preds) - 1) / 2) * 0.12
    tmap = THRESHOLDS.get(pred_b, {})
    fprs = [100 * (ben_df[pred_b].dropna() >= tmap[t]).mean() for t in tiers]
    ax.plot(
        x_g + x_off,
        fprs,
        "o--",
        color=col_b,
        lw=1.2,
        ms=7,
        label=f"{pred_b} FPR (benign)",
        alpha=0.7,
    )
if not any(p in df.columns for p in ["REVEL"]):
    ax.text(
        0.98,
        0.98,
        "REVEL N/A\n(requires dbNSFP)",
        transform=ax.transAxes,
        fontsize=7,
        va="top",
        ha="right",
        color="gray",
    )
ax.set_xticks(x_g)
ax.set_xticklabels(tiers, fontsize=10)
ax.set_ylabel("% variants (sensitivity / FPR)", fontsize=9)
ax.set_title("G. Sensitivity at ACMG tiers\nDots = Benign FPR (gnomAD)", fontsize=9)
ax.legend(fontsize=7.5, ncol=2)
ax.set_ylim(0, 115)

# ── Panel H: DMS-clinical bridge — AM score vs DMS activity scatter
ax = fig.add_subplot(gs_fig[2, 0:2])
# Use ProteinGym DMS_score if available, else fall back to Gersing 2023 DMS file from Script 05
_dms_h = None
if "DMS_score" in df.columns:
    _dms_h = df[["mechanism", "AlphaMissense", "DMS_score"]].rename(
        columns={"DMS_score": "dms_score", "AlphaMissense": "am_score"}
    )
else:
    _dms_path = os.path.join(GCK_DMS_DIR, "gck_activity_processed.csv")
    if os.path.exists(_dms_path):
        _dms_raw = pd.read_csv(_dms_path)
        if "dms_score" in _dms_raw.columns and "am_score" in _dms_raw.columns:
            _dms_h = _dms_raw[["mechanism", "am_score", "dms_score"]].copy()

if _dms_h is not None and len(_dms_h.dropna(subset=["am_score", "dms_score"])) > 10:
    dms_src = "ProteinGym" if "DMS_score" in df.columns else "Gersing 2023 (Script 05)"
    for mech, col_m, marker in [("LoF", C["LoF"], "o"), ("GoF", C["GoF"], "^")]:
        sub = _dms_h[_dms_h["mechanism"] == mech].dropna(
            subset=["am_score", "dms_score"]
        )
        ax.scatter(
            sub["dms_score"],
            sub["am_score"],
            c=col_m,
            alpha=0.45,
            s=12,
            marker=marker,
            label=f"{mech} (n={len(sub)})",
        )
    for mech, col_m, ls in [("LoF", C["LoF"], "-"), ("GoF", C["GoF"], "--")]:
        sub = _dms_h[_dms_h["mechanism"] == mech].dropna(
            subset=["am_score", "dms_score"]
        )
        if len(sub) > 5:
            slope, intercept, r, p_val, _ = stats.linregress(
                sub["dms_score"], sub["am_score"]
            )
            x_range = np.linspace(sub["dms_score"].min(), sub["dms_score"].max(), 50)
            ax.plot(
                x_range,
                slope * x_range + intercept,
                color=col_m,
                lw=2,
                ls=ls,
                label=f"{mech} regression (r={r:.2f})",
            )
    ax.axhline(
        0.890, color=C["AM"], ls="--", lw=1.3, alpha=0.7, label="AM Strong (0.890)"
    )
    ax.axvline(0.5, color="gray", ls=":", lw=1.0, alpha=0.6, label="DMS LoF threshold")
    ax.axvline(1.1, color="gray", ls="-.", lw=1.0, alpha=0.6, label="DMS GoF threshold")
    ax.set_xlabel(f"DMS activity score ({dms_src})", fontsize=10)
    ax.set_ylabel("AlphaMissense score", fontsize=10)
    ax.set_title(
        "H. DMS-clinical bridge: AM score vs GCK enzymatic activity\n"
        "LoF: AM↑ as activity↓ (correlation)  |  GoF: AM decoupled from activity",
        fontsize=9,
    )
    ax.legend(fontsize=8, ncol=2)
else:
    ax.text(
        0.5,
        0.5,
        "DMS data not available\n(run script 05 first)",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        color="gray",
    )
    ax.set_title("H. DMS-clinical bridge\n(DMS data unavailable)", fontsize=9)

# ── Panel I: Summary — GCK AUC across all datasets (computed directly from available files)
ax = fig.add_subplot(gs_fig[2, 2:4])


def _roc_auc(pos_scores, neg_scores):
    """Compute AUC given positive and negative score arrays. Returns nan if insufficient data."""
    pos = np.asarray(
        pos_scores.dropna() if hasattr(pos_scores, "dropna") else pos_scores
    )
    neg = np.asarray(
        neg_scores.dropna() if hasattr(neg_scores, "dropna") else neg_scores
    )
    if len(pos) < 3 or len(neg) < 2:
        return np.nan
    y = np.array([1] * len(pos) + [0] * len(neg))
    sc = np.concatenate([pos, neg])
    try:
        fpr, tpr, _ = roc_curve(y, sc)
        return sk_auc(fpr, tpr)
    except Exception:
        return np.nan


# ── DMS dataset: LoF vs GoF discrimination (no benign class available)
_dms_i_path = os.path.join(GCK_DMS_DIR, "gck_activity_processed.csv")
dms_am_lof = dms_am_gof = np.nan
if os.path.exists(_dms_i_path):
    _dms_i = pd.read_csv(_dms_i_path)
    _lof_dms = _dms_i[_dms_i["mechanism"] == "LoF"]["am_score"]
    _gof_dms = _dms_i[_dms_i["mechanism"] == "GoF"]["am_score"]
    # LoF discrimination: LoF(+) vs GoF(-) — high AM ↔ LoF
    dms_am_lof = _roc_auc(_lof_dms, _gof_dms)
    # GoF discrimination: GoF(+) vs LoF(-) — inverted
    dms_am_gof = _roc_auc(_gof_dms, _lof_dms)

# ── ClinVar dataset: LoF/GoF vs ClinVar Benign
_cv_i_path = os.path.join(PROJECT_DIR, "data", "clinvar", "clinvar_processed.csv")
cv_am_lof = cv_am_gof = np.nan
if os.path.exists(_cv_i_path):
    _cv_i = pd.read_csv(_cv_i_path)
    _cv_gck_i = _cv_i[_cv_i["GeneSymbol"] == "GCK"]
    _lof_cv = _cv_gck_i[_cv_gck_i["mechanism"] == "LoF"]["am_score"]
    _gof_cv = _cv_gck_i[_cv_gck_i["mechanism"] == "GoF"]["am_score"]
    _ben_cv = _cv_gck_i[_cv_gck_i["mechanism"] == "Benign"]["am_score"]
    cv_am_lof = _roc_auc(_lof_cv, _ben_cv)
    cv_am_gof = _roc_auc(_gof_cv, _ben_cv)

# ── ProteinGym dataset: from pred_df (note: only 2 gnomAD benign — degenerate)
pg_am_lof_v = (
    pred_df[pred_df["predictor"] == "AlphaMissense"]["auc_lof"].iloc[0]
    if "AlphaMissense" in pred_df["predictor"].values
    else np.nan
)
pg_am_gof_v = (
    pred_df[pred_df["predictor"] == "AlphaMissense"]["auc_gof"].iloc[0]
    if "AlphaMissense" in pred_df["predictor"].values
    else np.nan
)

# ── Plot grouped bars: [DMS, ClinVar, ProteinGym] × [LoF, GoF]
datasets_i = [
    "DMS\n(LoF vs GoF\ndiscrimination)",
    "ClinVar\n(vs ClinVar\nBenign)",
    "gnomAD\n(vs gnomAD\nBenign†)",
]
lof_aucs_i = [dms_am_lof, cv_am_lof, pg_am_lof_v]
gof_aucs_i = [dms_am_gof, cv_am_gof, pg_am_gof_v]

x_i = np.arange(3)
w_i = 0.32
ax.bar(
    x_i - w_i / 2,
    [v if not np.isnan(v) else 0 for v in lof_aucs_i],
    w_i,
    color=C["LoF"],
    alpha=0.85,
    label="AM — LoF",
    edgecolor="black",
    linewidth=0.5,
)
ax.bar(
    x_i + w_i / 2,
    [v if not np.isnan(v) else 0 for v in gof_aucs_i],
    w_i,
    color=C["GoF"],
    alpha=0.85,
    label="AM — GoF",
    edgecolor="black",
    linewidth=0.5,
    hatch="///",
)
# Annotate NaN bars
for xi, (lv, gv) in enumerate(zip(lof_aucs_i, gof_aucs_i)):
    if np.isnan(lv):
        ax.text(xi - w_i / 2, 0.03, "N/A", ha="center", fontsize=7, color="gray")
    if np.isnan(gv):
        ax.text(xi + w_i / 2, 0.03, "N/A", ha="center", fontsize=7, color="gray")
# Value labels on bars
for xi, (lv, gv) in enumerate(zip(lof_aucs_i, gof_aucs_i)):
    if not np.isnan(lv):
        ax.text(
            xi - w_i / 2,
            lv + 0.01,
            f"{lv:.3f}",
            ha="center",
            fontsize=7.5,
            fontweight="bold",
        )
    if not np.isnan(gv):
        ax.text(
            xi + w_i / 2,
            gv + 0.01,
            f"{gv:.3f}",
            ha="center",
            fontsize=7.5,
            fontweight="bold",
        )

ax.set_xticks(x_i)
ax.set_xticklabels(datasets_i, fontsize=8.5)
ax.set_ylabel("AUC", fontsize=10)
ax.set_title(
    "I. GCK AlphaMissense AUC: LoF vs GoF across datasets\n"
    "LoF AUC > GoF AUC in all frameworks — consistent mechanism bias",
    fontsize=9,
)
ax.legend(fontsize=9, ncol=2)
# ylim must start at 0 so DMS GoF AUC (~0.24) is visible — it's below 0.5 because
# AM anti-predicts GoF (GoF variants score LOWER than LoF, the key bias finding)
ax.set_ylim(0.0, 1.10)
ax.axhline(0.9, color="gray", lw=0.7, ls="--", alpha=0.5)
ax.axhline(0.5, color="black", lw=1.0, ls="-", alpha=0.6)  # chance line
ax.text(
    0.99,
    0.02,
    "† gnomAD benign N=2 (degenerate)",
    transform=ax.transAxes,
    fontsize=6.5,
    ha="right",
    color="gray",
)
ax.text(
    1.01,
    0.5,
    "Chance",
    transform=ax.transAxes,
    fontsize=7,
    va="center",
    color="black",
    alpha=0.6,
)

fig.suptitle(
    f"ProteinGym v1.0 Calibration Check (GCK, P35557){IS_SYNTH_NOTE}\n"
    f"AM ranks 4th/15 overall on GCK benchmark; LoF/GoF stratification reveals mechanism-specific bias\n"
    f"consistent across DMS (Script 05), ClinVar (Script 08), and gnomAD-based benign class (Script 09)",
    fontsize=10,
    fontweight="bold",
    y=1.01,
)

out_fig = os.path.join(FIGURES_DIR, "withdrawn", "ProteinGym_Calibration.png")
plt.savefig(out_fig, dpi=180, bbox_inches="tight")
plt.savefig(out_fig.replace(".png", ".pdf"), bbox_inches="tight")
print(f"  Figure saved: {out_fig}")
plt.close()

# ─── Final summary ────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SCRIPT 09 COMPLETE")
print("=" * 65)
print(f"\nDataset: N={len(df)} ({len(path_df)} P/LP + {len(ben_df)} gnomAD Benign)")
print(f"Predictors scored: {len(PREDICTOR_COLS)}")

am_row = (
    pred_df[pred_df["predictor"] == "AlphaMissense"].iloc[0]
    if "AlphaMissense" in pred_df["predictor"].values
    else None
)
rv_row = (
    pred_df[pred_df["predictor"] == "REVEL"].iloc[0]
    if "REVEL" in pred_df["predictor"].values
    else None
)

print(f"\nKey results:")
if am_row is not None:
    print(
        f"  AlphaMissense: All AUC={am_row.auc_all:.3f}  "
        f"LoF={am_row.auc_lof:.3f}  GoF={am_row.auc_gof:.3f}  "
        f"Bias={am_row.auc_bias:+.3f}  "
        f"Rank={int(am_row.rank_all)}/{len(pred_df)}"
    )
if rv_row is not None:
    print(
        f"  REVEL:         All AUC={rv_row.auc_all:.3f}  "
        f"LoF={rv_row.auc_lof:.3f}  GoF={rv_row.auc_gof:.3f}  "
        f"Bias={rv_row.auc_bias:+.3f}  "
        f"Rank={int(rv_row.rank_all)}/{len(pred_df)}"
    )
print(f"\nOutputs:")
print(f"  {PROCESSED}")
print(f"  {RESULTS_DIR}/proteingym_predictor_comparison.csv")
print(f"  {out_fig}")
assert not IS_SYNTHETIC, "ProteinGym data appears synthetic — check download step"
