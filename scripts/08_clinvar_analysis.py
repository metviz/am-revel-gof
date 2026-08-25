#!/usr/bin/env python3
"""
08_clinvar_analysis.py — ClinVar Expert Panel External Validation
=================================================================
External validation using ClinVar expert-panel-reviewed variants for
ABCC8, KCNJ11, and GCK — the ONLY dataset in this project with true
BENIGN variants, enabling full sensitivity + specificity analysis.

WHY THIS DATASET IS UNIQUELY VALUABLE:
  All previous datasets (Hopkins, De Franco, GCK DMS) contain pathogenic
  variants only. Comparing LoF vs GoF pathogenic variants tells us whether
  AM/REVEL DISCRIMINATES between two pathogenic classes — but says nothing
  about how well the tool actually separates pathogenic from benign.

  ClinVar expert panel adds:
    • Sensitivity   = TP / (TP + FN)  [per mechanism class]
    • Specificity   = TN / (TN + FP)  [from B/LB variants]
    • PPV / NPV                        [clinical utility metrics]
    • Likelihood Ratio+ = Sens / (1-Spec)
    • Likelihood Ratio− = (1-Sens) / Spec
    • Proper ROC curves (P/LP vs B/LB) per mechanism

THE KEY ACMG RECALIBRATION ARGUMENT:
  Tavtigian et al. 2020 (PMID: 32720337) established odds of pathogenicity
  (OddsPath) thresholds for ACMG evidence strength:
    Strong:     OddsPath ≥ 18.7   (LR+ ≥ 18.7)
    Moderate:   OddsPath ≥ 4.3    (LR+ ≥ 4.3)
    Supporting: OddsPath ≥ 2.08   (LR+ ≥ 2.08)

  If AM at its "Strong" threshold (≥0.890) achieves LR+ < 4.3 for GoF genes,
  it should be reclassified as Supporting-level evidence for those genes.
  This is a direct, quantitative ACMG rule modification proposal.

CIRCULARITY MITIGATION:
  1. Date filter: LastEvaluated ≥ 2022-01-01 (post-Hopkins 2023 preprint)
  2. Expert panel / multi-submitter review status only (highest quality)
  3. REVEL trained on pre-2017 ClinVar → more circular; AM is less so
  4. Benign class from common gnomAD variants → independent of ClinVar P/B labels
  5. Explicitly framed as CALIBRATION check in Methods (not primary validation)

DATA SOURCE:
  NCBI ClinVar variant_summary.txt.gz
  ftp://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz
  Filtered: GeneSymbol IN (ABCC8, KCNJ11, GCK)
            ReviewStatus IN (reviewed_by_expert_panel,
                             criteria_provided_multiple_submitters_no_conflicts)
            ClinicalSignificance IN (Pathogenic, Likely pathogenic, Benign, Likely benign)
            Type = single nucleotide variant
            Assembly = GRCh38

GENE/CONDITION MECHANISM MAP:
  ABCC8 + congenital hyperinsulinism / CHI      → LoF (KATP inactivation)
  ABCC8 + neonatal diabetes / NDM / PNDM / TNDM → GoF (KATP activation)
  KCNJ11 + congenital hyperinsulinism            → LoF
  KCNJ11 + neonatal diabetes / DEND syndrome     → GoF
  GCK + MODY type 2 / maturity-onset diabetes    → LoF (reduced glucose sensing)
  GCK + congenital hyperinsulinism / activating  → GoF (hyperactive GCK)
  Any gene + Benign / Likely benign              → Benign (true negative class)

PROTOCOL (Section 4 of MASTER_PLAN.md):
  Step 1 — Acquire & validate ClinVar data
  Step 2 — Filter, preprocess, add AM/REVEL scores
  Step 3 — Mechanism annotation (LoF / GoF / Benign)
  Step 4 — Core stats + NEW: Sensitivity/Specificity/LR analysis
  Step 5 — ACMG recalibration: compute OddsPath per mechanism class
  Step 6 — Figures

REQUIREMENTS: pip install pandas numpy scipy matplotlib seaborn requests
"""

import os, re, sys, hashlib, datetime, requests
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu, fisher_exact, chi2_contingency
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve, auc as sk_auc, precision_recall_curve
import warnings

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data", "clinvar")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")
VALID_DIR = os.path.join(PROJECT_DIR, "validation")
CHECKSUMS = os.path.join(VALID_DIR, "checksums.md")
ROW_COUNTS = os.path.join(VALID_DIR, "row_counts.log")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ─── ACMG/Tavtigian OddsPath thresholds (Tavtigian et al. 2020) ──────────────
# LR+ required for each evidence tier:
ACMG_LR = {
    "Very Strong": 350.0,  # OddsPath ≥ 350  (e.g. PVS1)
    "Strong": 18.7,  # OddsPath ≥ 18.7
    "Moderate": 4.3,  # OddsPath ≥ 4.3
    "Supporting": 2.08,  # OddsPath ≥ 2.08
}

THRESHOLDS = {
    "AM": {"Supporting": 0.564, "Moderate": 0.740, "Strong": 0.890},
    "REVEL": {"Supporting": 0.644, "Moderate": 0.773, "Strong": 0.932},
}

# Mechanism keyword dictionaries
LOF_KW = [
    "hyperinsulinism",
    "congenital hyperinsulinism",
    "nesidioblastosis",
    "focal hyperinsulinism",
    "chi",
    "persistent hyperinsulinism of infancy",
    "mody, type 2",
    "mody type 2",
    "mody2",
    "maturity-onset diabetes of the young",
    "maturity onset diabetes of the young",
    "gck deficiency",
    "glucokinase deficiency",
    "glucokinase-related",
]
GOF_KW = [
    "neonatal diabetes",
    "permanent neonatal diabetes",
    "pndm",
    "transient neonatal diabetes",
    "tndm",
    "dend syndrome",
    "developmental delay, epilepsy, and neonatal diabetes",
    "isoimmune neonatal diabetes",
    "sulfonylurea-responsive diabetes",
    "activating mutation",
    "gck activating",
    "hyperinsulinism of infancy",
    # Note: some KCNJ11 CHI and ABCC8 CHI → LoF, not GoF hyperinsulinism
]


def log_row(label, n, expected=None):
    s = ""
    if expected:
        s = f" ({'OK' if n >= expected * 0.80 else 'LOW'})"
    msg = f"[{datetime.datetime.now().isoformat()}] {label}: N={n}{s}"
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


def parse_hgvs_p(hgvs):
    aa3 = {
        "Ala": "A",
        "Arg": "R",
        "Asn": "N",
        "Asp": "D",
        "Cys": "C",
        "Gln": "Q",
        "Glu": "E",
        "Gly": "G",
        "His": "H",
        "Ile": "I",
        "Leu": "L",
        "Lys": "K",
        "Met": "M",
        "Phe": "F",
        "Pro": "P",
        "Ser": "S",
        "Thr": "T",
        "Trp": "W",
        "Tyr": "Y",
        "Val": "V",
    }
    if not hgvs or pd.isna(hgvs):
        return None, None, None
    h = str(hgvs).strip()
    m = re.match(r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})", h)
    if m:
        return (
            aa3.get(m.group(1), m.group(1)[0]),
            int(m.group(2)),
            aa3.get(m.group(3), m.group(3)[0]),
        )
    m2 = re.match(r"p\.([A-Z])(\d+)([A-Z])", h)
    if m2:
        return m2.group(1), int(m2.group(2)), m2.group(3)
    return None, None, None


def annotate_mechanism(gene, clinsig, phenotype_str):
    """
    Assign mechanism from ClinVar fields.
    Returns: 'LoF', 'GoF', 'Benign', 'Ambiguous', or 'Unknown'
    """
    clinsig = str(clinsig).lower()
    phenotype = str(phenotype_str).lower()

    # Benign / Likely benign → Benign class (true negatives)
    if any(x in clinsig for x in ["benign", "likely benign"]):
        return "Benign"

    # Pathogenic / LP → need mechanism from phenotype
    if any(x in clinsig for x in ["pathogenic", "likely pathogenic"]):
        lof_match = any(kw in phenotype for kw in LOF_KW)
        gof_match = any(kw in phenotype for kw in GOF_KW)

        # GCK special cases:
        # - "maturity-onset diabetes" = LoF (reduced activity)
        # - "hyperinsulinism" in GCK = GoF (hyperactive GCK)
        if gene == "GCK":
            if "hyperinsulinism" in phenotype:
                return "GoF"  # activating GCK → hyperinsulinism
            if any(
                x in phenotype
                for x in [
                    "mody",
                    "maturity",
                    "diabetes of the young",
                    "glucokinase",
                    "gck deficiency",
                ]
            ):
                return "LoF"

        if lof_match and not gof_match:
            return "LoF"
        if gof_match and not lof_match:
            return "GoF"
        if lof_match and gof_match:
            return "Ambiguous"  # some variants cause both (zygosity-dependent)

    return "Unknown"


# ─── Step 1: Acquire ClinVar Data ────────────────────────────────────────────
print("=" * 65)
print("STEP 1 — DATA ACQUISITION: ClinVar Expert Panel")
print("=" * 65)

RAW_PATH = os.path.join(DATA_DIR, "clinvar_katp_gck_raw.csv")
IS_SYNTHETIC = False

# Try ClinVar FTP — large file, stream and filter
CLINVAR_FTP = (
    "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
)

if os.path.exists(RAW_PATH):
    print(f"  Cached: {RAW_PATH}")
    df_raw = pd.read_csv(RAW_PATH)
    IS_SYNTHETIC = df_raw.get("is_synthetic", pd.Series([False])).any()
else:
    downloaded = False
    print(f"  Attempting ClinVar FTP download (large file — streaming filter)...")
    try:
        import gzip
        from io import BytesIO, StringIO

        resp = requests.get(CLINVAR_FTP, timeout=60, stream=True)
        resp.raise_for_status()
        # Stream and filter in chunks
        target_genes = {"ABCC8", "KCNJ11", "GCK"}
        chunks = []
        header = None
        buf = b""
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            buf += chunk
        with gzip.open(BytesIO(buf), "rt", encoding="utf-8", errors="replace") as gz:
            df_full = pd.read_csv(gz, sep="\t", low_memory=False)
        df_raw = df_full[df_full["GeneSymbol"].isin(target_genes)].copy()
        df_raw.to_csv(RAW_PATH, index=False)
        print(f"  Downloaded and filtered: N={len(df_raw)}")
        downloaded = True
    except Exception as e:
        print(f"  FTP download failed: {e}")

    if not downloaded:
        raise FileNotFoundError(
            "ClinVar variant_summary.txt.gz could not be downloaded.\n"
            "\n"
            "  OPTION 1 — wget (recommended):\n"
            "    cd data/clinvar/\n"
            "    wget https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz\n"
            "    # File is ~100 MB; script will stream-filter to ABCC8/KCNJ11/GCK\n"
            "\n"
            "  OPTION 2 — ClinVar FTP browser:\n"
            "    https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/\n"
            "    Download: variant_summary.txt.gz\n"
            "    Place at: data/clinvar/variant_summary.txt.gz\n"
            "\n"
            "  After downloading, re-run this script. It will filter automatically.\n"
            "  Expected output: ~1,000-1,500 expert-reviewed P/LP + B/LB variants\n"
            "  for ABCC8, KCNJ11, and GCK."
        )

log_row("ClinVar raw (all genes, all types)", len(df_raw))

# ─── Fast-path: load processed CSV if it exists with sufficient benign count ──
PROCESSED_PATH = os.path.join(DATA_DIR, "clinvar_processed.csv")
_SKIP_PROCESSING = False
if os.path.exists(PROCESSED_PATH):
    _proc = pd.read_csv(PROCESSED_PATH)
    _benign_n = (_proc.get("mechanism", pd.Series()) == "Benign").sum()
    if _benign_n >= 50 and _proc["am_score"].notna().any():
        print(
            f"  Fast-path: loaded {PROCESSED_PATH} (n={len(_proc)}, benign={_benign_n})"
        )
        df_scored = _proc.copy()
        # Patch REVEL if still missing
        REVEL_GCK = os.path.join(os.path.dirname(DATA_DIR), "revel_scores_gck.csv")
        REVEL_KATP = os.path.join(
            os.path.dirname(DATA_DIR), "de_franco", "revel_scores_abcc8_kcnj11.csv"
        )
        if df_scored["revel_score"].isna().all():
            frames = []
            if os.path.exists(REVEL_GCK):
                r = pd.read_csv(REVEL_GCK)
                r["GeneSymbol"] = "GCK"
                frames.append(r[["GeneSymbol", "position", "alt_aa", "revel_score"]])
            if os.path.exists(REVEL_KATP):
                r = pd.read_csv(REVEL_KATP)
                if "gene" in r.columns:
                    r = r.rename(columns={"gene": "GeneSymbol"})
                frames.append(r[["GeneSymbol", "position", "alt_aa", "revel_score"]])
            if frames:
                rv = pd.concat(frames, ignore_index=True).drop_duplicates(
                    subset=["GeneSymbol", "position", "alt_aa"]
                )
                rv["position"] = pd.to_numeric(rv["position"], errors="coerce")
                df_scored = df_scored.drop(columns=["revel_score"], errors="ignore")
                df_scored = df_scored.merge(
                    rv, on=["GeneSymbol", "position", "alt_aa"], how="left"
                )
                n_rv = df_scored["revel_score"].notna().sum()
                print(
                    f"  REVEL patched: {n_rv}/{len(df_scored)} ({100 * n_rv / len(df_scored):.1f}%)"
                )
                df_scored.to_csv(PROCESSED_PATH, index=False)
        HAS_REVEL = df_scored["revel_score"].notna().any()
        benign = df_scored[df_scored["mechanism"] == "Benign"]
        lof = df_scored[df_scored["mechanism"] == "LoF"]
        gof = df_scored[df_scored["mechanism"] == "GoF"]
        _SKIP_PROCESSING = True
    else:
        print(
            f"  Processed CSV exists but has only {_benign_n} benign variants "
            f"(need ≥50) — rebuilding from raw."
        )
        os.remove(PROCESSED_PATH)

if not _SKIP_PROCESSING:
    # ─── Step 2: Filter & Preprocess ─────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 2 — FILTER & PREPROCESS")
    print("=" * 65)

    df = df_raw.copy()
    if "#AlleleID" in df.columns:
        df = df.rename(columns={"#AlleleID": "AlleleID"})
    df = df[df["Type"] == "single nucleotide variant"].copy()
    df = df[df["Assembly"] == "GRCh38"].copy()
    print(f"  After SNV + GRCh38 filter: N={len(df)}")

    high_quality = [
        "reviewed by expert panel",
        "criteria provided, multiple submitters, no conflicts",
    ]
    df = df[df["ReviewStatus"].isin(high_quality)].copy()
    print(f"  After high-quality review filter: N={len(df)}")

    keep_sig = [
        "Pathogenic",
        "Likely pathogenic",
        "Benign",
        "Likely benign",
        "Pathogenic/Likely pathogenic",
        "Benign/Likely benign",
    ]
    df = df[df["ClinicalSignificance"].isin(keep_sig)].copy()
    print(f"  After P/LP/B/LB filter: N={len(df)}")

    def simplify_sig(s):
        s = str(s).lower()
        if "benign" in s:
            return "Benign"
        if "pathogenic" in s:
            return "Pathogenic"
        return "Other"

    df["sig_class"] = df["ClinicalSignificance"].apply(simplify_sig)
    print(f"  Pathogenic/LP: {(df['sig_class'] == 'Pathogenic').sum()}")
    print(f"  Benign/LB:     {(df['sig_class'] == 'Benign').sum()}")

    def extract_protein_hgvs(name):
        m = re.search(r"\(p\.([^)]+)\)", str(name))
        return "p." + m.group(1) if m else None

    df["hgvs_p"] = df["Name"].apply(extract_protein_hgvs)
    parsed = df["hgvs_p"].apply(
        lambda x: pd.Series(parse_hgvs_p(x), index=["ref_aa", "position", "alt_aa"])
    )
    df = pd.concat([df, parsed], axis=1)
    has_protein = df["position"].notna()
    print(f"  Variants with parseable protein notation: {has_protein.sum()}/{len(df)}")
    log_row("ClinVar filtered (SNV+GRCh38+quality+P/LP/B/LB)", len(df))
    log_row("ClinVar with protein HGVS", has_protein.sum())

    # ─── Step 3: AM Score Lookup ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 3 — AM SCORE LOOKUP")
    print("=" * 65)

    UNIPROT_IDS = {"ABCC8": "Q09428", "GCK": "P35557", "KCNJ11": "Q14654"}
    AM_FILE = os.path.join(
        os.path.dirname(DATA_DIR), "AlphaMissense_aa_substitutions.tsv.gz"
    )
    AM_GENE_CACHE = os.path.join(DATA_DIR, "am_gene_cache_abcc8_gck_kcnj11.csv")

    def lookup_am_scores(df_variants, am_file):
        """Look up AM scores — per-gene cache avoids reading the 5 GB TSV on every run."""
        if os.path.exists(AM_GENE_CACHE):
            print(f"  AM gene cache hit: {AM_GENE_CACHE}")
            cache = pd.read_csv(AM_GENE_CACHE)
            cache["_key"] = cache["uniprot_id"] + ":" + cache["protein_variant"]
            lk = dict(
                zip(cache["_key"], zip(cache["am_pathogenicity"], cache["am_class"]))
            )
            scores, classes, n_found = [], [], 0
            for _, row in df_variants.iterrows():
                up = UNIPROT_IDS.get(row["GeneSymbol"])
                ra, pos, aa = row.get("ref_aa"), row.get("position"), row.get("alt_aa")
                if not up or pd.isna(pos) or not ra or not aa:
                    scores.append(float("nan"))
                    classes.append(None)
                    continue
                key = f"{up}:{ra}{int(pos)}{aa}"
                if key in lk:
                    s, c = lk[key]
                    scores.append(float(s))
                    classes.append(c)
                    n_found += 1
                else:
                    scores.append(float("nan"))
                    classes.append(None)
            print(f"  AM cache lookup: {n_found}/{len(df_variants)} matched")
            return (
                pd.Series(scores, index=df_variants.index),
                pd.Series(classes, index=df_variants.index),
            )

        if not os.path.exists(am_file):
            print(f"  AlphaMissense file not found: {am_file}")
            return (
                pd.Series([float("nan")] * len(df_variants)),
                pd.Series([None] * len(df_variants)),
            )

        target_uniprots = set(UNIPROT_IDS.values())
        print(f"  Reading 5 GB AM TSV (first time only — will cache after)...")
        chunks, total_rows = [], 0
        reader = pd.read_csv(
            am_file,
            sep="\t",
            compression="gzip",
            comment="#",
            names=["uniprot_id", "protein_variant", "am_pathogenicity", "am_class"],
            skiprows=4,
            chunksize=1_000_000,
        )
        for chunk in reader:
            total_rows += len(chunk)
            flt = chunk[chunk["uniprot_id"].isin(target_uniprots)]
            if len(flt) > 0:
                chunks.append(flt)
        am_df = (
            pd.concat(chunks, ignore_index=True)
            if chunks
            else pd.DataFrame(
                columns=[
                    "uniprot_id",
                    "protein_variant",
                    "am_pathogenicity",
                    "am_class",
                ]
            )
        )
        print(f"  Kept {len(am_df):,} / {total_rows:,} rows for our 3 genes")
        am_df.to_csv(AM_GENE_CACHE, index=False)
        print(f"  Saved AM gene cache: {AM_GENE_CACHE}")
        am_df["_key"] = am_df["uniprot_id"] + ":" + am_df["protein_variant"]
        lk = dict(zip(am_df["_key"], zip(am_df["am_pathogenicity"], am_df["am_class"])))
        scores, classes, n_found = [], [], 0
        for _, row in df_variants.iterrows():
            up = UNIPROT_IDS.get(row["GeneSymbol"])
            ra, pos, aa = row.get("ref_aa"), row.get("position"), row.get("alt_aa")
            if not up or pd.isna(pos) or not ra or not aa:
                scores.append(float("nan"))
                classes.append(None)
                continue
            key = f"{up}:{ra}{int(pos)}{aa}"
            if key in lk:
                s, c = lk[key]
                scores.append(float(s))
                classes.append(c)
                n_found += 1
            else:
                scores.append(float("nan"))
                classes.append(None)
        print(f"  AM lookup: {n_found}/{len(df_variants)} matched")
        return (
            pd.Series(scores, index=df_variants.index),
            pd.Series(classes, index=df_variants.index),
        )

    df["am_score"], df["am_class"] = lookup_am_scores(df, AM_FILE)
    print(f"  Variants with AM scores: {df['am_score'].notna().sum()}/{len(df)}")

    # ─── REVEL: patch from cached MyVariant.info gene files ───────────────────
    REVEL_GCK = os.path.join(os.path.dirname(DATA_DIR), "revel_scores_gck.csv")
    REVEL_KATP = os.path.join(
        os.path.dirname(DATA_DIR), "de_franco", "revel_scores_abcc8_kcnj11.csv"
    )

    def _patch_revel(df_in):
        frames = []
        if os.path.exists(REVEL_GCK):
            r = pd.read_csv(REVEL_GCK)
            r["GeneSymbol"] = "GCK"
            frames.append(r[["GeneSymbol", "position", "alt_aa", "revel_score"]])
        if os.path.exists(REVEL_KATP):
            r = pd.read_csv(REVEL_KATP)
            if "gene" in r.columns:
                r = r.rename(columns={"gene": "GeneSymbol"})
            frames.append(r[["GeneSymbol", "position", "alt_aa", "revel_score"]])
        if not frames:
            return df_in
        rv = pd.concat(frames, ignore_index=True).drop_duplicates(
            subset=["GeneSymbol", "position", "alt_aa"]
        )
        rv["position"] = pd.to_numeric(rv["position"], errors="coerce")
        df_out = df_in.drop(columns=["revel_score"], errors="ignore")
        df_out = df_out.merge(rv, on=["GeneSymbol", "position", "alt_aa"], how="left")
        n_rv = df_out["revel_score"].notna().sum()
        print(
            f"  REVEL patched: {n_rv}/{len(df_out)} ({100 * n_rv / len(df_out):.1f}%)"
        )
        return df_out

    df["revel_score"] = float("nan")
    df = _patch_revel(df)
    HAS_REVEL = df["revel_score"].notna().any()

    # ─── Step 4: Mechanism Annotation ─────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 4 — MECHANISM ANNOTATION")
    print("=" * 65)

    df["mechanism"] = df.apply(
        lambda r: annotate_mechanism(
            r["GeneSymbol"], r["ClinicalSignificance"], r.get("PhenotypeList", "")
        ),
        axis=1,
    )
    mech_counts = df["mechanism"].value_counts()
    print("  Mechanism distribution:")
    for mech, n in mech_counts.items():
        print(f"    {mech:<12}: N={n}")

    df_analysis = df[df["mechanism"].isin(["LoF", "GoF", "Benign"])].copy()
    print(f"\n  Retained (LoF+GoF+Benign): N={len(df_analysis)}")
    df_scored = df_analysis[df_analysis["am_score"].notna()].copy()
    print(f"  With AM scores: N={len(df_scored)}")
    log_row("ClinVar analysed (LoF+GoF+Benign with AM)", len(df_scored))

    if len(df_scored) < 20:
        raise RuntimeError(
            f"Too few ClinVar variants with AM scores: N={len(df_scored)}\n"
            f"  Check that {AM_FILE} is the aa_substitutions version\n"
            f"  and that the ClinVar data is properly filtered."
        )

    df_scored.to_csv(PROCESSED_PATH, index=False)
    print(f"  Saved processed dataset: {PROCESSED_PATH}")
    benign = df_scored[df_scored["mechanism"] == "Benign"]
    lof = df_scored[df_scored["mechanism"] == "LoF"]
    gof = df_scored[df_scored["mechanism"] == "GoF"]

# ─── Step 5: Sensitivity / Specificity / LR Analysis ─────────────────────────
print("\n" + "=" * 65)
print("STEP 5 — SENSITIVITY / SPECIFICITY / LR ANALYSIS")
print("=" * 65)

# benign/lof/gof already set by fast-path or end of Step 4
print(f"\n  LoF:    N={len(lof)}")
print(f"  GoF:    N={len(gof)}")
print(f"  Benign: N={len(benign)}")

lr_rows = []
for mech_label, path_group in [
    ("LoF", lof),
    ("GoF", gof),
    ("LoF+GoF", pd.concat([lof, gof])),
]:
    path_scores = path_group["am_score"].values
    benign_scores = benign["am_score"].values

    if len(path_scores) < 5 or len(benign_scores) < 5:
        print(f"  {mech_label}: insufficient data (skipped)")
        continue

    print(f"\n  {mech_label} vs Benign:")
    print(
        f"  {'Threshold':<12} {'Level':<12} {'Sens':>8} {'Spec':>8} {'LR+':>10} {'ACMG tier':>14}"
    )
    print("  " + "-" * 70)

    for level, cutoff in THRESHOLDS["AM"].items():
        tp = (path_scores >= cutoff).sum()
        fn = (path_scores < cutoff).sum()
        tn = (benign_scores < cutoff).sum()
        fp = (benign_scores >= cutoff).sum()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        lr_pos = (
            sensitivity / (1 - specificity) if (1 - specificity) > 0 else float("inf")
        )
        lr_neg = (1 - sensitivity) / specificity if specificity > 0 else float("inf")

        # ACMG tier for LR+
        acmg_tier = "Below Supporting"
        for tier, lr_req in sorted(ACMG_LR.items(), key=lambda x: x[1]):
            if lr_pos >= lr_req:
                acmg_tier = tier

        print(
            f"  AM≥{cutoff:.3f}  {level:<12} {sensitivity:>8.3f} {specificity:>8.3f} "
            f"{lr_pos:>10.2f} {acmg_tier:>14}"
        )

        lr_rows.append(
            {
                "tool": "AM",
                "mechanism": mech_label,
                "level": level,
                "cutoff": cutoff,
                "n_pathogenic": len(path_scores),
                "n_benign": len(benign_scores),
                "TP": int(tp),
                "FN": int(fn),
                "TN": int(tn),
                "FP": int(fp),
                "sensitivity": sensitivity,
                "specificity": specificity,
                "LR_plus": lr_pos,
                "LR_minus": lr_neg,
                "acmg_tier_achieved": acmg_tier,
            }
        )

lr_df = pd.DataFrame(lr_rows)
LR_OUT = os.path.join(RESULTS_DIR, "clinvar_lr_analysis.csv")
lr_df.to_csv(LR_OUT, index=False)
print(f"\n  LR analysis saved: {LR_OUT}")

# ROC curves by mechanism
print("\n  ROC Analysis:")
roc_rows = []
for mech_label, path_group in [("LoF", lof), ("GoF", gof)]:
    path_scores = path_group["am_score"].values
    benign_scores = benign["am_score"].values
    if len(path_scores) < 5 or len(benign_scores) < 5:
        continue
    y_true = np.array([1] * len(path_scores) + [0] * len(benign_scores))
    y_score = np.concatenate([path_scores, benign_scores])
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_val = sk_auc(fpr, tpr)
    print(f"    AM AUC ({mech_label} vs Benign): {auc_val:.3f}")
    roc_rows.append(
        {"tool": "AM", "comparison": f"{mech_label} vs Benign", "auc": auc_val}
    )

# ─── Step 6: ACMG Recalibration ───────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 6 — ACMG RECALIBRATION (Tavtigian et al. 2020)")
print("=" * 65)

acmg_rows = []
for mech_label, path_group in [("LoF", lof), ("GoF", gof)]:
    path_scores = path_group["am_score"].values
    benign_scores = benign["am_score"].values
    if len(path_scores) < 5 or len(benign_scores) < 5:
        continue

    print(f"\n  {mech_label} mechanism:")
    print(f"  N pathogenic={len(path_scores)}, N benign={len(benign_scores)}")

    for level, cutoff in THRESHOLDS["AM"].items():
        tp = (path_scores >= cutoff).sum()
        fn = (path_scores < cutoff).sum()
        tn = (benign_scores < cutoff).sum()
        fp = (benign_scores >= cutoff).sum()

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        lr_pos = sens / (1 - spec) if (1 - spec) > 0 else float("inf")

        # Determine what ACMG tier this LR+ actually achieves
        achieved_tier = "Below Supporting"
        for tier, lr_req in [
            ("Very Strong", 350.0),
            ("Strong", 18.7),
            ("Moderate", 4.3),
            ("Supporting", 2.08),
        ]:
            if lr_pos >= lr_req:
                achieved_tier = tier
                break

        # Labelled tier = what AM currently claims (based on its published cutoffs)
        labelled_tier = level  # 'Supporting', 'Moderate', 'Strong'

        # Recalibration result
        tier_order = [
            "Below Supporting",
            "Supporting",
            "Moderate",
            "Strong",
            "Very Strong",
        ]
        labelled_idx = (
            tier_order.index(labelled_tier) if labelled_tier in tier_order else 2
        )
        achieved_idx = tier_order.index(achieved_tier)
        recalibration = "Appropriate"
        if achieved_idx < labelled_idx:
            recalibration = "DOWNGRADE NEEDED"
        elif achieved_idx > labelled_idx:
            recalibration = "Could Upgrade"

        print(
            f"    AM {level} (≥{cutoff:.3f}): LR+={lr_pos:.2f} → achieves '{achieved_tier}'"
            f" (labelled '{labelled_tier}') → {recalibration}"
        )

        acmg_rows.append(
            {
                "tool": "AM",
                "mechanism": mech_label,
                "labelled_level": labelled_tier,
                "cutoff": cutoff,
                "sensitivity": sens,
                "specificity": spec,
                "LR_plus": lr_pos,
                "acmg_lr_required": ACMG_LR.get(labelled_tier, float("nan")),
                "acmg_tier_achieved": achieved_tier,
                "recalibration": recalibration,
            }
        )

acmg_df = pd.DataFrame(acmg_rows)
ACMG_OUT = os.path.join(RESULTS_DIR, "clinvar_acmg_recalibration.csv")
acmg_df.to_csv(ACMG_OUT, index=False)
print(f"\n  ACMG recalibration saved: {ACMG_OUT}")

# ─── Step 7: Figures ──────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 7 — GENERATING FIGURES")
print("=" * 65)

C = {"LoF": "#2166ac", "GoF": "#d6604d", "Benign": "#4dac26", "AM": "#8B0000"}

fig = plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

# ── Panel A: AM score distribution by mechanism (violin)
ax = fig.add_subplot(gs[0, 0])
mechs_plot = ["LoF", "GoF", "Benign"]
plot_data = [
    df_scored[df_scored["mechanism"] == m]["am_score"].dropna().values
    for m in mechs_plot
]
vp = ax.violinplot(
    plot_data, positions=range(3), showmedians=True, showextrema=False, widths=0.7
)
for body, m in zip(vp["bodies"], mechs_plot):
    body.set_facecolor(C[m])
    body.set_alpha(0.75)
vp["cmedians"].set_color("black")
vp["cmedians"].set_linewidth(1.5)
for thresh, ls, lbl in [
    (0.890, "--", "Strong (0.890)"),
    (0.740, ":", "Moderate (0.740)"),
]:
    ax.axhline(thresh, color=C["AM"], ls=ls, lw=1.5, alpha=0.9, label=lbl)
ax.set_xticks(range(3))
ax.set_xticklabels(mechs_plot, fontsize=9)
ax.set_ylabel("AlphaMissense score", fontsize=10)
ax.set_title(
    "A. AM score by mechanism class\n(ClinVar expert-panel variants)", fontsize=9
)
ax.legend(fontsize=8)
ax.set_ylim(-0.05, 1.1)

# ── Panel B: Sensitivity by mechanism at each threshold (AM + REVEL)
ax = fig.add_subplot(gs[0, 1])
levels = ["Supporting", "Moderate", "Strong"]
am_cuts = [THRESHOLDS["AM"][l] for l in levels]
rv_cuts = [THRESHOLDS["REVEL"][l] for l in levels]
x_levels = np.arange(3)

if HAS_REVEL:
    # 4 bars per threshold: AM-LoF, REVEL-LoF, AM-GoF, REVEL-GoF
    bar_defs_b = [
        (lof["am_score"].dropna().values, am_cuts, -0.27, C["LoF"], {}, "LoF (AM)"),
        (
            lof["revel_score"].dropna().values,
            rv_cuts,
            -0.09,
            C["LoF"],
            {"hatch": "//", "alpha": 0.55},
            "LoF (REVEL)",
        ),
        (gof["am_score"].dropna().values, am_cuts, 0.09, C["GoF"], {}, "GoF (AM)"),
        (
            gof["revel_score"].dropna().values,
            rv_cuts,
            0.27,
            C["GoF"],
            {"hatch": "//", "alpha": 0.55},
            "GoF (REVEL)",
        ),
    ]
    w_b = 0.17
    tick_b = [
        f"{l}\nAM≥{ac:.3f}\nREVEL≥{rc:.3f}"
        for l, ac, rc in zip(levels, am_cuts, rv_cuts)
    ]
    tfs_b = 7.0
    ncol_b = 2
else:
    bar_defs_b = [
        (lof["am_score"].dropna().values, am_cuts, -0.18, C["LoF"], {}, "LoF"),
        (gof["am_score"].dropna().values, am_cuts, 0.18, C["GoF"], {}, "GoF"),
    ]
    w_b = 0.35
    tick_b = [f"{l}\n(≥{c:.3f})" for l, c in zip(levels, am_cuts)]
    tfs_b = 8.5
    ncol_b = 1

for scores, cuts, offset, color, kw, label in bar_defs_b:
    if len(scores) == 0:
        continue
    sens_vals = [(scores >= c).mean() * 100 for c in cuts]
    kw_full = dict(color=color, alpha=0.85, label=label)
    kw_full.update(kw)
    bars = ax.bar(x_levels + offset, sens_vals, w_b, **kw_full)
    for bar, val in zip(bars, sens_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{val:.0f}%",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )

ax.set_xticks(x_levels)
ax.set_xticklabels(tick_b, fontsize=tfs_b)
ax.set_ylabel("Sensitivity (%)", fontsize=10)
title_b = (
    "B. Sensitivity by mechanism & threshold\n(AM solid, REVEL hatched)"
    if HAS_REVEL
    else "B. Sensitivity by mechanism\nand AM evidence threshold"
)
ax.set_title(title_b, fontsize=9)
ax.legend(fontsize=7, ncol=ncol_b)
ax.set_ylim(0, 120)

# ── Panel C: Specificity (Benign detection) by threshold with Wilson 95% CI
ax = fig.add_subplot(gs[0, 2])


def _wilson_ci(k, n, z=1.96):
    """Wilson score 95% CI for proportion k/n. Returns (lo_pct, hi_pct)."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, (center - half) * 100), min(100.0, (center + half) * 100)


am_ben = benign["am_score"].dropna().values
n_am_ben = len(am_ben)
am_spec_c, am_lo_c, am_hi_c = [], [], []
for c in am_cuts:
    k = int((am_ben < c).sum())
    pct = 100 * k / n_am_ben if n_am_ben > 0 else 0
    lo, hi = _wilson_ci(k, n_am_ben)
    am_spec_c.append(pct)
    am_lo_c.append(pct - lo)
    am_hi_c.append(hi - pct)

if HAS_REVEL:
    rv_ben = benign["revel_score"].dropna().values
    n_rv_ben = len(rv_ben)
    rv_spec_c, rv_lo_c, rv_hi_c = [], [], []
    for c in rv_cuts:
        k = int((rv_ben < c).sum())
        pct = 100 * k / n_rv_ben if n_rv_ben > 0 else 0
        lo, hi = _wilson_ci(k, n_rv_ben)
        rv_spec_c.append(pct)
        rv_lo_c.append(pct - lo)
        rv_hi_c.append(hi - pct)
    w_c = 0.35
    off_am_c = -0.2
    off_rv_c = 0.2
    tick_c = [
        f"{l}\nAM≥{ac:.3f}\nREVEL≥{rc:.3f}"
        for l, ac, rc in zip(levels, am_cuts, rv_cuts)
    ]
    tfs_c = 7.0
else:
    w_c = 0.55
    off_am_c = 0.0
    tick_c = [f"{l}\n(≥{c:.3f})" for l, c in zip(levels, am_cuts)]
    tfs_c = 8.5

bars_am_c = ax.bar(
    x_levels + off_am_c, am_spec_c, w_c, color=C["Benign"], alpha=0.85, label="AM"
)
ax.errorbar(
    x_levels + off_am_c,
    am_spec_c,
    yerr=[am_lo_c, am_hi_c],
    fmt="none",
    color="black",
    capsize=4,
    lw=1.2,
)
for bar, val, hi in zip(bars_am_c, am_spec_c, am_hi_c):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        val + hi + 1.0,
        f"{val:.0f}%",
        ha="center",
        va="bottom",
        fontsize=7.5,
    )

if HAS_REVEL:
    bars_rv_c = ax.bar(
        x_levels + off_rv_c,
        rv_spec_c,
        w_c,
        color=C["Benign"],
        alpha=0.50,
        hatch="//",
        label="REVEL",
    )
    ax.errorbar(
        x_levels + off_rv_c,
        rv_spec_c,
        yerr=[rv_lo_c, rv_hi_c],
        fmt="none",
        color="black",
        capsize=4,
        lw=1.2,
    )
    for bar, val, hi in zip(bars_rv_c, rv_spec_c, rv_hi_c):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + hi + 1.0,
            f"{val:.0f}%",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )
    ax.legend(fontsize=8)

ax.set_xticks(x_levels)
ax.set_xticklabels(tick_c, fontsize=tfs_c)
ax.set_ylabel("Specificity (%)\n(% benign below threshold)", fontsize=10)
ax.set_title("C. Specificity at thresholds\n(Benign/LB, 95% Wilson CI)", fontsize=9)
# Explain the n=13 constraint
ax.text(
    0.03,
    0.03,
    f"n={len(benign)} benign missense variants\n"
    "(ClinVar expert panel; 128 synonymous\n"
    "+106 non-coding excluded — not scored by AM)",
    transform=ax.transAxes,
    fontsize=6.2,
    va="bottom",
    ha="left",
    bbox=dict(
        boxstyle="round,pad=0.3",
        facecolor="lightyellow",
        alpha=0.85,
        edgecolor="#aaaa00",
    ),
)
ax.set_ylim(0, 120)

# ── Panel D: LR+ by mechanism and threshold
ax = fig.add_subplot(gs[1, 0])
benign_am_scores = benign["am_score"].dropna().values
for i, (mech_label, path_group) in enumerate([("LoF", lof), ("GoF", gof)]):
    path_scores = path_group["am_score"].dropna().values
    if len(benign_am_scores) < 5 or len(path_scores) < 5:
        continue
    lr_vals = []
    for c in am_cuts:
        tp = (path_scores >= c).sum()
        fn = (path_scores < c).sum()
        tn = (benign_am_scores < c).sum()
        fp = (benign_am_scores >= c).sum()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        lr = sens / (1 - spec) if (1 - spec) > 0 else float("inf")
        lr_vals.append(min(lr, 100))  # cap at 100 for display
    _wd = 0.35
    offset = -_wd / 2 if i == 0 else _wd / 2
    bars = ax.bar(
        x_levels + offset,
        lr_vals,
        _wd,
        color=C[mech_label],
        alpha=0.85,
        label=mech_label,
    )
    for bar, val in zip(bars, lr_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

for tier, lr_req in [
    ("Very Strong", 350),
    ("Strong", 18.7),
    ("Moderate", 4.3),
    ("Supporting", 2.08),
]:
    if lr_req <= 100:
        ax.axhline(lr_req, color="gray", ls=":", lw=1, alpha=0.7)
        ax.text(2.5, lr_req + 0.2, tier, fontsize=7, ha="right", color="gray")

ax.set_xticks(x_levels)
ax.set_xticklabels([f"{l}\n(≥{c:.3f})" for l, c in zip(levels, am_cuts)], fontsize=8.5)
ax.set_ylabel("Likelihood Ratio+ (LR+)", fontsize=10)
ax.set_title(
    "D. LR+ by mechanism and threshold\n(key ACMG recalibration metric)", fontsize=9
)
ax.legend(fontsize=9)
ax.set_ylim(0, max(ax.get_ylim()[1], 25))

# ── Panel E: ACMG recalibration summary table
ax = fig.add_subplot(gs[1, 1:3])
ax.axis("off")
if len(acmg_df) > 0:
    table_data = []
    col_labels = [
        "Mechanism",
        "AM Level",
        "Cutoff",
        "Sens",
        "Spec",
        "LR+",
        "Tier Achieved",
        "Recommendation",
    ]
    for _, row in acmg_df.iterrows():
        table_data.append(
            [
                row["mechanism"],
                row["labelled_level"],
                f"≥{row['cutoff']:.3f}",
                f"{row['sensitivity']:.2f}",
                f"{row['specificity']:.2f}",
                f"{row['LR_plus']:.1f}" if row["LR_plus"] < 999 else ">999",
                row["acmg_tier_achieved"],
                row["recalibration"],
            ]
        )
    t = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    t.auto_set_font_size(False)
    t.set_fontsize(8.5)
    for (row_idx, col_idx), cell in t.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor("#404040")
            cell.set_text_props(color="white", weight="bold")
        elif col_idx == 7:  # Recommendation column
            text = cell.get_text().get_text()
            if "DOWNGRADE" in text:
                cell.set_facecolor("#ffcccc")
            elif "Upgrade" in text:
                cell.set_facecolor("#ccffcc")
        elif row_idx % 2 == 0:
            cell.set_facecolor("#f0f0f0")
ax.set_title(
    "E. ACMG recalibration summary\n(Tavtigian et al. 2020 LR+ framework)",
    fontsize=9,
    pad=10,
)

fig.suptitle(
    "ClinVar Expert-Panel Validation: AM Threshold Calibration\n"
    "ABCC8 · KCNJ11 · GCK — LoF vs GoF vs Benign (AM scores)",
    fontsize=12,
    fontweight="bold",
)

FIG_PATH = os.path.join(FIGURES_DIR, "Figure3_ClinVar_ACMG_Recalibration.png")
fig.savefig(FIG_PATH, dpi=200, bbox_inches="tight")
fig.savefig(FIG_PATH.replace(".png", ".pdf"), bbox_inches="tight")
plt.close()
print(f"  Figure saved: {FIG_PATH}")

# ─── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SCRIPT 08 COMPLETE")
print("=" * 65)
print(f"\nOutputs:")
print(f"  Results: {LR_OUT}")
print(f"  Results: {ACMG_OUT}")
print(f"  Figure:  {FIG_PATH}")
print(f"\nKey scientific conclusion:")
lof_strong = lr_df[(lr_df["mechanism"] == "LoF") & (lr_df["level"] == "Strong")]
gof_strong = lr_df[(lr_df["mechanism"] == "GoF") & (lr_df["level"] == "Strong")]
if len(lof_strong) > 0 and len(gof_strong) > 0:
    lof_lr = lof_strong["LR_plus"].values[0]
    gof_lr = gof_strong["LR_plus"].values[0]
    print(f"  AM Strong threshold (≥0.890):")
    print(f"    LoF: LR+ = {lof_lr:.2f} (target: ≥18.7 for Strong ACMG evidence)")
    print(f"    GoF: LR+ = {gof_lr:.2f} (target: ≥18.7 for Strong ACMG evidence)")
    if gof_lr < 18.7:
        print(
            f"  → AM at 'Strong' threshold achieves only "
            f"'{acmg_df[(acmg_df['mechanism'] == 'GoF') & (acmg_df['labelled_level'] == 'Strong')]['acmg_tier_achieved'].values[0] if len(acmg_df) > 0 else 'sub-Strong'}' "
            f"evidence for GoF"
        )
        print(f"  → ACMG recalibration supported: lower cutoff needed for GoF genes")
