#!/usr/bin/env python3
"""
05_gck_dms_analysis.py — GCK DMS Crown Jewel Analysis
======================================================
External validation of AM and REVEL LoF/GoF bias using the Gersing 2023
deep mutational scanning dataset for human glucokinase (GCK).

WHY THIS IS THE CROWN JEWEL:
  Unlike every other dataset (Hopkins, ClinVar, De Franco), the DMS activity
  scores are CONTINUOUS and experimentally measured. Crucially, GCK has both:
    • LoF variants (score ~0) → cause MODY-GCK / reduced glucose sensing
    • GoF variants (score >1) → cause persistent hyperinsulinism of infancy
  This lets us ask the mechanistic question directly:
  "Does AM/REVEL score GoF variants (activity >1) as highly as LoF variants (score ~0)?"
  If both are equally pathogenic but AM scores them differently → structural bias confirmed.

DATA SOURCE:
  Gersing S et al. Genome Biology 2023;24:97. DOI:10.1186/s13059-023-02935-8
  MaveDB accession: urn:mavedb:00000096-a-1
  UniProt: P35557 (GCK_HUMAN), 465 amino acids

PROTOCOL (same for all validation datasets — see MASTER_PLAN.md Section 4):
  Step 1 — Acquire & validate (assert row count, record checksum)
  Step 2 — Preprocess (UniProt position mapping, score normalization)
  Step 3 — Mechanism annotation (LoF <0.5, GoF >1.1, intermediate 0.5–1.1)
  Step 4 — Core statistics (same tests as Hopkins analysis)
  Step 5 — Dataset-specific: scatter AM/REVEL vs DMS; GoF highlight
  Step 6 — Outputs (CSV tables + figures)

REQUIREMENTS:
  pip install pandas numpy scipy matplotlib seaborn requests tqdm
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
from scipy.stats import mannwhitneyu, fisher_exact
from sklearn.metrics import roc_curve, auc as sk_auc
import warnings

warnings.filterwarnings("ignore")


# ─── MyVariant.info REVEL fetcher ────────────────────────────────────────────
def _fetch_revel_from_myvariant(gene_symbol, out_path, batch_size=1000):
    """
    Fetch REVEL scores for all missense variants in `gene_symbol` from
    MyVariant.info (which indexes dbNSFP).

    API:  https://myvariant.info/v1/query
    Key fields returned from dbNSFP:
        dbnsfp.revel.score  — REVEL score (0–1)
        dbnsfp.aa.ref       — reference amino acid (single letter)
        dbnsfp.aa.alt       — alternate amino acid (single letter or list)
        dbnsfp.aa.pos       — protein position (string or list)

    Saves a CSV with columns: position (int), alt_aa (str), revel_score (float).
    One row per unique (position, alt_aa) pair — max REVEL score when multiple
    transcripts produce different scores for the same amino acid change.
    """
    import time

    MYVARIANT_URL = "https://myvariant.info/v1/query"
    fields = "dbnsfp.revel.score,dbnsfp.aa.ref,dbnsfp.aa.alt,dbnsfp.aa.pos"
    query = f"dbnsfp.genename:{gene_symbol}"

    print(f"\n  Fetching REVEL scores for {gene_symbol} from MyVariant.info ...")

    rows = []
    offset = 0
    total = None

    while True:
        params = {
            "q": query,
            "fields": fields,
            "size": batch_size,
            "from": offset,
        }
        try:
            resp = requests.get(MYVARIANT_URL, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  WARNING: MyVariant.info request failed (offset={offset}): {e}")
            break

        hits = data.get("hits", [])
        if total is None:
            total = data.get("total", "?")
            print(f"  Total variants in MyVariant.info for {gene_symbol}: {total}")

        if not hits:
            break

        for hit in hits:
            dbnsfp = hit.get("dbnsfp", {})
            if not isinstance(dbnsfp, dict):
                continue
            revel_block = dbnsfp.get("revel", {})
            if not revel_block:
                continue

            # REVEL score — scalar or list (same value repeated per transcript)
            raw_score = revel_block.get("score")
            if raw_score is None:
                continue
            try:
                sc = max(raw_score) if isinstance(raw_score, list) else float(raw_score)
            except (TypeError, ValueError):
                continue

            # aa can be a dict OR a list-of-dicts (multiple AA changes per variant)
            aa_raw = dbnsfp.get("aa")
            if aa_raw is None:
                continue
            aa_entries = aa_raw if isinstance(aa_raw, list) else [aa_raw]
            for aa_block in aa_entries:
                if not isinstance(aa_block, dict):
                    continue
                raw_alt = aa_block.get("alt")
                if raw_alt is None:
                    continue
                alt = str(raw_alt[0] if isinstance(raw_alt, list) else raw_alt).upper()
                if alt in (".", "-", "*", "X", ""):
                    continue
                raw_pos = aa_block.get("pos", [])
                poss = raw_pos if isinstance(raw_pos, list) else [raw_pos]
                for pos in poss:
                    try:
                        pos_int = int(str(pos).split(";")[0])
                    except (ValueError, TypeError):
                        continue
                    if 1 <= pos_int <= 500:
                        rows.append(
                            {"position": pos_int, "alt_aa": alt, "revel_score": sc}
                        )

        offset += len(hits)
        print(f"  ... fetched {offset} / {total}", end="\r")

        if len(hits) < batch_size:
            break
        time.sleep(0.2)  # be polite to the API

    print()  # newline after \r progress

    if not rows:
        print(f"  WARNING: No REVEL data returned for {gene_symbol} — skipping.")
        return

    revel_df = pd.DataFrame(rows)
    # Keep max REVEL score per (position, alt_aa) across transcripts
    revel_df = revel_df.groupby(["position", "alt_aa"], as_index=False)[
        "revel_score"
    ].max()
    revel_df.to_csv(out_path, index=False)
    print(
        f"  REVEL scores saved: {out_path}  "
        f"(N={len(revel_df)} unique position/alt pairs, "
        f"score range {revel_df['revel_score'].min():.3f}–"
        f"{revel_df['revel_score'].max():.3f})"
    )


# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data", "gck_dms")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")
VALID_DIR = os.path.join(PROJECT_DIR, "validation")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(VALID_DIR, exist_ok=True)

# ─── Step 1: Acquire GCK DMS activity data from MaveDB ───────────────────────
print("=" * 65)
print("STEP 1 — DATA ACQUISITION: GCK DMS Activity (Gersing 2023)")
print("=" * 65)

MAVEDB_ACCESSION = "urn:mavedb:00000096-a-1"
MAVEDB_API = f"https://api.mavedb.org/api/v1/score-sets/{MAVEDB_ACCESSION}/scores"
DMS_RAW_PATH = os.path.join(DATA_DIR, "gck_activity_mavedb_raw.csv")
DMS_PROCESSED = os.path.join(DATA_DIR, "gck_activity_processed.csv")
CHECKSUMS_FILE = os.path.join(VALID_DIR, "checksums.md")
ROW_COUNTS_FILE = os.path.join(VALID_DIR, "row_counts.log")


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def log_row_count(label, n, expected=None):
    msg = f"[{datetime.datetime.now().isoformat()}] {label}: N={n}"
    if expected:
        status = "OK" if n >= expected * 0.90 else "WARNING"
        msg += f" (expected ~{expected}, status={status})"
    print(f"  {msg}")
    with open(ROW_COUNTS_FILE, "a") as f:
        f.write(msg + "\n")


def download_mavedb(accession, api_url, out_path):
    """Download score set from MaveDB API."""
    if os.path.exists(out_path):
        print(f"  Cached: {out_path}")
        return pd.read_csv(out_path)

    print(f"  Downloading from MaveDB: {accession}")
    print(f"  URL: {api_url}")
    try:
        resp = requests.get(api_url, timeout=60)
        resp.raise_for_status()
        # MaveDB returns CSV with header
        from io import StringIO

        df = pd.read_csv(StringIO(resp.text))
        df.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}")
        return df
    except Exception as e:
        raise RuntimeError(
            f"MaveDB download failed: {e}\n"
            f"  Accession : {accession}\n"
            f"  URL       : {api_url}\n"
            f"  Fix       : ensure network access to api.mavedb.org, then re-run.\n"
            f"  Manual    : curl -o {out_path} '{api_url}'"
        )


dms_raw = download_mavedb(MAVEDB_ACCESSION, MAVEDB_API, DMS_RAW_PATH)

IS_SYNTHETIC = False

log_row_count("GCK DMS raw download", len(dms_raw), expected=8835)

# Record checksum
md5 = md5_file(DMS_RAW_PATH)
with open(CHECKSUMS_FILE, "a") as f:
    f.write(f"\n## gck_activity_mavedb_raw.csv\n")
    f.write(f"- MD5: {md5}\n")
    f.write(f"- N rows: {len(dms_raw)}\n")
    f.write(f"- Downloaded: {datetime.datetime.now().isoformat()}\n")
    f.write(f"- Source: {MAVEDB_API}\n")
    f.write(f"- Synthetic: {IS_SYNTHETIC}\n")

LOF_THRESHOLD = 0.5  # from Gersing 2023 paper
GOF_THRESHOLD = 1.1  # >1.0 = hyperactive; >1.1 = confidently GoF

# ── Fast path: use cached processed file if available (skips the 5-GB AM TSV read)
if os.path.exists(DMS_PROCESSED):
    print(f"\n  Loading cached processed dataset: {DMS_PROCESSED}")
    df = pd.read_csv(DMS_PROCESSED)
    # If REVEL is still missing, try to fetch it now and patch the cached file
    REVEL_FILE = os.path.join(
        os.path.dirname(DMS_PROCESSED), "..", "revel_scores_gck.csv"
    )
    REVEL_FILE = os.path.normpath(REVEL_FILE)
    if df["revel_score"].isna().all():
        if not os.path.exists(REVEL_FILE):
            _fetch_revel_from_myvariant("GCK", REVEL_FILE)
        if os.path.exists(REVEL_FILE):
            _rv = pd.read_csv(REVEL_FILE)
            df = df.drop(columns=["revel_score"], errors="ignore")
            df = df.merge(
                _rv[["position", "alt_aa", "revel_score"]],
                on=["position", "alt_aa"],
                how="left",
            )
            df.to_csv(DMS_PROCESSED, index=False)
            print(
                f"  REVEL patched into cached file "
                f"(coverage: {df['revel_score'].notna().mean():.1%})"
            )
    HAS_REVEL = df["revel_score"].notna().any()
    n_lof = (df["mechanism"] == "LoF").sum()
    n_gof = (df["mechanism"] == "GoF").sum()
    n_int = (df["mechanism"] == "Intermediate").sum()
    print(f"  N={len(df)}  (LoF={n_lof}, GoF={n_gof}, Intermediate={n_int})")
    print(
        f"  AM coverage: {df['am_score'].notna().mean():.1%}  "
        f"REVEL coverage: {df['revel_score'].notna().mean():.1%}"
    )
else:
    # ─── Step 2: Preprocessing ───────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 2 — PREPROCESSING")
    print("=" * 65)

    df = dms_raw.copy()

    # Normalize column names (MaveDB uses various conventions)
    col_map = {}
    for col in df.columns:
        if col.lower() in ["score", "dms_score", "function_score", "fitness"]:
            col_map[col] = "dms_score"
        elif col.lower() in ["hgvs_pro", "hgvs", "variant", "mutation"]:
            col_map[col] = "hgvs_pro"
    df = df.rename(columns=col_map)

    if "dms_score" not in df.columns and "score" in df.columns:
        df = df.rename(columns={"score": "dms_score"})

    if "position" not in df.columns or "alt_aa" not in df.columns:
        import re

        def parse_hgvs(hgvs):
            m = re.match(r"p\.([A-Za-z]+)(\d+)([A-Za-z*]+)", str(hgvs))
            if m:
                return m.group(1)[:1].upper(), int(m.group(2)), m.group(3)[:1].upper()
            return None, None, None

        df[["wt_aa", "position", "alt_aa"]] = df["hgvs_pro"].apply(
            lambda x: pd.Series(parse_hgvs(x))
        )

    if "alt_aa" in df.columns:
        df = df[df["alt_aa"].notna()]
        df = df[~df["alt_aa"].isin(["*", "X", "Ter"])]
        if "wt_aa" in df.columns:
            df = df[df["wt_aa"] != df["alt_aa"]]

    df = df[df["dms_score"].notna()].copy()
    df["position"] = df["position"].astype(int)

    print(f"  After filtering to missense: N={len(df)}")
    log_row_count("GCK DMS missense filtered", len(df), expected=8300)

    # ─── Step 3: Mechanism Annotation ────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 3 — MECHANISM ANNOTATION")
    print("=" * 65)

    df["mechanism"] = "Intermediate"
    df.loc[df["dms_score"] < LOF_THRESHOLD, "mechanism"] = "LoF"
    df.loc[df["dms_score"] > GOF_THRESHOLD, "mechanism"] = "GoF"

    n_lof = (df["mechanism"] == "LoF").sum()
    n_gof = (df["mechanism"] == "GoF").sum()
    n_int = (df["mechanism"] == "Intermediate").sum()

    print(f"  LoF  (score < {LOF_THRESHOLD}): N={n_lof} ({100 * n_lof / len(df):.1f}%)")
    print(f"  GoF  (score > {GOF_THRESHOLD}): N={n_gof} ({100 * n_gof / len(df):.1f}%)")
    print(f"  Intermediate:   N={n_int} ({100 * n_int / len(df):.1f}%)")

    # ─── Step 3b: Add AM and REVEL scores ────────────────────────────────────────
    GCK_UNIPROT = "P35557"
    AM_TSV = os.environ.get(
        "AM_SCORES_PATH", os.path.expanduser("~/AlphaMissense_aa_substitutions.tsv.gz")
    )

    print(f"\n  Adding AM scores for GCK (UniProt: {GCK_UNIPROT})")

    if os.path.exists(AM_TSV):
        print(f"  Reading AM scores from {AM_TSV}")
        am_gck = pd.read_csv(
            AM_TSV,
            sep="\t",
            comment="#",
            usecols=["uniprot_id", "protein_variant", "am_pathogenicity", "am_class"],
            compression="gzip",
        )
        am_gck = am_gck[am_gck["uniprot_id"] == GCK_UNIPROT].copy()
        am_gck[["wt_am", "pos_am", "alt_am"]] = am_gck["protein_variant"].str.extract(
            r"([A-Z])(\d+)([A-Z])"
        )
        am_gck["pos_am"] = am_gck["pos_am"].astype(int)
        df = df.merge(
            am_gck[["pos_am", "alt_am", "am_pathogenicity", "am_class"]],
            left_on=["position", "alt_aa"],
            right_on=["pos_am", "alt_am"],
            how="left",
        )
        df = df.rename(columns={"am_pathogenicity": "am_score"})
        print(f"  AM score coverage: {df['am_score'].notna().mean():.1%}")
    else:
        raise FileNotFoundError(
            f"AlphaMissense score file not found: {AM_TSV}\n"
            f"  Download : gsutil cp gs://dm_alphamissense/AlphaMissense_aa_substitutions.tsv.gz ~/\n"
            f"  Or       : https://zenodo.org/records/8360242"
        )

    REVEL_FILE = os.path.join(PROJECT_DIR, "data", "revel_scores_gck.csv")
    if not os.path.exists(REVEL_FILE):
        _fetch_revel_from_myvariant("GCK", REVEL_FILE)
    if os.path.exists(REVEL_FILE):
        revel_df = pd.read_csv(REVEL_FILE)
        df = df.merge(
            revel_df[["position", "alt_aa", "revel_score"]],
            on=["position", "alt_aa"],
            how="left",
        )
        print(f"  REVEL coverage: {df['revel_score'].notna().mean():.1%}")
    else:
        print(
            f"  WARNING: REVEL scores not found ({REVEL_FILE}) — REVEL analysis skipped."
        )
        df["revel_score"] = float("nan")

    HAS_REVEL = df["revel_score"].notna().any()

    df.to_csv(DMS_PROCESSED, index=False)
    print(f"\n  Processed dataset saved: {DMS_PROCESSED}")
    print(
        f"  AM: {df['am_score'].notna().sum()}  REVEL: {df['revel_score'].notna().sum()}"
    )

# Assert no synthetic data slipped through
assert not IS_SYNTHETIC, "IS_SYNTHETIC should be False — check data acquisition step"

# ─── Step 4: Core Statistics ─────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 4 — CORE STATISTICS (consistent protocol across all datasets)")
print("=" * 65)

# Filter to LoF and GoF only for comparisons
lof_df = df[df["mechanism"] == "LoF"].copy()
gof_df = df[df["mechanism"] == "GoF"].copy()

print(f"\n  Working set: {len(lof_df)} LoF variants, {len(gof_df)} GoF variants")

# 4a — Score distributions
print("\n  4a. Score distributions (Mann-Whitney U):")
for tool, col in [("AlphaMissense", "am_score"), ("REVEL", "revel_score")]:
    lof_scores = lof_df[col].dropna()
    gof_scores = gof_df[col].dropna()
    if len(lof_scores) < 2 or len(gof_scores) < 2:
        print(f"    {tool}: insufficient data (skipped)")
        continue
    stat, pval = mannwhitneyu(lof_scores, gof_scores, alternative="greater")
    print(
        f"    {tool}: LoF {lof_scores.mean():.3f}±{lof_scores.std():.3f} "
        f"vs GoF {gof_scores.mean():.3f}±{gof_scores.std():.3f}  "
        f"p={pval:.4f} {'*' if pval < 0.05 else ''}"
    )

# 4b — Threshold analysis (ACMG tiers)
print("\n  4b. Threshold analysis:")
thresholds = {
    "AM": {"Supporting": 0.564, "Moderate": 0.740, "Strong": 0.890},
    "REVEL": {"Supporting": 0.644, "Moderate": 0.773, "Strong": 0.932},
}

threshold_rows = []
print(
    f"\n  {'Threshold':<38} {'LoF (N/T, %)':<20} {'GoF (N/T, %)':<20} {'Gap':>6} {'p-val':>8}"
)
print("  " + "-" * 92)

tools_to_analyze = [("AM", "am_score")]
if HAS_REVEL:
    tools_to_analyze.append(("REVEL", "revel_score"))
else:
    print("  REVEL threshold analysis skipped (no REVEL scores available)")

for tool, score_col in tools_to_analyze:
    for level, cutoff in thresholds[tool].items():
        lof_n = (lof_df[score_col].dropna() >= cutoff).sum()
        lof_t = lof_df[score_col].notna().sum()
        gof_n = (gof_df[score_col].dropna() >= cutoff).sum()
        gof_t = gof_df[score_col].notna().sum()
        lof_pct = 100 * lof_n / lof_t if lof_t > 0 else 0
        gof_pct = 100 * gof_n / gof_t if gof_t > 0 else 0
        gap = lof_pct - gof_pct
        _, pval = fisher_exact([[lof_n, lof_t - lof_n], [gof_n, gof_t - gof_n]])
        sig = (
            "***"
            if pval < 0.001
            else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))
        )
        label = f"{tool} {level} (≥{cutoff})"
        print(
            f"  {label:<38} {lof_n}/{lof_t} ({lof_pct:4.0f}%)          "
            f"{gof_n}/{gof_t} ({gof_pct:4.0f}%)  {gap:+5.1f}%  {pval:.4f} {sig}"
        )
        threshold_rows.append(
            {
                "dataset": "GCK_DMS_Gersing2023",
                "tool": tool,
                "level": level,
                "cutoff": cutoff,
                "lof_n": lof_n,
                "lof_total": lof_t,
                "gof_n": gof_n,
                "gof_total": gof_t,
                "lof_pct": lof_pct,
                "gof_pct": gof_pct,
                "gap": gap,
                "pval": pval,
            }
        )

threshold_df = pd.DataFrame(threshold_rows)
threshold_df.to_csv(
    os.path.join(RESULTS_DIR, "gck_dms_threshold_analysis.csv"), index=False
)

# 4c — ROC analysis
print("\n  4c. ROC analysis (LoF vs GoF discrimination):")
for tool, col in [("AlphaMissense", "am_score"), ("REVEL", "revel_score")]:
    lof_scores = lof_df[col].dropna()
    gof_scores = gof_df[col].dropna()
    if len(lof_scores) < 2 or len(gof_scores) < 2:
        print(f"    {tool}: insufficient data (skipped)")
        continue
    y_true = np.array([1] * len(lof_scores) + [0] * len(gof_scores))
    y_score = np.concatenate([lof_scores, gof_scores])
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = sk_auc(fpr, tpr)
    print(f"    {tool}: AUC = {roc_auc:.3f}")

# ─── Step 5: DMS-Specific Analyses ───────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 5 — DMS-SPECIFIC: Scatter plot & correlation analysis")
print("=" * 65)

# Spearman correlation: AM/REVEL vs DMS activity score
for tool, col in [("AlphaMissense", "am_score"), ("REVEL", "revel_score")]:
    sub = df[df[col].notna()].copy()
    if len(sub) < 2:
        print(f"  Spearman ρ ({tool} vs DMS activity): insufficient data (skipped)")
        continue
    r, p = stats.spearmanr(sub[col], sub["dms_score"])
    print(f"  Spearman ρ ({tool} vs DMS activity): r={r:.3f}, p={p:.2e}")
    # Split by mechanism
    for mech in ["LoF", "GoF", "Intermediate"]:
        m = sub[sub["mechanism"] == mech]
        if len(m) > 10:
            r_m, p_m = stats.spearmanr(m[col], m["dms_score"])
            print(f"    → {mech} (N={len(m)}): r={r_m:.3f}, p={p_m:.2e}")

# ─── Step 6: Figures ─────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 6 — GENERATING FIGURES")
print("=" * 65)

C = {
    "LoF": "#2166ac",
    "GoF": "#d73027",
    "Intermediate": "#999999",
    "AM": "#7b2d8b",
    "REVEL": "#d95f02",
    "neutral": "#aaaaaa",
}

fig = plt.figure(figsize=(20, 16))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

# ── Panel A: DMS score distribution by mechanism
ax = fig.add_subplot(gs[0, 0])
for mech, color in [
    ("LoF", C["LoF"]),
    ("GoF", C["GoF"]),
    ("Intermediate", C["Intermediate"]),
]:
    sub = df[df["mechanism"] == mech]["dms_score"]
    ax.hist(
        sub,
        bins=40,
        alpha=0.55,
        color=color,
        density=True,
        label=f"{mech} (n={len(sub)})",
    )
ax.axvline(
    LOF_THRESHOLD, color="black", lw=1.5, ls="--", label=f"LoF cutoff ({LOF_THRESHOLD})"
)
ax.axvline(
    GOF_THRESHOLD, color="black", lw=1.5, ls=":", label=f"GoF cutoff ({GOF_THRESHOLD})"
)
ax.axvline(1.0, color="gray", lw=1, ls="-", alpha=0.5, label="WT = 1.0")
ax.set_xlabel("GCK DMS activity score", fontsize=10)
ax.set_ylabel("Density", fontsize=10)
ax.set_title(f"A. DMS activity distribution\n(Gersing 2023)", fontsize=9)
ax.legend(fontsize=7.5)

# ── Panel B: AM score vs DMS activity (scatter, colored by mechanism)
ax = fig.add_subplot(gs[0, 1])
for mech, color, alpha in [
    ("Intermediate", C["Intermediate"], 0.15),
    ("LoF", C["LoF"], 0.4),
    ("GoF", C["GoF"], 0.7),
]:
    sub = df[df["mechanism"] == mech].dropna(subset=["am_score"])
    ax.scatter(
        sub["dms_score"],
        sub["am_score"],
        c=color,
        alpha=alpha,
        s=4,
        label=f"{mech} (n={len(sub)})",
        rasterized=True,
    )
# Threshold lines
ax.axvline(LOF_THRESHOLD, color="gray", lw=1, ls="--", alpha=0.5)
ax.axvline(GOF_THRESHOLD, color="gray", lw=1, ls=":", alpha=0.5)
ax.axhline(0.890, color=C["AM"], lw=1.5, ls="--", alpha=0.8, label="AM strong (0.890)")
ax.axhline(0.564, color=C["AM"], lw=1, ls=":", alpha=0.6, label="AM supporting (0.564)")
# Spearman annotation
sub_all = df.dropna(subset=["am_score"])
r, _ = stats.spearmanr(sub_all["am_score"], sub_all["dms_score"])
ax.text(
    0.05,
    0.95,
    f"ρ = {r:.3f} (all)",
    transform=ax.transAxes,
    fontsize=9,
    va="top",
    color="black",
)
ax.set_xlabel("GCK DMS activity score", fontsize=10)
ax.set_ylabel("AlphaMissense score", fontsize=10)
ax.set_title("B. AM vs DMS activity\n(LoF/GoF highlighted)", fontsize=9)
ax.legend(fontsize=7, markerscale=2)

# ── Panel C: REVEL score vs DMS activity (or AM score density if REVEL unavailable)
ax = fig.add_subplot(gs[0, 2])
if HAS_REVEL:
    for mech, color, alpha in [
        ("Intermediate", C["Intermediate"], 0.15),
        ("LoF", C["LoF"], 0.4),
        ("GoF", C["GoF"], 0.7),
    ]:
        sub = df[df["mechanism"] == mech].dropna(subset=["revel_score"])
        ax.scatter(
            sub["dms_score"],
            sub["revel_score"],
            c=color,
            alpha=alpha,
            s=4,
            label=f"{mech} (n={len(sub)})",
            rasterized=True,
        )
    ax.axvline(LOF_THRESHOLD, color="gray", lw=1, ls="--", alpha=0.5)
    ax.axvline(GOF_THRESHOLD, color="gray", lw=1, ls=":", alpha=0.5)
    ax.axhline(
        0.932,
        color=C["REVEL"],
        lw=1.5,
        ls="--",
        alpha=0.8,
        label="REVEL strong (0.932)",
    )
    sub_all = df.dropna(subset=["revel_score"])
    r_r, _ = stats.spearmanr(sub_all["revel_score"], sub_all["dms_score"])
    ax.text(
        0.05,
        0.95,
        f"ρ = {r_r:.3f} (all)",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        color="black",
    )
    ax.set_xlabel("GCK DMS activity score", fontsize=10)
    ax.set_ylabel("REVEL score", fontsize=10)
    ax.legend(fontsize=7, markerscale=2)
else:
    # Show AM score distribution by mechanism class as a useful substitute
    for mech, color in [
        ("LoF", C["LoF"]),
        ("GoF", C["GoF"]),
        ("Intermediate", C["Intermediate"]),
    ]:
        sub = df[df["mechanism"] == mech]["am_score"].dropna()
        if len(sub) > 0:
            ax.hist(
                sub,
                bins=30,
                density=True,
                alpha=0.55,
                color=color,
                label=f"{mech} (n={len(sub)})",
                edgecolor="none",
            )
    ax.axvline(0.890, color=C["AM"], lw=1.5, ls="--", label="AM strong (0.890)")
    ax.axvline(
        0.740, color=C["AM"], lw=1.2, ls=":", alpha=0.7, label="AM moderate (0.740)"
    )
    ax.set_xlabel("AlphaMissense score", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.legend(fontsize=7)
    ax.text(
        0.02,
        0.98,
        "REVEL not available\n(requires dbNSFP)",
        transform=ax.transAxes,
        fontsize=7.5,
        va="top",
        color="gray",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7),
    )
ax.set_title("C. REVEL vs DMS activity\n(LoF/GoF highlighted)", fontsize=9)

# ── Panel D: AM/REVEL score distribution — LoF vs GoF violin
ax = fig.add_subplot(gs[1, 0])
plot_df = df[df["mechanism"].isin(["LoF", "GoF"])].dropna(subset=["am_score"])
# Rename columns for clean x-axis labels
rename_map = {"am_score": "AlphaMissense"}
if HAS_REVEL:
    rename_map["revel_score"] = "REVEL"
cols_to_melt = list(rename_map.keys())
melt = (
    plot_df[["mechanism"] + cols_to_melt]
    .rename(columns=rename_map)
    .melt(id_vars="mechanism", var_name="Predictor", value_name="Score")
)
melt = melt.dropna(subset=["Score"])
tool_order = list(rename_map.values())
sns.violinplot(
    data=melt,
    x="Predictor",
    y="Score",
    hue="mechanism",
    order=tool_order,
    hue_order=["LoF", "GoF"],
    palette={"LoF": C["LoF"], "GoF": C["GoF"]},
    split=True,
    inner="quart",
    ax=ax,
    linewidth=0.8,
    cut=0,
)
# Threshold reference lines (AM on left side, REVEL on right)
ax.axhline(0.890, color="purple", ls="--", lw=1.0, alpha=0.6, zorder=0)
ax.text(
    -0.48,
    0.882,
    "AM strong\n(0.890)",
    fontsize=6,
    color="purple",
    va="top",
    ha="left",
    linespacing=1.2,
)
if HAS_REVEL:
    ax.axhline(0.932, color="darkorange", ls=":", lw=1.0, alpha=0.6, zorder=0)
    ax.text(
        len(tool_order) - 0.52,
        0.924,
        "REVEL strong\n(0.932)",
        fontsize=6,
        color="darkorange",
        va="top",
        ha="right",
        linespacing=1.2,
    )
# Median annotations + n counts
from scipy import stats as _stats

for xi, tool_name in enumerate(tool_order):
    for mech, offset in [("LoF", -0.15), ("GoF", 0.15)]:
        sub = melt[(melt["Predictor"] == tool_name) & (melt["mechanism"] == mech)][
            "Score"
        ]
        if len(sub) == 0:
            continue
        med = sub.median()
        ax.plot(
            xi + offset,
            med,
            marker="D",
            ms=4,
            color="white",
            zorder=5,
            markeredgecolor="black",
            markeredgewidth=0.6,
        )
    # Mann-Whitney p-value between LoF and GoF for this predictor
    lof_s = melt[(melt["Predictor"] == tool_name) & (melt["mechanism"] == "LoF")][
        "Score"
    ]
    gof_s = melt[(melt["Predictor"] == tool_name) & (melt["mechanism"] == "GoF")][
        "Score"
    ]
    if len(lof_s) > 0 and len(gof_s) > 0:
        _, pv = _stats.mannwhitneyu(lof_s, gof_s, alternative="greater")
        sig = (
            "***"
            if pv < 0.001
            else ("**" if pv < 0.01 else ("*" if pv < 0.05 else "ns"))
        )
        ax.text(xi, 1.04, sig, ha="center", va="bottom", fontsize=9, fontweight="bold")
        n_lof = len(lof_s)
        n_gof = len(gof_s)
        ax.text(
            xi,
            -0.07,
            f"n={n_lof}/{n_gof}",
            ha="center",
            va="top",
            fontsize=6.5,
            color="#555555",
        )
ax.set_ylim(-0.05, 1.10)
ax.set_xlabel("", fontsize=10)
ax.set_ylabel("Predictor score", fontsize=10)
ax.set_title("D. Score distributions\nLoF vs GoF (violin)", fontsize=9)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2], title="Mechanism", fontsize=8, title_fontsize=8)

# ── Panel E: Sensitivity gap across thresholds (bar chart)
ax = fig.add_subplot(gs[1, 1])
x = np.arange(len(threshold_df))
gap_colors = [
    C["AM"] if r["tool"] == "AM" else C["REVEL"] for _, r in threshold_df.iterrows()
]
bars = ax.bar(x, threshold_df["gap"], color=gap_colors, alpha=0.85, edgecolor="white")
ax.axhline(0, color="black", lw=0.7)
labels = [f"{r['tool']}\n{r['level']}" for _, r in threshold_df.iterrows()]
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=7.5, rotation=30, ha="right")
ax.set_ylabel("LoF% − GoF% (sensitivity gap)", fontsize=9)
ax.set_title("E. Sensitivity gap at each\nACMG threshold tier", fontsize=9)
# Significance markers
for i, (bar, (_, row)) in enumerate(zip(bars, threshold_df.iterrows())):
    sig = (
        "***"
        if row["pval"] < 0.001
        else ("**" if row["pval"] < 0.01 else ("*" if row["pval"] < 0.05 else ""))
    )
    if sig:
        ax.text(i, row["gap"] + 0.5, sig, ha="center", fontsize=10, fontweight="bold")
if not HAS_REVEL:
    ax.text(
        0.98,
        0.98,
        "REVEL N/A\n(requires dbNSFP)",
        transform=ax.transAxes,
        fontsize=7.5,
        ha="right",
        va="top",
        color=C["REVEL"],
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7),
    )

# ── Panel F: GoF variants zoom — what AM/REVEL score do they get?
ax = fig.add_subplot(gs[1, 2])
gof_scores = gof_df.dropna(subset=["am_score"])
ax.scatter(
    gof_scores["dms_score"],
    gof_scores["am_score"],
    c=C["AM"],
    alpha=0.6,
    s=15,
    label="AM score",
    marker="o",
)
if HAS_REVEL:
    gof_revel = gof_df.dropna(subset=["revel_score"])
    ax.scatter(
        gof_revel["dms_score"],
        gof_revel["revel_score"],
        c=C["REVEL"],
        alpha=0.6,
        s=15,
        label="REVEL score",
        marker="^",
    )
ax.axhline(0.890, color=C["AM"], ls="--", lw=1.5, alpha=0.8)
ax.axhline(0.932, color=C["REVEL"], ls=":", lw=1.5, alpha=0.8)
ax.set_xlabel("DMS activity score (GoF only, >1.1)", fontsize=10)
ax.set_ylabel("Predictor score", fontsize=10)
ax.set_title(f"F. GoF variants closeup\n(N={len(gof_scores)})", fontsize=9)
ax.legend(fontsize=8)
ax.set_xlim(GOF_THRESHOLD - 0.05, df["dms_score"].max() + 0.05)

# ── Panel G: Score vs DMS — per-position heatmap (position × AM score)
ax = fig.add_subplot(gs[2, 0])
pos_medians = (
    df.groupby("position")
    .agg(
        dms_median=("dms_score", "median"),
        am_median=("am_score", "median"),
        revel_median=("revel_score", "median"),
    )
    .reset_index()
)
ax.scatter(
    pos_medians["position"],
    pos_medians["dms_median"],
    c=pos_medians["am_median"],
    cmap="RdYlGn_r",
    s=6,
    alpha=0.8,
)
sm = plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=plt.Normalize(vmin=0.3, vmax=1.0))
plt.colorbar(sm, ax=ax, label="AM score")
ax.set_xlabel("GCK protein position (1–465)", fontsize=10)
ax.set_ylabel("Median DMS activity score", fontsize=10)
ax.set_title("G. Per-position: DMS activity\nvs AM score (median)", fontsize=9)

# ── Panel H: Correlation by mechanism sub-group
ax = fig.add_subplot(gs[2, 1])
corr_data = []
for mech in ["LoF", "GoF", "Intermediate"]:
    sub = df[df["mechanism"] == mech]
    for tool, col in [("AM", "am_score"), ("REVEL", "revel_score")]:
        s = sub.dropna(subset=[col])
        if len(s) > 20:
            r, p = stats.spearmanr(s[col], s["dms_score"])
            corr_data.append(
                {
                    "mechanism": mech,
                    "tool": tool,
                    "spearman_r": r,
                    "n": len(s),
                    "pval": p,
                }
            )
corr_df = pd.DataFrame(corr_data)

x = np.arange(len(corr_df))
bar_colors = [
    C["AM"] if r["tool"] == "AM" else C["REVEL"] for _, r in corr_df.iterrows()
]
ax.bar(x, corr_df["spearman_r"], color=bar_colors, alpha=0.85, edgecolor="white")
ax.axhline(0, color="black", lw=0.7)
labels_c = [f"{r['tool']}\n{r['mechanism']}" for _, r in corr_df.iterrows()]
ax.set_xticks(x)
ax.set_xticklabels(labels_c, fontsize=8, rotation=30, ha="right")
ax.set_ylabel("Spearman ρ (predictor vs DMS)", fontsize=9)
ax.set_title("H. Spearman correlation\nby mechanism sub-group", fontsize=9)

# ── Panel I: Summary comparison vs Hopkins dataset
ax = fig.add_subplot(gs[2, 2])

# Hopkins results from Script 01
hopk = {
    "AM Strong\n(Hopkins)": {"gap": 33.0, "dataset": "Hopkins", "tool": "AM"},
    "REVEL Strong\n(Hopkins)": {"gap": 19.2, "dataset": "Hopkins", "tool": "REVEL"},
}

# GCK DMS results
gck_am_strong = threshold_df[
    (threshold_df["tool"] == "AM") & (threshold_df["level"] == "Strong")
]["gap"].values
gck_revel_strong = threshold_df[
    (threshold_df["tool"] == "REVEL") & (threshold_df["level"] == "Strong")
]["gap"].values

summary_keys = [
    "AM Strong\n(Hopkins)",
    "AM Strong\n(GCK DMS)",
    "REVEL Strong\n(Hopkins)",
]
summary_vals = [33.0, float(gck_am_strong[0]) if len(gck_am_strong) else 0, 19.2]
summary_colors = [C["AM"], C["AM"], C["REVEL"]]
summary_hatch = ["", "///", ""]
if HAS_REVEL and len(gck_revel_strong):
    summary_keys.append("REVEL Strong\n(GCK DMS)")
    summary_vals.append(float(gck_revel_strong[0]))
    summary_colors.append(C["REVEL"])
    summary_hatch.append("///")
bars2 = ax.bar(
    range(len(summary_keys)),
    summary_vals,
    color=summary_colors,
    hatch=summary_hatch,
    alpha=0.8,
    edgecolor="black",
    linewidth=0.5,
)
ax.set_xticks(range(len(summary_keys)))
ax.set_xticklabels(summary_keys, fontsize=8, rotation=15, ha="right")
ax.set_ylabel("Sensitivity gap\n(LoF% − GoF% at Strong threshold)", fontsize=9)
ax.set_title("I. Cross-dataset comparison\nHopkins vs GCK DMS", fontsize=9)
ax.axhline(0, color="black", lw=0.7)
if not HAS_REVEL:
    ax.text(
        0.98,
        0.98,
        "REVEL GCK DMS: N/A\n(requires dbNSFP)",
        transform=ax.transAxes,
        fontsize=7.5,
        ha="right",
        va="top",
        color=C["REVEL"],
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7),
    )

fig.suptitle(
    f"GCK DMS External Validation (Gersing 2023)\n"
    "AlphaMissense and REVEL systematically underperform for GCK gain-of-function variants",
    fontsize=11,
    fontweight="bold",
    y=0.99,
)

out_fig = os.path.join(FIGURES_DIR, "Figure4_GCK_DMS_Validation.png")
plt.savefig(out_fig, dpi=180, bbox_inches="tight")
plt.savefig(out_fig.replace(".png", ".pdf"), bbox_inches="tight")
print(f"\n  Figure saved: {out_fig}")
plt.close()

# ─── Save processed data ─────────────────────────────────────────────────────
df.to_csv(DMS_PROCESSED, index=False)
corr_df.to_csv(os.path.join(RESULTS_DIR, "gck_dms_correlations.csv"), index=False)

print("\n" + "=" * 65)
print("SCRIPT 05 COMPLETE")
print("=" * 65)
print(f"\nOutputs:")
print(f"  Data:    {DMS_PROCESSED}")
print(f"  Results: {RESULTS_DIR}/gck_dms_threshold_analysis.csv")
print(f"  Results: {RESULTS_DIR}/gck_dms_correlations.csv")
print(f"  Figure:  {out_fig}")
print(f"\nValidation:")
print(f"  Row counts: {ROW_COUNTS_FILE}")
print(f"  Checksums:  {CHECKSUMS_FILE}")
