"""
00_download_data.py — Raw data download for absolute reproducibility
=====================================================================
Creates data/ directory tree and downloads all raw data sources.

Usage:
    python scripts/00_download_data.py

Data sources:
    1. Hopkins cohort     — copied from results/all_variants_annotated.csv
    2. GCK DMS            — MaveDB API (urn:mavedb:00000096-a-1)
    3. ClinVar            — NCBI FTP (variant_summary.txt.gz)
    4. ProteinGym v1.0    — Zenodo 13936340 (clinical_substitutions.parquet)
    5. GoFCards v1.0      — GoFCards REST API (ABCC8/KCNJ11 GOF variants)
    6. AlphaMissense      — MANUAL: zenodo.org/records/8360242 (~3.5 GB)
    7. REVEL v1.3         — MANUAL: sites.google.com/site/revelgenomics

Run time: ~5-15 min depending on network (excludes large AM file).
"""

import csv
import hashlib
import os
import shutil
import time
from pathlib import Path

import requests

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"

SUBDIRS = [
    DATA_DIR,
    DATA_DIR / "gck_dms",
    DATA_DIR / "clinvar",
    DATA_DIR / "proteingym",
    DATA_DIR / "gofcards",
    DATA_DIR / "alphamissense",
]

MANIFEST = DATA_DIR / "MANIFEST.md"


# ── helpers ──────────────────────────────────────────────────────────────────


def md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk_data in iter(lambda: f.read(chunk), b""):
            h.update(chunk_data)
    return h.hexdigest()


def download(
    url: str, dest: Path, desc: str, headers: dict = None, timeout: int = 120
) -> bool:
    """Stream-download url → dest with progress dots."""
    if dest.exists():
        print(f"  [SKIP] {desc} — already exists ({dest.name})")
        return True
    print(f"  [DL]   {desc}")
    print(f"         {url}")
    try:
        r = requests.get(url, stream=True, headers=headers, timeout=timeout)
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            written = 0
            for chunk_data in r.iter_content(chunk_size=1 << 20):
                f.write(chunk_data)
                written += len(chunk_data)
                print(".", end="", flush=True)
        print(f" {written / 1e6:.1f} MB")
        return True
    except Exception as e:
        print(f"\n  [FAIL] {e}")
        if dest.exists():
            dest.unlink()
        return False


# ── 0. Create directories ─────────────────────────────────────────────────────

print("=" * 60)
print("00_download_data.py — reproducibility data download")
print("=" * 60)

for d in SUBDIRS:
    d.mkdir(parents=True, exist_ok=True)
print(f"Created {len(SUBDIRS)} data subdirectories under {DATA_DIR}/\n")

status = {}

# ── 1. Hopkins cohort (copy from results/) ───────────────────────────────────

print("── 1. Hopkins cohort (131 variants) ────────────────────────")
src = RESULTS_DIR / "all_variants_annotated.csv"
dest = DATA_DIR / "am_revel_scores_combined.csv"

if dest.exists():
    print("  [SKIP] already exists: data/am_revel_scores_combined.csv")
elif src.exists():
    shutil.copy2(src, dest)
    print(
        "  [OK]   Copied results/all_variants_annotated.csv → data/am_revel_scores_combined.csv"
    )
else:
    print(f"  [FAIL] Source not found: {src}")

status["hopkins"] = dest.exists()
print()

# ── 2. GCK DMS — MaveDB urn:mavedb:00000096-a-1 ──────────────────────────────

print("── 2. GCK DMS (MaveDB urn:mavedb:00000096-a-1) ─────────────")
MAVEDB_URL = "https://api.mavedb.org/api/v1/score-sets/urn:mavedb:00000096-a-1/scores"
MAVEDB_DEST = DATA_DIR / "gck_dms" / "gck_activity_mavedb_raw.csv"

ok = download(
    MAVEDB_URL,
    MAVEDB_DEST,
    "GCK DMS activity scores (MaveDB)",
    headers={"Accept": "text/csv"},
)
status["gck_dms"] = ok and MAVEDB_DEST.exists()
print()

# ── 3. ClinVar variant summary ────────────────────────────────────────────────

print("── 3. ClinVar (NCBI FTP) ───────────────────────────────────")
CLINVAR_URL = (
    "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
)
CLINVAR_DEST = DATA_DIR / "clinvar" / "variant_summary.txt.gz"

ok = download(CLINVAR_URL, CLINVAR_DEST, "ClinVar variant_summary.txt.gz", timeout=300)
status["clinvar"] = ok and CLINVAR_DEST.exists()
print()

# ── 4. ProteinGym v1.0 clinical substitutions ─────────────────────────────────

print("── 4. ProteinGym v1.0 (Zenodo 13936340) ───────────────────")
PG_URLS = [
    "https://zenodo.org/records/13936340/files/clinical_substitutions.parquet",
    "https://proteingym.s3.amazonaws.com/clinical_substitutions.parquet",
]
PG_DEST = DATA_DIR / "proteingym" / "clinical_substitutions.parquet"

ok = False
if PG_DEST.exists():
    print(f"  [SKIP] already exists ({PG_DEST.name})")
    ok = True
else:
    for url in PG_URLS:
        ok = download(
            url, PG_DEST, "ProteinGym clinical_substitutions.parquet", timeout=300
        )
        if ok:
            break
        time.sleep(1)
status["proteingym"] = ok and PG_DEST.exists()
print()

# ── 5. GoFCards v1.0 — REST API (ABCC8 + KCNJ11 GOF variants) ───────────────

print("── 5. GoFCards v1.0 (REST API) ─────────────────────────────")
GOFCARDS_DEST = DATA_DIR / "gofcards" / "gofcards_validation_final.csv"

if GOFCARDS_DEST.exists():
    print(f"  [SKIP] already exists ({GOFCARDS_DEST.name})")
    status["gofcards"] = True
else:
    GOFCARDS_API = "https://www.genemed.tech/gofcards/api/variants/"
    genes = ["ABCC8", "KCNJ11"]
    rows = []
    api_ok = True

    for gene in genes:
        url = f"{GOFCARDS_API}?gene={gene}&page_size=500"
        print(f"  [DL]   GoFCards {gene} GOF variants")
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            data = r.json()
            results_key = (
                "results" if "results" in data else "data" if "data" in data else None
            )
            items = (
                data[results_key]
                if results_key
                else (data if isinstance(data, list) else [])
            )
            print(f"         {len(items)} variants retrieved")
            for item in items:
                rows.append(
                    {
                        "gene": item.get("gene", gene),
                        "protein": item.get("protein_change", item.get("hgvs_p", "")),
                        "pscore": item.get("pscore", item.get("p_score", "")),
                        "am_score": item.get("alphamissense", item.get("am_score", "")),
                        "revel_score": item.get("revel", item.get("revel_score", "")),
                        "phenotype": item.get("phenotype", ""),
                        "pmid": item.get("pmid", ""),
                    }
                )
        except Exception as e:
            print(f"  [FAIL] GoFCards API: {e}")
            api_ok = False
            break

    if api_ok and rows:
        GOFCARDS_DEST.parent.mkdir(parents=True, exist_ok=True)
        with open(GOFCARDS_DEST, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(
            f"  [OK]   Wrote {len(rows)} GoFCards variants → data/gofcards/gofcards_validation_final.csv"
        )
        status["gofcards"] = True
    else:
        # API unavailable — write instructions; do NOT fabricate scores
        instructions = (
            "GoFCards v1.0 — data/gofcards/gofcards_validation_final.csv\n"
            "=============================================================\n\n"
            "The GoFCards REST API was unavailable during download.\n\n"
            "Download the full dataset directly:\n\n"
            "  Option A — All GOF mutations (Excel, ~easiest):\n"
            "    http://download.genemed.tech/upload/GainFunCards/gofcards_data_download.xlsx\n"
            "    Convert ABCC8 + KCNJ11 rows to CSV and save as:\n"
            "      data/gofcards/gofcards_validation_final.csv\n\n"
            "  Option B — Full annotations (ZIP with extended fields):\n"
            "    http://download.genemed.tech/upload/GainFunCards/gofcards_annotations_download.zip\n"
            "    Extract and filter ABCC8 + KCNJ11 rows.\n\n"
            "  Option C — Browse/export via web UI:\n"
            "    http://www.genemed.tech/gofcards/#/genemed/download\n\n"
            "Required columns in the CSV:\n"
            "  gene, protein, pscore, am_score, revel_score, phenotype, pmid\n\n"
            "Reference: Zhao W et al. GoFCards. Nucleic Acids Res. 2024.\n"
            "DOI: 10.1093/nar/gkae1079\n"
        )
        notice_path = DATA_DIR / "gofcards" / "DOWNLOAD_INSTRUCTIONS.txt"
        notice_path.write_text(instructions)
        print(
            "  [WARN] API unavailable — instructions written to data/gofcards/DOWNLOAD_INSTRUCTIONS.txt"
        )
        print(
            "         Direct download: http://download.genemed.tech/upload/GainFunCards/gofcards_data_download.xlsx"
        )
        status["gofcards"] = False

print()

# ── 6. AlphaMissense — manual download notice ─────────────────────────────────

print("── 6. AlphaMissense (MANUAL — ~3.5 GB) ────────────────────")
AM_NOTICE = DATA_DIR / "alphamissense" / "DOWNLOAD_INSTRUCTIONS.txt"
AM_EXPECTED = DATA_DIR / "alphamissense" / "AlphaMissense_aa_substitutions.tsv.gz"

if AM_EXPECTED.exists():
    print(f"  [OK]   AlphaMissense file found: {AM_EXPECTED.name}")
    status["alphamissense"] = True
else:
    instructions = (
        "AlphaMissense aa-substitutions file (~3.5 GB)\n"
        "================================================\n\n"
        "Download one of:\n"
        "  A) Zenodo (recommended):\n"
        "     wget https://zenodo.org/records/8360242/files/AlphaMissense_aa_substitutions.tsv.gz\n"
        "     mv AlphaMissense_aa_substitutions.tsv.gz data/alphamissense/\n\n"
        "  B) Google Cloud Storage:\n"
        "     gsutil cp gs://dm_alphamissense/AlphaMissense_aa_substitutions.tsv.gz data/alphamissense/\n\n"
        "Then run:\n"
        "  python scripts/02_fetch_am_scores.py --lookup --method uniprot\n\n"
        "NOTE: AM scores are already embedded in data/am_revel_scores_combined.csv\n"
        "      (am_fetched=True). Re-fetching is optional for independent verification.\n"
    )
    AM_NOTICE.write_text(instructions)
    print("  [WARN] Not downloaded (large file) — instructions written to")
    print("         data/alphamissense/DOWNLOAD_INSTRUCTIONS.txt")
    print("  NOTE:  AM scores already embedded in am_revel_scores_combined.csv")
    status["alphamissense"] = "manual"

print()

# ── 7. REVEL — manual download notice ────────────────────────────────────────

print("── 7. REVEL v1.3 (MANUAL) ──────────────────────────────────")
REVEL_NOTICE = DATA_DIR / "revel_DOWNLOAD_INSTRUCTIONS.txt"
if not REVEL_NOTICE.exists():
    instructions = (
        "REVEL v1.3 scores\n"
        "==================\n\n"
        "Download from:\n"
        "  https://sites.google.com/site/revelgenomics/downloads\n"
        "  File: revel_with_transcript_ids (tab-separated, ~7 GB)\n\n"
        "NOTE: REVEL scores are already embedded in data/am_revel_scores_combined.csv\n"
        "      Re-fetching is not required to replicate manuscript analyses.\n"
    )
    REVEL_NOTICE.write_text(instructions)
print("  [INFO] REVEL scores already in am_revel_scores_combined.csv")
print("         Full database instructions: data/revel_DOWNLOAD_INSTRUCTIONS.txt")
status["revel"] = "embedded"

print()

# ── Manifest ─────────────────────────────────────────────────────────────────

print("── Writing MANIFEST.md ─────────────────────────────────────")

manifest_lines = [
    "# Data Manifest — am_revel_project",
    "",
    f"Generated: {time.strftime('%Y-%m-%d')}",
    "",
    "## Files",
    "",
    "| File | Source | Status | MD5 |",
    "|------|--------|--------|-----|",
]

file_map = {
    "am_revel_scores_combined.csv": (
        "Hopkins cohort (131 vars)",
        DATA_DIR / "am_revel_scores_combined.csv",
    ),
    "gck_dms/gck_activity_mavedb_raw.csv": (
        "MaveDB urn:mavedb:00000096-a-1",
        DATA_DIR / "gck_dms" / "gck_activity_mavedb_raw.csv",
    ),
    "clinvar/variant_summary.txt.gz": (
        "NCBI FTP",
        DATA_DIR / "clinvar" / "variant_summary.txt.gz",
    ),
    "proteingym/clinical_substitutions.parquet": (
        "Zenodo 13936340",
        DATA_DIR / "proteingym" / "clinical_substitutions.parquet",
    ),
    "gofcards/gofcards_validation_final.csv": (
        "GoFCards v1.0 API",
        DATA_DIR / "gofcards" / "gofcards_validation_final.csv",
    ),
    "alphamissense/AlphaMissense_aa_substitutions.tsv.gz": (
        "Zenodo 8360242 (MANUAL)",
        DATA_DIR / "alphamissense" / "AlphaMissense_aa_substitutions.tsv.gz",
    ),
}

for label, (source, path) in file_map.items():
    if path.exists():
        checksum = md5(path)
        size_mb = path.stat().st_size / 1e6
        manifest_lines.append(
            f"| `{label}` | {source} | ✓ present ({size_mb:.1f} MB) | `{checksum}` |"
        )
    else:
        manifest_lines.append(
            f"| `{label}` | {source} | ⚠ absent — see instructions | — |"
        )

manifest_lines += [
    "",
    "## Canonical DOIs",
    "",
    "- AlphaMissense: https://doi.org/10.5281/zenodo.8360242",
    "- REVEL: https://sites.google.com/site/revelgenomics/downloads",
    "- GCK DMS (MaveDB): urn:mavedb:00000096-a-1",
    "- ProteinGym: https://doi.org/10.5281/zenodo.13936340",
    "- GoFCards: https://doi.org/10.1093/nar/gkae1079",
    "- De Franco 2020: https://doi.org/10.1002/humu.23995",
    "- Hopkins 2023: https://doi.org/10.1002/humu.24595",
    "",
    "## Reproducibility notes",
    "",
    "All manuscript analyses can be replicated from `data/am_revel_scores_combined.csv` alone.",
    "Full raw data enables independent score re-annotation and verification.",
    "GoFCards data requires manual export from genemed.tech/gofcards/ if the API is unavailable.",
]

MANIFEST.write_text("\n".join(manifest_lines) + "\n")
print(f"  Written: data/MANIFEST.md")
print()

# ── Summary ───────────────────────────────────────────────────────────────────

print("=" * 60)
print("Summary")
print("=" * 60)
for k, v in status.items():
    if v is True:
        icon = "✓"
    elif v in ("manual", "embedded"):
        icon = "⚠"
    else:
        icon = "✗"
    print(f"  {icon}  {k}: {v}")
print()
print("Next step: python scripts/01_main_analysis.py")
