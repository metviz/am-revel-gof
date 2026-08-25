# AM vs REVEL — KATP/GCK Channelopathy Study

Replication and extension of Hopkins et al. (2023) Human Mutation.
Comparative analysis of AlphaMissense vs REVEL performance on
Loss-of-Function vs Gain-of-Function pathogenic variants in monogenic
diabetes channelopathy genes (ABCC8, KCNJ11, GCK).

**Companion paper:** Cross-gene channelopathy study (`am_channelopathy/`), targeting Genome Medicine.

## Quick start

### Step 1 — Download raw data
```bash
python scripts/00_download_data.py
```
Downloads all automatable sources into `data/` and writes `data/MANIFEST.md` with MD5 checksums.
Takes ~5–15 min depending on network (ClinVar is 436 MB).

### Step 2 — Run the pipeline
```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

## Data directory (`data/`)

| Path | Source | Auto? | Size |
|------|--------|-------|------|
| `am_revel_scores_combined.csv` | Hopkins cohort (131 variants) | ✓ copied from results/ | 15 KB |
| `gck_dms/gck_activity_mavedb_raw.csv` | MaveDB `urn:mavedb:00000096-a-1` | ✓ API | 0.9 MB |
| `clinvar/variant_summary.txt.gz` | NCBI FTP | ✓ auto | 436 MB |
| `proteingym/clinical_substitutions.parquet` | Zenodo 13936340 | ✓ auto | 10 MB |
| `gofcards/gofcards_validation_final.csv` | GoFCards v1.0 (genemed.tech) | ⚠ manual export | — |
| `alphamissense/AlphaMissense_aa_substitutions.tsv.gz` | Zenodo 8360242 | ⚠ manual (~3.5 GB) | 3.5 GB |

## Manual downloads

### GoFCards (script 11)
Direct bulk download (no authentication required):
- **All GOF mutations (Excel):** `http://download.genemed.tech/upload/GainFunCards/gofcards_data_download.xlsx`
- **Full annotations (ZIP):** `http://download.genemed.tech/upload/GainFunCards/gofcards_annotations_download.zip`
- **Web download page:** `http://www.genemed.tech/gofcards/#/genomed/download`

Filter for ABCC8 + KCNJ11 rows, convert to CSV, save as `data/gofcards/gofcards_validation_final.csv`.
Required columns: `gene, protein, pscore, am_score, revel_score, phenotype, pmid`

### AlphaMissense (~3.5 GB, one-time — optional)
AM scores are already embedded in `data/am_revel_scores_combined.csv`. Download only for independent verification:
```bash
wget https://zenodo.org/records/8360242/files/AlphaMissense_aa_substitutions.tsv.gz
mv AlphaMissense_aa_substitutions.tsv.gz data/alphamissense/
python scripts/02_fetch_am_scores.py --lookup --method uniprot
```

### De Franco 2020 (paywall — requires institutional access)
- URL: https://onlinelibrary.wiley.com/doi/10.1002/humu.23995
- Download Table S1 (ABCC8) and Table S2 (KCNJ11) Excel files
- Convert: `python scripts/07_de_franco_analysis.py --convert-excel --abcc8 TableS1.xlsx --kcnj11 TableS2.xlsx`

## Pipeline
```
01_main_analysis.py          Hopkins replication (131 variants)
02_fetch_am_scores.py        AM score verification
03_supplementary_stats.py    Supplementary statistics
04_dynamic_thresholds.py     GMM threshold optimisation
05_gck_dms_analysis.py       GCK DMS activity (Gersing 2023, MaveDB)
06_gck_abundance_analysis.py GCK DMS abundance (Gersing 2024, MaveDB)
07_de_franco_analysis.py     De Franco 2020 ABCC8/KCNJ11 (PAYWALL)
08_clinvar_analysis.py       ClinVar expert panel ACMG recalibration
09_proteingym_analysis.py    ProteinGym v1.0 calibration check
10_meta_analysis.py          Meta-analysis forest plot (D-L random effects)
11_gofcards_validation.py    GoFCards cross-validation
12_manuscript_draft.py       Manuscript generation
```

## ACMG thresholds (consistent across all scripts)
| Tool  | Supporting | Moderate | Strong |
|-------|-----------|---------|--------|
| AM    | ≥0.564    | ≥0.740  | ≥0.890 |
| REVEL | ≥0.644    | ≥0.773  | ≥0.932 |

## Key result
AlphaMissense pooled LoF/GoF gap: +35.1 pp (95% CI +31.0 to +39.3; k=7, I²=0%), ~1.7× larger than REVEL (+20.2 pp).
