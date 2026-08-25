# Data Manifest — am_revel_project

Generated: 2026-04-04

## Files

| File | Source | Status | MD5 |
|------|--------|--------|-----|
| `am_revel_scores_combined.csv` | Hopkins cohort (131 vars) | ✓ present (0.0 MB) | `937f54ddaa4447bfe9d7103584c1f4ff` |
| `gck_dms/gck_activity_mavedb_raw.csv` | MaveDB urn:mavedb:00000096-a-1 | ✓ present (0.9 MB) | `2d7563a59848b4f2c3e4b20fa0cfb825` |
| `clinvar/variant_summary.txt.gz` | NCBI FTP | ✓ present (435.6 MB) | `6f4331f178f524f747de70653fb3fd0b` |
| `proteingym/clinical_substitutions.parquet` | Zenodo 13936340 | ✓ present (10.2 MB) | `e15e55307aa67772febaf66fcaa6baee` |
| `gofcards/gofcards_validation_final.csv` | GoFCards v1.0 API | ✓ present (0.0 MB) | `bc863613e8a85f059deb2e3ef68d36c2` |
| `alphamissense/AlphaMissense_aa_substitutions.tsv.gz` | Zenodo 8360242 (MANUAL) | ⚠ absent — see instructions | — |

## Canonical DOIs

- AlphaMissense: https://doi.org/10.5281/zenodo.8360242
- REVEL: https://sites.google.com/site/revelgenomics/downloads
- GCK DMS (MaveDB): urn:mavedb:00000096-a-1
- ProteinGym: https://doi.org/10.5281/zenodo.13936340
- GoFCards: https://doi.org/10.1093/nar/gkae1079
- De Franco 2020: https://doi.org/10.1002/humu.23995
- Hopkins 2023: https://doi.org/10.1002/humu.24595

## Reproducibility notes

All manuscript analyses can be replicated from `data/am_revel_scores_combined.csv` alone.
Full raw data enables independent score re-annotation and verification.
GoFCards data requires manual export from genemed.tech/gofcards/ if the API is unavailable.
