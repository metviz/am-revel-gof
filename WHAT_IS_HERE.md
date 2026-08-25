# What is in this repository, and what is not

This is the analysis code and the derived tables behind the AlphaMissense-versus-REVEL
comparison in ABCC8, KCNJ11 and GCK. It is meant to let someone re-derive every
reported number from the sources.

## Included

| Path | |
|---|---|
| `scripts/00`–`17` | the numbered analysis pipeline, in dependency order |
| `scripts/run_all.py` | driver; `--check` verifies inputs, `--analysis` runs the chain |
| `scripts/test_*.py` | regression checks, including the guard against the silent-fallback bug described in `REPRODUCE.md` |
| `results/*.csv` | every derived table the analysis produces |
| `data/**/*.csv` | the small parsed inputs: cohort variant lists, the gene-level AlphaMissense cache, gnomAD controls, the GCK MAVE tables, GoFCards |
| `validation/` | checksums and row counts for the inputs |
| `REPRODUCE.md` | manifest: each script with its exact inputs and outputs |

## Deliberately not included

**Manuscript files, and everything that writes them.** No manuscript, figure, cover
letter or supplementary document is here, nor any of the scripts that generate or edit
them, nor any test that reads them. That is deliberate and permanent: this repository
carries the analysis and its derived tables, and nothing of the paper's text. Use
`run_all.py --analysis` and `--tests`.

**Bulk third-party data**, because of size and redistribution terms. Re-download with
`scripts/00_download_data.py`, which writes instructions where a manual step is needed:

- AlphaMissense all-substitutions predictions (1.2 GB; CC BY-NC-SA 4.0)
- the ClinVar `variant_summary` dump (416 MB; re-downloadable from NCBI)
- REVEL, which requires accepting its own terms before download

**Published articles and their supplementary files.** Copyright rests with the
publishers; the references in `REPRODUCE.md` identify each one.

## A note on the small score tables

`data/dbnsfp/dbnsfp_panel_raw.csv` and the per-variant score columns in
`data/am_revel_scores_combined.csv` and `results/*.csv` are subsets covering the 131
cohort variants and the 578-variant predictor panel, retained so the analysis can be
re-run without re-querying. dbNSFP and REVEL each carry their own terms; consult them
before redistributing these files further.
