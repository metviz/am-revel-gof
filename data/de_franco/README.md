# De Franco 2020 — ABCC8/KCNJ11 Variant Data

**Source:** De Franco E et al. *Human Mutation* 2020;41(4):884-905.  
**DOI:** https://doi.org/10.1002/humu.23995  
**Supplementary file:** humu23995-sup-0001-supp_mat_tables_1-6_final_r.docx  
(Tables S1–S6: pathogenic, VUS, benign variants for KCNJ11 and ABCC8)

## Files

| File | Description |
|---|---|
| `humu23995-sup-0001-supp_mat_tables_1-6_final_r.docx` | Original supplementary DOCX (parsed to extract variants) |
| `de_franco_processed.csv` | 547 missense variants (LoF+GoF) with AM and REVEL scores |

## Processing methodology

1. Tables S1 (KCNJ11 pathogenic) and S4 (ABCC8 pathogenic) parsed from DOCX
2. Mechanism assigned from phenotype: NDM/PNDM/TNDM/DEND/later-onset diabetes = GoF; HI = LoF
3. AM and REVEL scores retrieved from myvariant.info (dbNSFP4) using protein position queries
4. Variants with missing scores excluded (n=7 unusual substitutions, 1 frameshift)

## Coverage vs. original analysis

| | This dataset | Paper (De Franco analysis) |
|---|---|---|
| Total | 547 | 555 |
| ABCC8 LoF | 261 | 261 |
| ABCC8 GoF | 120 | 123 |
| KCNJ11 LoF | 74 | 77 |
| KCNJ11 GoF | 92 | 94 |
| AM Strong gap (all) | +24.7 pp | +20.0 pp |
| REVEL Strong gap (all) | +13.0 pp | +12.6 pp |

The 8-variant shortfall is due to AM score unavailability in dbNSFP4 for 7 rare
substitutions. The original analysis used the full AlphaMissense aa_substitutions TSV
(Zenodo 8360242) via direct UniProt lookup, which covers more positions.
For exact replication, download the AlphaMissense TSV and re-run script 07.
