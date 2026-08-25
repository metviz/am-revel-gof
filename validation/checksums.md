
# Data Provenance Audit — AM vs REVEL Project

## Synthetic: True entries (development phase — NOT used in manuscript)

Entries dated before 2026-03-12T12:38 carry `Synthetic: True`. These are from the
initial pipeline development phase when placeholder data was used to validate script
logic. All entries from 2026-03-12T12:38 onwards carry `Synthetic: False` and represent
the real downloaded datasets used in all reported analyses.

Final analysis checksums (real data):
- gck_activity_mavedb_raw.csv  MD5: 3f72e4a1d59b651915a673d48292fdd6  N=9,362
- gck_abundance_mavedb_raw.csv MD5: de51ec73727353330b13445a6df0a84a  N=9,201
- de_franco_2020_raw.csv       MD5: 3fc5571f832d444077810606f5be5596  N=557
- proteingym_gck_clinical.csv  MD5: 817c6e535c9b37a12763c9aa0369e2e3  N=149

No synthetic data is present in any result, figure, or table in the manuscript.
All `assert not IS_SYNTHETIC` guards in scripts 05, 06, and 09 passed on the final runs.

---

## Full run log (append-only)

## gck_activity_mavedb_raw.csv
- MD5: e506d903fedf5488caec3585dcd09e81
- N rows: 8835
- Downloaded: 2026-03-12T00:05:34.492222
- Source: https://api.mavedb.org/api/v1/score-sets/urn:mavedb:00000097-a-1/scores
- Synthetic: True

## gck_abundance_mavedb_raw.csv
- MD5: 306be6403512e439d93747419d4d42cf
- N rows: 8835
- Date: 2026-03-12T00:19:39.311205
- Synthetic: True

## de_franco_2020_raw.csv
- MD5: 719a859933aacb3224c8da02af143681
- N: 497
- Date: 2026-03-12T00:33:16.424963
- Synthetic: True

## clinvar_katp_gck_raw.csv
- MD5: ea4d0a90bd3865ba2c60f657d45d385e
- N: 1117
- Date: 2026-03-12T00:46:42.577804
- Synthetic: True

## clinvar_katp_gck_raw.csv
- MD5: ea4d0a90bd3865ba2c60f657d45d385e
- N: 1117
- Date: 2026-03-12T00:47:09.157077
- Synthetic: True

## clinvar_katp_gck_raw.csv
- MD5: ea4d0a90bd3865ba2c60f657d45d385e
- N: 1117
- Date: 2026-03-12T00:47:20.399075
- Synthetic: True

## proteingym_gck_clinical.csv
- MD5: 3556091b30df5c4586b8f1dc85ab3ba9
- N: 263
- Date: 2026-03-12T01:42:12.997646
- Synthetic: True

## proteingym_gck_clinical.csv
- MD5: 74e645e88a413be620ca04ee95b213ca
- N: 263
- Date: 2026-03-12T01:43:20.892826
- Synthetic: True

## gck_activity_mavedb_raw.csv
- MD5: 836b036e6da5299d55f11e8578425720
- N rows: 312
- Downloaded: 2026-03-12T12:24:57.124936
- Source: https://api.mavedb.org/api/v1/score-sets/urn:mavedb:00000097-a-1/scores
- Synthetic: False

## gck_activity_mavedb_raw.csv
- MD5: 3f72e4a1d59b651915a673d48292fdd6
- N rows: 9362
- Downloaded: 2026-03-12T12:38:56.915736
- Source: https://api.mavedb.org/api/v1/score-sets/urn:mavedb:00000096-a-1/scores
- Synthetic: False

## gck_activity_mavedb_raw.csv
- MD5: 3f72e4a1d59b651915a673d48292fdd6
- N rows: 9362
- Downloaded: 2026-03-12T12:44:51.984531
- Source: https://api.mavedb.org/api/v1/score-sets/urn:mavedb:00000096-a-1/scores
- Synthetic: False

## gck_abundance_mavedb_raw.csv
- MD5: de51ec73727353330b13445a6df0a84a
- N rows: 9201
- Date: 2026-03-12T12:48:02.111503
- Synthetic: False

## gck_activity_mavedb_raw.csv
- MD5: 3f72e4a1d59b651915a673d48292fdd6
- N rows: 9362
- Downloaded: 2026-03-12T12:50:04.430071
- Source: https://api.mavedb.org/api/v1/score-sets/urn:mavedb:00000096-a-1/scores
- Synthetic: False

## gck_abundance_mavedb_raw.csv
- MD5: de51ec73727353330b13445a6df0a84a
- N rows: 9201
- Date: 2026-03-12T12:53:16.016943
- Synthetic: False

## gck_activity_mavedb_raw.csv
- MD5: 3f72e4a1d59b651915a673d48292fdd6
- N rows: 9362
- Downloaded: 2026-03-12T13:01:20.383580
- Source: https://api.mavedb.org/api/v1/score-sets/urn:mavedb:00000096-a-1/scores
- Synthetic: False

## gck_abundance_mavedb_raw.csv
- MD5: de51ec73727353330b13445a6df0a84a
- N rows: 9201
- Date: 2026-03-12T13:04:32.392818
- Synthetic: False

## gck_activity_mavedb_raw.csv
- MD5: 3f72e4a1d59b651915a673d48292fdd6
- N rows: 9362
- Downloaded: 2026-03-12T13:06:34.296483
- Source: https://api.mavedb.org/api/v1/score-sets/urn:mavedb:00000096-a-1/scores
- Synthetic: False

## gck_abundance_mavedb_raw.csv
- MD5: de51ec73727353330b13445a6df0a84a
- N rows: 9201
- Date: 2026-03-12T13:09:45.382749
- Synthetic: False

## gck_abundance_mavedb_raw.csv
- MD5: de51ec73727353330b13445a6df0a84a
- N rows: 9201
- Date: 2026-03-12T16:23:44.193343
- Synthetic: False

## gck_activity_mavedb_raw.csv
- MD5: 3f72e4a1d59b651915a673d48292fdd6
- N rows: 9362
- Downloaded: 2026-03-12T18:36:49.543897
- Source: https://api.mavedb.org/api/v1/score-sets/urn:mavedb:00000096-a-1/scores
- Synthetic: False

## gck_abundance_mavedb_raw.csv
- MD5: de51ec73727353330b13445a6df0a84a
- N rows: 9201
- Date: 2026-03-12T18:40:05.208772
- Synthetic: False

## de_franco_2020_raw.csv
- MD5: 3fc5571f832d444077810606f5be5596
- N: 557
- Date: 2026-03-12T22:51:50.523983
- Synthetic: False

## de_franco_2020_raw.csv
- MD5: 3fc5571f832d444077810606f5be5596
- N: 557
- Date: 2026-03-12T22:59:13.356919
- Synthetic: False

## de_franco_2020_raw.csv
- MD5: 3fc5571f832d444077810606f5be5596
- N: 557
- Date: 2026-03-13T00:22:41.668236
- Synthetic: False

## de_franco_2020_raw.csv
- MD5: 3fc5571f832d444077810606f5be5596
- N: 557
- Date: 2026-03-13T00:29:58.655531
- Synthetic: False

## proteingym_gck_clinical.csv
- MD5: 817c6e535c9b37a12763c9aa0369e2e3
- N: 149
- Date: 2026-03-13T00:46:33.386663
- Synthetic: False

## proteingym_gck_clinical.csv
- MD5: 817c6e535c9b37a12763c9aa0369e2e3
- N: 149
- Date: 2026-03-13T00:53:10.813321
- Synthetic: False

## gck_activity_mavedb_raw.csv
- MD5: 3f72e4a1d59b651915a673d48292fdd6
- N rows: 9362
- Downloaded: 2026-03-13T01:12:13.184211
- Source: https://api.mavedb.org/api/v1/score-sets/urn:mavedb:00000096-a-1/scores
- Synthetic: False

## gck_abundance_mavedb_raw.csv
- MD5: de51ec73727353330b13445a6df0a84a
- N rows: 9201
- Date: 2026-03-13T01:15:08.203816
- Synthetic: False

## de_franco_2020_raw.csv
- MD5: 3fc5571f832d444077810606f5be5596
- N: 557
- Date: 2026-03-13T01:15:15.275684
- Synthetic: False

## proteingym_gck_clinical.csv
- MD5: 817c6e535c9b37a12763c9aa0369e2e3
- N: 149
- Date: 2026-03-13T01:20:31.718557
- Synthetic: False

## gck_activity_mavedb_raw.csv
- MD5: 3f72e4a1d59b651915a673d48292fdd6
- N rows: 9362
- Downloaded: 2026-03-13T12:21:50.100897
- Source: https://api.mavedb.org/api/v1/score-sets/urn:mavedb:00000096-a-1/scores
- Synthetic: False

## gck_abundance_mavedb_raw.csv
- MD5: de51ec73727353330b13445a6df0a84a
- N rows: 9201
- Date: 2026-03-13T12:50:51.635605
- Synthetic: False

## de_franco_2020_raw.csv
- MD5: 3fc5571f832d444077810606f5be5596
- N: 557
- Date: 2026-03-13T12:53:38.434888
- Synthetic: False

## proteingym_gck_clinical.csv
- MD5: 817c6e535c9b37a12763c9aa0369e2e3
- N: 149
- Date: 2026-03-13T12:53:42.017131
- Synthetic: False

## de_franco_2020_raw.csv
- MD5: 3fc5571f832d444077810606f5be5596
- N: 557
- Date: 2026-03-13T17:04:17.281495
- Synthetic: False

## proteingym_gck_clinical.csv
- MD5: 817c6e535c9b37a12763c9aa0369e2e3
- N: 149
- Date: 2026-03-13T17:18:26.487633
- Synthetic: False

## proteingym_gck_clinical.csv
- MD5: 817c6e535c9b37a12763c9aa0369e2e3
- N: 149
- Date: 2026-03-13T17:24:04.260473
- Synthetic: False

## de_franco_2020_raw.csv
- MD5: 3fc5571f832d444077810606f5be5596
- N: 557
- Date: 2026-03-13T17:25:18.112380
- Synthetic: False

## proteingym_gck_clinical.csv
- MD5: 817c6e535c9b37a12763c9aa0369e2e3
- N: 149
- Date: 2026-03-13T18:07:59.617753
- Synthetic: False

## proteingym_gck_clinical.csv
- MD5: 817c6e535c9b37a12763c9aa0369e2e3
- N: 149
- Date: 2026-03-13T18:21:43.829249
- Synthetic: False

## proteingym_gck_clinical.csv
- MD5: 817c6e535c9b37a12763c9aa0369e2e3
- N: 149
- Date: 2026-03-13T19:51:29.083773
- Synthetic: False

## proteingym_gck_clinical.csv
- MD5: 817c6e535c9b37a12763c9aa0369e2e3
- N: 149
- Date: 2026-03-13T21:47:18.678044
- Synthetic: False

## gck_activity_mavedb_raw.csv
- MD5: 3f72e4a1d59b651915a673d48292fdd6
- N rows: 9362
- Downloaded: 2026-03-13T22:15:16.849638
- Source: https://api.mavedb.org/api/v1/score-sets/urn:mavedb:00000096-a-1/scores
- Synthetic: False

## gck_activity_mavedb_raw.csv
- MD5: 3f72e4a1d59b651915a673d48292fdd6
- N rows: 9362
- Downloaded: 2026-03-13T22:22:48.606073
- Source: https://api.mavedb.org/api/v1/score-sets/urn:mavedb:00000096-a-1/scores
- Synthetic: False

## gck_activity_mavedb_raw.csv
- MD5: 3f72e4a1d59b651915a673d48292fdd6
- N rows: 9362
- Downloaded: 2026-03-13T22:26:46.476869
- Source: https://api.mavedb.org/api/v1/score-sets/urn:mavedb:00000096-a-1/scores
- Synthetic: False

## gck_activity_mavedb_raw.csv
- MD5: 3f72e4a1d59b651915a673d48292fdd6
- N rows: 9362
- Downloaded: 2026-03-13T22:33:47.463805
- Source: https://api.mavedb.org/api/v1/score-sets/urn:mavedb:00000096-a-1/scores
- Synthetic: False

## gck_activity_mavedb_raw.csv
- MD5: 3f72e4a1d59b651915a673d48292fdd6
- N rows: 9362
- Downloaded: 2026-03-13T22:40:24.219238
- Source: https://api.mavedb.org/api/v1/score-sets/urn:mavedb:00000096-a-1/scores
- Synthetic: False

## gck_activity_mavedb_raw.csv
- MD5: 3f72e4a1d59b651915a673d48292fdd6
- N rows: 9362
- Downloaded: 2026-03-13T22:48:41.551526
- Source: https://api.mavedb.org/api/v1/score-sets/urn:mavedb:00000096-a-1/scores
- Synthetic: False

## gck_activity_mavedb_raw.csv
- MD5: 3f72e4a1d59b651915a673d48292fdd6
- N rows: 9362
- Downloaded: 2026-03-13T22:59:12.536592
- Source: https://api.mavedb.org/api/v1/score-sets/urn:mavedb:00000096-a-1/scores
- Synthetic: False

## de_franco_2020_raw.csv
- MD5: 3fc5571f832d444077810606f5be5596
- N: 557
- Date: 2026-03-13T23:30:44.837240
- Synthetic: False

## de_franco_2020_raw.csv
- MD5: 3fc5571f832d444077810606f5be5596
- N: 557
- Date: 2026-03-13T23:34:52.860780
- Synthetic: False

## de_franco_2020_raw.csv
- MD5: 3fc5571f832d444077810606f5be5596
- N: 557
- Date: 2026-03-13T23:40:41.703903
- Synthetic: False

## de_franco_2020_raw.csv
- MD5: 3fc5571f832d444077810606f5be5596
- N: 557
- Date: 2026-03-13T23:49:22.504141
- Synthetic: False

## de_franco_2020_raw.csv
- MD5: 3fc5571f832d444077810606f5be5596
- N: 557
- Date: 2026-03-13T23:56:24.681081
- Synthetic: False

## de_franco_2020_raw.csv
- MD5: 3fc5571f832d444077810606f5be5596
- N: 557
- Date: 2026-03-14T00:30:53.078236
- Synthetic: False

## de_franco_2020_raw.csv
- MD5: 3fc5571f832d444077810606f5be5596
- N: 557
- Date: 2026-03-14T00:44:53.177164
- Synthetic: False

## de_franco_2020_raw.csv
- MD5: 3fc5571f832d444077810606f5be5596
- N: 557
- Date: 2026-03-14T00:53:37.268314
- Synthetic: False

## gck_activity_mavedb_raw.csv
- MD5: 3f72e4a1d59b651915a673d48292fdd6
- N rows: 9362
- Downloaded: 2026-03-14T00:58:32.618281
- Source: https://api.mavedb.org/api/v1/score-sets/urn:mavedb:00000096-a-1/scores
- Synthetic: False

## gck_activity_mavedb_raw.csv
- MD5: 3f72e4a1d59b651915a673d48292fdd6
- N rows: 9362
- Downloaded: 2026-03-14T01:10:39.445881
- Source: https://api.mavedb.org/api/v1/score-sets/urn:mavedb:00000096-a-1/scores
- Synthetic: False

## gck_activity_mavedb_raw.csv
- MD5: 3f72e4a1d59b651915a673d48292fdd6
- N rows: 9362
- Downloaded: 2026-03-14T01:12:08.591111
- Source: https://api.mavedb.org/api/v1/score-sets/urn:mavedb:00000096-a-1/scores
- Synthetic: False

## gck_abundance_mavedb_raw.csv
- MD5: de51ec73727353330b13445a6df0a84a
- N rows: 9201
- Date: 2026-03-14T01:21:16.236236
- Synthetic: False

## gck_abundance_mavedb_raw.csv
- MD5: de51ec73727353330b13445a6df0a84a
- N rows: 9201
- Date: 2026-03-14T01:28:52.136457
- Synthetic: False

## gck_abundance_mavedb_raw.csv
- MD5: de51ec73727353330b13445a6df0a84a
- N rows: 9201
- Date: 2026-03-14T01:39:12.710394
- Synthetic: False

## gck_activity_mavedb_raw.csv
- MD5: 2d7563a59848b4f2c3e4b20fa0cfb825
- N rows: 9362
- Downloaded: 2026-04-04T13:38:03.842478
- Source: https://api.mavedb.org/api/v1/score-sets/urn:mavedb:00000096-a-1/scores
- Synthetic: False

## gck_abundance_mavedb_raw.csv
- MD5: de51ec73727353330b13445a6df0a84a
- N rows: 9201
- Date: 2026-04-04T13:38:11.730028
- Synthetic: False

## proteingym_gck_clinical.csv
- MD5: f65a9a630d3992ce68fcfca659cfc06f
- N: 194
- Date: 2026-04-04T13:44:04.117459
- Synthetic: False

## de_franco_2020_raw.csv
- MD5: cead6d9a473f4ca998907f107f0c298e
- N: 547
- Date: 2026-04-04T14:37:20.323973
- Synthetic: False

## de_franco_2020_raw.csv
- MD5: cead6d9a473f4ca998907f107f0c298e
- N: 547
- Date: 2026-04-04T14:38:04.801846
- Synthetic: False

## de_franco_2020_raw.csv
- MD5: cead6d9a473f4ca998907f107f0c298e
- N: 547
- Date: 2026-04-04T14:38:30.368445
- Synthetic: False
