"""
17 — Supporting analyses.

Three supporting statistics, none needing new data. Each is
computed here and written to a CSV so the write-up prose in
scripts/the document stage.py reads from a file rather than from a literal.

  A. STRONG-SHORTFALL TEST.
     an earlier version conceded that a low LR+ ceiling meant the data "lack the power" to test
     the 18.7 Strong boundary. That inverts the logic. ceiling = n_benign / FP is
     the MEASURED false-positive rate at that cutoff: a ceiling below 18.7 says
     that even a perfectly sensitive predictor could not reach Strong against
     these controls. It is evidence against Strong, not absence of evidence.
     Two independent statements are computed per Strong cutoff:
       - the log-method 95% CI on LR+ (already in tier2_lr_table.csv), and
       - an exact one-sided binomial test of the false-positive count against
         the largest false-positive rate compatible with LR+ >= 18.7,
         sens / 18.7. The UPPER 95% (Wilson) bound of sensitivity is used, so
         the test is conservative in favour of reaching Strong.

  B. PANEL GROUP MEANS.
     The per-row screen (>=100 distinct rankscores AND |achieved - target LoF
     sensitivity| <= 2.0 pp) is already uniform across all 90 rows; what the
     summary did not report is how few rows survive per group. The
     conservation-only mean rests on three PROVEAN rows plus one SIFT4G row.
     n, range and a leave-one-row-out range of the group mean are written out,
     which is the statistic that answers "does this hang on one row".

  C. ABCC8 ISOFORM OFFSET.
     Cohort and UniProt numbering diverge for ABCC8; script 16 assigns each
     variant the offset whose cache wild-type residue matches, trying 0 first.
     Four variants match at BOTH offsets, so their assignment is arbitrary. The
     source cohort carries no transcript accession, so the ambiguity cannot be
     resolved from the data. Both the published rule and a uniform alternative
     are run and reported.

Outputs
    results/v32_strong_shortfall.csv
    results/v32_panel_group_means.csv
    results/v32_abcc8_offset_sensitivity.csv
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest, mannwhitneyu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib

_p16 = importlib.import_module("16_positional_and_dynamics")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
DATA = os.path.join(ROOT, "data")
LR_REQUIRED = 18.7


def wilson_hi(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre + half


# ------------------------------------------------------------------ part A
def strong_shortfall():
    lt = pd.read_csv(os.path.join(RES, "tier2_lr_table.csv"))
    lt = lt[(lt.control == "gnomAD AF>=1e-5") & (lt.tier == "Strong")
            & (lt.mech == "LoF")]
    rows = []
    for scheme in ("v26 (as submitted)", "Bergquist 2025"):
        for tool in ("AM", "REVEL"):
            r = lt[(lt.scheme == scheme) & (lt.tool == tool)]
            if r.empty:
                continue
            r = r.iloc[0]
            n_b, fp, n_d = int(r.n_benign), int(r.fp), int(r.n_disease)
            k = int(round(r.sens * n_d))
            sens_hi = wilson_hi(k, n_d)
            fpr_max = sens_hi / LR_REQUIRED
            bt = binomtest(fp, n_b, fpr_max, alternative="greater")
            rows.append(dict(
                scheme=scheme, tool=tool, cutoff=r.cutoff,
                sens=r.sens, sens_hi=sens_hi, spec=r.spec,
                n_disease=n_d, n_benign=n_b, fp=fp,
                lr=r.lr, lr_lo=r.lr_lo, lr_hi=r.lr_hi,
                lr_ceiling=r.lr_ceiling,
                max_fp_for_strong=fpr_max * n_b,
                ci_excludes_required=bool(r.lr_hi < LR_REQUIRED),
                p_binom=bt.pvalue,
            ))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "v32_strong_shortfall.csv"), index=False)
    print("A. Strong-threshold shortfall, LoF, 425-variant gnomAD control set")
    for _, r in out.iterrows():
        print(f"    {r.tool:5s} >={r.cutoff:.3f} ({r.scheme[:9]})  "
              f"LR+ {r.lr:4.2f} ({r.lr_lo:.2f}-{r.lr_hi:.2f})  ceiling {r.lr_ceiling:5.2f}  "
              f"needs <={r.max_fp_for_strong:4.1f} FP/{r.n_benign}, observed {r.fp}  "
              f"p={r.p_binom:.2g}")
    assert out.ci_excludes_required.all(), "a Strong CI includes 18.7"
    assert (out.p_binom < 0.05).all(), "a Strong cutoff is not rejected"
    return out


# ------------------------------------------------------------------ part B
def panel_group_means():
    g = pd.read_csv(os.path.join(RES, "tier4_panel_gaps.csv"))
    pooled = g[g.dataset == "ALL"]
    ok = pooled[pooled.usable]
    rows = []
    for grp, d in ok.groupby("group"):
        gaps = d.gap.values
        loo = [np.mean(np.delete(gaps, i)) for i in range(len(gaps))]
        rows.append(dict(
            group=grp,
            n_rows=len(d),
            n_predictors=d.predictor.nunique(),
            predictors="; ".join(
                f"{p} x{c}" for p, c in d.predictor.value_counts().items()),
            mean=gaps.mean(),
            row_min=gaps.min(),
            row_max=gaps.max(),
            loo_min=min(loo),
            loo_max=max(loo),
            n_rows_screened_out=int((~pooled[pooled.group == grp].usable).sum()),
        ))
    out = pd.DataFrame(rows).sort_values("mean", ascending=False)
    out.to_csv(os.path.join(RES, "v32_panel_group_means.csv"), index=False)
    print("\nB. Panel group means (pooled cohort, rows surviving the uniform screen)")
    for _, r in out.iterrows():
        print(f"    {r.group:26s} {r['mean']:6.2f} pp  n={int(r.n_rows):2d} rows "
              f"({int(r.n_rows_screened_out)} screened out)  "
              f"rows {r.row_min:6.2f} to {r.row_max:6.2f}  "
              f"leave-one-row-out {r.loo_min:6.2f} to {r.loo_max:6.2f}   [{r.predictors}]")
    # the number the prose has to carry: SIFT4G's largest sensitivity overshoot
    s = g[g.predictor == "SIFT4G"]
    print(f"    SIFT4G LoF-sensitivity overshoot across all rows: "
          f"{s.sens_overshoot.min():+.2f} to {s.sens_overshoot.max():+.2f} pp")
    return out


# ------------------------------------------------------------------ part C
def abcc8_offset():
    am = pd.read_csv(os.path.join(DATA, "clinvar",
                                  "am_gene_cache_abcc8_gck_kcnj11.csv"))
    u2g = {v: k for k, v in _p16.UNIPROT.items()}
    am["gene"] = am.uniprot_id.map(u2g)
    am["pos"] = am.protein_variant.str.extract(r"^[A-Z](\d+)[A-Z]$")[0].astype(int)
    am["ref"] = am.protein_variant.str[0]
    agg = am.groupby(["gene", "pos"]).am_pathogenicity.agg(["sum", "count"])
    wt = am.set_index(["gene", "pos"]).ref.groupby(level=[0, 1]).first().to_dict()
    cache_score = am.set_index(
        ["gene", "protein_variant"]).am_pathogenicity.to_dict()

    hop = pd.read_csv(os.path.join(RES, "all_variants_annotated.csv"))
    hop["short"] = hop.protein.map(_p16.short)
    hop = hop.dropna(subset=["short"]).copy()
    hop["pos"] = hop.short.str.extract(r"^[A-Z](\d+)[A-Z]$")[0].astype(int)

    def matches(gene, pos, off, ref):
        return wt.get((gene, pos + off)) == ref

    # which ABCC8 variants are ambiguous, and where does each offset actually apply
    amb, only0, onlym1 = [], [], []
    for _, r in hop[hop.gene == "ABCC8"].iterrows():
        m0 = matches("ABCC8", r.pos, 0, r.short[0])
        m1 = matches("ABCC8", r.pos, -1, r.short[0])
        if m0 and m1:
            amb.append((r["short"], r.pos))
        elif m0:
            only0.append(r.pos)
        elif m1:
            onlym1.append(r.pos)

    def prior_loo(rule):
        """rule(gene, pos, ref) -> offset or None."""
        vals = []
        for g_, p_, s_ in zip(hop.gene, hop.pos, hop.short):
            off = rule(g_, p_, s_[0])
            if off is None:
                vals.append(np.nan)
                continue
            cpos = p_ + off
            own = cache_score.get((g_, f"{s_[0]}{cpos}{s_[-1]}"))
            key = (g_, cpos)
            if own is None or key not in agg.index:
                vals.append(np.nan)
                continue
            tot, n = agg.loc[key, "sum"], agg.loc[key, "count"]
            vals.append((tot - own) / (n - 1) if n > 1 else np.nan)
        return np.array(vals, dtype=float)

    def published(gene, pos, ref):
        for off in (0, -1, 1):
            if matches(gene, pos, off, ref):
                return off
        return None

    def alt_minus1_above_600(gene, pos, ref):
        """Ambiguous ABCC8 residues above 600 take -1 instead of 0."""
        if gene == "ABCC8" and pos > 600 and matches(gene, pos, -1, ref) \
                and matches(gene, pos, 0, ref):
            return -1
        return published(gene, pos, ref)

    # Externally resolved assignment. Each of the four ambiguous ABCC8 variants has
    # a c. notation in an external record, and a c. position fixes the residue by
    # arithmetic (codon of residue r spans c.3r-2 .. c.3r):
    #   N24K   c.72C>A    ClinVar NM_000352.6  -> residue   24  (offset  0)
    #   L225P  c.674T>C   ClinVar NM_000352.6  -> residue  225  (offset  0)
    #   S1386P c.4153T>C  ClinVar NM_000352.6  -> residue 1385  (offset -1)
    #   G1478V c.4433G>T  De Franco 2020 supplementary table    -> residue 1478 (0)
    # The two sites above the breakpoint resolve in OPPOSITE directions, so no single
    # offset rule is right for both; the tie-break is per variant, not per region.
    RESOLVED = {"N24K": 0, "L225P": 0, "S1386P": -1, "G1478V": 0}

    def _resolved_rule(gene, pos, sub_):
        if gene == "ABCC8" and sub_ in RESOLVED:
            return RESOLVED[sub_]
        return published(gene, pos, sub_[0])

    def alt_minus1_all(gene, pos, ref):
        """Every ambiguous ABCC8 residue takes -1 instead of 0."""
        if gene == "ABCC8" and matches(gene, pos, -1, ref) \
                and matches(gene, pos, 0, ref):
            return -1
        return published(gene, pos, ref)

    def prior_loo_bysub(rule_bysub):
        vals = []
        for g_, p_, s_ in zip(hop.gene, hop.pos, hop.short):
            off = rule_bysub(g_, p_, s_)
            if off is None:
                vals.append(np.nan); continue
            cpos = p_ + off
            own = cache_score.get((g_, f"{s_[0]}{cpos}{s_[-1]}"))
            key = (g_, cpos)
            if own is None or key not in agg.index:
                vals.append(np.nan); continue
            tot, n = agg.loc[key, "sum"], agg.loc[key, "count"]
            vals.append((tot - own) / (n - 1) if n > 1 else np.nan)
        return np.array(vals, dtype=float)

    rows = []
    for label, rule in (("published (offset 0 first)", published),
                        ("ambiguous >600 -> -1", alt_minus1_above_600),
                        ("all ambiguous -> -1", alt_minus1_all),
                        ("externally resolved", None)):
        hop["_p"] = (prior_loo_bysub(_resolved_rule) if rule is None
                     else prior_loo(rule))
        for gene in ("ABCC8", "ALL"):
            s = hop if gene == "ALL" else hop[hop.gene == gene]
            s = s.dropna(subset=["_p"])
            lof = s[s.mechanism == "LoF"]._p
            gof = s[s.mechanism == "GoF"]._p
            u, p = mannwhitneyu(lof, gof, alternative="two-sided")
            rows.append(dict(rule=label, gene=gene, n_lof=len(lof), n_gof=len(gof),
                             median_lof=lof.median(), median_gof=gof.median(),
                             p=p))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "v32_abcc8_offset_sensitivity.csv"), index=False)
    print("\nC. ABCC8 isoform-offset sensitivity")
    print(f"    ambiguous at both offsets (n={len(amb)}): "
          + ", ".join(f"{v}" for v, _ in amb))
    print(f"    offset 0 only:  {min(only0)}-{max(only0)} (n={len(only0)})")
    print(f"    offset -1 only: {min(onlym1)}-{max(onlym1)} (n={len(onlym1)})")
    for _, r in out.iterrows():
        print(f"    {r.rule:26s} {r.gene:6s} LoF {r.median_lof:.3f} "
              f"GoF {r.median_gof:.3f}  p={r.p:.4f}")
    return out, amb, (min(only0), max(only0)), (min(onlym1), max(onlym1))


if __name__ == "__main__":
    strong_shortfall()
    panel_group_means()
    abcc8_offset()
    print("\nwrote results/v32_{strong_shortfall,panel_group_means,"
          "abcc8_offset_sensitivity}.csv")
