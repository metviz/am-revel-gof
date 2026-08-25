"""
16 — Two mechanistic analyses.

A. POSITIONAL PRIOR, with KCNJ11 recovered and the part-whole correlation fixed.

   an earlier version-an earlier version report a site-level AM prior (mean AM over all 19 substitutions at a
   position) for ABCC8 and GCK only, stating that KCNJ11 was excluded because its
   aa-substitutions file was unavailable. Two problems were raised:

     - KCNJ11 supplies BOTH of the paper's headline discordant variants
       (p.Gln52Arg, p.Val59Met), so a site-level claim that omits it cannot carry
       the weight placed on it. The exclusion is also unnecessary: the local
       all-substitutions cache (data/clinvar/am_gene_cache_abcc8_gck_kcnj11.csv)
       contains 7,448 KCNJ11 substitutions covering all 390 positions. This was a
       data-plumbing problem, not a data-availability problem.

     - The observed variant is itself one of the 19 substitutions entering its own
       positional mean, so the reported variant-vs-prior correlation (r=0.65) is
       inflated by construction. Here the prior is computed LEAVE-ONE-OUT.

B. DYNAMICS-LoF vs STABILITY-LoF.

   the write-up mechanistic story is that AM keys on destabilisation and so
   under-rates variants acting through other routes. The GCK abundance assay
   already separates Stability-LoF (loss of cellular abundance) from Dynamics-LoF
   (activity lost with abundance retained) -- and this stratification sits unused
   in Figure S3. It is the discriminating experiment: if AM misses Dynamics-LoF as
   badly as it misses GoF, the finding is not "AM misses gain-of-function" but
   "AM misses variants that act through altered dynamics rather than through
   destabilisation", which is a more general and more mechanistic claim.

Outputs
    results/tier4_positional_prior.csv
    results/tier4_dynamics_stratification.csv
"""

import os
import re

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
DATA = os.path.join(ROOT, "data")

UNIPROT = {"ABCC8": "Q09428", "KCNJ11": "Q14654", "GCK": "P35557"}
AA3 = {
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


def strip_p(s):
    r"""Remove an HGVS 'p.' prefix and surrounding parentheses -- and nothing else.

    The obvious one-liner, re.sub(r"[p\.\(\)]", "", s), deletes EVERY 'p'
    in the string, so Asp209Glu becomes As209Glu and Trp becomes Tr. That silently
    mangles every variant at an Asp or Trp residue.
    """
    s = str(s).strip()
    s = re.sub(r"^p\.", "", s)
    return s.strip("()").strip()


def short(s):
    b = strip_p(s)
    m = re.fullmatch(r"([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})", b)
    if m and m.group(1) in AA3 and m.group(3) in AA3:
        return f"{AA3[m.group(1)]}{m.group(2)}{AA3[m.group(3)]}"
    return b if re.fullmatch(r"[A-Z]\d+[A-Z]", b) else None


# ------------------------------------------------------------------ part A
def positional_prior():
    am = pd.read_csv(
        os.path.join(DATA, "clinvar", "am_gene_cache_abcc8_gck_kcnj11.csv")
    )
    u2g = {v: k for k, v in UNIPROT.items()}
    am["gene"] = am.uniprot_id.map(u2g)
    am["pos"] = am.protein_variant.str.extract(r"^[A-Z](\d+)[A-Z]$")[0].astype(int)
    am["ref"] = am.protein_variant.str[0]

    print("  all-substitutions cache:")
    for g in UNIPROT:
        s = am[am.gene == g]
        print(
            f"    {g:7s} {len(s):6d} substitutions across {s.pos.nunique()} positions"
        )

    # site totals, for a leave-one-out mean
    agg = am.groupby(["gene", "pos"]).am_pathogenicity.agg(["sum", "count"])

    hop = pd.read_csv(os.path.join(RES, "all_variants_annotated.csv"))
    hop["short"] = hop.protein.map(short)
    hop = hop.dropna(subset=["short"]).copy()
    hop["pos"] = hop.short.str.extract(r"^[A-Z](\d+)[A-Z]$")[0].astype(int)

    # ABCC8 is numbered on a different isoform in the cohort than in UniProt
    # Q09428: below ~residue 600 the two agree, above it the cohort numbering runs
    # one higher (the SUR1 isoform carrying an extra residue). Matching on position
    # alone therefore reads 22 of 47 ABCC8 variants at the wrong residue -- silently,
    # because a site mean exists at every position. Rather than hard-code a
    # breakpoint, each variant is mapped to the offset whose cache wild-type residue
    # matches its own, and any variant that matches at no offset is dropped.
    wt = am.set_index(["gene", "pos"]).ref.groupby(level=[0, 1]).first().to_dict()

    def resolve(gene, pos, sub_):
        for off in (0, -1, 1):
            if wt.get((gene, pos + off)) == sub_[0]:
                return pos + off, off
        return None, None

    # The value subtracted for the leave-one-out mean must be the cache's OWN score
    # for that substitution, not the score carried in the annotation file. The two
    # sources agree to within rounding but are not identical, and subtracting the
    # annotation value from a cache-derived sum let two ABCC8 site priors exceed
    # 1.0 -- impossible for a mean of AlphaMissense scores.
    cache_score = am.set_index(["gene", "protein_variant"]).am_pathogenicity.to_dict()

    loo, naive, offsets, n_missing = [], [], [], 0
    for g, p, sub_ in zip(hop.gene, hop.pos, hop.short):
        cpos, off = resolve(g, p, sub_)
        key = (g, cpos)
        own = cache_score.get((g, f"{sub_[0]}{cpos}{sub_[-1]}")) if cpos else None
        offsets.append(off)
        if cpos is None or key not in agg.index or own is None:
            n_missing += 1
            loo.append(np.nan)
            naive.append(np.nan)
            continue
        tot, n = agg.loc[key, "sum"], agg.loc[key, "count"]
        naive.append(tot / n)
        loo.append((tot - own) / (n - 1) if n > 1 else np.nan)
    hop["cache_offset"] = offsets
    if n_missing:
        print(f"    [!] {n_missing} variants could not be matched to the cache")
    used = pd.Series(offsets).value_counts().to_dict()
    print(f"    isoform offset applied (cohort -> UniProt): {used}")
    hop["prior_naive"] = naive
    hop["prior_loo"] = loo

    bad = hop[hop.prior_loo > 1.0]
    assert bad.empty, f"site prior above 1.0 for {list(bad.protein)}"

    cov = hop.prior_loo.notna().mean()
    print(
        f"\n  positional prior recovered for {hop.prior_loo.notna().sum()}/{len(hop)} "
        f"variants ({cov * 100:.0f}%), all three genes"
    )

    rows = []
    for g in list(UNIPROT) + ["ALL"]:
        s = hop if g == "ALL" else hop[hop.gene == g]
        s = s.dropna(subset=["prior_loo"])
        lof = s[s.mechanism == "LoF"].prior_loo
        gof = s[s.mechanism == "GoF"].prior_loo
        if len(lof) < 3 or len(gof) < 3:
            continue
        u, p = mannwhitneyu(lof, gof, alternative="two-sided")
        rows.append(
            dict(
                gene=g,
                n_lof=len(lof),
                n_gof=len(gof),
                median_lof=lof.median(),
                median_gof=gof.median(),
                delta=lof.median() - gof.median(),
                p=p,
            )
        )
    pri = pd.DataFrame(rows)
    print("\n  leave-one-out positional prior, LoF vs GoF sites:")
    for _, r in pri.iterrows():
        star = "*" if r.p < 0.05 else " "
        print(
            f"    {r.gene:7s} LoF {r.median_lof:.3f} (n={int(r.n_lof):3d})  "
            f"GoF {r.median_gof:.3f} (n={int(r.n_gof):3d})  "
            f"delta {r.delta:+.3f}  p={r.p:.4f}{star}"
        )

    # part-whole inflation: naive vs leave-one-out correlation
    print("\n  variant AM vs site prior (the r=0.65 reported previously):")
    for mech in ("LoF", "GoF"):
        s = hop[(hop.mechanism == mech)].dropna(subset=["prior_loo"])
        rn = spearmanr(s.am_score, s.prior_naive)
        rl = spearmanr(s.am_score, s.prior_loo)
        print(
            f"    {mech}: naive r={rn.statistic:.3f} (p={rn.pvalue:.1e})   "
            f"leave-one-out r={rl.statistic:.3f} (p={rl.pvalue:.1e})   "
            f"inflation {rn.statistic - rl.statistic:+.3f}"
        )

    hop.to_csv(os.path.join(RES, "tier4_positional_prior.csv"), index=False)
    print("  wrote tier4_positional_prior.csv")
    return hop, pri


# ------------------------------------------------------------------ part B
def dynamics_stratification():
    d = pd.read_csv(os.path.join(DATA, "gck_dms", "gck_activity_abundance_merged.csv"))
    d = d[d.mechanism_4class.isin(["Stability-LoF", "Dynamics-LoF", "GoF"])].copy()
    print("\n  GCK abundance-assay classes:")
    print("   ", d.mechanism_4class.value_counts().to_dict())

    rows = []
    for tool, col in (("AM", "am_score"), ("REVEL", "revel_score")):
        s = d.dropna(subset=[col])
        ref = s[s.mechanism_4class == "Stability-LoF"][col].values
        for target in (0.60, 0.50, 0.40):
            cut = np.quantile(ref, 1 - target)
            for cls in ("Stability-LoF", "Dynamics-LoF", "GoF"):
                v = s[s.mechanism_4class == cls][col].values
                rows.append(
                    dict(
                        tool=tool,
                        target_stability_sens=target * 100,
                        threshold=cut,
                        mechanism=cls,
                        n=len(v),
                        detected=float((v >= cut).mean() * 100),
                    )
                )
    out = pd.DataFrame(rows)

    print(
        "\n  detection rate by mechanism, threshold set to a fixed "
        "Stability-LoF sensitivity:"
    )
    for tool in ("AM", "REVEL"):
        print(f"    {tool}")
        for target in (0.60, 0.50, 0.40):
            s = out[(out.tool == tool) & (out.target_stability_sens == target * 100)]
            g = {r.mechanism: r.detected for r in s.itertuples()}
            print(
                f"      stability-LoF fixed at {target * 100:.0f}%:  "
                f"dynamics-LoF {g['Dynamics-LoF']:5.1f}%   GoF {g['GoF']:5.1f}%   "
                f"(dyn deficit {g['Stability-LoF'] - g['Dynamics-LoF']:5.1f} pp, "
                f"GoF deficit {g['Stability-LoF'] - g['GoF']:5.1f} pp)"
            )

    out.to_csv(os.path.join(RES, "tier4_dynamics_stratification.csv"), index=False)
    print("  wrote tier4_dynamics_stratification.csv")
    return out


if __name__ == "__main__":
    print("A. Positional prior (KCNJ11 recovered, leave-one-out)")
    positional_prior()
    print("\nB. Dynamics-LoF vs Stability-LoF")
    dynamics_stratification()
