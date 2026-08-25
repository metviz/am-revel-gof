"""
14 — Tier 2 analysis. THRESHOLD PROVENANCE.
    an earlier version/an earlier version Methods say the AM Moderate (0.740) and Strong (0.890) thresholds
    "were set by analogous LR+ calibration consistent with the Tavtigian
    framework", with no calibration set, no LR+ estimate and no citation, while
    Table 2's legend credits them to Cheng et al. an earlier check called this blocking:
    the headline finding "AM's Strong threshold does not reach Strong-level LR+"
    tests a threshold the authors chose themselves.

    A published calibration exists and should be used instead:

        Bergquist T, Stenton SL, Nadeau EAW, Byrne AB, Greenblatt MS,
        Harrison SM, Tavtigian SV, O'Donnell-Luria A, Biesecker LG,
        Radivojac P, Brenner SE, Pejaver V; ClinGen Sequence Variant
        Interpretation Working Group. Calibration of additional computational
        tools expands ClinGen recommendation options for variant classification
        with PP3/BP4 criteria. Genet Med. 2025;27(6):101402.
        doi:10.1016/j.gim.2025.101402. PMID 40084623. PMCID PMC12208618.

    It calibrates both AM and REVEL against the same ClinVar 2019 reference set
    using local posterior probability, so using it makes the AM-vs-REVEL
    comparison like-for-like -- which is also ITEM 11.

    Consequence for this write-up's central claim: under the published
    calibration, AM >=0.890 falls inside the Moderate (+2) band [0.792, 0.905].
    the write-up empirical finding that AM's "Strong" cutoff performs at
    Moderate strength is therefore INDEPENDENTLY CORROBORATED rather than
    circular. The AM Strong (+4) band does not begin until 0.972.

    Note also that the write-up REVEL cutoffs (0.644 / 0.773 / 0.932), which
    it attributes to Pejaver et al. 2022 and labels Supporting / Moderate /
    Strong, correspond under the newer Bergquist calibration to the Moderate,
    3-point and Very Strong boundaries respectively.

the interaction test the conclusion rests on but never ran. Both tools
    score the same variants, so the AM-vs-REVEL difference in LoF-GoF gap is a
    paired quantity. Estimated by a stratified cluster bootstrap over variants.

LR+ for BOTH tools, with confidence intervals, against three control
    sets of increasing size, replacing the "ACMG tier achieved" / "Verdict"
    columns that were a deterministic recoding of sensitivity.

Benjamini-Hochberg adjustment across the reported test family, and the
    GCK DMS comparison recomputed complete-case on variants both tools score.

Outputs: results/tier2_*.csv and a printed summary.
"""

import itertools
import json
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
DATA = os.path.join(ROOT, "data")

RNG = np.random.default_rng(20260820)
N_BOOT = 10000

def strip_p(s):
    r"""Remove an HGVS 'p.' prefix and surrounding parentheses -- and nothing else.

    The obvious one-liner, re.sub(r"[p\.\(\)]", "", s), deletes EVERY 'p'
    in the string, so Asp209Glu becomes As209Glu and Trp becomes Tr. That silently
    mangles every variant at an Asp or Trp residue.
    """
    s = str(s).strip()
    s = re.sub(r"^p\.", "", s)
    return s.strip("()").strip()


# ---------------------------------------------------------------- thresholds
# As published in Bergquist et al. 2025 (Genet Med 27:101402), lower bound of
# each evidence-strength interval on the pathogenic side.
BERGQUIST = {
    "AM": {"Supporting": 0.170, "Moderate": 0.792, "Strong": 0.972},
    "REVEL": {"Supporting": 0.291, "Moderate": 0.644, "Strong": 0.879},
}
# What v26/v27 used. AM Moderate/Strong are author-derived; REVEL is Pejaver 2022.
MANUSCRIPT = {
    "AM": {"Supporting": 0.564, "Moderate": 0.740, "Strong": 0.890},
    "REVEL": {"Supporting": 0.644, "Moderate": 0.773, "Strong": 0.932},
}
TIERS = ["Supporting", "Moderate", "Strong"]
ACMG_LR = {"Supporting": 2.08, "Moderate": 4.3, "Strong": 18.7}
COL = {"AM": "am_score", "REVEL": "revel_score"}


# ---------------------------------------------------------------- helpers
def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    m = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return c - m, c + m


def lr_plus(tp, n_dis, fp, n_ben):
    """LR+ with a log-method 95% CI. Returns (lr, lo, hi, ceiling)."""
    sens = tp / n_dis
    spec = (n_ben - fp) / n_ben
    ceiling = np.inf if fp == 0 else n_ben / fp
    if fp == 0 or sens == 0:
        return (np.inf if fp == 0 else 0.0), np.nan, np.nan, ceiling
    lr = sens / (1 - spec)
    var = (1 - sens) / tp + spec / fp
    se = np.sqrt(var)
    return lr, lr * np.exp(-1.96 * se), lr * np.exp(1.96 * se), ceiling


def bh(pvals):
    """Benjamini-Hochberg adjusted p-values, order preserved."""
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 1.0
    for rank, idx in enumerate(order[::-1]):
        i = n - rank
        prev = min(prev, p[idx] * n / i)
        adj[idx] = prev
    return adj


# ---------------------------------------------------------------- datasets
def load_paired():
    """Per-variant frames with both tool scores, one per dataset x gene stratum."""
    out = []

    hop = pd.read_csv(os.path.join(RES, "all_variants_annotated.csv"))
    hop = hop.rename(columns={"protein": "vid"})
    hop["dataset"] = "Hopkins"
    out.append(hop[["dataset", "gene", "vid", "mechanism", "am_score", "revel_score"]])

    dfr = pd.read_csv(os.path.join(DATA, "de_franco", "de_franco_processed.csv"))
    # de-duplicate against Hopkins, as established in 10c
    import re

    norm = strip_p
    hk = set(zip(hop.gene, hop.vid.map(norm)))
    dfr["vid"] = dfr.hgvs_p
    dfr = dfr[~pd.Series(list(zip(dfr.gene, dfr.hgvs_p.map(norm)))).isin(hk).values]
    dfr["dataset"] = "De Franco (novel)"
    out.append(dfr[["dataset", "gene", "vid", "mechanism", "am_score", "revel_score"]])

    dms = pd.read_csv(os.path.join(DATA, "gck_dms", "gck_activity_processed.csv"))
    dms = dms[dms.mechanism.isin(["LoF", "GoF"])].copy()
    dms["dataset"] = "GCK DMS"
    dms["gene"] = "GCK"
    dms["vid"] = dms.hgvs_pro
    out.append(dms[["dataset", "gene", "vid", "mechanism", "am_score", "revel_score"]])

    return pd.concat(out, ignore_index=True)


def gap(df, tool, cut):
    c = COL[tool]
    d = df.dropna(subset=[c])
    lof, gof = d[d.mechanism == "LoF"], d[d.mechanism == "GoF"]
    if len(lof) == 0 or len(gof) == 0:
        return np.nan, 0, 0, 0, 0
    lk, gk = int((lof[c] >= cut).sum()), int((gof[c] >= cut).sum())
    return (lk / len(lof) - gk / len(gof)) * 100, lk, len(lof), gk, len(gof)


# ---------------------------------------------------------------- item 10/11
def tier_table(paired):
    """Detection rates at each tool's own published-calibrated tier."""
    rows = []
    for scheme, TH in (
        ("Bergquist 2025", BERGQUIST),
        ("v26 (as submitted)", MANUSCRIPT),
    ):
        for ds, sub in paired.groupby("dataset", sort=False):
            for tool in ("AM", "REVEL"):
                for tier in TIERS:
                    g, lk, ln, gk, gn = gap(sub, tool, TH[tool][tier])
                    if ln == 0:
                        continue
                    p = fisher_exact([[lk, ln - lk], [gk, gn - gk]])[1]
                    rows.append(
                        dict(
                            scheme=scheme,
                            dataset=ds,
                            tool=tool,
                            tier=tier,
                            cutoff=TH[tool][tier],
                            lof_pct=lk / ln * 100,
                            gof_pct=gk / gn * 100,
                            gap=g,
                            lof_n=ln,
                            gof_n=gn,
                            p=p,
                        )
                    )
    return pd.DataFrame(rows)


def matched_operating_points(paired):
    """Item 11: hold LoF sensitivity equal across tools, then compare gaps."""
    hop = paired[paired.dataset == "Hopkins"]
    rows = []
    for tier in TIERS:
        cut_am = BERGQUIST["AM"][tier]
        g_am, lk, ln, gk, gn = gap(hop, "AM", cut_am)
        target = lk / ln
        # REVEL cutoff giving the closest LoF sensitivity to AM's
        rv = hop.dropna(subset=["revel_score"])
        lof_r = np.sort(rv[rv.mechanism == "LoF"].revel_score.values)[::-1]
        k = max(1, int(round(target * len(lof_r))))
        cut_rv = lof_r[min(k, len(lof_r)) - 1]
        g_rv, lk2, ln2, gk2, gn2 = gap(hop, "REVEL", cut_rv)
        rows.append(
            dict(
                tier=tier,
                am_cutoff=cut_am,
                am_lof_sens=lk / ln * 100,
                am_gap=g_am,
                revel_cutoff_matched=cut_rv,
                revel_lof_sens=lk2 / ln2 * 100,
                revel_gap=g_rv,
                delta_gap=g_am - g_rv,
            )
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- item 12
def interaction_test(paired, TH, label):
    """Paired cluster bootstrap on delta = gap(AM) - gap(REVEL)."""
    d = paired.dropna(subset=["am_score", "revel_score"]).copy()
    strata = list(d.groupby(["dataset", "gene"], sort=False).indices.items())
    idx_by_stratum = [np.asarray(ix) for _, ix in strata]
    am_s, rv_s = d.am_score.values, d.revel_score.values
    is_lof = (d.mechanism == "LoF").values

    def delta(sel, cam, crv):
        lof, gof = sel[is_lof[sel]], sel[~is_lof[sel]]
        if len(lof) == 0 or len(gof) == 0:
            return np.nan
        g_am = (am_s[lof] >= cam).mean() - (am_s[gof] >= cam).mean()
        g_rv = (rv_s[lof] >= crv).mean() - (rv_s[gof] >= crv).mean()
        return (g_am - g_rv) * 100

    rows = []
    for tier in TIERS:
        cam, crv = TH["AM"][tier], TH["REVEL"][tier]
        allidx = np.arange(len(d))
        obs = delta(allidx, cam, crv)
        boot = np.empty(N_BOOT)
        for b in range(N_BOOT):
            pick = np.concatenate(
                [RNG.choice(ix, size=len(ix), replace=True) for ix in idx_by_stratum]
            )
            boot[b] = delta(pick, cam, crv)
        boot = boot[~np.isnan(boot)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        # two-sided bootstrap p: proportion of resamples on the other side of 0
        p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
        p = min(1.0, max(p, 1 / len(boot)))
        rows.append(
            dict(
                scheme=label,
                tier=tier,
                am_cutoff=cam,
                revel_cutoff=crv,
                delta_gap_pp=obs,
                ci_lo=lo,
                ci_hi=hi,
                p_boot=p,
                n_variants=len(d),
            )
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- item 13/14
def lr_table():
    """LR+ for both tools against three control sets, with CIs. No verdicts."""
    cv = pd.read_csv(os.path.join(RES, "clinvar_with_revel.csv"), low_memory=False)
    ben_gn = pd.read_csv(os.path.join(RES, "benign_control_set.csv"))

    controls = {
        "ClinVar 3-star benign (as submitted)": cv[cv.mechanism == "Benign"],
        "gnomAD AF>=1e-4": ben_gn[ben_gn.tier == "strict"],
        "gnomAD AF>=1e-5": ben_gn,
    }
    rows = []
    for scheme, TH in (
        ("Bergquist 2025", BERGQUIST),
        ("v26 (as submitted)", MANUSCRIPT),
    ):
        for cname, ben in controls.items():
            for tool in ("AM", "REVEL"):
                c = COL[tool]
                b = ben.dropna(subset=[c])
                for mech in ("LoF", "GoF"):
                    dis = cv[(cv.mechanism == mech)].dropna(subset=[c])
                    for tier in TIERS:
                        cut = TH[tool][tier]
                        tp = int((dis[c] >= cut).sum())
                        fp = int((b[c] >= cut).sum())
                        lr, lo, hi, ceil = lr_plus(tp, len(dis), fp, len(b))
                        srange = wilson(len(b) - fp, len(b))
                        rows.append(
                            dict(
                                scheme=scheme,
                                control=cname,
                                n_benign=len(b),
                                tool=tool,
                                mech=mech,
                                n_disease=len(dis),
                                tier=tier,
                                cutoff=cut,
                                sens=tp / len(dis) if len(dis) else np.nan,
                                spec=(len(b) - fp) / len(b),
                                spec_lo=srange[0],
                                spec_hi=srange[1],
                                fp=fp,
                                lr=lr,
                                lr_lo=lo,
                                lr_hi=hi,
                                lr_ceiling=ceil,
                                lr_required=ACMG_LR[tier],
                            )
                        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- item 15
def dms_complete_case(paired):
    """REVEL covers 81.7% of the DMS; recompute both tools on the shared subset."""
    dms = paired[paired.dataset == "GCK DMS"]
    both = dms.dropna(subset=["am_score", "revel_score"])
    rows = []
    for scheme, TH in (
        ("Bergquist 2025", BERGQUIST),
        ("v26 (as submitted)", MANUSCRIPT),
    ):
        for basis, sub in (("all AM-scored", dms), ("complete-case", both)):
            for tool in ("AM", "REVEL"):
                g, lk, ln, gk, gn = gap(sub, tool, TH[tool]["Strong"])
                if ln == 0:
                    continue
                rows.append(
                    dict(
                        scheme=scheme,
                        basis=basis,
                        tool=tool,
                        n=len(sub),
                        lof_n=ln,
                        gof_n=gn,
                        gap=g,
                    )
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    paired = load_paired()
    print(
        f"loaded {len(paired)} variants across "
        f"{paired.groupby(['dataset', 'gene']).ngroups} dataset x gene strata"
    )

    print("\n=== ITEM 10/11: detection at each tool's own published tier ===")
    tt = tier_table(paired)
    tt.to_csv(os.path.join(RES, "tier2_tier_table.csv"), index=False)
    for scheme in tt.scheme.unique():
        print(f"\n  {scheme}")
        for ds in tt.dataset.unique():
            s = tt[(tt.scheme == scheme) & (tt.dataset == ds)]
            if s.empty:
                continue
            print(f"    {ds}")
            for tier in TIERS:
                r = {t: s[(s.tool == t) & (s.tier == tier)] for t in ("AM", "REVEL")}
                if r["AM"].empty or r["REVEL"].empty:
                    continue
                a, v = r["AM"].iloc[0], r["REVEL"].iloc[0]
                print(
                    f"      {tier:11s} AM@{a.cutoff:.3f} gap {a.gap:5.1f} pp   "
                    f"REVEL@{v.cutoff:.3f} gap {v.gap:5.1f} pp   "
                    f"Δ {a.gap - v.gap:5.1f}"
                )

    print("\n=== ITEM 11: LoF-sensitivity-matched comparison (Hopkins) ===")
    mo = matched_operating_points(paired)
    mo.to_csv(os.path.join(RES, "tier2_matched_operating_points.csv"), index=False)
    print(mo.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print(f"\n=== ITEM 12: paired interaction test ({N_BOOT} cluster bootstraps) ===")
    it = pd.concat(
        [
            interaction_test(paired, BERGQUIST, "Bergquist 2025"),
            interaction_test(paired, MANUSCRIPT, "v26 (as submitted)"),
        ],
        ignore_index=True,
    )
    it.to_csv(os.path.join(RES, "tier2_interaction_test.csv"), index=False)
    for _, r in it.iterrows():
        star = "*" if r.p_boot < 0.05 else " "
        print(
            f"  {r.scheme:20s} {r.tier:11s} Δgap {r.delta_gap_pp:6.1f} pp  "
            f"95% CI [{r.ci_lo:6.1f}, {r.ci_hi:6.1f}]  p={r.p_boot:.4f}{star}"
        )

    print("\n=== ITEM 13/14: LR+ with CIs, both tools, three control sets ===")
    lt = lr_table()
    lt.to_csv(os.path.join(RES, "tier2_lr_table.csv"), index=False)
    for scheme in ("Bergquist 2025", "v26 (as submitted)"):
        print(f"\n  {scheme}")
        for cname in lt.control.unique():
            s = lt[(lt.scheme == scheme) & (lt.control == cname)]
            ceil = s.lr_ceiling.iloc[0]
            print(
                f"    {cname}  (n_benign={s.n_benign.iloc[0]}, "
                f"LR+ ceiling {'inf' if np.isinf(ceil) else f'{ceil:.0f}'})"
            )
            for tool in ("AM", "REVEL"):
                for mech in ("LoF", "GoF"):
                    r = s[(s.tool == tool) & (s.mech == mech) & (s.tier == "Strong")]
                    if r.empty:
                        continue
                    r = r.iloc[0]
                    lo = "nan" if np.isnan(r.lr_lo) else f"{r.lr_lo:.1f}"
                    hi = "inf" if np.isnan(r.lr_hi) else f"{r.lr_hi:.1f}"
                    lr = "inf" if np.isinf(r.lr) else f"{r.lr:.1f}"
                    print(
                        f"      {tool:5s} {mech} Strong@{r.cutoff:.3f}: "
                        f"sens {r.sens:.3f} spec {r.spec:.3f} "
                        f"LR+ {lr} [{lo}, {hi}]  (need {r.lr_required})"
                    )

    print("\n=== ITEM 15: GCK DMS complete-case ===")
    dc = dms_complete_case(paired)
    dc.to_csv(os.path.join(RES, "tier2_dms_complete_case.csv"), index=False)
    print(dc.to_string(index=False, float_format=lambda x: f"{x:.1f}"))

    print("\n=== ITEM 15: Benjamini-Hochberg across the reported family ===")
    gene_tests = []
    for (ds, g), sub in paired.groupby(["dataset", "gene"], sort=False):
        for tool in ("AM", "REVEL"):
            for tier in TIERS:
                _, lk, ln, gk, gn = gap(sub, tool, MANUSCRIPT[tool][tier])
                if ln == 0 or gn == 0:
                    continue
                gene_tests.append(dict(
                    dataset=f"{ds}:{g}", tool=tool, tier=tier,
                    p=fisher_exact([[lk, ln - lk], [gk, gn - gk]])[1]))
    fam = pd.concat(
        [tt[tt.scheme == "v26 (as submitted)"][["dataset", "tool", "tier", "p"]],
         pd.DataFrame(gene_tests)], ignore_index=True)
    fam["p_bh"] = bh(fam.p.values)
    fam.to_csv(os.path.join(RES, "tier2_bh_adjusted.csv"), index=False)
    flipped = fam[(fam.p < 0.05) & (fam.p_bh >= 0.05)]
    print(
        f"  {len(fam)} tests; {(fam.p < 0.05).sum()} nominally significant, "
        f"{(fam.p_bh < 0.05).sum()} survive BH"
    )
    if len(flipped):
        print("  lose significance under BH:")
        for _, r in flipped.iterrows():
            print(
                f"    {r.dataset:20s} {r.tool:5s} {r.tier:11s} "
                f"p={r.p:.4f} -> q={r.p_bh:.4f}"
            )

    json.dump(
        {"n_boot": N_BOOT, "bergquist": BERGQUIST, "manuscript": MANUSCRIPT},
        open(os.path.join(RES, "tier2_config.json"), "w"),
        indent=2,
    )
    print("\nwrote results/tier2_*.csv")
