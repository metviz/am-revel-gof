"""
Regression checks for the Tier 4 pass (scripts 15, 16, fix_tier4_v30).

Run: /home/raghu/tools/.venv/bin/python scripts/test_tier4.py

The check that matters most here is the HGVS parser. The bug it guards against --
re.sub(r"[p\\.\\(\\)]", "", s) deleting every letter 'p', so p.Asp209Glu became
As209Glu -- was silent: it produced a well-formed-looking string, the joins still
"worked" wherever both sides were mangled identically, and it only surfaced as a
23/131 coverage shortfall in one analysis. Anything that parses HGVS gets a test.
"""

import importlib.util
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")


def load(name):
    spec = importlib.util.spec_from_file_location(
        name.replace(".py", ""), os.path.join(ROOT, "scripts", name)
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_hgvs_parser_handles_p_containing_residues():
    """Asp and Trp both contain 'p'. This is the bug that shipped."""
    m = load("16_positional_and_dynamics.py")
    cases = {
        "p.Asp209Glu": "D209E",  # Asp -> was mangled to 'As'
        "p.(Trp1339Arg)": "W1339R",  # Trp -> was mangled to 'Tr'
        "p.Val59Met": "V59M",
        "p.Gln52Arg": "Q52R",
        "p.(Ser3Cys)": "S3C",
        "V59M": "V59M",
        "p.Pro254Leu": "P254L",  # two p's, one in the prefix
    }
    for raw, want in cases.items():
        got = m.short(raw)
        assert got == want, f"short({raw!r}) = {got!r}, expected {want!r}"
    assert m.short("p.?") is None
    print(f"  ok  HGVS parser correct on {len(cases)} cases incl. Asp/Trp/Pro")


def test_positional_prior_full_coverage():
    """Every Hopkins variant must get a site prior; 108/131 was the bug's symptom."""
    d = pd.read_csv(os.path.join(RES, "tier4_positional_prior.csv"))
    assert len(d) == 131, f"expected 131 Hopkins variants, got {len(d)}"
    assert d.prior_loo.notna().all(), (
        f"{d.prior_loo.isna().sum()} variants without a leave-one-out site prior"
    )
    assert set(d.gene) == {"ABCC8", "KCNJ11", "GCK"}, "a gene is missing"
    n_kcnj11 = int((d.gene == "KCNJ11").sum())
    assert n_kcnj11 > 50, f"KCNJ11 under-represented ({n_kcnj11}); was it excluded?"
    print(f"  ok  site prior for 131/131 variants, KCNJ11 included (n={n_kcnj11})")


def test_site_prior_in_range_and_isoform_matched():
    """Site priors are means of AM scores: they cannot exceed 1.0."""
    d = pd.read_csv(os.path.join(RES, "tier4_positional_prior.csv"))
    assert (d.prior_loo <= 1.0).all(), (
        f"site prior above 1.0 for {list(d[d.prior_loo > 1.0].protein)}"
    )
    # ABCC8 is numbered one residue higher than UniProt above ~600; if the offset
    # resolution regresses, those variants silently read the wrong site.
    off = d.cache_offset.value_counts().to_dict()
    assert off.get(-1, 0) == 20, f"isoform offset counts changed: {off}"
    assert d.prior_loo.notna().all(), "offset resolution dropped variants"
    print(f"  ok  all site priors <= 1.0; isoform offset applied to {off[-1]} variants")


def test_loo_is_not_inflated():
    """The leave-one-out prior must exclude the variant, so it differs from naive."""
    d = pd.read_csv(os.path.join(RES, "tier4_positional_prior.csv"))
    assert not (d.prior_loo == d.prior_naive).all(), "leave-one-out is not leaving out"
    from scipy.stats import spearmanr

    naive = spearmanr(d.am_score, d.prior_naive).statistic
    loo = spearmanr(d.am_score, d.prior_loo).statistic
    assert loo < naive, f"LOO r ({loo:.3f}) should be below naive r ({naive:.3f})"
    print(
        f"  ok  part-whole inflation removed (naive r={naive:.3f} -> LOO r={loo:.3f})"
    )


def test_panel_covers_full_cohort():
    d = pd.read_csv(os.path.join(RES, "tier4_panel_scores.csv"))
    assert len(d) == 578, f"expected 578 cohort variants, got {len(d)}"
    for col in ("AlphaMissense", "REVEL", "PrimateAI", "MPC", "SIFT4G"):
        cov = d[col].notna().mean()
        assert cov > 0.95, f"{col} coverage only {cov:.0%}"
    print(f"  ok  predictor panel covers {len(d)} variants")


def test_panel_conclusion_holds():
    """The Tier 4 claim: shared across deleteriousness predictors, absent from two."""
    g = pd.read_csv(os.path.join(RES, "tier4_panel_gaps.csv"))
    g = g[(g.dataset == "ALL") & g.usable]
    by = g.groupby("predictor").gap.mean()
    for p in ("AlphaMissense", "REVEL", "SIFT4G", "VEST4", "PROVEAN"):
        assert by[p] > 10, f"{p} gap collapsed to {by[p]:.1f} pp"
    assert by["PrimateAI"] < 5, f"PrimateAI gap rose to {by['PrimateAI']:.1f} pp"
    assert by["MPC"] < 10, f"MPC gap rose to {by['MPC']:.1f} pp"
    assert by["AlphaMissense"] > by["REVEL"], "AM no longer exceeds REVEL"
    assert not g[g.predictor == "LRT"].shape[0], "degenerate LRT back in the summary"
    # the screen must be on achieved-vs-target sensitivity, not distinct-value count
    raw = pd.read_csv(os.path.join(RES, "tier4_panel_gaps.csv"))
    bad = raw[raw.usable & (raw.sens_overshoot.abs() > 2.0)]
    assert bad.empty, f"unmatched rows marked usable: {list(bad.predictor)}"
    print(
        f"  ok  AM {by['AlphaMissense']:.1f} pp > REVEL {by['REVEL']:.1f} pp; "
        f"PrimateAI {by['PrimateAI']:.1f}, MPC {by['MPC']:.1f}"
    )


def test_dedup_unaffected_by_parser_fix():
    """The parser bug did not change de-duplication; confirm it still doesn't."""
    e = pd.read_csv(os.path.join(RES, "meta_analysis_effects_dedup.csv"))
    df = e[e.dataset.str.startswith("De Franco") & (e.tool == "AM")]
    total = int(df.lof_n.sum() + df.gof_n.sum())
    assert total == 447, f"De Franco strata hold {total} variants, expected 447"
    print("  ok  de-duplication still yields 447 De Franco variants")



if __name__ == "__main__":
    print("test_tier4.py")
    test_hgvs_parser_handles_p_containing_residues()
    test_positional_prior_full_coverage()
    test_site_prior_in_range_and_isoform_matched()
    test_loo_is_not_inflated()
    test_panel_covers_full_cohort()
    test_panel_conclusion_holds()
    test_dedup_unaffected_by_parser_fix()
    print("all checks passed")
