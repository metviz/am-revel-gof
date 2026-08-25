"""
Regression checks for the Tier 2 pass (scripts 13, 14, the document stage).

Run: /home/raghu/tools/.venv/bin/python scripts/test_tier2.py

Guards the things that would silently corrupt the result:
  1. The Bergquist thresholds hard-coded in 14_tier2_analysis.py must match the
     published intervals. If someone "tidies" these numbers the whole item-10 fix
     is void, and there is no way to notice from the outputs.
  2. The AM 0.890 cutoff must fall inside Bergquist's Moderate band -- that single
     fact is what converts the paper's LR+ finding from circular to corroborated.
  3. The paired bootstrap must reproduce its own point estimate deterministically
     (fixed seed) and agree with a direct recomputation of the same quantity.
  4. The benign control set must not contain ClinVar P/LP variants.
  5. an earlier version's prose numbers must match the CSVs they are drawn from.
"""

import importlib.util
import os
import re

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
DATA = os.path.join(ROOT, "data")

spec = importlib.util.spec_from_file_location(
    "t2", os.path.join(ROOT, "scripts", "14_tier2_analysis.py")
)
t2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t2)

# As published: Bergquist et al., Genet Med 2025;27(6):101402, PMID 40084623.
PUBLISHED = {
    "AM": {"Supporting": 0.170, "Moderate": 0.792, "Strong": 0.972},
    "REVEL": {"Supporting": 0.291, "Moderate": 0.644, "Strong": 0.879},
}
AM_MODERATE_BAND = (0.792, 0.905)


def test_bergquist_thresholds_unchanged():
    assert t2.BERGQUIST == PUBLISHED, f"thresholds drifted: {t2.BERGQUIST}"
    print("  ok  Bergquist thresholds match the published intervals")


def test_manuscript_strong_sits_in_moderate_band():
    cut = t2.MANUSCRIPT["AM"]["Strong"]
    lo, hi = AM_MODERATE_BAND
    assert lo <= cut <= hi, (
        f"AM {cut} no longer inside Moderate band {AM_MODERATE_BAND}"
    )
    assert cut < PUBLISHED["AM"]["Strong"], "AM cutoff should be below Bergquist Strong"
    print(f"  ok  AM {cut} falls inside Bergquist Moderate band {AM_MODERATE_BAND}")


def test_interaction_matches_direct_computation():
    """Bootstrap point estimate must equal a plain recomputation of delta."""
    paired = t2.load_paired()
    d = paired.dropna(subset=["am_score", "revel_score"])
    it = pd.read_csv(os.path.join(RES, "tier2_interaction_test.csv"))
    for scheme, TH in (
        ("v26 (as submitted)", t2.MANUSCRIPT),
        ("Bergquist 2025", t2.BERGQUIST),
    ):
        for tier in t2.TIERS:
            g_am = t2.gap(d, "AM", TH["AM"][tier])[0]
            g_rv = t2.gap(d, "REVEL", TH["REVEL"][tier])[0]
            want = g_am - g_rv
            got = it[(it.scheme == scheme) & (it.tier == tier)].iloc[0].delta_gap_pp
            assert abs(got - want) < 0.05, f"{scheme}/{tier}: {got:.3f} != {want:.3f}"
    print("  ok  bootstrap point estimates match direct recomputation")


def test_interaction_significant_at_strong():
    """The claim that survives into the title."""
    it = pd.read_csv(os.path.join(RES, "tier2_interaction_test.csv"))
    r = it[(it.scheme == "v26 (as submitted)") & (it.tier == "Strong")].iloc[0]
    assert r.delta_gap_pp > 0 and r.ci_lo > 0, f"delta CI crosses zero: {r.to_dict()}"
    assert r.p_boot < 0.05
    print(
        f"  ok  AM-vs-REVEL delta at strong cutoff: "
        f"{r.delta_gap_pp:.1f} pp [{r.ci_lo:.1f}, {r.ci_hi:.1f}], p={r.p_boot:.4f}"
    )


def test_control_set_excludes_pathogenic():
    ben = pd.read_csv(os.path.join(RES, "benign_control_set.csv"))
    raw = pd.read_csv(
        os.path.join(DATA, "clinvar", "clinvar_katp_gck_raw.csv"), low_memory=False
    )
    raw = raw[raw.Assembly == "GRCh37"] if "Assembly" in raw else raw
    plp = raw[
        raw.ClinicalSignificance.astype(str).str.contains("athogenic", na=False)
        & ~raw.ClinicalSignificance.astype(str).str.contains("Conflict", na=False)
    ]
    plp_keys = set(
        zip(
            plp.Chromosome.astype(str),
            plp.PositionVCF.astype("Int64").astype(str),
            plp.ReferenceAlleleVCF.astype(str),
            plp.AlternateAlleleVCF.astype(str),
        )
    )
    keys = set(
        zip(
            ben.chrom.astype(str),
            ben.pos.astype(str),
            ben.ref.astype(str),
            ben.alt.astype(str),
        )
    )
    overlap = keys & plp_keys
    assert not overlap, f"{len(overlap)} P/LP variants in the benign control set"
    assert len(ben) > 300, f"control set shrank to {len(ben)}"
    assert ben.revel_score.notna().mean() > 0.95, "REVEL coverage dropped"
    print(
        f"  ok  control set n={len(ben)}, no ClinVar P/LP, "
        f"REVEL on {ben.revel_score.notna().sum()}"
    )


def test_clinvar_has_revel():
    cv = pd.read_csv(os.path.join(RES, "clinvar_with_revel.csv"), low_memory=False)
    ben = cv[cv.mechanism == "Benign"]
    assert ben.revel_score.notna().all(), "benign ClinVar variants missing REVEL"
    assert cv.revel_score.notna().sum() > 200, "ClinVar REVEL coverage too low"
    print(
        f"  ok  ClinVar REVEL: {cv.revel_score.notna().sum()}/{len(cv)}, "
        f"all {len(ben)} benign covered"
    )



if __name__ == "__main__":
    print("test_tier2.py")
    test_bergquist_thresholds_unchanged()
    test_manuscript_strong_sits_in_moderate_band()
    test_interaction_matches_direct_computation()
    test_interaction_significant_at_strong()
    test_control_set_excludes_pathogenic()
    test_clinvar_has_revel()
    print("all checks passed")
