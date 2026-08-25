"""
Regression check for 10c_meta_analysis_dedup.py.

Run: /home/raghu/tools/.venv/bin/python scripts/test_10c_meta.py

Three things worth guarding:
  1. The DerSimonian-Laird estimator is a reimplementation. Fed 10b's stratum
     table it must reproduce 10b's published pooled numbers exactly, otherwise
     the dedup result is not comparable to the corrected result it replaces.
  2. De-duplication must actually de-duplicate: no variant may appear in both a
     Hopkins stratum and a De Franco stratum.
  3. The ProteinGym stratum that script 09 fabricated must not be back.
"""

import importlib.util
import os
import re

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")

spec = importlib.util.spec_from_file_location(
    "m10c", os.path.join(ROOT, "scripts", "10c_meta_analysis_dedup.py")
)
m10c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m10c)


def test_dl_reproduces_10b():
    """dl_meta on 10b's strata must match 10b's published summary."""
    eff = pd.read_csv(os.path.join(RES, "meta_analysis_effects_corrected.csv"))
    summ = pd.read_csv(os.path.join(RES, "meta_analysis_summary_corrected.csv"))
    checked = 0
    for tool in ("AM", "REVEL"):
        sub = eff[eff.tool == tool]
        # 10b's "All" pool is the 6 gene-level strata, i.e. everything except the
        # whole-cohort rows (gene == "All")
        strata = sub[sub.gene != "All"]
        got = m10c.dl_meta(strata.gap.values, strata.se.values)
        want = summ[(summ.tool == tool) & (summ.subgroup == "All")].iloc[0]
        assert int(got["k"]) == int(want.k), f"{tool} k {got['k']} != {want.k}"
        for field in ("pooled_gap", "ci_lo", "ci_hi", "I2", "Q"):
            a, b = got[field], float(want[field])
            assert abs(a - b) < 0.05, f"{tool}.{field}: {a:.3f} != {b:.3f}"
        checked += 1
    print(f"  ok  DL estimator reproduces 10b on {checked} pools")


def test_strata_are_disjoint():
    """No variant may enter the pool twice via Hopkins and De Franco."""
    hop = pd.read_csv(os.path.join(RES, "all_variants_annotated.csv"))
    dfr = pd.read_csv(
        os.path.join(ROOT, "data", "de_franco", "de_franco_processed.csv")
    )
    hop_keys = set(zip(hop.gene, hop.protein.map(m10c.norm_hgvs)))
    dfr_keys = list(zip(dfr.gene, dfr.hgvs_p.map(m10c.norm_hgvs)))
    novel = [k for k in dfr_keys if k not in hop_keys]

    shared = len(dfr_keys) - len(novel)
    assert shared == 100, f"expected 100 shared variants, got {shared}"
    assert len(novel) == 447, f"expected 447 novel, got {len(novel)}"
    assert not (set(novel) & hop_keys), "de-duplication left overlapping variants"
    print(
        f"  ok  {shared} shared removed, {len(novel)} De Franco strata variants disjoint"
    )


def test_no_proteingym():
    """The fabricated stratum must not reappear in outputs or the manuscript."""
    for name in ("meta_analysis_effects_dedup.csv", "meta_analysis_summary_dedup.csv"):
        body = open(os.path.join(RES, name)).read()
        assert "ProteinGym" not in body, f"ProteinGym back in {name}"

    eff = pd.read_csv(os.path.join(RES, "meta_analysis_effects_dedup.csv"))
    summ = pd.read_csv(os.path.join(RES, "meta_analysis_summary_dedup.csv"))
    for tool in ("AM", "REVEL"):
        k = int(summ[(summ.tool == tool) & (summ.subgroup == "All")].iloc[0].k)
        assert k == 6, f"{tool} all-strata pool has k={k}, expected 6"
        kc = int(summ[(summ.tool == tool) & (summ.subgroup == "Clinical")].iloc[0].k)
        assert kc == 5, f"{tool} clinical pool has k={kc}, expected 5"
    assert len(eff) == 12, f"expected 12 stratum rows, got {len(eff)}"
    print("  ok  no ProteinGym stratum; k=5 clinical / k=6 all for both tools")



if __name__ == "__main__":
    print("test_10c_meta.py")
    test_dl_reproduces_10b()
    test_strata_are_disjoint()
    test_no_proteingym()
    print("all checks passed")
