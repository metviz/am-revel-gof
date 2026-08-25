"""Regression test for the script-09 silent-fallback bug.

The bug: 09_proteingym_analysis.py wrapped its data acquisition in a bare
`except Exception`, so when pyarrow was missing (ImportError from read_parquet)
it silently substituted ClinVar GCK data and wrote it to files named
`proteingym_*`. That produced a meta-analysis stratum labelled "ProteinGym v1.0"
whose rows were actually ClinVar — and it reached the write-up.

These tests assert the script now FAILS LOUDLY instead. Run:

    python3 scripts/test_09_no_silent_fallback.py
"""

import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "09_proteingym_analysis.py")
RAW = os.path.join(ROOT, "data", "proteingym", "proteingym_gck_clinical.csv")

CLINVAR_HEADER = (
    "AlleleID,Type,Name,GeneID,GeneSymbol,ClinicalSignificance,ReviewStatus,"
    "RCVaccession,mutant,AlphaMissense,mechanism\n"
    "31172,single nucleotide variant,NM_000162.5(GCK),2645,GCK,Pathogenic,"
    "reviewed by expert panel,RCV000123,R186C,0.97,LoF\n"
)


def run_script(env=None):
    e = dict(os.environ)
    if env:
        e.update(env)
    p = subprocess.run(
        [sys.executable, SCRIPT],
        cwd=ROOT,
        env=e,
        capture_output=True,
        text=True,
        timeout=600,
    )
    return p.returncode, p.stdout + p.stderr


def test_source_has_no_fallback():
    """The ClinVar substitution code must not exist at all."""
    src = open(SCRIPT, encoding="utf-8").read()
    banned = [
        "using ClinVar GCK expert panel data as benchmark",
        "Fall back to ClinVar GCK data as the benchmark",
    ]
    for b in banned:
        assert b not in src, f"silent-fallback code is back: {b!r}"
    assert "class ProteinGymUnavailable" in src, "the fatal error class is gone"
    print("  ok: no ClinVar fallback code in source")


def test_rejects_clinvar_masquerading_as_proteingym():
    """A cache carrying ClinVar-only columns must be refused, not reused."""
    backup = None
    if os.path.exists(RAW):
        backup = RAW + ".testbak"
        os.replace(RAW, backup)
    try:
        with open(RAW, "w", encoding="utf-8") as fh:
            fh.write(CLINVAR_HEADER)
        rc, out = run_script()
        assert rc != 0, "script exited 0 on a ClinVar-poisoned cache"
        assert "ProteinGymUnavailable" in out, out[-400:]
        assert "masquerading as ProteinGym" in out, out[-400:]
        print("  ok: ClinVar-poisoned cache is rejected")
    finally:
        if os.path.exists(RAW):
            os.remove(RAW)
        if backup:
            os.replace(backup, RAW)


def test_raises_when_proteingym_cannot_answer_the_question():
    """The real parquet has no predictor scores and no LoF/GoF labels -> must raise."""
    backup = None
    if os.path.exists(RAW):
        backup = RAW + ".testbak"
        os.replace(RAW, backup)
    try:
        rc, out = run_script()
        assert rc != 0, "script exited 0 despite ProteinGym lacking mechanism labels"
        assert "ProteinGymUnavailable" in out, out[-400:]
        assert "No LoF/GoF mechanism labels" in out, out[-400:]
        print("  ok: refuses to fabricate a mechanism stratum from ProteinGym")
    finally:
        if backup:
            os.replace(backup, RAW)


if __name__ == "__main__":
    print("script 09 — silent-fallback regression tests")
    test_source_has_no_fallback()
    test_rejects_clinvar_masquerading_as_proteingym()
    test_raises_when_proteingym_cannot_answer_the_question()
    print("\nALL PASS — script 09 fails loudly and cannot launder ClinVar data.")
