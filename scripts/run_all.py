"""
run_all.py — re-run the AM/REVEL analysis and write-up chain end to end.

    /home/raghu/tools/.venv/bin/python scripts/run_all.py            # everything
    /home/raghu/tools/.venv/bin/python scripts/run_all.py --tests    # tests only
    /home/raghu/tools/.venv/bin/python scripts/run_all.py --check    # verify inputs, run nothing
    /home/raghu/tools/.venv/bin/python scripts/run_all.py --offline  # fail rather than hit the network

Stops at the first failure. See REPRODUCE.md for what each stage does and why.

the write-up steps are NOT idempotent: each rewrites specific sentences of the
version before it and raises SystemExit if an anchor is missing. That is deliberate --
a silent no-op is how an earlier version ended up carrying two different meta-analyses. Re-running
the analysis chain is deterministic; re-running a single step out of order will
fail loudly.
"""

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = "/home/raghu/tools/.venv/bin/python"

# (script, description, needs_network)
ANALYSIS = [
    ("10c_meta_analysis_dedup.py", "de-duplicated meta-analysis + forest plot", False),
    (
        "13_benign_control_set.py",
        "gnomAD benign control set + ClinVar REVEL annotation",
        True,
    ),
    (
        "14_tier2_analysis.py",
        "thresholds, matched operating points, interaction test, LR+",
        False,
    ),
    ("15_predictor_panel.py", "ten-predictor dbNSFP panel", True),
    (
        "16_positional_and_dynamics.py",
        "positional prior + dynamics-LoF stratification",
        False,
    ),
    (
        "17_v32_supporting.py",
        "Strong-shortfall test, panel group means, ABCC8 offset sensitivity",
        False,
    ),
]

MANUSCRIPT = []  # not part of the public pipeline

TESTS = [
    ("test_10c_meta.py", "meta-analysis and de-duplication", False),
    ("test_tier2.py", "thresholds, control set, interaction test, v32 prose", False),
    ("test_tier3.py", "claims and references (add --online for DOI resolution)", False),
    ("test_tier4.py", "HGVS parser, site priors, isoform offset, panel", False),
    ("test_v32.py", "third-round acceptance checks on v32", False),
]

# Inputs the chain reads but never writes. Missing any of these is fatal.
REQUIRED_INPUTS = [
    "results/all_variants_annotated.csv",
    "data/de_franco/de_franco_processed.csv",
    "data/gck_dms/gck_activity_processed.csv",
    "data/gck_dms/gck_activity_abundance_merged.csv",
    "data/clinvar/clinvar_processed.csv",
    "data/clinvar/clinvar_katp_gck_raw.csv",
    "data/clinvar/am_gene_cache_abcc8_gck_kcnj11.csv",
    "results/meta_analysis_effects_corrected.csv",
]

# Present => that stage runs offline. Absent => it fetches.
CACHES = {
    "data/gnomad/gnomad_missense_raw.csv": "gnomAD v2.1.1 missense (13)",
    "data/dbnsfp/dbnsfp_panel_raw.csv": "dbNSFP predictor panel (15)",
}


def check_inputs(offline: bool) -> bool:
    print("Inputs")
    ok = True
    for f in REQUIRED_INPUTS:
        p = os.path.join(ROOT, f)
        if os.path.exists(p):
            print(f"  ok       {os.path.getsize(p) / 1e3:8.0f}K  {f}")
        else:
            print(f"  MISSING            {f}")
            ok = False

    print("\nCaches (absent = that stage will use the network)")
    cold = []
    for f, what in CACHES.items():
        if os.path.exists(os.path.join(ROOT, f)):
            print(f"  warm     {what}")
        else:
            print(f"  cold     {what}  <- will fetch")
            cold.append(what)

    if cold and offline:
        print("\n  --offline was given but these caches are cold:")
        for c in cold:
            print(f"    {c}")
        ok = False

    # ClinVar REVEL is re-fetched every run; there is no cache for it.
    if not offline:
        print("\n  note     13_benign_control_set.py re-queries myvariant.info for the")
        print("           260 ClinVar variants on every run (no cache for that step)")
    return ok


def run(stage, args):
    script, desc, needs_net = stage
    if needs_net and args.offline:
        print(f"\n[skip]  {script}  ({desc}) -- needs network, --offline given")
        return True
    path = os.path.join(ROOT, "scripts", script)
    if not os.path.exists(path):
        print(f"\n[FAIL]  {script} not found")
        return False
    print(f"\n{'=' * 72}\n{script}  --  {desc}\n{'=' * 72}")
    t0 = time.time()
    cmd = [PY, path]
    if script == "test_tier3.py" and args.online:
        cmd.append("--online")
    r = subprocess.run(cmd, cwd=ROOT)
    dt = time.time() - t0
    if r.returncode != 0:
        print(f"\n[FAIL]  {script} exited {r.returncode} after {dt:.0f}s")
        return False
    print(f"[ok]    {script} ({dt:.0f}s)")
    return True


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--analysis", action="store_true", help="analysis only")
    ap.add_argument("--manuscript", action="store_true", help="manuscript chain only")
    ap.add_argument("--tests", action="store_true", help="tests only")
    ap.add_argument("--check", action="store_true", help="check inputs and exit")
    ap.add_argument("--offline", action="store_true", help="skip network stages")
    ap.add_argument(
        "--online",
        action="store_true",
        help="run test_tier3 with --online (resolves every DOI)",
    )
    args = ap.parse_args()

    print("AM/REVEL reproduction\n" + "-" * 72)
    if not check_inputs(args.offline):
        print("\nInput check failed. See REPRODUCE.md section 1.")
        return 1
    if args.check:
        print("\nInputs OK.")
        return 0

    selected = args.analysis or args.manuscript or args.tests
    stages = []
    if args.analysis or not selected:
        stages += ANALYSIS
    if args.manuscript or not selected:
        stages += MANUSCRIPT
    if args.tests or not selected:
        stages += TESTS

    t0 = time.time()
    for stage in stages:
        if not run(stage, args):
            print("\nStopped at the first failure. Nothing after this point ran.")
            return 1

    print(f"\n{'=' * 72}")
    print(f"All {len(stages)} stages completed in {(time.time() - t0) / 60:.1f} min")
    if os.path.exists(final):
        print(
            f"({os.path.getsize(final) / 1e6:.1f} MB)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
