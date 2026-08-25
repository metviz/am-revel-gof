"""
15 — Predictor panel: is the LoF/GoF gap an AlphaMissense property, or a property
of the whole class of missense predictors?

an earlier check 3 raised the decisive objection to the mechanistic story: the write-up
attributes AM's reduced GoF detection to AM's training signal, but tests only two
predictors, and the paper's own mechanistic account (predictors track evolutionary
constraint; GoF sites are less constrained in the way predictors measure) predicts
the same deficit in ANY conservation-driven tool. If every predictor shows it, the
correct conclusion is "lower the GoF threshold", not "switch tools".

This script tests that directly. Ten predictors are pulled from dbNSFP via
myvariant.info and grouped by inductive bias:

    conservation / alignment only     PROVEAN, SIFT4G, LRT
      (no human disease labels, no structure -- the pure "sequence channel")
    population deep learning          PrimateAI
      (trained on primate population variation, structure-free)
    regional missense constraint      MPC
    supervised ensembles              REVEL, VEST4, MetaRNN, ClinPred
      (trained on curated human pathogenic/benign labels)
    structure + population frequency  AlphaMissense

All comparisons use dbNSFP RANKSCORES, which are oriented so that higher is more
damaging for every tool regardless of the native score direction. That makes a
single operating point comparable across predictors: for each predictor we take
the rankscore threshold giving a fixed LoF sensitivity, then read off GoF
sensitivity at that same threshold. This is the matched-operating-point design
already adopted for the AM-vs-REVEL comparison, extended to the panel.

Outputs
    data/dbnsfp/dbnsfp_panel_raw.csv    one row per dbNSFP record for the 3 genes
    results/tier4_panel_scores.csv      cohort variants x predictor rankscores
    results/tier4_panel_gaps.csv        LoF-GoF gap per predictor at matched sensitivity
"""

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
DATA = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(DATA, "dbnsfp")

GENES = ["ABCC8", "KCNJ11", "GCK"]

# field path -> (column name, inductive-bias group)
PREDICTORS = {
    "provean.converted_rankscore": ("PROVEAN", "conservation only"),
    "sift4g.converted_rankscore": ("SIFT4G", "conservation only"),
    "lrt.converted_rankscore": ("LRT", "conservation only"),
    "primateai.rankscore": ("PrimateAI", "population deep learning"),
    "mpc.rankscore": ("MPC", "regional constraint"),
    "revel.rankscore": ("REVEL", "supervised ensemble"),
    "vest4.rankscore": ("VEST4", "supervised ensemble"),
    "metarnn.rankscore": ("MetaRNN", "supervised ensemble"),
    "clinpred.rankscore": ("ClinPred", "supervised ensemble"),
    "alphamissense.rankscore": ("AlphaMissense", "structure + frequency"),
}
FIELDS = ",".join(
    ["dbnsfp.genename", "dbnsfp.hgvsp"] + [f"dbnsfp.{p}" for p in PREDICTORS]
)

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


def _get(url, tries=5):
    for a in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=90) as fh:
                return json.load(fh)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            if a == tries - 1:
                raise
            print(f"    retry {a + 1} after {e}")
            time.sleep(4 * (a + 1))


def first_num(v):
    """dbNSFP repeats values per isoform; take the first finite number."""
    if v is None:
        return np.nan
    if isinstance(v, list):
        for x in v:
            n = first_num(x)
            if not np.isnan(n):
                return n
        return np.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def short_forms(hgvsp):
    """All 1-letter forms of a record's protein changes, e.g. {'M288K','M375K'}."""
    out = set()
    for h in hgvsp if isinstance(hgvsp, list) else [hgvsp]:
        if not isinstance(h, str) or not h.startswith("p."):
            continue
        b = h[2:]
        m = re.fullmatch(r"([A-Z])(\d+)([A-Z])", b)
        if m:
            out.add(b)
            continue
        m = re.fullmatch(r"([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})", b)
        if m and m.group(1) in AA3 and m.group(3) in AA3:
            out.add(f"{AA3[m.group(1)]}{m.group(2)}{AA3[m.group(3)]}")
    return out


def scroll_gene(gene):
    """Pull every dbNSFP missense record for one gene."""
    base = "https://myvariant.info/v1/query?"
    q = {
        "q": f"dbnsfp.genename:{gene} AND _exists_:dbnsfp.revel",
        "fields": FIELDS,
        "size": 1000,
        "fetch_all": "true",
    }
    r = _get(base + urllib.parse.urlencode(q))
    rows, total = [], r.get("total", 0)
    while True:
        for h in r.get("hits", []):
            d = h.get("dbnsfp") or {}
            rec = {
                "gene": gene,
                "_id": h["_id"],
                "short": "|".join(sorted(short_forms(d.get("hgvsp")))),
            }
            for path, (name, _) in PREDICTORS.items():
                node, key = path.split(".")
                rec[name] = first_num((d.get(node) or {}).get(key))
            rows.append(rec)
        sid = r.get("_scroll_id")
        if not sid or len(rows) >= total:
            break
        r = _get(base + urllib.parse.urlencode({"scroll_id": sid}))
        if not r.get("hits"):
            break
        time.sleep(0.3)
    print(f"  {gene}: {len(rows)}/{total} records")
    return rows


def fetch_panel():
    os.makedirs(OUT_DIR, exist_ok=True)
    cache = os.path.join(OUT_DIR, "dbnsfp_panel_raw.csv")
    if os.path.exists(cache):
        print(f"  using cached {os.path.basename(cache)}")
        return pd.read_csv(cache)
    rows = []
    for g in GENES:
        rows += scroll_gene(g)
    df = pd.DataFrame(rows)
    df.to_csv(cache, index=False)
    print(f"  wrote {os.path.basename(cache)} ({len(df)} rows)")
    return df


# ---------------------------------------------------------------- cohorts
def norm(s):
    """'p.(Ser3Cys)' / 'p.Ser3Cys' -> 'S3C'."""
    b = strip_p(s)
    m = re.fullmatch(r"([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})", b)
    if m and m.group(1) in AA3 and m.group(3) in AA3:
        return f"{AA3[m.group(1)]}{m.group(2)}{AA3[m.group(3)]}"
    return b if re.fullmatch(r"[A-Z]\d+[A-Z]", b) else None


def load_cohorts():
    hop = pd.read_csv(os.path.join(RES, "all_variants_annotated.csv"))
    hop["short"] = hop.protein.map(norm)
    hop["dataset"] = "Hopkins"

    dfr = pd.read_csv(os.path.join(DATA, "de_franco", "de_franco_processed.csv"))
    dfr["short"] = dfr.hgvs_p.map(norm)
    hk = set(zip(hop.gene, hop.short))
    dfr = dfr[~pd.Series(list(zip(dfr.gene, dfr.short))).isin(hk).values].copy()
    dfr["dataset"] = "De Franco (novel)"

    cols = ["dataset", "gene", "short", "mechanism"]
    return pd.concat([hop[cols], dfr[cols]], ignore_index=True).dropna(subset=["short"])


def join_panel(cohort, panel):
    """Explode the panel's multi-isoform 'short' field, then join on (gene, short)."""
    p = panel.copy()
    p["short"] = p.short.fillna("").str.split("|")
    p = p.explode("short")
    p = p[p.short != ""]
    names = [n for n, _ in PREDICTORS.values()]
    p = p.groupby(["gene", "short"], as_index=False)[names].max()
    return cohort.merge(p, on=["gene", "short"], how="left")


# ---------------------------------------------------------------- analysis
# A rankscore with few distinct values cannot support a quantile-matched
# operating point: the threshold lands on a large tie block and the resulting
# "gap" reflects where the ties fall rather than the predictor's behaviour.
# Predictors below this resolution are still computed, but are excluded from the
# summary with the reason recorded. LRT fails this test (42 distinct values, 70%
# of variants sharing one of them).
# Screening on distinct-value COUNT was the wrong diagnostic: it asks how many
# rankscores exist, not how much probability mass sits on the tie block the
# quantile lands in. SIFT4G has 149 distinct values and still overshoots its
# target LoF sensitivity by up to 6.4 pp, inflating its gap by the same mechanism
# LRT was excluded for. The screen is therefore on the ACHIEVED LoF sensitivity:
# if the threshold cannot be placed within tolerance of the target, the matched
# comparison is not being made and the row is not summarised.
MIN_DISTINCT = 100
MAX_SENS_OVERSHOOT = 2.0  # percentage points
RNG = np.random.default_rng(20260820)
N_BOOT = 2000


def matched_gap(df, col, target_lof_sens, boot=True):
    """Threshold giving ~target LoF sensitivity, then the LoF-GoF gap there."""
    d = df.dropna(subset=[col])
    lof = d[d.mechanism == "LoF"][col].values
    gof = d[d.mechanism == "GoF"][col].values
    if len(lof) < 10 or len(gof) < 10:
        return None
    ndist = int(len(np.unique(np.concatenate([lof, gof]))))

    def one(lv, gv):
        cut = np.quantile(lv, 1 - target_lof_sens)
        return (lv >= cut).mean(), (gv >= cut).mean(), cut

    ls, gs, cut = one(lof, gof)
    lo = hi = float("nan")
    if boot:
        b = np.empty(N_BOOT)
        for i in range(N_BOOT):
            a, c, _ = one(RNG.choice(lof, len(lof), replace=True),
                          RNG.choice(gof, len(gof), replace=True))
            b[i] = (a - c) * 100
        lo, hi = np.percentile(b, [2.5, 97.5])
    return dict(
        threshold=cut,
        lof_sens=ls * 100,
        gof_sens=gs * 100,
        gap=(ls - gs) * 100,
        ci_lo=lo,
        ci_hi=hi,
        n_lof=len(lof),
        n_gof=len(gof),
        n_distinct=ndist,
        sens_overshoot=ls * 100 - target_lof_sens * 100,
        usable=(ndist >= MIN_DISTINCT
                and abs(ls * 100 - target_lof_sens * 100) <= MAX_SENS_OVERSHOOT),
    )


if __name__ == "__main__":
    print("dbNSFP panel:")
    panel = fetch_panel()
    cohort = load_cohorts()
    merged = join_panel(cohort, panel)
    names = [n for n, _ in PREDICTORS.values()]
    cov = merged[names].notna().mean().sort_values(ascending=False)
    print(f"\ncohort variants: {len(merged)}  (coverage per predictor)")
    for n, c in cov.items():
        print(f"    {n:15s} {c * 100:5.1f}%")
    merged.to_csv(os.path.join(RES, "tier4_panel_scores.csv"), index=False)

    rows = []
    for target in (0.60, 0.50, 0.40):
        for ds in ("Hopkins", "De Franco (novel)", "ALL"):
            sub = merged if ds == "ALL" else merged[merged.dataset == ds]
            for name, group in PREDICTORS.values():
                r = matched_gap(sub, name, target)
                if r:
                    rows.append(
                        dict(
                            dataset=ds,
                            predictor=name,
                            group=group,
                            target_lof_sens=target * 100,
                            **r,
                        )
                    )
    gaps = pd.DataFrame(rows)
    gaps.to_csv(os.path.join(RES, "tier4_panel_gaps.csv"), index=False)

    print("\nLoF-GoF gap at matched LoF sensitivity (pooled cohorts)")
    for target in (0.60, 0.50, 0.40):
        s = gaps[(gaps.dataset == "ALL") & (gaps.target_lof_sens == target * 100)]
        s = s.sort_values("gap", ascending=False)
        print(f"\n  LoF sensitivity fixed at {target * 100:.0f}%")
        for _, r in s.iterrows():
            if r.usable:
                note = ""
            elif r.n_distinct < MIN_DISTINCT:
                note = f"   [excluded: {int(r.n_distinct)} distinct rankscores]"
            else:
                note = (f"   [excluded: LoF sensitivity overshoots target by "
                        f"{r.sens_overshoot:+.1f} pp]")
            print(
                f"    {r.predictor:15s} {r.group:26s} gap {r.gap:5.1f} pp "
                f"[{r.ci_lo:5.1f}, {r.ci_hi:5.1f}]  (GoF {r.gof_sens:4.1f}%){note}"
            )

    print("\nby inductive-bias group, averaged over the three operating points:")
    pooled = gaps[gaps.dataset == "ALL"]
    ok = pooled[pooled.usable]
    drop = sorted(set(pooled[~pooled.usable].predictor))
    if drop:
        reasons = []
        for d in drop:
            rows = pooled[(pooled.predictor == d) & ~pooled.usable]
            why = ("rankscore resolution" if rows.n_distinct.min() < MIN_DISTINCT
                   else "LoF sensitivity overshoot")
            reasons.append(f"{d} ({why})")
        print(f"  (excluded from the group means: {', '.join(reasons)})")
    summary_order = list(ok.groupby("predictor").gap.mean()
                         .sort_values(ascending=False).index)
    g = ok.groupby("group").gap.agg(["mean", "min", "max"])
    for grp, r in g.sort_values("mean", ascending=False).iterrows():
        print(
            f"    {grp:26s} mean {r['mean']:5.1f} pp  "
            f"(range {r['min']:5.1f} to {r['max']:5.1f})"
        )
    print("\nper dataset (the pooled figure hides real disagreement):")
    hdr = f"    {'predictor':15s} " + " ".join(f"{d:>20s}" for d in
                                               ("Hopkins", "De Franco (novel)", "ALL"))
    print(hdr)
    for pred in summary_order:
        cells = []
        for ds in ("Hopkins", "De Franco (novel)", "ALL"):
            sub = gaps[(gaps.predictor == pred) & (gaps.dataset == ds) & gaps.usable]
            cells.append(f"{sub.gap.mean():>20.1f}" if len(sub) else f"{'n/a':>20s}")
        print(f"    {pred:15s} " + " ".join(cells))

    print("\nwrote results/tier4_panel_{scores,gaps}.csv")
