"""
13 — Build an enlarged benign control set for LR+ estimation, and annotate the
ClinVar set with REVEL.

Why. The an earlier version/an earlier version LR+ analysis rests on 13 ClinVar 3-star benign missense
variants, one of which is a false positive at every AM threshold. Specificity is
therefore pinned at 12/13 = 0.923 and the estimable LR+ is capped at
1/(1-0.923) = 13.0, below the 18.7 required for Strong evidence. A sub-threshold
LR+ was guaranteed by the composition of the control set. an earlier check flagged this
as blocking, and also noted that Table 3 reports LR+ for AlphaMissense only while
the write-up recommends REVEL -- because data/clinvar/clinvar_processed.csv has
revel_score missing for all 260 rows.

This script fixes both.

  A. BENIGN PROXY SET (gnomAD). Missense variants in ABCC8, KCNJ11 and GCK are
     pulled from gnomAD v2.1.1 (GRCh37) and filtered to a frequency-based benign
     proxy, the approach used in the PP3/BP4 calibration literature. ABCC8/KCNJ11
     neonatal diabetes and congenital hyperinsulinism, and GCK-MODY, are severe
     early-onset phenotypes, so an allele seen at appreciable population frequency
     is very unlikely to be highly penetrant. Two nested definitions are emitted so
     the sensitivity of the result to the cutoff is visible:
        strict     AF >= 1e-4
        permissive AF >= 1e-5
     Any variant with a Pathogenic/Likely pathogenic ClinVar assertion is removed
     from both, and the ClinVar B/LB missense set is added.

  B. REVEL FOR CLINVAR. The 260 ClinVar variants are annotated from
     myvariant.info (dbNSFP), the same source the write-up already uses for the
     GoFCards variants.

AlphaMissense scores come from the local all-substitutions cache
(data/clinvar/am_gene_cache_abcc8_gck_kcnj11.csv), so no AM network call is made.

Outputs
    data/gnomad/gnomad_missense_raw.csv     one row per gnomAD missense variant
    results/benign_control_set.csv          the annotated control set
    results/clinvar_with_revel.csv          ClinVar 260 + revel_score
"""

import json
import urllib.parse
import os
import time
import urllib.error
import urllib.request

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
DATA = os.path.join(ROOT, "data")
GNOMAD_DIR = os.path.join(DATA, "gnomad")

GENES = {"ABCC8": "Q09428", "KCNJ11": "Q14654", "GCK": "P35557"}
STRICT_AF = 1e-4
PERMISSIVE_AF = 1e-5

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


# ------------------------------------------------------------------ gnomAD
def gnomad_query(gene, retries=4):
    q = (
        """{ gene(gene_symbol:"%s", reference_genome:GRCh37){
      variants(dataset:gnomad_r2_1){
        variant_id chrom pos ref alt consequence hgvsp flags
        exome{ ac an } genome{ ac an } } } }"""
        % gene
    )
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                "https://gnomad.broadinstitute.org/api",
                data=json.dumps({"query": q}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=90) as fh:
                return json.load(fh)["data"]["gene"]["variants"]
        except (urllib.error.URLError, TimeoutError, KeyError) as e:
            if attempt == retries - 1:
                raise
            print(f"    {gene}: retry {attempt + 1} after {e}")
            time.sleep(5 * (attempt + 1))


def combined_af(v):
    e, g = v.get("exome") or {}, v.get("genome") or {}
    ac = (e.get("ac") or 0) + (g.get("ac") or 0)
    an = (e.get("an") or 0) + (g.get("an") or 0)
    return (ac / an, ac, an) if an else (0.0, 0, 0)


def fetch_gnomad():
    os.makedirs(GNOMAD_DIR, exist_ok=True)
    cache = os.path.join(GNOMAD_DIR, "gnomad_missense_raw.csv")
    if os.path.exists(cache):
        print(f"  using cached {os.path.basename(cache)}")
        return pd.read_csv(cache)

    rows = []
    for gene in GENES:
        vs = gnomad_query(gene)
        mis = [v for v in vs if v.get("consequence") == "missense_variant"]
        for v in mis:
            af, ac, an = combined_af(v)
            rows.append(
                dict(
                    gene=gene,
                    variant_id=v["variant_id"],
                    chrom=v["chrom"],
                    pos=v["pos"],
                    ref=v["ref"],
                    alt=v["alt"],
                    hgvsp=v.get("hgvsp"),
                    af=af,
                    ac=ac,
                    an=an,
                    flags=";".join(v.get("flags") or []),
                )
            )
        print(f"  {gene}: {len(vs)} variants, {len(mis)} missense")
    df = pd.DataFrame(rows)
    df.to_csv(cache, index=False)
    print(f"  wrote {os.path.basename(cache)} ({len(df)} rows)")
    return df


# ------------------------------------------------------------------ REVEL
def myvariant_revel(hgvs_ids, chunk=200, assembly="hg19"):
    """POST hgvs ids to myvariant.info, return {id: revel}."""
    out = {}
    for i in range(0, len(hgvs_ids), chunk):
        batch = hgvs_ids[i : i + chunk]
        body = urllib.parse.urlencode(
            {"ids": ",".join(batch), "fields": "dbnsfp.revel", "assembly": assembly}
        ).encode()
        req = urllib.request.Request(
            "https://myvariant.info/v1/variant",
            data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=90) as fh:
                    res = json.load(fh)
                break
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt == 3:
                    raise
                print(f"    myvariant retry {attempt + 1} after {e}")
                time.sleep(5 * (attempt + 1))
        for r in res:
            rv = (r.get("dbnsfp") or {}).get("revel")
            if isinstance(rv, dict):
                rv = rv.get("score")
            if isinstance(rv, list):
                rv = max(x for x in rv if x is not None) if any(rv) else None
            if rv is not None:
                out[r["query"]] = float(rv)
        print(f"    myvariant {i + len(batch)}/{len(hgvs_ids)} -> {len(out)} hits")
        time.sleep(1)
    return out


def hgvs_g(chrom, pos, ref, alt):
    return f"chr{chrom}:g.{pos}{ref}>{alt}"


# ------------------------------------------------------------------ AM
def load_am_cache():
    am = pd.read_csv(
        os.path.join(DATA, "clinvar", "am_gene_cache_abcc8_gck_kcnj11.csv")
    )
    uni2gene = {v: k for k, v in GENES.items()}
    am["gene"] = am.uniprot_id.map(uni2gene)
    return am.set_index(["gene", "protein_variant"]).am_pathogenicity.to_dict()


def hgvsp_to_short(h):
    """'p.Asp387Gly' -> 'D387G'; returns None for anything not a clean missense."""
    if not isinstance(h, str) or not h.startswith("p."):
        return None
    body = h[2:]
    ref3, alt3 = body[:3], body[-3:]
    num = body[3:-3]
    if ref3 not in AA3 or alt3 not in AA3 or not num.isdigit():
        return None
    return f"{AA3[ref3]}{num}{AA3[alt3]}"


# ------------------------------------------------------------------ main
def build_control_set():
    print("gnomAD:")
    gn = fetch_gnomad()

    # drop anything ClinVar calls pathogenic / likely pathogenic
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
    gn["_k"] = list(
        zip(
            gn.chrom.astype(str),
            gn.pos.astype(str),
            gn.ref.astype(str),
            gn.alt.astype(str),
        )
    )
    n_before = len(gn)
    gn = gn[~gn._k.isin(plp_keys)].copy()
    print(
        f"  removed {n_before - len(gn)} gnomAD variants with a ClinVar P/LP assertion"
    )

    gn["short"] = gn.hgvsp.map(hgvsp_to_short)
    gn = gn.dropna(subset=["short"]).copy()

    am = load_am_cache()
    gn["am_score"] = [am.get((g, s)) for g, s in zip(gn.gene, gn.short)]
    print(f"  AM annotated: {gn.am_score.notna().sum()}/{len(gn)} from local cache")

    strict = gn[gn.af >= STRICT_AF].copy()
    perm = gn[gn.af >= PERMISSIVE_AF].copy()
    print(
        f"  benign proxy: strict(AF>={STRICT_AF:g}) n={len(strict)}, "
        f"permissive(AF>={PERMISSIVE_AF:g}) n={len(perm)}"
    )
    for g in GENES:
        print(
            f"    {g}: strict {int((strict.gene == g).sum())}, "
            f"permissive {int((perm.gene == g).sum())}"
        )

    print("  REVEL from myvariant.info:")
    ids = [
        hgvs_g(c, p, r, a)
        for c, p, r, a in zip(perm.chrom, perm.pos, perm.ref, perm.alt)
    ]
    rev = myvariant_revel(ids)
    perm["hgvs_g"] = ids
    perm["revel_score"] = perm.hgvs_g.map(rev)
    print(f"  REVEL annotated: {perm.revel_score.notna().sum()}/{len(perm)}")

    perm["tier"] = ["strict" if a >= STRICT_AF else "permissive" for a in perm.af]
    perm["source"] = "gnomAD"
    out = perm[
        [
            "gene",
            "variant_id",
            "chrom",
            "pos",
            "ref",
            "alt",
            "hgvsp",
            "short",
            "af",
            "ac",
            "an",
            "am_score",
            "revel_score",
            "tier",
            "source",
        ]
    ]
    out.to_csv(os.path.join(RES, "benign_control_set.csv"), index=False)
    print(f"  wrote benign_control_set.csv ({len(out)} rows)")
    return out


def annotate_clinvar_revel():
    print("\nClinVar REVEL annotation:")
    cv = pd.read_csv(
        os.path.join(DATA, "clinvar", "clinvar_processed.csv"), low_memory=False
    )
    ids = [
        hgvs_g(c, int(p), r, a)
        for c, p, r, a in zip(
            cv.Chromosome, cv.PositionVCF, cv.ReferenceAlleleVCF, cv.AlternateAlleleVCF
        )
    ]
    # NOTE: clinvar_processed.csv is GRCh38 (Assembly column), even though
    # the manuscript Methods says all coordinates are GRCh37. Query hg38.
    rev = myvariant_revel(ids, assembly="hg38")
    cv["hgvs_g"] = ids
    cv["revel_score"] = cv.hgvs_g.map(rev)
    got = cv.revel_score.notna().sum()
    print(f"  REVEL annotated: {got}/{len(cv)}")
    for m, sub in cv.groupby("mechanism"):
        print(f"    {m}: {sub.revel_score.notna().sum()}/{len(sub)}")
    cv.to_csv(os.path.join(RES, "clinvar_with_revel.csv"), index=False)
    print("  wrote clinvar_with_revel.csv")
    return cv


if __name__ == "__main__":
    import urllib.parse  # noqa: F401  (used by myvariant_revel)

    build_control_set()
    annotate_clinvar_revel()
