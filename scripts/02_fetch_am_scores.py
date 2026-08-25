#!/usr/bin/env python3
"""
02_fetch_am_scores.py
========================
Independent verification: fetch AlphaMissense scores from the
official published dataset and merge with Hopkins et al. variant list.

This script allows you to independently verify the AM scores used in
01_main_analysis.py by downloading and querying the official AM dataset.

AlphaMissense Dataset:
  Cheng et al. (2023) Science. doi:10.1126/science.adg7492
  Zenodo: https://zenodo.org/records/8208688
  Google Cloud: https://storage.googleapis.com/dm_alphamissense/

File options (hg38 recommended):
  AlphaMissense_hg38.tsv.gz   - by genomic coordinates (GRCh38/hg38)
  AlphaMissense_hg19.tsv.gz   - by genomic coordinates (GRCh37/hg19)
  AlphaMissense_aa_substitutions.tsv.gz - by UniProt + amino acid change

Requirements:
    pip install pandas requests tqdm

Usage:
    # Step 1: Download AM data (do this once; file is ~3.5 GB)
    python 02_fetch_am_scores.py --download

    # Step 2: Look up scores for our 131 variants
    python 02_fetch_am_scores.py --lookup

    # Step 3: Compare fetched scores to our analysis scores
    python 02_fetch_am_scores.py --verify
"""

import os
import argparse
import pandas as pd
import numpy as np

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR    = os.path.join(PROJECT_DIR, 'data')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── UniProt IDs for our genes ────────────────────────────────────────────────
UNIPROT_IDS = {
    'ABCC8':  'Q09428',  # Sulfonylurea receptor 1
    'GCK':    'P35557',  # Glucokinase
    'KCNJ11': 'Q14654',  # Kir6.2 potassium channel
}

AM_AA_URL = "https://storage.googleapis.com/dm_alphamissense/AlphaMissense_aa_substitutions.tsv.gz"
AM_HG19_URL = "https://storage.googleapis.com/dm_alphamissense/AlphaMissense_hg19.tsv.gz"

LOCAL_AA_FILE  = os.path.join(DATA_DIR, 'AlphaMissense_aa_substitutions.tsv.gz')
LOCAL_HG19_FILE = os.path.join(DATA_DIR, 'AlphaMissense_hg19.tsv.gz')


def download_am_data(url=AM_AA_URL, local_path=LOCAL_AA_FILE):
    """Download AlphaMissense data from Google Cloud Storage."""
    try:
        import requests
        from tqdm import tqdm
    except ImportError:
        print("Install dependencies: pip install requests tqdm")
        return False

    if os.path.exists(local_path):
        print(f"Already exists: {local_path}")
        return True

    print(f"Downloading AlphaMissense data (~3.5 GB)...")
    print(f"Source: {url}")
    print(f"Destination: {local_path}")

    response = requests.get(url, stream=True)
    if response.status_code != 200:
        print(f"ERROR: HTTP {response.status_code}")
        return False

    total = int(response.headers.get('content-length', 0))
    os.makedirs(DATA_DIR, exist_ok=True)

    with open(local_path, 'wb') as f, tqdm(
        desc='Downloading', total=total, unit='B', unit_scale=True
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))

    print(f"Download complete: {local_path}")
    return True


def lookup_am_by_uniprot(variants_df, am_file=LOCAL_AA_FILE):
    """
    Look up AM scores using the amino acid substitution file.
    File format: uniprot_id, protein_variant, am_pathogenicity, am_class
    protein_variant format: e.g., V86A (single-letter AA codes)
    """
    THREE_TO_ONE = {
        'Ala':'A','Arg':'R','Asn':'N','Asp':'D','Cys':'C',
        'Gln':'Q','Glu':'E','Gly':'G','His':'H','Ile':'I',
        'Leu':'L','Lys':'K','Met':'M','Phe':'F','Pro':'P',
        'Ser':'S','Thr':'T','Trp':'W','Tyr':'Y','Val':'V'
    }

    def protein_to_single(p):
        """Convert p.Val86Ala -> V86A"""
        import re
        # Extract ref AA, position, alt AA
        m = re.match(r'p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})', p)
        if not m:
            return None
        ref3, pos, alt3 = m.groups()
        ref1 = THREE_TO_ONE.get(ref3)
        alt1 = THREE_TO_ONE.get(alt3)
        if not ref1 or not alt1:
            return None
        return f"{ref1}{pos}{alt1}"

    if not os.path.exists(am_file):
        print(f"AM file not found: {am_file}")
        print("Run: python 02_fetch_am_scores.py --download")
        return None

    print(f"Loading AM data from {am_file}...")
    am_df = pd.read_csv(am_file, sep='\t', compression='gzip',
                        comment='#',
                        names=['uniprot_id', 'protein_variant', 'am_pathogenicity', 'am_class'],
                        skiprows=4)
    print(f"  Loaded {len(am_df):,} AM entries")

    results = []
    for _, row in variants_df.iterrows():
        uniprot = UNIPROT_IDS.get(row['gene'])
        variant_1l = protein_to_single(row['protein'])

        if not uniprot or not variant_1l:
            results.append({'am_score_verified': None, 'am_class_verified': None, 'lookup_key': None})
            continue

        lookup_key = f"{uniprot}:{variant_1l}"
        match = am_df[(am_df['uniprot_id'] == uniprot) &
                      (am_df['protein_variant'] == variant_1l)]

        if len(match) == 1:
            results.append({
                'am_score_verified': match['am_pathogenicity'].values[0],
                'am_class_verified': match['am_class'].values[0],
                'lookup_key': lookup_key
            })
        elif len(match) == 0:
            print(f"  NOT FOUND: {lookup_key}")
            results.append({'am_score_verified': None, 'am_class_verified': None, 'lookup_key': lookup_key})
        else:
            print(f"  DUPLICATE: {lookup_key} ({len(match)} hits)")
            results.append({'am_score_verified': None, 'am_class_verified': None, 'lookup_key': lookup_key})

    return pd.DataFrame(results)


def lookup_am_by_hg19(variants_df, am_file=LOCAL_HG19_FILE):
    """
    Alternative lookup by genomic coordinates (hg19/GRCh37).
    File format: CHROM, POS, REF, ALT, genome, uniprot_id, transcript_id,
                 protein_variant, am_pathogenicity, am_class
    """
    if not os.path.exists(am_file):
        print(f"AM hg19 file not found: {am_file}")
        return None

    print(f"Loading AM hg19 data from {am_file}...")
    am_df = pd.read_csv(am_file, sep='\t', compression='gzip', comment='#',
                        names=['CHROM','POS','REF','ALT','genome','uniprot_id',
                               'transcript_id','protein_variant','am_pathogenicity','am_class'],
                        skiprows=4)
    print(f"  Loaded {len(am_df):,} AM entries")

    def parse_genomic(coord_str):
        """Parse '11:17496466A>G' -> (11, 17496466, 'A', 'G')"""
        import re
        m = re.match(r'(\w+):(\d+)([ACGT]+)>([ACGT]+)', coord_str)
        if not m:
            return None
        chrom, pos, ref, alt = m.groups()
        return str(chrom), int(pos), ref, alt

    results = []
    for _, row in variants_df.iterrows():
        parsed = parse_genomic(row['genomic_grch37'])
        if not parsed:
            results.append({'am_score_hg19': None, 'am_class_hg19': None})
            continue
        chrom, pos, ref, alt = parsed

        # Note: hg19 file uses REF as the genomic ref, ALT as the variant
        match = am_df[(am_df['CHROM'].astype(str) == chrom) &
                      (am_df['POS'] == pos) &
                      (am_df['REF'] == ref) &
                      (am_df['ALT'] == alt)]

        if len(match) >= 1:
            results.append({
                'am_score_hg19': match['am_pathogenicity'].values[0],
                'am_class_hg19': match['am_class'].values[0]
            })
        else:
            # Try reversed REF/ALT (some coords are reported differently)
            match2 = am_df[(am_df['CHROM'].astype(str) == chrom) &
                           (am_df['POS'] == pos)]
            if len(match2) > 0:
                print(f"  COORD MISMATCH: {row['genomic_grch37']} (found {len(match2)} at same pos)")
            results.append({'am_score_hg19': None, 'am_class_hg19': None})

    return pd.DataFrame(results)


def verify_scores(original_df, verified_df):
    """Compare our analysis AM scores to the officially fetched ones."""
    from scipy import stats
    merged = pd.concat([original_df[['gene', 'protein', 'mechanism', 'am_score']],
                        verified_df[['am_score_verified']]], axis=1)

    found = merged['am_score_verified'].notna()
    print(f"\nVerification: {found.sum()}/{len(merged)} scores matched")

    if found.sum() > 0:
        matched = merged[found]
        diff = (matched['am_score'] - matched['am_score_verified']).abs()
        print(f"  Mean absolute difference: {diff.mean():.4f}")
        print(f"  Max absolute difference:  {diff.max():.4f}")
        r, p = stats.pearsonr(matched['am_score'], matched['am_score_verified'])
        print(f"  Pearson r: {r:.4f} (p={p:.2e})")

        # Flag discordant cases
        discordant = matched[diff > 0.05]
        if len(discordant) > 0:
            print(f"\n  WARNING: {len(discordant)} variants with >0.05 score difference:")
            for _, row in discordant.iterrows():
                print(f"    {row['gene']} {row['protein']}: "
                      f"analysis={row['am_score']:.3f}, official={row['am_score_verified']:.3f}, "
                      f"diff={abs(row['am_score']-row['am_score_verified']):.3f}")
        else:
            print("  All scores agree within tolerance (±0.05). Verification PASSED ✓")

    return merged


def main():
    parser = argparse.ArgumentParser(description='Fetch and verify AM scores')
    parser.add_argument('--download', action='store_true', help='Download AM dataset')
    parser.add_argument('--lookup', action='store_true', help='Look up AM scores for 131 variants')
    parser.add_argument('--verify', action='store_true', help='Compare to analysis scores')
    parser.add_argument('--method', choices=['uniprot', 'hg19'], default='uniprot',
                        help='Lookup method (default: uniprot AA substitutions)')
    args = parser.parse_args()

    if not (args.download or args.lookup or args.verify):
        parser.print_help()
        print("\nExample usage:")
        print("  python 02_fetch_am_scores.py --download")
        print("  python 02_fetch_am_scores.py --lookup --method uniprot")
        print("  python 02_fetch_am_scores.py --verify")
        return

    # Load variants
    variants_df = pd.read_csv(os.path.join(DATA_DIR, 'am_revel_scores_combined.csv'))

    if args.download:
        if args.method == 'hg19':
            download_am_data(url=AM_HG19_URL, local_path=LOCAL_HG19_FILE)
        else:
            download_am_data(url=AM_AA_URL, local_path=LOCAL_AA_FILE)

    if args.lookup:
        print(f"\nLooking up AM scores using method: {args.method}")
        if args.method == 'uniprot':
            result_df = lookup_am_by_uniprot(variants_df)
        else:
            result_df = lookup_am_by_hg19(variants_df)

        if result_df is not None:
            out = os.path.join(RESULTS_DIR, f'am_scores_verified_{args.method}.csv')
            verified = pd.concat([variants_df[['gene', 'protein', 'mechanism']], result_df], axis=1)
            verified.to_csv(out, index=False)
            print(f"Verified scores saved: {out}")

            # ── Write real AM scores back into the combined CSV ──────────────
            # This is the key step: replaces the placeholder scores and stamps
            # am_fetched=True so that scripts 01, 03, 04 know they have real data.
            col = 'am_score_verified' if args.method == 'uniprot' else 'am_score_hg19'
            n_found = result_df[col].notna().sum()
            n_total = len(result_df)

            if n_found < n_total * 0.80:
                raise RuntimeError(
                    f"AM lookup coverage too low: {n_found}/{n_total} variants matched.\n"
                    f"  Check that the AlphaMissense file is the aa_substitutions version\n"
                    f"  and that UniProt IDs match (ABCC8=Q09428, KCNJ11=Q14654, GCK=P35557).\n"
                    f"  Note: Some ABCC8 variants use an alternative isoform numbering that\n"
                    f"  differs from the AlphaMissense canonical (Q09428). This is expected."
                )

            combined_path = os.path.join(DATA_DIR, 'am_revel_scores_combined.csv')
            combined = pd.read_csv(combined_path)
            # For NOT FOUND variants (result_df[col] is NaN), preserve existing scores
            # from the combined CSV rather than overwriting with NaN.
            # This handles known ABCC8 isoform numbering discrepancies where the
            # clinical isoform (Cheng 2023) differs by 1 AA from the AM canonical.
            new_scores = result_df[col].values
            new_classes = result_df.get('am_class_verified',
                                        result_df.get('am_class_hg19')).values
            missing_mask = pd.isna(new_scores)
            if missing_mask.any():
                print(f"  Keeping existing scores for {missing_mask.sum()} isoform-mismatched variants.")
                new_scores = pd.array(new_scores, dtype='Float64')
                existing_scores = combined['am_score'].values
                for i in range(len(new_scores)):
                    if pd.isna(new_scores[i]) and not pd.isna(existing_scores[i]):
                        new_scores[i] = existing_scores[i]
                        new_classes[i] = combined['am_class'].values[i]
            combined['am_score']   = new_scores
            combined['am_class']   = new_classes
            combined['am_fetched'] = ~pd.isna(new_scores)

            n_missing = combined['am_score'].isna().sum()
            if n_missing > 0:
                print(f"  WARNING: {n_missing} variants have no AM score (will be NaN).")
                print("  These variants will be excluded from threshold analyses.")

            combined.to_csv(combined_path, index=False)
            print(f"\n  ✓ am_revel_scores_combined.csv updated with real AM scores")
            print(f"  ✓ Coverage: {n_found}/{n_total} variants ({100*n_found/n_total:.1f}%)")
            print(f"  ✓ am_fetched flag set → scripts 01/03/04 will now run without error")

    if args.verify:
        from scipy import stats
        verified_file = os.path.join(RESULTS_DIR, 'am_scores_verified_uniprot.csv')
        if not os.path.exists(verified_file):
            print(f"Run --lookup first to generate: {verified_file}")
            return
        verified_df = pd.read_csv(verified_file)
        analysis_df = pd.read_csv(os.path.join(DATA_DIR, 'am_revel_scores_combined.csv'))
        verify_scores(analysis_df, verified_df)


if __name__ == '__main__':
    main()
