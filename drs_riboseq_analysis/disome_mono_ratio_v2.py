#!/usr/bin/env python3
"""
disome_mono_ratio.py  -- v2  (2025-06-24)

1.  Collapse the monosome- and disome-length count tables
      (made with simple_rpf_counter_v2.py)
2.  Compute
      • mono_nReads,  diso_nReads
      • mono_TPM,     diso_TPM
      • disome_ratio  = diso_TPM / mono_TPM     (NaN if mono_TPM==0)
3.  Flag low-depth genes ( <min_mono or <min_di )
4.  Write a tidy CSV whose column names are **stable**
      --> ready for downstream merge_ribo_rna.py

Example
-------
# ❶  Make the disome count table (50–70 nt)
python simple_rpf_counter_v2.py  B48_4_clean_dedup.sorted.bam \
        clean_gpcr_minigenome.gtf  --min_len 50 --max_len 70 \
        --keep-zeros -o B48_4_clean_rpf_counts_disome.csv

# ❷  Compute the ratios
python disome_mono_ratio.py \
        B48_4_clean_rpf_counts.csv \
        B48_4_clean_rpf_counts_disome.csv \
        --min_mono 100 --min_di 15 \
        -o B48_4_clean_RPF_D_M_ratio.csv
"""

import argparse, pandas as pd, numpy as np

# ---------------------------------------------------------------- helpers
def collapse_totals(path, label):
    """
    Read <path> → dataframe with columns:
      transcript | <label>_nReads | <label>_TPM
    TPM is computed *within that library*.
    """
    df = pd.read_csv(path)

    # total reads per transcript
    tot = (df.groupby("transcript", as_index=False)["nReads"]
             .sum()
             .rename(columns={"nReads": f"{label}_nReads"}))

    # library-size normalisation
    lib_m = tot[f"{label}_nReads"].sum() / 1_000_000
    tot[f"{label}_TPM"] = tot[f"{label}_nReads"] / lib_m

    return tot

# ---------------------------------------------------------------- main
def main(mono_csv, di_csv, min_mono, min_di, out_csv):
    mono = collapse_totals(mono_csv, "mono")
    di   = collapse_totals(di_csv,   "diso")

    df = (mono.merge(di, on="transcript", how="outer")
              .fillna(0))

    # ratio (TPM-based; NaN if mono_TPM=0 to avoid Inf)
    df["disome_ratio"] = df["diso_TPM"] / df["mono_TPM"].replace(0, np.nan)

    # QC flag
    df["include"] = (
        (df["mono_nReads"] >= min_mono) |
        (df["diso_nReads"] >= min_di)
    )

    out_cols = [
        "transcript",
        "mono_nReads", "mono_TPM",
        "diso_nReads", "diso_TPM",
        "disome_ratio",
        "include"
    ]
    df[out_cols].to_csv(out_csv, index=False)

    kept = df["include"].sum()
    print(f"✓ wrote {out_csv}   ({kept}/{len(df)} transcripts pass depth filter)")

# ---------------------------------------------------------------- CLI
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compute disome/monosome ratios")
    p.add_argument("mono_counts", help="monosome count CSV (28-38 nt)")
    p.add_argument("di_counts",   help="disome   count CSV (50-70 nt)")
    p.add_argument("--min_mono", type=int, default=100,
                   help="min monosome reads to keep (default 100)")
    p.add_argument("--min_di",   type=int, default=15,
                   help="min disome reads to keep (default 15)")
    p.add_argument("-o","--out", default="mono_di_ratio.csv")
    args = p.parse_args()

    main(args.mono_counts, args.di_counts,
         args.min_mono, args.min_di, args.out)
