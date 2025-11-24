#!/usr/bin/env python3
"""
disome_mono_ratio.py
--------------------
Merge monosome- and disome-length RPF count tables and compute
  • D_over_M   (disome / monosome reads)
  • keep/discard flags based on minimum depth

Inputs  : 1) monosome CSV   from simple_rpf_counter_v2  (28–38 nt)
          2) disome   CSV   from simple_rpf_counter_v2  (50–70 nt, say)
          3) optional thresholds
Output  : <prefix>_mono_di_ratio.csv      (new file)

Example
-------
# step 1 – make a disome count table
python simple_rpf_counter_v2.py \
       B48_4_clean_dedup.sorted.bam \
       clean_gpcr_minigenome.gtf     \
       --min_len 50 --max_len 70 \
       --keep-zeros \
       -o B48_4_clean_rpf_counts_disome.csv

# step 2 – compute the D/M ratio
python disome_mono_ratio.py \
       B48_4_clean_rpf_counts.csv \
       B48_4_clean_rpf_counts_disome.csv \
       --min_mono 100 \
       --min_di   15 \
       --out B48_4_clean_RPF_D_M_ratio.csv
"""

import argparse, pandas as pd, numpy as np

def load_totals(path):
    """collapse length bins → total reads per transcript"""
    df = pd.read_csv(path)
    return (df.groupby("transcript", as_index=False)
              ["nReads"].sum()
              .rename(columns={"nReads": "reads"}))

def main(mono_csv, di_csv, min_mono, min_di, out_csv):
    mono = load_totals(mono_csv).rename(columns={"reads": "mono_reads"})
    di   = load_totals(di_csv)  .rename(columns={"reads": "di_reads"})

    merged = mono.merge(di, on="transcript", how="outer").fillna(0)

    # D / M ratio (NaN if mono_reads == 0)
    merged["D_over_M"] = merged["di_reads"] / merged["mono_reads"].replace(0, np.nan)

    # QC flag
    merged["include"] = (
        (merged["mono_reads"] >= min_mono) |
        (merged["di_reads"]  >= min_di)
    )

    merged.to_csv(out_csv, index=False)
    kept = merged["include"].sum()
    total = len(merged)
    print(f"✓ wrote {out_csv}   ({kept}/{total} transcripts pass depth filter)")

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
    main(args.mono_counts, args.di_counts, args.min_mono, args.min_di, args.out)
