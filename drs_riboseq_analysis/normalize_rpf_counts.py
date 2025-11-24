#!/usr/bin/env python3
"""
normalise_rpf_counts.py
-----------------------
Take the counts-with-zeros CSV from simple_rpf_counter.py
+ the same GTF                         → length column
Return: 
  * RPKM   (reads / kb / million) 
  * TPM    (transcripts per million)

Usage
-----
python normalise_rpf_counts.py \
       B48_4_rpf_counts_with_zeros.csv \
       gpcr_minigenome_v3.gtf \
       -o B48_4_rpf_tpm.csv
"""

import argparse, gzip, csv
import pandas as pd
from collections import defaultdict

# ------------------------------------------------------------------ helpers
def load_lengths(gtf_path):
    """Return dict {transcript : cds_length_nt}"""
    lens = {}
    open_fn = gzip.open if str(gtf_path).endswith((".gz", ".bgz")) else open
    with open_fn(gtf_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            chrom, _, feat, s, e, *_rest, attr = line.rstrip().split("\t")
            if feat != "CDS":
                continue
            start, end = int(s), int(e)
            tid = [f.split('"')[1] for f in attr.split(";") 
                   if f.strip().startswith("transcript_id")][0]
            lens[tid] = end - start + 1
    return lens

# ------------------------------------------------------------------ main
def main(count_csv, gtf, out_csv):
    df = pd.read_csv(count_csv)
    # collapse over length column to total per transcript
    tot = df.groupby("transcript", as_index=False)["nReads"].sum()

    lens = load_lengths(gtf)
    tot["length_nt"] = tot["transcript"].map(lens)
    tot.dropna(inplace=True)

    mism = tot[tot["transcript"].map(lens).isna()]
    print(f"#transcripts in counts  : {tot.shape[0]}")
    print(f"#found matching length  : {(~tot['transcript'].map(lens).isna()).sum()}")
    print("Example of missing IDs :", mism["transcript"].head().tolist())



    tot["length_kb"] = tot["length_nt"] / 1_000
    libsize_m = tot["nReads"].sum() / 1_000_000           # total mapped (M)

    tot["RPK"]  = tot["nReads"] / tot["length_kb"]
    tot["RPKM"] = tot["RPK"] / libsize_m

    # TPM: scale RPKs so they sum to 1 000 000
    per_m = tot["RPK"].sum() / 1_000_000
    tot["TPM"] = tot["RPK"] / per_m

    out_cols = ["transcript", "nReads", "length_nt", "RPKM", "TPM"]
    tot[out_cols].to_csv(out_csv, index=False)
    print(f"✓ wrote {out_csv}  ({len(tot)} transcripts)")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("counts_csv")
    p.add_argument("gtf")
    p.add_argument("-o", "--out", default="rpf_tpm.csv")
    args = p.parse_args()
    main(args.counts_csv, args.gtf, args.out)
