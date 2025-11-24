#!/usr/bin/env python3
"""
merge_ribo_rna.py  (v1.1 – adds translational efficiency)

…<previous docstring text>…
"""

import argparse, pandas as pd, numpy as np, sys
from pathlib import Path
pd.options.mode.copy_on_write = True

def add_gene_id(df, col="transcript"):
    # old →  df["gene_id"] = df[col].str.split("_", 1).str[0]
    df["gene_id"] = df[col].astype(str).str.split("_", n=1).str[0]
    return df

def main(rpf_csv, dis_csv, xlsx, sheet, out_csv, log2):
    # ── load tables ───────────────────────────────────────────────────
    rpf = add_gene_id(pd.read_csv(rpf_csv))\
          .rename(columns={"nReads": "mono_nReads", "TPM": "mono_TPM"})

    dis = add_gene_id(pd.read_csv(dis_csv))\
          .rename(columns={"nReads_diso":"diso_nReads",
                           "TPM_diso":  "diso_TPM"})

    rna = pd.read_excel(xlsx, sheet_name=sheet,
                        usecols=["gene_id", "total_coverage"])\
           .rename(columns={"total_coverage":"rna_total_coverage"})

    # ── merge ─────────────────────────────────────────────────────────
    merged = (rpf
              .merge(dis[["transcript","diso_nReads","diso_TPM","disome_ratio"]],
                     on="transcript", how="left")
              .merge(rna, on="gene_id", how="left"))

    # ── translational efficiency (simple proxy) ───────────────────────
    merged["TE"] = merged["mono_TPM"] / merged["rna_total_coverage"]
    if log2:
        merged["log2TE"] = np.log2(merged["TE"].replace(0, np.nan))

    # ── save ──────────────────────────────────────────────────────────
    merged.to_csv(out_csv, index=False)
    print(f"✓ wrote {out_csv}  ({len(merged)} rows)")

# ── CLI ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Merge Ribo-seq + RNA-seq and compute translational efficiency")
    ap.add_argument("--rpf", required=True, help="monosome TPM CSV")
    ap.add_argument("--dis", required=True, help="disome ratio CSV")
    ap.add_argument("--rna", required=True, help="RNA-seq Excel workbook")
    ap.add_argument("--sheet", default="rnaseq", help="sheet name [rnaseq]")
    ap.add_argument("--out", required=True, help="output CSV")
    ap.add_argument("--log2", action="store_true",
                    help="add log2TE column (log2 of TE)")
    args = ap.parse_args()

    for f in (args.rpf, args.dis, args.rna):
        if not Path(f).exists():
            sys.exit(f"✗ file not found: {f}")

    main(args.rpf, args.dis, args.rna, args.sheet, args.out, args.log2)
