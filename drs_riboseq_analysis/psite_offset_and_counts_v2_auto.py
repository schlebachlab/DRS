#!/usr/bin/env python3
"""
psite_offset_and_counts_v2.py
-----------------------------
1. Learn optimal P-site offsets per read-length from high-coverage contigs
2. Fill in missing lengths      ⟶   linear interpolation / fallback
3. Apply offsets to **all** reads and output per-transcript P-site counts

Input  : dedup-sorted BAM  + single-exon GTF
Output : <prefix>_offsets_by_length.csv
         <prefix>_psite_counts.csv
"""

import argparse, csv, gzip
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from scipy.stats import linregress          # conda install scipy
import pysam                                # conda install pysam
import pandas as pd                         # conda install pandas
from intervaltree import IntervalTree, Interval


# ---------------------------------------------------------------- helpers
def load_cds(gtf):
    cds_coords = {}
    trees      = defaultdict(IntervalTree)
    opener = gzip.open if str(gtf).endswith((".gz",".bgz")) else open
    with opener(gtf,"rt") as fh:
        for ln in fh:
            if ln.startswith("#"): continue
            chrom, _, feat, s, e, *_ , attr = ln.rstrip().split('\t')
            if feat.strip() != "CDS":         # ← strip() !
                continue
            s,e = int(s)-1, int(e)
            cds_coords[chrom] = (s,e)
            trees[chrom].add(Interval(s,e))
    return cds_coords, trees


def best_offset(reads, cds_start, cds_end, offsets):
    """return (offset, frac_in_frame0, nReads) or (None, 0, 0)"""
    best = (None, 0.0, 0)                    # (off, frac, n)
    for off in offsets:
        f0 = tot = 0
        for r in reads:
            psite = r.reference_start + off
            if cds_start <= psite < cds_end:
                tot += 1
                if (psite - cds_start) % 3 == 0:
                    f0 += 1
        if tot and (f0/tot) > best[1]:
            best = (off, f0/tot, tot)
    return best


def collect_reads_by_len(bam, tree):
    """dict {length : [AlignedSegment,…]}"""
    by_len = defaultdict(list)
    for r in bam.fetch(until_eof=True):
        if r.is_unmapped or r.is_secondary or r.is_supplementary: continue
        if r.reference_name not in tree: continue
        by_len[r.query_length].append(r)
    return by_len


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bam", help="dedup & coordinate-sorted BAM")
    ap.add_argument("gtf", help="single-exon GTF")
    ap.add_argument("--min_reads", type=int, default=500,
                    help="min reads per contig to learn offsets [500]")
    ap.add_argument("--min_len",   type=int, default=28)
    ap.add_argument("--max_len",   type=int, default=38)
    ap.add_argument("--fallback",  type=int,
                    help="constant offset to use for missing lengths")
    ap.add_argument("--out_prefix", default="sample")
    args = ap.parse_args()

    cds_coords, trees = load_cds(args.gtf)
    print(f"Loaded {len(cds_coords)} CDS entries from GTF")

    bam = pysam.AlignmentFile(args.bam, "rb")

    # ---------- 1) pool reads by contig & length --------------------------
    contig_reads = defaultdict(list)
    for r in bam.fetch(until_eof=True):
        if r.is_unmapped or r.is_secondary or r.is_supplementary: continue
        contig_reads[r.reference_name].append(r)

    # pick training contigs
    train = {ctg: rs for ctg, rs in contig_reads.items()
             if len(rs) >= args.min_reads and ctg in cds_coords}
    n_train_reads = sum(len(v) for v in train.values())
    print(f"Using {len(train)} contigs to learn offsets "
          f"({n_train_reads:,} reads)")

    # ---------- 2) learn offsets length-by-length -------------------------
    off_dict = {}            # length ➜ (offset, in-frame frac, nReads)
    for length in range(args.min_len, args.max_len+1):
        all_reads = [r for rs in train.values() for r in rs
                      if r.query_length == length]
        if not all_reads: continue
        ctg = all_reads[0].reference_name
        cds_s, cds_e = cds_coords[ctg]       # same for all (single-exon)
        best_off, frac0, n = best_offset(all_reads, cds_s, cds_e,
                                         offsets=range(6, 20))
        if best_off is not None:
            off_dict[length] = (best_off, frac0, n)

    # ---------- 3) fill missing lengths ----------------------------------
    if args.fallback is None:
        known = sorted(off_dict.items())         # [(len, (off, frac, n)), …]
        if len(known) >= 2:
            xs  = np.array([k for k,_ in known])
            ys  = np.array([v[0] for _,v in known])
            slope, intercept, *_ = linregress(xs, ys)
            def infer(L): return int(round(slope*L + intercept))
            for L in range(args.min_len, args.max_len+1):
                if L not in off_dict:
                    off_dict[L] = (infer(L), np.nan, 0)
            print(f"   → filled gaps by linear fit (slope ≈ {slope:.2f})")
        else:
            raise RuntimeError("Not enough lengths to fit fallback. "
                               "Use --fallback N instead.")
    else:
        for L in range(args.min_len, args.max_len+1):
            off_dict.setdefault(L, (args.fallback, np.nan, 0))
        print(f"   → constant fallback offset {args.fallback} applied")

    # save offsets table
    off_rows = [(L, off, frac, n)
                for L,(off,frac,n) in sorted(off_dict.items())]
    pd.DataFrame(off_rows, columns=["length","offset","in_frame_frac","nReads"]
                ).to_csv(f"{args.out_prefix}_offsets_by_length.csv", index=False)
    print(f"✓ offsets → {args.out_prefix}_offsets_by_length.csv")

    # ---------- 4) apply offsets to *all* reads ---------------------------
    psite_counts = defaultdict(int)          # transcript ➜ n P-sites
    skipped = 0
    bam.reset()
    for r in bam.fetch(until_eof=True):
        L = r.query_length
        if L < args.min_len or L > args.max_len: continue
        off = off_dict[L][0]
        psite = r.reference_start + off
        ctg = r.reference_name
        if ctg not in cds_coords:
            skipped += 1
            continue
        cds_s, cds_e = cds_coords[ctg]
        if cds_s <= psite < cds_e:
            psite_counts[ctg] += 1
    bam.close()

    out_psite = pd.DataFrame(sorted(psite_counts.items()),
                             columns=["transcript","nPsites"])
    out_psite.to_csv(f"{args.out_prefix}_psite_counts.csv", index=False)
    print(f"✓ p-site counts → {args.out_prefix}_psite_counts.csv  "
          f"({len(out_psite)} transcripts)")
    if skipped:
        print(f"  (skipped {skipped:,} reads with lengths lacking an offset)")

if __name__ == "__main__":
    main()
