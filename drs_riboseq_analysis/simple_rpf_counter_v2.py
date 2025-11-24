#!/usr/bin/env python3
"""
Count raw RPFs per transcript (optionally report zeros).
"""

import argparse, gzip, csv
from collections import defaultdict, Counter
from intervaltree import IntervalTree, Interval
import pysam, pandas as pd

# -------------------------------------------------------------------------
def load_gtf(gtf_file):
    trees = defaultdict(IntervalTree)
    opener = gzip.open if str(gtf_file).endswith((".gz",".bgz")) else open
    with opener(gtf_file,"rt") as fh:
        for ln in fh:
            if ln.startswith("#"): continue
            chrom, _, feat, s, e, *_ , attr = ln.rstrip().split('\t')
            if feat.strip() != "exon":        # ← strip() !
                continue
            tid = None
            for f in attr.split(';'):
                f = f.strip()
                if f.startswith("transcript_id"):
                    tid = f.split('"')[1]
                    break
            if tid:
                trees[chrom].add(Interval(int(s)-1, int(e), tid))
    return trees

def assign_read(r, trees):
    if r.is_unmapped: return None
    if r.reference_name not in trees: return None
    hits = trees[r.reference_name][r.reference_start]
    return list(hits)[0].data if hits else None
# -------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("bam");   p.add_argument("gtf")
    p.add_argument("-o","--out", default="rpf_counts.csv")
    p.add_argument("--min_len", type=int, default=28)
    p.add_argument("--max_len", type=int, default=38)
    p.add_argument("--keep-zeros", action="store_true")
    a = p.parse_args()

    trees = load_gtf(a.gtf)
    print(f"Loaded {len(trees)} contigs from GTF")

    counts = defaultdict(Counter)
    bam = pysam.AlignmentFile(a.bam,"rb")
    for r in bam.fetch(until_eof=True):
        if r.is_unmapped or r.is_secondary or r.is_supplementary: continue
        L = r.query_length
        if a.min_len <= L <= a.max_len:
            tid = assign_read(r, trees)
            if tid:
                counts[tid][L] += 1
    bam.close()

    rows = []
    for tid, c in counts.items():
        for L, n in c.items():
            rows.append((tid,L,n))
    if a.keep_zeros:
        # add zero rows for transcripts with no reads
        for chrom in trees.keys():
            if chrom not in counts:
                rows.append((chrom, None, 0))
    pd.DataFrame(rows, columns=["transcript","length","nReads"]
                ).to_csv(a.out,index=False)
    print(f"✓ wrote {a.out}  ({len(rows)} rows)")
# -------------------------------------------------------------------------
if __name__ == "__main__":
    main()
