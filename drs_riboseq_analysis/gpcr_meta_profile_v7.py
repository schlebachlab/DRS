#!/usr/bin/env python3
"""
gpcr_meta_profile.py

Given:
  - dedup-sorted BAM of your minigenome Ribo-seq,
  - single-exon GTF (with CDS lines),
  - a per-length offset table CSV (length,offset),
  - a topology table TSV (one row per transcript, with TMD1_start, TMD1_end, … TMD7_start, TMD7_end),
  - a class map CSV (transcript, gpcr_class),

Produce:
  a) domain_counts.csv:
     transcript, gpcr_class, domain, mono_count, di_count, D_over_M

  b) percent_profile.csv:
     gpcr_class, percent_bin (0–100), mono_count, di_count
"""

import argparse, sys
from collections import defaultdict
import pandas as pd
import numpy as np
import pysam
import re

DOMAINS = [
    "N-term",
    "TMD1","IL1","TMD2","EL1","TMD3","IL2","TMD4","EL2","TMD5","IL3","TMD6","EL3","TMD7",
    "C-term"
]

def load_offsets(path):
    df = pd.read_csv(path)
    return dict(zip(df.length, df.offset))

import pandas as pd

def load_topology(path, cds_len_codons):
    """
    Load per-transcript topology in NT-coordinates, shifting for HA tag.

    Input table (tab-sep) must have columns:
      - transcript_id
      - TMD1_start … TMD7_start  (1-based codon indices)
      - TMD1_end   … TMD7_end    (1-based codon indices)
      - CDS_length                  (1-based codon index for last AA)

    Returns dict:
      { transcript_id : [
          ("N-term",   start_nt, end_nt),
          ("TMD1",     start_nt, end_nt),
          ("IL1",      start_nt, end_nt),
          ("TMD2",     start_nt, end_nt),
          ("EL1",      …       ),
          …,
          ("TMD7",     …       ),
          ("C-term",   start_nt, end_nt)
      ] }
    where all positions are 1-based nucleotide.
    """
    df = pd.read_csv(path, sep="\t")
    topo = {}

    ha_nt = 27 # Add HA tag length to TMD coordinates (in nucleotides)

    for row in df.itertuples(index=False):
        tx = row.transcript_id
        if tx not in cds_len_codons:
            continue
        coords = []

        # N-term runs from first nt of codon1 up to just before TMD1:
        # codon1 starts at nt (1−1)*3+1 = 1, so after HA tag it is 1+ha_nt
        first_tmd1_codon = row.TMD1_start
        first_tmd1_nt    = (first_tmd1_codon - 1)*3 + 1 + ha_nt
        coords.append(("N-term", 1 + ha_nt, first_tmd1_nt - 1))

        # now each TMD and the loops between them
        for i in range(1, 8):
            # codon -> nt conversion
            codon_s = getattr(row, f"TMD{i}_start")
            codon_e = getattr(row, f"TMD{i}_end")

            nt_s = (codon_s - 1)*3 + 1 + ha_nt
            nt_e = codon_e*3     + ha_nt
            coords.append((f"TMD{i}", nt_s, nt_e))

            # add loop after TMDi if not the last TMD
            if i < 7:
                # start of next TMD
                next_codon = getattr(row, f"TMD{i+1}_start")
                next_nt    = (next_codon - 1)*3 + 1 + ha_nt

                # intracellular loops follow odd-numbered TMDs, else extracellular
                loop_name = "IL" if i % 2 == 1 else "EL"
                loop_idx  = (i + 1) // 2
                loop_start = nt_e + 1
                loop_end   = next_nt - 1
                coords.append((f"{loop_name}{loop_idx}", loop_start, loop_end))

        # finally the C-term (from end of TMD7 to end of CDS)
        # row.CDS_length is in codons; last nt is codon*3
        codon_len = cds_len_codons[tx]
        last_nt = codon_len * 3 + ha_nt
        # C-term starts right after last TMD7 end
        cterm_start = coords[-1][2] + 1
        coords.append(("C-term", cterm_start, last_nt))

        topo[tx] = coords

    return topo

def load_classes(path):
    df = pd.read_csv(path, sep="\t")
    return dict(zip(df.transcript_id, df.gpcr_class))

def which_domain(domains, pos):
    """domains: list of (name,start,end). return name or None."""
    for name,s,e in domains:
        if s <= pos <= e:
            return name
    return None

import re

def parse_cds_lengths_from_gtf(path_gtf):
    """
      - cds_len_nt     : CDS end coordinate in nucleotides
      - cds_len_codons : CDS length in codons (end_nt // 3)
    """
    cds_len_nt     = {}
    cds_len_codons = {}
 
    raw_gtf_data = []
    gtf_fh = open(path_gtf, 'r')

    for line in gtf_fh:
        ln = line.split('\t')
        raw_gtf_data.append(ln)
    
    cds_lines = []
    for x in raw_gtf_data:
        if x[0] != 'HA_TAG_INSERT':
            if x[2].strip() == 'CDS':
                cds_line = [y.strip() for y in x]
                cds_lines.append(cds_line)
    
    for entry in cds_lines:
        gene_name = entry[0].split('_')[0]
        transcript_id_stable = entry[0].split('_')[1]
        nuc_len = int(entry[4])
        codon_len = nuc_len // 3
        cds_len_nt[transcript_id_stable] = nuc_len
        cds_len_codons[transcript_id_stable] = codon_len

    return cds_len_nt, cds_len_codons

def main():
    import argparse
    import pandas as pd
    import numpy as np
    from collections import defaultdict

    p = argparse.ArgumentParser()
    p.add_argument("--mono_csv", required=True,
                   help="CSV of mono codon counts (transcript_id, codon_1, codon_2, …)")
    p.add_argument("--di_csv",   required=True,
                   help="CSV of di   codon counts (transcript_id, codon_1, codon_2, …)")
    p.add_argument("--gtf",      required=True,
                   help="single‑exon GTF (for CDS lengths)")
    p.add_argument("--topo",     required=True,
                   help="TSV topology table")
    #p.add_argument("--classes",  required=True,
    #               help="TSV transcript_id<TAB>gpcr_class")
    p.add_argument("--out_dom",  default="domain_counts.csv")
    p.add_argument("--out_pct",  default="percent_profile.csv")
    args = p.parse_args()

    # 0) parse CDS lengths from GTF
    cds_len_nt, cds_len_codons = parse_cds_lengths_from_gtf(args.gtf)

    # 1) load topology and classes
    topo    = load_topology(args.topo, cds_len_codons)
    classes = load_classes(args.topo)

    # 2) read our mono/di codon‑count matrices
    mono_df = pd.read_csv(args.mono_csv)
    di_df   = pd.read_csv(args.di_csv)

    # 3) tally domain and percent‑along for each transcript
    dom_counts = defaultdict(lambda: {"mono": 0, "di": 0})
    pct_counts = defaultdict(lambda: {"mono": 0, "di": 0, "class": None, 
                                      "hist_mono": np.zeros(101, int),
                                      "hist_di":   np.zeros(101, int)})

    for df, cat in [(mono_df, "mono"), (di_df, "di")]:
        for row in df.itertuples(index=False):
            # split "gene_transcript" → transcript ID
            full_id = row.transcript_id
            tx = full_id.split("_", 1)[1]

            # assign class
            pct_counts[tx]["class"] = classes.get(tx, "UNK")

            # iterate codon columns
            for col in df.columns:
                if not col.startswith("codon_"):
                    continue
                codon = int(col.split("_", 1)[1])
                count = getattr(row, col)
                if count == 0:
                    continue

                # compute nucleotide‐position of P‑site (1‐based)
                psite = (codon - 1) * 3 + 1

                # 3a) domain tally
                domains = topo.get(tx)
                if domains:
                    dom = which_domain(domains, psite)
                    if dom:
                        dom_counts[(tx, dom)][cat] += count

                # 3b) percent‐along tally
                total_nt = cds_len_nt.get(tx)
                if total_nt:
                    pct = int((psite / total_nt) * 100)
                    pct = min(max(pct, 0), 100)
                    pct_counts[tx][f"hist_{cat}"][pct] += count

    # 4) aggregate domain densities per class
    #    first compute per‐tx densities
    domain_lengths = {
        (tx, dom): end - start + 1
        for tx, coords in topo.items()
        for dom, start, end in coords
    }
    per_tx = defaultdict(lambda: {"class": None, "densities": {}})
    for (tx, dom), ct in dom_counts.items():
        cl = classes.get(tx, "UNK")
        per_tx[tx]["class"] = cl
        length = domain_lengths.get((tx, dom), 1)
        per_tx[tx]["densities"][dom] = {
            "mono_den": ct["mono"] / length,
            "di_den":   ct["di"]   / length,
        }

    #    then average across transcripts in each class/domain
    agg = defaultdict(lambda: {"mono_den": [], "di_den": []})
    for tx, payload in per_tx.items():
        cl = payload["class"]
        for dom, dens in payload["densities"].items():
            agg[(cl, dom)]["mono_den"].append(dens["mono_den"])
            agg[(cl, dom)]["di_den"].append(dens["di_den"])

    dom_rows = []
    for (cl, dom), lists in agg.items():
        mono_avg = np.mean(lists["mono_den"]) if lists["mono_den"] else 0.0
        di_avg   = np.mean(lists["di_den"])   if lists["di_den"]   else 0.0
        ratio    = (di_avg / mono_avg) if mono_avg else np.nan
        dom_rows.append({
            "gpcr_class":   cl,
            "domain":       dom,
            "mono_density": mono_avg,
            "di_density":   di_avg,
            "D_over_M":     ratio,
        })

    pd.DataFrame(dom_rows) \
      .sort_values(["gpcr_class", "domain"]) \
      .to_csv(args.out_dom, index=False)

    # 5) aggregate percent‐along profiles per class
    class_bins = defaultdict(lambda: {"mono": [], "di": []})
    for tx, data in pct_counts.items():
        cl = data["class"]
        mono_hist = data["hist_mono"].astype(float)
        di_hist   = data["hist_di"].astype(float)
        total_m   = mono_hist.sum()
        total_d   = di_hist.sum()

        mono_norm = mono_hist/total_m if total_m > 0 else np.zeros_like(mono_hist)
        di_norm   = di_hist/total_d   if total_d > 0 else np.zeros_like(di_hist)

        for pct in range(101):
            class_bins[(cl, pct)]["mono"].append(mono_norm[pct])
            class_bins[(cl, pct)]["di"].append(di_norm[pct])

    pct_rows = []
    for (cl, pct), lists in class_bins.items():
        m = np.mean(lists["mono"])
        d = np.mean(lists["di"])
        pct_rows.append({
            "gpcr_class": cl,
            "percent_bin": pct,
            "mono_frac":   m,
            "di_frac":     d,
            "D_over_M":    (d/m if m else np.nan),
        })

    pd.DataFrame(pct_rows) \
      .sort_values(["gpcr_class", "percent_bin"]) \
      .to_csv(args.out_pct, index=False)

    print("Wrote", args.out_dom, "and", args.out_pct)

if __name__=="__main__":
    main()