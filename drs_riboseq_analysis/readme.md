The following steps were performed using the riboseq analysis on the GPCR library. Before these steps, reads were processed as described in the manuscript to trim, remove RNA contaminants, etc.

1. Index genome using STAR
STAR --runThreadN 16 \
     --runMode genomeGenerate \
     --genomeDir ./clean_index \
     --genomeFastaFiles  clean_minigenome.fa \
     --sjdbGTFfile       clean_minigenome.gtf \
     --genomeSAindexNbases 8

2. Perform alignment of reads to GPCR library transcript sequences
STAR --runThreadN 16 \
     --genomeDir ./clean_index \
     --readFilesIn no_contam_trimmed_IMGN_B48_4.fastq \
     --outFileNamePrefix B48_4_clean_ \
     --outFilterMultimapNmax 1 \
     --alignIntronMax 1 \
     --alignEndsType EndToEnd \
     --outFilterMismatchNoverReadLmax 0.04 \
     --outFilterMismatchNmax 2 \
     --outFilterScoreMinOverLread 0.30 \
     --outFilterMatchNminOverLread 0.30 \
     --readNameSeparator " " \
     --outSAMattributes NH HI AS nM \
     --outSAMtype BAM SortedByCoordinate \
     --quantMode GeneCounts

3. Sort the bam file using samtools
samtools index -@8 B48_4_clean_Aligned.sortedByCoord.out.bam

4. Remove duplicates using UMI tools
umi_tools dedup \
    -I B48_4_clean_Aligned.sortedByCoord.out.bam \
    --output-stats=B48_4_clean_umi_dedup \
    --log=dedup_B48_4_clean.log \
    -S B48_4_clean_dedup.bam

5. Re-sort and index deduplicated bam
samtools sort -@8 -o B48_4_clean_dedup.sorted.bam  B48_4_clean_dedup.bam
samtools index -@8 B48_4_clean_dedup.sorted.bam

6. Count monosome and disome counts in bam file using custom Python script

Monosome counts
python simple_rpf_counter_v2.py \
       B48_4_clean_dedup.sorted.bam \
       clean_gpcr_minigenome.gtf     \
       --keep-zeros \
       -o B48_4_clean_rpf_counts.csv
   
Disome counts
python simple_rpf_counter_v2.py \
       B48_4_clean_dedup.sorted.bam \
       clean_gpcr_minigenome.gtf     \
       --min_len 50 --max_len 70 \
       --keep-zeros \
       -o B48_4_clean_rpf_counts_disome.csv

7. Learn codon offsets and produce per-codon p-site counts using a custom Python script
python psite_offset_and_counts_v2_auto.py \
       B48_4_clean_dedup.sorted.bam \
       clean_gpcr_minigenome.gtf     \
       --min_reads 500               \
       --out_prefix B48_4_clean_auto

8. Normalize RPF counts using a custom Python script
python normalize_rpf_counts.py \
       B48_4_clean_rpf_counts.csv \
       clean_gpcr_minigenome.gtf \
       -o B48_4_clean_rpf_TPM.csv

9. Compute monosome disome ratio using a custom Python script
python disome_mono_ratio.py \
       B48_4_clean_rpf_counts.csv \
       B48_4_clean_rpf_counts_disome.csv \
       --min_mono 0 --min_di 0 \
       --out B48_4_clean_RPF_D_M_ratio.csv

Run version 2 of this custom script to get TPMs instead of raw counts
python disome_mono_ratio_v2.py \
        B48_4_clean_rpf_counts.csv \
        B48_4_clean_rpf_counts_disome.csv \
        --min_mono 0 --min_di 0 \
        -o B48_4_clean_RPF_disome_monosome_ratio_tpm.csv

10. Merge RNAseq and riboseq to get a rough estimate of translation efficiency using a custom Python script
python psite_offset_and_counts_v3_auto.py \
       B48_4_clean_dedup.sorted.bam \
       clean_gpcr_minigenome.gtf     \
       --min_reads 500               \
       --out_prefix B48_4_clean_auto_vector

11. Create GPCR metaprofile using topological properties of GPCRs with a custom Python script
python gpcr_meta_profile_v7.py --mono_csv B48_4_clean_auto_codon_vectors_mono_codon_counts.csv --di_csv B48_4_clean_auto_codon_vectors_di_codon_counts.csv --gtf clean_gpcr_minigenome.gtf --topo gpcr_topological_features_consensus_tmds.tsv --out_dom gpcr_domain_counts_v7.csv --out_pct gpcr_percent_profile_v7.csv


