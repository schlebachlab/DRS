#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import re
import textwrap
import string
import random
from itertools import groupby
from collections import Counter
import difflib
import numpy as np
import pandas as pd
import csv
import math
from bisect import bisect_right


def genetic_code_func():
    genetic_code  = """
        TTT F      CTT L      ATT I      GTT V
        TTC F      CTC L      ATC I      GTC V
        TTA L      CTA L      ATA I      GTA V
        TTG L      CTG L      ATG M      GTG V
        TCT S      CCT P      ACT T      GCT A
        TCC S      CCC P      ACC T      GCC A
        TCA S      CCA P      ACA T      GCA A
        TCG S      CCG P      ACG T      GCG A
        TAT Y      CAT H      AAT N      GAT D
        TAC Y      CAC H      AAC N      GAC D
        TAA *      CAA Q      AAA K      GAA E
        TAG *      CAG Q      AAG K      GAG E
        TGT C      CGT R      AGT S      GGT G
        TGC C      CGC R      AGC S      GGC G
        TGA *      CGA R      AGA R      GGA G
        TGG W      CGG R      AGG R      GGG G
        """
    codon_finder = re.compile(r'[ATCG]{3}')
    amino_acid_finder = re.compile(r'\ \w{1}[\ |\n]|\*')
    codon_list = codon_finder.findall(genetic_code)
    amino_acid_list = [x.strip() for x in amino_acid_finder.findall(genetic_code)]
    genetic_code_dict = {}
    i = 0
    while i < len(codon_list):
        genetic_code_dict[codon_list[i]] = amino_acid_list[i]
        i += 1
    return genetic_code_dict


def ribosome(domain_codons):
    genetic_code_dict = genetic_code_func()
    domain_aminoacids = []
    for codon in domain_codons:
        amino_acid = genetic_code_dict[codon]
        domain_aminoacids.append(amino_acid)
    return domain_aminoacids




def import_text_file(path_to_file):
    file_obj = open(path_to_file)
    data_raw = []
    for line in file_obj:
        data_raw.append(line.split())
    file_obj.close()
    seqs_str_list = []
    for line in data_raw:
        item = line[0]
        seqs_str_list.append(item)
    return seqs_str_list

def import_tsv_file(path_tsv):
    tsv_data = []
    with open(path_tsv, "r") as tsv_infile:
        for line in tsv_infile:
            tsv_data.append(line.strip('\n').split('\t'))
    return tsv_data



def build_geneblocks(final_output, ordered_transcripts_list, backup_umi_seqs, spacer_positions_dict, new_cleavage_sites_dict, ha_insertion_site_dict):

    upstream_of_barcode = 'TAGGCG'
    
    #umi = final_output[x][6]
    
    upstream_gpcr_input = 'CTTCGCGATGTACGGGCCAGATATACGCGTTCCGGCTTGCCGGCTTGtcgacgacggcggtctccgtcgtcaggatcatccGCTAGCGTTTAAACTTAAGCTTGGTGCC'
    ha_tag_seq = 'TACCCATACGACGTACCAGATTACGCT'
    upstream_of_gpcr = upstream_gpcr_input.upper()
    
    #gpcr_seq = final_output[x][13][3:-3]

    downstream_of_gpcr = 'GGGTAGAGACG'
    
    # Plasmid components
    upstream_geneblock = 'GGGGATCTCATGCTGGAGTTCTTCGCCCACCCCAACTTGTTTATTGCAGCTTATAATGGTTACAAATAAAGCAATAGCATCACAAATTTCACAAATAAAGCATTTTTTTCACTGCATTCTAGTTGTGGTTTGTCCAAACTCATCAATGTATCTTATCATGTCTGTATACCGTCGACCTCTAGCTAGAGCTTGGCGTAATCATGGTCATAGCTGTTTCCTGTGTGAAATTGTTATCCGCTCACAATTCCACACAACATACGAGCCGGAAGCATAAAGTGTAAAGCCTGGGGTGCCTAATGAGTGAGCTAACTCACATTAATTGCGTTGCGCTCACTGCCCGCTTTCCTCAGCGGAAACCTGTCGTGCCAGCTGCATTAATGAATCGGCCAACGCGCGGGGAGAGGCGGTTTGCGTATTGGGCGCTCTTCCGCTTCCTCGCTCACTGACTCGCTGCGCTCGGTCGTTCGGCTGCGGCGAGCGGTATCAGCTCACTCAAAGGCGGTAATACGGTTATCCACAGAATCAGGGGATAACGCAGGAAAGAACATGTGAGCAAAAGGCCAGCAAAAGGCCAGGAACCGTAAAAAGGCCGCGTTGCTGGCGTTTTTCCATAGGCTCCGCCCCCCTGACGAGCATCACAAAAATCGACGCTCAAGTCAGAGGTGGCGAAACCCGACAGGACTATAAAGATACCAGGCGTTTCCCCCTGGAAGCTCCCTCGTGCGCTCTCCTGTTCCGACCCTGCCGCTTACCGGATACCTGTCCGCCTTTCTCCCTTCGGGAAGCGTGGCGCTTTCTCATAGCTCACGCTGTAGGTATCTCAGTTCGGTGTAGGTCGTTCGCTCCAAGCTGGGCTGTGTGCACGAACCCCCCGTTCAGCCCGACCGCTGCGCCTTATCCGGTAACTATCGTCTTGAGTCCAACCCGGTAAGACACGACTTATCGCCACTGGCAGCAGCCACTGGTAACAGGATTAGCAGAGCGAGGTATGTAGGCGGTGCTACAGAGTTCTTGAAGTGGTGGCCTAACTACGGCTACACTAGAAGAACAGTATTTGGTATCTGCGCTCTGCTGAAGCCAGTTACCTTCGGAAAAAGAGTTGGTAGCTCTTGATCCGGCAAACAAACCACCGCTGGTAGCGGTGGTTTTTTTGTTTGCAAGCAGCAGATTACGCGCAGAAAAAAAAGGATCTCAAGAAGATCCTTTGATCTTTTCTACGGGGTCTGACGCTCAGTGGAACGAAAACTCACGTTAAGGGATTTTGGTCATGAGATTATCAAAAAGGATCTTCACCTAGATCCTTTTAAATTAAAAATGAAGTTTTAAATCAATCTAAAGTATATATGAGTAAACTTGGTCTGACAGTTACCAATGCTTAATCAGTGAGGCACCTATCTCAGCGATCTGTCTATTTCGTTCATCCATAGTTGCCTGACTCCCCGTCGTGTAGATAACTACGATACGGGAGGGCTTACCATCTGGCCCCAGTGCTGCAATGATACCGCGAGACCCACGCTCACCGGCTCCAGATTTATCAGCAATAAACCAGCCAGCCGGAAGGGCCGAGCGCAGAAGTGGTCCTGCAACTTTATCCGCCTCCATCCAGTCTATTAATTGTTGCCGGGAAGCTAGAGTAAGTAGTTCGCCAGTTAATAGTTTGCGCAACGTTGTTGCCATTGCTACAGGCATCGTGGTGTCACGCTCGTCGTTTGGTATGGCTTCATTCAGCTCCGGTTCCCAACGATCAAGGCGAGTTACATGATCCCCCATGTTGTGCAAAAAAGCGGTTAGCTCCTTCGGTCCTCCGATCGTTGTCAGAAGTAAGTTGGCCGCAGTGTTATCACTCATGGTTATGGCAGCACTGCATAATTCTCTTACTGTCATGCCATCCGTAAGATGCTTTTCTGTGACTGGTGAGTACTCAACCAAGTCATTCTGAGAATAGTGTATGCGGCGACCGAGTTGCTCTTGCCCGGCGTCAATACGGGATAATACCGCGCCACATAGCAGAACTTTAAAAGTGCTCATCATTGGAAAACGTTCTTCGGGGCGAAAACTCTCAAGGATCTTACCGCTGTTGAGATCCAGTTCGATGTAACCCACTCGTGCACCCAACTGATCTTCAGCATCTTTTACTTTCACCAGCGTTTCTGGGTGAGCAAAAACAGGAAGGCAAAATGCCGCAAAAAAGGGAATAAGGGCGACACGGAAATGTTGAATACTCATACTCTTCCTTTTTCAATATTATTGAAGCATTTATCAGGGTTATTGTCTCATGAGCGGATACATATTTGAATGTATTTAGAAAAATAAACAAATAGGGGTTCCGCGCACATTTCCCCGAAAAGTGCCACCTGACGTCGACGGATCGGGAGATCTCCCGATCCCCTATGGTGCACTCTCAGTACAATCTGCTCTGATGCCGCATAGTTAAGCCAGTATCTGCTCCCTGCTTGTGTGTTGGAGGTCGCTGAGTAGTGCGCGAGCAAAATTTAAGCTACAACAAGGCAAGCGTCTCCCGACAATTGCATGAAGAATCTGCTTAGGGT'
    downstream_geneblock = 'GAGGTTAAGGTCTCTAAAATTCCGCCCCCCCCCCTAACGTTACTGGCCGAAGCCGCTTGGAATAAGGCCGGTGTGCGTTTGTCTATATGTTATTTTCCACCATATTGCCGTCTTTTGGCAATGTGAGGGCCCGGAAACCTGGCCCTGTCTTCTTGACGAGCATTCCTAGGGGTCTTTCCCCTCTCGCCAAAGGAATGCAAGGTCTGTTGAATGTCGTGAAGGAAGCAGTTCCTCTGGAAGCTTCTTGAAGACAAACAACGTCTGTAGCGACCCTTTGCAGGCAGCGGAACCCCCCACCTGGCGACAGGTGCCTCTGCGGCCAAAAGCCACGTGTATAAGATACACCTGCAAAGGCGGCACAACCCCAGTGCCACGTTGTGAGTTGGATAGTTGTGGAAAGAGTCAAATGGCTCTCCTCAAGCGTATTCAACAAGGGGCTGAAGGATGCCCAGAAGGTACCCCATTGTATGGGATCTGATCTGGGGCCTCGGTACACATGCTTTACATGTGTTTAGTCGAGGTTAAAAAACGTCTAGGCCCCCCGAACCACGGGGACGTGGTTTTCCTTTGAAAAACACGATGATAATATGGCCACAACCATGACGGCCCTGACAGAAGGTGCGAAGCTGTTCGAGAAGGAGATTCCCTATATCACAGAATTGGAGGGGGATGTAGAGGGTATGAAGTTTATCATCAAAGGCGAAGGGACAGGGGATGCAACAACTGGAACAATTAAGGCTAAGTACATTTGCACGACCGGCGACGTCCCGGTGCCCTGGTCCACGCTCGTCACCACGCTCACGTACGGAGCCCAGTGCTTTGCCAAATATGGCCCTGAACTTAAAGACTTCTACAAGTCGTGTATGCCGGAGGGATACGTGCAAGAGAGGACGATCACCTTTGAAGGTGACGGAGTATTCAAAACAAGAGCGGAGGTGACGTTCGAGAATGGATCGGTCTATAACCGGGTCAAGCTCAACGGACAGGGCTTTAAGAAAGATGGACACGTCCTTGGGAAGAATTTGGAGTTCAATTTCACCCCGCATTGTCTTTACATCTGGGGTGATCAGGCGAATCACGGGTTGAAATCAGCGTTCAAGATCATGCACGAGATTACGGGGAGCAAAGAGGACTTTATCGTGGCAGACCACACTCAGATGAACACTCCAATCGGAGGGGGTCCCGTACACGTACCCGAGTATCATCACCTGACCGTCTGGACATCGTTTGGAAAAGACCCTGACGACGATGAAACTGATCATCTCAACATTGTGGAAGTGATCAAGGCGGTGGACTTGGAAACATACCGGTGAGCGGCCGCTCGAGTCTAGAGGGCCCGTTTAAACCCGCTGATCAGCCTCGACTGTGCCTTCTAGTTGCCAGCCATCTGTTGTTTGCCCCTCCCCCGTGCCTTCCTTGACCCTGGAAGGTGCCACTCCCACTGTCCTTTCCTAATAAAATGAGGAAATTGCATCGCATTGTCTGAGTAGGTGTCATTCTATTCTGGGGGGTGGGGTGGGGCAGGACAGCAAGGGGGAGGATTGGGAAGACAATAGCAGGCATGCTGGGGATGCGGTGGGCTCTATGGCTTCTGAGGCGGAAAGAACCAGCTGGGGCTCTAGGGGGTATCCCCACGCGCCCTGTAGCGGCGCATTAAGCGCGGCGGGTGTGGTGGTTACGCGCAGCGTGACCGCTACACTTGCCAGCGCCCTAGCGCCCGCTCCTTTCGCTTTCTTCCCTTCCTTTCTCGCCACGTTCGCCGGCTTTCCCCGTCAAGCTCTAAATCGGGGGCTCCCTTTAGGGTTCCGATTTAGTGCTTTACGGCACCTCGACCCCAAAAAACTTGATTAGGGTGATGGTTCACGTACACGAGATTTCGATTCCACCGCCGCCTTCTATGAAAGGTTGGGCTTCGGAATCGTTTTCCGGGACGCCGGCTGGATGATCCTCCAGCGC'
    
    #constructs_dict = {}
    
    constructs_built = [['final_gene_name','barcode','gene_block_seq']]
    plasmids_built = [['final_gene_name','barcode','vector_seq']]

    #updated_table = [['gene_id','transcript_id','final_gene_name','uniprot_id','gpcr_class','approved_name','barcode','protein_seq','prot_seqlen','original_transcript_seq','nuc_seqlen','uniprot_transcript_identity','num_goldengate_sites','goldengate_free_seq','gene_block_seq','len_gene_block','complete_vector_seq']]
    #updated_table = [['gene_id', 'transcript_id', 'final_gene_name', 'uniprot_id', 'gpcr_class', 'approved_name', 'barcode', 'alt_isoform?', 'compound_name', 'protein_seq', 'prot_seqlen', 'cds_complete_transcript_seq', 'original_transcript_seq', 'nuc_seqlen','uniprot_transcript_identity', 'num_goldengate_sites', 'goldengate_free_seq','gene_block_seq','len_gene_block','complete_vector_seq','gly_fix']]

    #                        0       1            2               3                 4            5             6            7             8           9              10           11               12                    13              14               15             16             17                     18                         19                   20                  21                         22                   23                    24                  25             26                 27                 28             29                  30                         31                            32                33                 34                       35                            36
    updated_table =      [['idx','gene_id','transcript_id','final_gene_name','signalP_name','uniprot_id','gpcr_class','approved_name','barcode','alt_isoform?','compound_name','num_tmds','signal_peptide_topcons','signalP_pred', 'signalP_prob', 'signalP_cutsite','protein_seq','prot_seqlen','cds_complete_transcript_seq','original_transcript_seq','nuc_seqlen','uniprot_transcript_identity','isoform_unmodified','num_goldengate_sites','goldengate_free_seq','gene_block_seq','len_gene_block','complete_vector_seq','gly_fix?','already_ordered?','signal_pep_trunc_prot_seq', 'original_sigpep_cutsite', 'new_sigpep_cutsite', 'spacer_length', 'ha_insert_position', 'length_upstream_residues', 'res_between_cleavage_and_ha']]


    #                        0       1            2               3                 4            5             6             7            8           9              10           11               12                    13              14                 15            16            17                   18                            19                 20                   21                         22                   23                    24                  25             26                 27                 28             29                  30
    #updated_table_opt2 = [['idx','gene_id','transcript_id','final_gene_name','signalP_name','uniprot_id','gpcr_class','approved_name','barcode','alt_isoform?','compound_name','num_tmds','signal_peptide_topcons','signalP_pred', 'signalP_prob', 'signalP_cutsite','protein_seq','prot_seqlen','cds_complete_transcript_seq','original_transcript_seq','nuc_seqlen','uniprot_transcript_identity','isoform_unmodified','num_goldengate_sites','goldengate_free_seq','gene_block_seq','len_gene_block','complete_vector_seq','gly_fix?','already_ordered?','signal_pep_trunc_prot_seq']]


    backup_barcodes_used = []

    for entry in final_output[1:]:
    
        signalP_pred = entry[13]
        if signalP_pred == 'NO_SP':
            continue
    
        cut_site_raw = entry[15]
        if cut_site_raw == 'N/A':
            continue

        #cutsite_positions = cut_site_raw.split(' ')[2].split('.')[0]

        # Only process transcripts smaller than 3200 bp ORFs
        nuc_seqlen = int(entry[20])
        if nuc_seqlen > 3200:
            continue


        hgnc_symbol = entry[3]
        transcript_id = entry[2]
        gene_name = hgnc_symbol+'_'+transcript_id+'_SP_HA'

        if transcript_id not in spacer_positions_dict:
            continue

        print(f"Transcript ID: {transcript_id}, Spacer Length: {spacer_positions_dict.get(transcript_id)}")


        spacer_length = spacer_positions_dict[transcript_id]
        new_cleavage_site = new_cleavage_sites_dict[transcript_id]
        ha_insertion_site = ha_insertion_site_dict[transcript_id]

        

        if spacer_length is None:
            print(f"Warning: Spacer length for {transcript_id} is None. Skipping...")
            continue



        cutsite_positions = cut_site_raw.split(' ')[2].split('.')[0]
        # Set length of spacer between cleavage site and HA insertion site
        #spacer_length = 4

        # Explanation of spacer_length:
        # Parse cut site: Get second number after dash; subtract one going from position to index, add spacer, add +1 to go from second position of cleavage to first amino acid after cleavage
        # Now spacer_length = 0 will yield HA insertion after the second codon in the cleavage site 
        # QQQ XXX -- YYY | AAA BBB CCC
        # Here, XXX -- YYY represents the signal peptidase cleavage site
        # YYY | AAA represents the HA insertion site at spacer_length = 0.
        # So spacer_length = 0 represents no spaces between the second codon after signal peptidase cleavage and HA.
        # THERE WILL ALWAYS BE AT LEAST ONE RESIDUE AFTER CLEAVAGE AND BEFORE HA INSERTION - WHEN spacer_length == 0
        # When spacer_length = 1, there will be two residues - and so on.
        cut_idx = int(cut_site_raw.split(' ')[2].split('-')[1].split('.')[0]) - 1 + spacer_length + 1
        ha_insert_position = cut_idx + 1

        # Obtain the barcode seq unless the transcript has already been ordered then fetch a new barcode  
        if transcript_id in ordered_transcripts_list:
            already_ordered = '1'
            barcode_idx = 0
            barcode_pass = False
            while barcode_pass == False:
                test_barcode = backup_umi_seqs[barcode_idx]
                if test_barcode not in backup_barcodes_used:
                    barcode_seq = test_barcode
                    barcode_pass = True
                    backup_barcodes_used.append(barcode_seq)
                else:
                    barcode_idx += 1
        
        else:
            already_ordered = '0'
            #gene_name = entry[10]
            barcode_seq = entry[8]

        # Remove STOP codons
        gpcr_seq = entry[24][0:-3]

        # Remove C-terminal glycines
        gpcr_seq_codons = textwrap.wrap(gpcr_seq, 3)
        if gpcr_seq_codons[-1] in ['GGT', 'GGC', 'GGA', 'GGG']:
            #gpcr_seq_processed = gpcr_seq[:-3]
            gpcr_seq_codons_processed = gpcr_seq_codons[:-1]
            gly_fix = '1'
        else:
            #gpcr_seq_processed = gpcr_seq
            gpcr_seq_codons_processed = gpcr_seq_codons
            gly_fix = '0'
    
        
        # Convert HA tag to codons
        ha_tag_codons = textwrap.wrap(ha_tag_seq, 3)


        # Find HA insert site and break up sequence there
        gpcr_seq_codons_after_SP = gpcr_seq_codons_processed[cut_idx:]
        gpcr_seq_codons_before_SP = gpcr_seq_codons_processed[0:cut_idx]
        gpcr_seq_codons_ha_insert = gpcr_seq_codons_before_SP + ha_tag_codons + gpcr_seq_codons_after_SP
        gpcr_seq_processed = ''.join(gpcr_seq_codons_ha_insert)
    
        # Make new protein sequence
        gpcr_seq_processed_pep = ribosome(gpcr_seq_codons_ha_insert)
        gpcr_seq_processed_pep_str = ''.join(gpcr_seq_processed_pep)
        
        # Find residues after cleavage site but before HA tag
        #cleavage_idx = int(new_cleavage_site.split('-')[0]) - 1
        cleavage_idx = int(new_cleavage_site.split('-')[1]) - 1 # Change to second position of cleavage site not first
        #ha_idx = int(ha_insertion_site.split('-')[0]) - 1
        ha_idx = int(ha_insertion_site.split('-')[1]) - 1 # Change to second position of HA insertion position site not first
        # A B C D E F = seq
        # 0 1 2 3 4 5
        # B-C is cleavage site
        # D-E is insertion site
        # cleavage index is 2
        # HA index i 4
        # seq[2:4] = C D
        #length_upstream_residues = int(ha_insertion_site.split('-')[0]) - int(new_cleavage_site.split('-')[0])
        #length_upstream_residues = int(ha_insertion_site.split('-')[1]) - int(new_cleavage_site.split('-')[0]) + 1 # ?
        residues_between_cleavage_and_ha = gpcr_seq_processed_pep_str[cleavage_idx:ha_idx]
        length_upstream_residues= len(residues_between_cleavage_and_ha)
        
        # Build gene blocks
        construct_list = [upstream_of_barcode, barcode_seq, upstream_of_gpcr, gpcr_seq_processed, downstream_of_gpcr]
        construct_seq = ''.join(construct_list)
        
        complete_vector_list = [upstream_geneblock, construct_seq, downstream_geneblock]
        complete_vector_seq = ''.join(complete_vector_list)
        
        geneblock_output_line = [gene_name, barcode_seq, construct_seq]
        vector_output_line = [gene_name, barcode_seq, complete_vector_seq]
        
        constructs_built.append(geneblock_output_line)
        plasmids_built.append(vector_output_line)
        
        updated_table_entry = entry[0:8] + [barcode_seq] + [entry[9]] + [gene_name] + entry[11:] + [construct_seq] + [str(len(construct_seq))] + [complete_vector_seq] + [gly_fix] + [already_ordered] + [gpcr_seq_processed_pep_str] + [cutsite_positions] + [new_cleavage_site] + [spacer_length] + [ha_insertion_site] + [length_upstream_residues] + [residues_between_cleavage_and_ha]
        
        
        updated_table.append(updated_table_entry)
        
        #constructs_dict[barcode_seq] = entry + [construct_seq] + [complete_vector_seq]
        
    return constructs_built, plasmids_built, updated_table



def print_fasta_gene_blocks(filtered_list, outname):
     myfasta = open(outname+'.fasta', 'w')
     for record in filtered_list[1:]:
         print('>'+record[0]+'|'+record[1], file = myfasta)
         idx = 0
         columns = 60
         while idx < len(record[2]):
             if len(record[2][idx:]) > columns:
                 print(record[2][idx:idx+columns], file = myfasta)
             else:
                 print(record[2][idx:], file = myfasta)
             idx += columns
     myfasta.close()
    
#                    0       1           2               3               4               5          6               7            8           9               10             11           12                        13             14              15           16                17                            18                   19                   20                        21                   22                    23                   24               25               26                 27           28                      29
#updated_table = [['idx','gene_id','transcript_id','final_gene_name','signalP_name','uniprot_id','gpcr_class','approved_name','barcode','alt_isoform?','compound_name','num_tmds','signal_peptide_topcons','signalP_pred','signalP_cutsite','protein_seq','prot_seqlen','cds_complete_transcript_seq','original_transcript_seq','nuc_seqlen','uniprot_transcript_identity','isoform_unmodified','num_goldengate_sites','goldengate_free_seq','gene_block_seq','len_gene_block','complete_vector_seq','gly_fix?','already_ordered?','signal_pep_trunc_prot_seq']]
def print_fasta_final_output_pep(filtered_list, outname):
     myfasta = open(outname+'_pep.fasta', 'w')
     for record in filtered_list[1:]:
         print('>'+record[10]+'|'+record[1]+'|'+record[2]+'|'+record[3]+'|'+record[5]+'|'+record[8]+'|'+record[9], file = myfasta)
         idx = 0
         columns = 60
         while idx < len(record[30]):
             if len(record[30][idx:]) > columns:
                 print(record[30][idx:idx+columns], file = myfasta)
             else:
                 print(record[30][idx:], file = myfasta)
             idx += columns
     myfasta.close()


def print_results(output_data, outname):
    myfile = open(outname+'.csv', 'w')
    for line in output_data:
        line_string = [str(x) for x in line]
        csv_line = ','.join(line_string)
        print(csv_line, file = myfile)
    myfile.close()



def safe_float(string_value, default=0.0):
    try:
        return float(string_value)
    except ValueError:
        return default

def import_csv_file(path_csv):
    csv_data = []
    with open(path_csv, "r") as csv_infile:
        for line in csv_infile:
            csv_data.append(line.strip('\n').split(','))
    return csv_data


#output_data = [transcript_id, best_position, new_cleavage_site_output, ha_insertion_site_output]
def import_spacer_dict(path_spacer_dict_csv):
    positions_dict = {}
    new_cleavage_sites_dict = {}
    ha_insertion_site_dict = {}
    spacer_dict_raw_data = import_csv_file(path_spacer_dict_csv)
    for entry in spacer_dict_raw_data:
        transcript_id = entry[0]
        spacer_length = int(entry[1])
        new_cleavage_site = entry[2]
        ha_insertion_site = entry[3]
        positions_dict[transcript_id] = spacer_length
        new_cleavage_sites_dict[transcript_id] = new_cleavage_site
        ha_insertion_site_dict[transcript_id] = ha_insertion_site
    return positions_dict, new_cleavage_sites_dict, ha_insertion_site_dict


if __name__ == '__main__':

    '''
    # Weights and cutoffs used in insertion site selection script
    cut_threshold_abs = 0.9    # Cutite existence probability
    cut_threshold_loc = 0.75   # Cutsite position probability
    disorder_threshold = 0.5   # Probability of disorder requirement on both sides of HA site
    rsa_threshold = 0.25       # RSA on both sides of HA site required
    coil_threshold = 0.5       # Probability of coil required on both sides of HA site
    '''

    #logfile='standard_wts_0_0_0_100_s1'
    #logfile='standard_wts_0_0_0_100_s1_v2b'   # Don't forget to change logfile too
    logfile = 'standard_wts_final'
    path_csv = 'final_output_with_signalP_pred3.csv'
    path_transcripts = 'submitted_transcripts.txt'
    path_barcodes = 'backup_umi_sequences.txt'
    #path_spacer_dict = 'cut_site_spacer_dict.csv'
    #path_spacer_dict = 'HA_insert_search_wts_0_0_0_100_s1.csv'    # Canonical weights, no check for length of HA-upstream peptide
    path_spacer_dict = 'HA_insert_search_wts_0_0_0_100_s1_v2b.csv' # Has the check for length of peptide upstream from HA tag
    
    positions_dict, new_cleavage_sites_dict, ha_insertion_site_dict = import_spacer_dict(path_spacer_dict)
    transcript_list = import_text_file(path_transcripts)
    alt_barcodes_list = import_text_file(path_barcodes)
    csv_data = import_csv_file(path_csv)
    
    

    constructs_built, plasmids_built, updated_table = build_geneblocks(csv_data, transcript_list, alt_barcodes_list, positions_dict, new_cleavage_sites_dict, ha_insertion_site_dict)
    
    job_suffix = logfile

    print_results(constructs_built, 'gene_blocks_data_gpcr_lib_sigpep_ins'+'_'+job_suffix)
    print_results(plasmids_built, 'complete_vector_data_gpcr_lib_sigpep_ins'+'_'+job_suffix)
    print_results(updated_table, 'updated_table_final_gpcr_lib_sigpep_ins'+'_'+job_suffix)
    print_fasta_gene_blocks(constructs_built, 'gene_blocks_gpcr_lib_sigpep_ins'+'_'+job_suffix)
    print_fasta_gene_blocks(plasmids_built, 'complete_vectors_gpcr_lib_sigpep_ins'+'_'+job_suffix)
    print_fasta_final_output_pep(updated_table, 'gpcr_lib_pep_seqs_sigpep_ins'+'_'+job_suffix)
