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

def import_csv_file(path_csv):
    csv_data = []
    with open(path_csv, "r") as csv_infile:
        for line in csv_infile:
            csv_data.append(line.strip('\n').split(','))
    return csv_data


def build_geneblocks(final_output):

    upstream_of_barcode = 'TAGGCG'
    
    #umi = final_output[x][6]
    
    upstream_gpcr_input = 'CTTCGCGATGTACGGGCCAGATATACGCGTTCCGGCTTGCCGGCTTGtcgacgacggcggtctccgtcgtcaggatcatccGCTAGCGTTTAAACTTAAGCTTGGTGCCATGTACCCATACGACGTACCAGATTACGCT'
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
    updated_table = [['gene_id', 'transcript_id', 'final_gene_name', 'uniprot_id', 'gpcr_class', 'approved_name', 'barcode', 'alt_isoform?', 'compound_name', 'protein_seq', 'prot_seqlen', 'cds_complete_transcript_seq', 'original_transcript_seq', 'nuc_seqlen','uniprot_transcript_identity', 'num_goldengate_sites', 'goldengate_free_seq','gene_block_seq','len_gene_block','complete_vector_seq','gly_fix']]

    for entry in final_output[1:]:
    
        gene_name = entry[8]
        barcode_seq = entry[6]

        # Remove START AUG and STOP codons
        gpcr_seq = entry[16][3:-3]

        # Remove C-terminal glycines
        gpcr_seq_codons = textwrap.wrap(gpcr_seq, 3)
        if gpcr_seq_codons[-1] in ['GGT', 'GGC', 'GGA', 'GGG']:
            gpcr_seq_processed = gpcr_seq[:-3]
            gly_fix = '1'
        else:
            gpcr_seq_processed = gpcr_seq
            gly_fix = '0'
    
        construct_list = [upstream_of_barcode, barcode_seq, upstream_of_gpcr, gpcr_seq_processed, downstream_of_gpcr]
        construct_seq = ''.join(construct_list)
        
        complete_vector_list = [upstream_geneblock, construct_seq, downstream_geneblock]
        complete_vector_seq = ''.join(complete_vector_list)
        
        geneblock_output_line = [gene_name, barcode_seq, construct_seq]
        vector_output_line = [gene_name, barcode_seq, complete_vector_seq]
        
        constructs_built.append(geneblock_output_line)
        plasmids_built.append(vector_output_line)
        
        updated_table_entry = entry + [construct_seq] + [str(len(construct_seq))] + [complete_vector_seq] + [gly_fix]
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
    

def print_results(output_data, outname):
    myfile = open(outname+'.csv', 'w')
    for line in output_data:
        line_string = [str(x) for x in line]
        csv_line = ','.join(line_string)
        print(csv_line, file = myfile)
    myfile.close()


def interactive_configuration():
    path_csv = str(input("Please enter a path to the csv containing the GPCR library sequences (bsmbI edited versions): "))
    #jobname = str(input("Please enter a name for this job: "))
    return path_csv


if __name__ == '__main__':

    path_csv = interactive_configuration()
    
    csv_data = import_csv_file(path_csv)
    
    constructs_built, plasmids_built, updated_table = build_geneblocks(csv_data)
    
    print_results(constructs_built, 'gene_blocks_data_gpcr_library')
    print_results(plasmids_built, 'complete_vector_data_gpcr_library')
    print_results(updated_table, 'updated_table_final_gpcr_library')
    print_fasta_gene_blocks(constructs_built, 'gene_blocks_gpcr_library')
    print_fasta_gene_blocks(plasmids_built, 'complete_vectors_gpcr_library')



