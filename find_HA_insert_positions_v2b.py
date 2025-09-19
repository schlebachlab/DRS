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
#from Bio import PDB

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


def import_csv_file(path_csv):
    csv_data = []
    with open(path_csv, "r") as csv_infile:
        for line in csv_infile:
            csv_data.append(line.strip('\n').split(','))
    return csv_data

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

def get_netsurfp_data(path_list_csv):

    # Column 0: HA insertion position: 99 is no insertion, -1, 0, 1, 2, 3, 4, 5, 6, 7
    # Column 1: Class assignment - B for buried or E for Exposed - Threshold: 25% exposure, but not based on RSA
    # Column 2: Amino acid
    # Column 3: Sequence name
    # Column 4: Amino acid number
    # Column 5: Relative Surface Accessibility - RSA
    # Column 6: Absolute Surface Accessibility
    # Column 7: Not used
    # Column 8: Probability for Alpha-Helix
    # Column 9: Probability for Beta-strand
    # Column 10: Probability for Coil
    # Column 11: Probability of Disorder

    netsurfp_metadata_list = import_text_file(path_list_csv)
    combined_master_list_netsurfp = []
    
    for path_csv in netsurfp_metadata_list:
        netsurfp_data_position = import_csv_file(path_csv)
        for entry in netsurfp_data_position[1:]:
            combined_master_list_netsurfp.append(entry)

    netsurfp_dict = {}

    # Here, the outer key is uniprot_id (string) and the inner key is residue number (int)
    for line in combined_master_list_netsurfp:
        ha_insert_position = int(line[0])
        seq_name = line[3]
        transcript_id = seq_name.split('_')[1]
        #uniprot_id = seq_name.split('_')[4]
        res_num = int(line[4])
        rsa = float(line[5])
        prob_helix = float(line[8])
        prob_strand = float(line[9])
        prob_coil = float(line[10])
        prob_disorder = float(line[11])
        # Make note of order of data types arranged here - rsa, helix, coil
        data = [rsa, prob_helix, prob_strand, prob_coil, prob_disorder]

        if transcript_id not in netsurfp_dict:
            netsurfp_dict[transcript_id] = {}

        if ha_insert_position not in netsurfp_dict[transcript_id]:
            netsurfp_dict[transcript_id][ha_insert_position] = {}

        netsurfp_dict[transcript_id][ha_insert_position][res_num] = data

    # Accessing data:
    # rsa = netsurfp_dict[transcript_id][position][res_num][0]
    # prob_helix = netsurfp_dict[transcript_id][position][res_num][1]
    # prob_strand = netsurfp_dict[transcript_id][position][res_num][2]
    # prob_coil = netsurfp_dict[transcript_id][position][res_num][3]
    # prob_disorder = netsurfp_dict[transcript_id][position][res_num][4]
    # or
    # [rsa, prob_helix, prob_strand, prob_coil, prob_disorder] = netsurfp_dict[transcript_id][position][res_num]

    return netsurfp_dict

'''
# ipython testing session
import find_HA_insert_positions_v2 as ha
path_signalp = 'master_sheet_signalP.csv'
path_netsurfp_metadata = 'netsurfp_metadata.txt'
signalp_dict = ha.get_signalp_data(path_signalp)
netsurfp_dict = ha.get_netsurfp_data(path_netsurfp_metadata)
transcript_id = 'ENST00000240093'
ha_tag_seq = 'TACCCATACGACGTACCAGATTACGCT'
len(ha_tag_seq) / 3
position0 = 22
position1 = 23
position = 0
res_num = position0 + 1 + position
netsurfp_dict[transcript_id][position][res_num]

'''



def get_signalp_data(path_csv_file):

    # Column 0: Sequence name
    # Column 1: HA insertion position: 99 is no insertion, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8
    # Column 2: Probability of Signal Peptidase Cut Site
    # Column 3: Cut site string (Only != "" when Column 2 is over 0.5)

    signalp_data_raw = import_csv_file(path_csv_file)
    signalp_dict = {}

    # Here, the outer key is uniprot_id (string) and the inner key is residue number (int)
    for line in signalp_data_raw[1:]:

        ha_insert_position = int(line[1]) # Remember 99 is no insertion
        seq_name = line[0]
        transcript_id = seq_name.split('_')[1]

        cut_site_existence_prob = float(line[2])
        cutsite_string = line[3]

        data = [cut_site_existence_prob, cutsite_string]

        if transcript_id not in signalp_dict:
            signalp_dict[transcript_id] = {}

        signalp_dict[transcript_id][ha_insert_position] = data


    # Accessing data:
    # cut_site_exists_prob = signalp_dict[transcript_id][position][0]
    # cut_site_string = signalp_dict[transcript_id][position][1]
    # or
    # [cut_site_exists_prob, cut_site_string] = signalp_dict[transcript_id][position]

    return signalp_dict

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

# This applies a bonus to the score if the distance is 3-5 codons downstream from cleavage site
#find_insert_positions_dist_func(path_netsurfp_data, weight_cut_prob, weight_rsa, weight_dis, weight_dist)
def find_insert_positions_dist_func_gaussian(transcript_list, signalp_dict, netsurfp_dict, cut_threshold_abs, cut_threshold_loc, disorder_threshold, rsa_threshold, coil_threshold, logname, weight_rsa, weight_dis, weight_coil, weight_dist, stdev):

    logfile = open("insert_search_log_"+logname+".txt", "w")

    positions_dict = {}
    new_cleavage_sites_dict = {}
    ha_insertion_site_dict = {}
    
    final_output_data = []

    for transcript_id in transcript_list:

        # Initialize best_score and best_position
        best_score = 0
        best_position = None

        # Define cut site positions of original sequence before HA insertion
        [original_seq_cut_site_exists_prob, original_seq_cut_site_string] = signalp_dict[transcript_id][99] # Remember 99 is index of original sequence

        original_seq_cut_site_positions = original_seq_cut_site_string.split(' ')[2].split('.')[0]
        # Let us call positions x--y the original seq cut site positions, where -- is the cleaved amide bond
        # Parse the string to get x and y
        x_cut = int(original_seq_cut_site_positions.split('-')[0])
        y_cut = int(original_seq_cut_site_positions.split('-')[1]) # y = x + 1 by definition

        position_ha_insert_site_dict = {}
        position_new_cut_site_dict = {}

        print(f"Processing {transcript_id}", file=logfile)
        print(f"Coil probabiliy weight: {weight_coil}, RSA weight: {weight_rsa}, Disorder weight: {weight_dis}, Distance weight: {weight_dist}", file=logfile)

        # Save positions above the cut threshold
        fallback_data = []
        
        for position in range(8, -1, -1):

            # Define new HA insertion site and add it to per-position dictionary for later retrieval
            x_ha = x_cut + 1 + position # Amino acid before HA insertion site candidate
            y_ha_orig = y_cut + 1 + position # Amino acid after HA insertion site candidate (before HA has been inserted: original seq)
            y_ha_inserted = y_cut + 1 + position + 9 # Amino acid after HA insertion site candidate (after HA has been inserted: new seq); 9 is the length of the HA tag
            ha_insertion_site = str(x_ha)+'-'+str(y_ha_orig)
            position_ha_insert_site_dict[position] = ha_insertion_site
            
            # Extract cut probabilities (both site location and whether there is still a cut site) from signalP data dict
            [cut_probability_abs, cut_data] = signalp_dict[transcript_id][position]

            assert type(cut_probability_abs) == float, f"Expected float but got {type(entry[2])} for cut_probabiliy_abs: {cut_probability_abs}"


            if cut_data.strip() == "":
                cut_probability_loc = 0.0
                new_cutsite_positions = None
            else:
                cut_probability_loc = safe_float(cut_data.split(":")[-1].strip().split(" ")[-1])
                new_cutsite_positions = cut_data.split(' ')[2].split('.')[0]
                second_position_new_cut_site = int(new_cutsite_positions.split('-')[1])

            # Add new cleavage site to per-position dictionary for later retrieval
            position_new_cut_site_dict[position] = new_cutsite_positions

            # Extract biophysical data from netsurfp

            # x_ha_orig
            [rsa_x_orig, prob_helix_x_orig, prob_strand_x_orig, prob_coil_x_orig, prob_disorder_x_orig] = netsurfp_dict[transcript_id][99][x_ha]
            # y_ha_orig
            [rsa_y_orig, prob_helix_y_orig, prob_strand_y_orig, prob_coil_y_orig, prob_disorder_y_orig] = netsurfp_dict[transcript_id][99][y_ha_orig]
            # x_ha_inserted
            [rsa_x_ins, prob_helix_x_ins, prob_strand_x_ins, prob_coil_x_ins, prob_disorder_x_ins] = netsurfp_dict[transcript_id][position][x_ha]
            # y_ha_inserted
            [rsa_y_ins, prob_helix_y_ins, prob_strand_y_ins, prob_coil_y_ins, prob_disorder_y_ins] = netsurfp_dict[transcript_id][position][y_ha_inserted]


            # Print for x_ha_orig
            print(f"\tResidue before HA insert site in original sequence for position {position} ({x_ha}):\n"
                  f"\tRSA: {rsa_x_orig}\n"
                  f"\tProbability Helix: {prob_helix_x_orig}\n"
                  f"\tProbability Strand: {prob_strand_x_orig}\n"
                  f"\tProbability Coil: {prob_coil_x_orig}\n"
                  f"\tProbability Disorder: {prob_disorder_x_orig}\n", 
                  file=logfile)

            # Print for y_ha_orig
            print(f"\tResidue after HA insert site in original sequence for position {position} ({y_ha_orig}):\n"
                  f"\tRSA: {rsa_y_orig}\n"
                  f"\tProbability Helix: {prob_helix_y_orig}\n"
                  f"\tProbability Strand: {prob_strand_y_orig}\n"
                  f"\tProbability Coil: {prob_coil_y_orig}\n"
                  f"\tProbability Disorder: {prob_disorder_y_orig}\n", 
                  file=logfile)

            # Print for x_ha_inserted
            print(f"\tResidue before HA insert site in HA-inserted sequence for position {position} ({x_ha}):\n"
                  f"\tRSA: {rsa_x_ins}\n"
                  f"\tProbability Helix: {prob_helix_x_ins}\n"
                  f"\tProbability Strand: {prob_strand_x_ins}\n"
                  f"\tProbability Coil: {prob_coil_x_ins}\n"
                  f"\tProbability Disorder: {prob_disorder_x_ins}\n", 
                  file=logfile)

            # Print for y_ha_inserted
            print(f"\tResidue after HA insert site in HA-inserted sequence for position {position} ({y_ha_inserted}):\n"
                  f"\tRSA: {rsa_y_ins}\n"
                  f"\tProbability Helix: {prob_helix_y_ins}\n"
                  f"\tProbability Strand: {prob_strand_y_ins}\n"
                  f"\tProbability Coil: {prob_coil_y_ins}\n"
                  f"\tProbability Disorder: {prob_disorder_y_ins}\n", 
                  file=logfile)


            # take average x and y parameters for RSA
            rsa_xy = [rsa_x_orig, rsa_y_orig, rsa_x_ins, rsa_y_ins]    # Here is all the RSA values to use the same cutoff
            rsa_xy_avg = np.mean(rsa_xy)                               # Or you could get an average of the RSA values
            
            rsa_x = [rsa_x_orig, rsa_x_ins]  # First position could have one cutoff
            rsa_y = [rsa_y_orig, rsa_y_ins]  # Second position could have a different cutoff
            rsa_x_avg = np.mean(rsa_x)       # Or you could use average for first position
            rsa_y_avg = np.mean(rsa_y)       # And average for second position

            

            # Get x and y positions for prob_coil (make prob_coil stay high - lower prob_coil indicates increased prob_strand or prob_helix)
            prob_coil_xy = [prob_coil_x_orig, prob_coil_y_orig, prob_coil_x_ins, prob_coil_y_ins]    # Here is all the coil values to use the same cutoff
            prob_coil_xy_avg = np.mean(prob_coil_xy)                               # Or you could get an average of the coil values
            
            prob_coil_x = [prob_coil_x_orig, prob_coil_x_ins]  # First position could have one cutoff
            prob_coil_y = [prob_coil_y_orig, prob_coil_y_ins]  # Second position could have a different cutoff
            prob_coil_x_avg = np.mean(prob_coil_x)       # Or you could use average for first position
            prob_coil_y_avg = np.mean(prob_coil_y)       # And average for second position

            # Get x and y positions for prob_disorder
            prob_disorder_xy = [prob_disorder_x_orig, prob_disorder_y_orig, prob_disorder_x_ins, prob_disorder_y_ins]    # Here is all the RSA values to use the same cutoff
            prob_disorder_xy_avg = np.mean(prob_disorder_xy)                               # Or you could get an average of the RSA values
            
            prob_disorder_x = [prob_disorder_x_orig, prob_disorder_x_ins]  # First position could have one cutoff
            prob_disorder_y = [prob_disorder_y_orig, prob_disorder_y_ins]  # Second position could have a different cutoff
            prob_disorder_x_avg = np.mean(prob_disorder_x)       # Or you could use average for first position
            prob_disorder_y_avg = np.mean(prob_disorder_y)       # And average for second position



            # Check that new cut site does not cleave th HA tag
            if second_position_new_cut_site >= y_ha_orig:
                fallback_data.append([position, prob_disorder_x_avg, cut_probability_abs, 'HA_CLEAVED'])
                print(f"Skipping position {position} due to possible cleavage of HA tag at positions {new_cutsite_positions}.", file=logfile)
                continue
            else:
                fallback_data.append([position, prob_disorder_x_avg, cut_probability_abs, 'HA_SAFE'])

            # Check if new cut site yields a sequence upstream of HA tag longer or shorter than spacer_length + 1\
            # If no sequences are found satisfying the cutoffs the regular fallback conditions will apply 
            len_ha_upstream_peptide = y_ha_orig - second_position_new_cut_site
            if position + 3 < len_ha_upstream_peptide: # position + 1 gives expected length of peptide upstream from HA tag if cleavage site didn't move
                print(f"Skipping position {position} due to cleavage site movement after HA insertion yielding long peptide upstream of HA tag: {len_ha_upstream_peptide}.", file=logfile)
                continue

            # Check the absolute cut_threshold
            if cut_probability_abs < cut_threshold_abs: # Should be set to 0.9 or 0.95
                #cut_probability_passed.append(position)
                print(f"Skipping position {position} due to cut_probability_abs {cut_probability_abs} below threshold {cut_threshold_abs}.", file=logfile)
                continue

            if cut_probability_loc < cut_threshold_loc:# Should be set to 0.7 or 0.75
                print(f"Skipping position {position} due to cut_probability_loc {cut_probability_loc} below threshold {cut_threshold_loc}.", file=logfile)
                continue


            # Skip RSA if an insert position is below 0.25
            rsa = min(rsa_xy)
            if rsa < rsa_threshold and position != 0:
                print(f"Skipping position {position} due to rsa {rsa} below threshold {rsa_threshold}.", file=logfile)
                continue

            # Skip position if prob_disorder is below 0.5 
            disorder = min(prob_disorder_xy)
            if disorder < disorder_threshold and position != 0:
                print(f"Skipping position {position} due to probability disorder {disorder} below threshold {disorder_threshold}.", file=logfile)
                continue

            # Skip position if prob_coil is below 0.5 
            coil = min(prob_coil_xy)
            if coil < coil_threshold and position != 0:
                print(f"Skipping position {position} due to probability of coil {coil} below threshold {coil_threshold}.", file=logfile)
                continue
            


            # Calculate the distance score using Gaussian function
            # Distance_score = math.exp(-(position - 4)**2 / 2) # sigma=1 gaussian function
            # stdev = 2 # standard deviation of gaussian function for gentler dropoff
            m = 4 # mean of the gaussian function. 3-5 with peak on 4 has high scores
            distance_score = math.exp(-((position - m)**2) / (2 * stdev**2))

            # Calculate the distance score using the logistic function
            #L = 1 # This sets the maximum value of the output of the function. We want it to be 1
            #x0 = 3 # This is the inflection point, is equal to 0.5 here
            #k = 1 # This controls the steepness of the curve, 2 is steeper, 1 is nearly linear
            #distance_score = L / (1 + np.exp(-k*(position-x0)))


            print(f"For position {position}, extracted values: Cut site existence probability: {cut_probability_abs}, Cut site location probability: {cut_probability_loc}", file=logfile)
            print(f"Average RSA: {rsa_xy_avg}, Average Prob Disorder: {prob_disorder_xy_avg}, Average Prob Coil: {prob_coil_xy_avg}, Distance Contribution: {distance_score}", file=logfile)

            # Weighted sum
            score = weight_coil * prob_coil_xy_avg + weight_rsa * rsa_xy_avg + weight_dis * prob_disorder_xy_avg + weight_dist * distance_score
            
            print(f"Calculated score for position {position}: {score}", file=logfile)

            # Compare the score with the best score
            if score > best_score:
                best_score = score
                best_position = position
                print(f"Updating best position to {position} with score {score}", file=logfile)


        for entry in fallback_data:
            assert isinstance(entry[2], float), f"Expected a float, but got {type(entry[2])} with value: {entry[2]}"
        positions_with_sufficient_cut_prob = [entry for entry in fallback_data if entry[2] >= cut_threshold_abs and entry[3] == 'HA_SAFE']
        safe_positions = [entry for entry in fallback_data if entry[3] == 'HA_SAFE']


        # Use fallback data
        if best_position == None:
            #positions_with_sufficient_cut_prob = [entry for entry in fallback_data if entry[2] >= cut_threshold_abs and entry[3] == 'HA_SAFE']
            #safe_positions = [entry for entry in fallback_data if entry[3] == 'HA_SAFE']
    
            if positions_with_sufficient_cut_prob:
                # Select the position with highest disorder among those that have sufficient cut probability
                best_position = max(positions_with_sufficient_cut_prob, key=lambda x: x[1])[0]
                print(f"No positions passed all thresholds; updating best position to the one with highest disorder: {best_position}", file=logfile)
    
            elif safe_positions: # We should check if there are any 'HA_SAFE' positions before trying to select one
                # Select the 'HA_SAFE' position with the highest cut site probability from all positions
                best_position = max(safe_positions, key=lambda x: x[2])[0]
                print(f"No positions passed cut site probability threshold; updating best position to the one with highest cut site probability: {best_position}", file=logfile)

            else:
                # If you want to handle cases where there are no 'HA_SAFE' positions at all, you can add an additional block here.
                print("No 'HA_SAFE' positions available.", file=logfile)



        # Add spacer position to dictionary
        positions_dict[transcript_id] = best_position

        # Add new cut site to dictionary
        #new_cleavage_sites_dict[transcript_id] = position_new_cut_site_dict[best_position]
        new_cleavage_site_output = position_new_cut_site_dict[best_position]

        # Add new HA insertion positions to dictionary
        #ha_insertion_site_dict[transcript_id] = position_ha_insert_site_dict[best_position]
        ha_insertion_site_output = position_ha_insert_site_dict[best_position]

        # For use in building new script
        output_data = [transcript_id, best_position, new_cleavage_site_output, ha_insertion_site_output]
        
        print(f"Optimal position for {transcript_id}: {best_position}", file=logfile)
        print("="*30, file=logfile)  # For clarity

        final_output_data.append(output_data)

    logfile.close()


    return final_output_data



# for logistic function distance function - iterate over k
def grid_search_with_k(path_netsurfp_data, path_preferred, resolution=0.05, k_min = 1.0, k_max = 2.0):
    best_weights = None
    best_k = None

    desired_outcomes_list = import_csv_file(path_preferred)
    desired_outcomes = {line[0]: int(line[1]) for line in desired_outcomes_list[1:]}

    closest_match = 0

    # Iterate through possible weights and k values
    for k in np.arange(k_min, k_max + resolution, resolution):
        for weight_rsa in np.arange(0, 1 + resolution, resolution):
            for weight_dis in np.arange(0, 1 - weight_rsa + resolution, resolution):
                weight_dist = 1.0 - weight_rsa - weight_dis

                assert np.isclose(weight_rsa + weight_dis + weight_dist, 1.0), "Weights do not sum to 1.0!"

                # Call the function with the new k parameter
                positions = find_insert_positions_dist_func_logistic(path_netsurfp_data, 0.8, 0.5, 0.25, 'defaultwts', 0.0, weight_rsa, weight_dis, weight_dist, k)

                # Check outcomes
                matches = sum([1 for key, val in desired_outcomes.items() if positions.get(key) == val])

                # Update best weights and k value
                if matches > closest_match:
                    closest_match = matches
                    best_weights = (0.0, weight_rsa, weight_dis, weight_dist)
                    best_k = k

    return best_weights, best_k

# For gaussian function distance score - iterate over sigma
def grid_search_with_s(path_netsurfp_data, path_preferred, resolution=0.05, s_min = 1.0, s_max = 3.0):
    best_weights = None
    best_k = None

    desired_outcomes_list = import_csv_file(path_preferred)
    desired_outcomes = {line[0]: int(line[1]) for line in desired_outcomes_list[1:]}

    closest_match = 0

    # Iterate through possible weights and k values
    for s in np.arange(s_min, s_max + resolution, resolution):
        for weight_rsa in np.arange(0, 1 + resolution, resolution):
            for weight_dis in np.arange(0, 1 - weight_rsa + resolution, resolution):
                weight_dist = 1.0 - weight_rsa - weight_dis

                assert np.isclose(weight_rsa + weight_dis + weight_dist, 1.0), "Weights do not sum to 1.0!"

                # Call the function with the new k parameter
                positions = find_insert_positions_dist_func_gaussian(path_netsurfp_data, 0.8, 0.5, 0.25, 'defaultwts', 0.0, weight_rsa, weight_dis, weight_dist, s)

                # Check outcomes
                matches = sum([1 for key, val in desired_outcomes.items() if positions.get(key) == val])

                # Update best weights and k value
                if matches > closest_match:
                    closest_match = matches
                    best_weights = (0.0, weight_rsa, weight_dis, weight_dist)
                    best_s = s

    return best_weights, best_s


if __name__ == '__main__':

    path_transcript_list = 'transcript_list.txt'
    path_signalp_data = 'master_sheet_signalP.csv'
    path_netsurfp_data = 'netsurfp_metadata.txt'

    transcript_list = import_text_file(path_transcript_list)
    signalp_dict = get_signalp_data(path_signalp_data)
    netsurfp_dict = get_netsurfp_data(path_netsurfp_data)

    # Set parameters here
    cut_threshold_abs = 0.9    # Cutite existence probability
    cut_threshold_loc = 0.75   # Cutsite position probability
    disorder_threshold = 0.5   # Probability of disorder requirement on both sides of HA site
    rsa_threshold = 0.25       # RSA on both sides of HA site required
    coil_threshold = 0.5       # Probability of coil required on both sides of HA site
    logname = 'HA_insert_search_wts_0_0_0_100_s1_v2b' # Ran a soft check on the length of the upstream peptide.
    #logname = 'HA_insert_search_wts_10_20_20_50_s1'
    weight_rsa, weight_dis, weight_coil, weight_dist = 0.0, 0.0, 0.0, 1.0 # wts_100
    #weight_rsa, weight_dis, weight_coil, weight_dist = 0.1, 0.2, 0.2, 0.5 # wts_10_20_20_50
    stdev = 1

    # Run the algorithm for HA search
    ha_insert_position_data = find_insert_positions_dist_func_gaussian(transcript_list, signalp_dict, netsurfp_dict, cut_threshold_abs, cut_threshold_loc, disorder_threshold, rsa_threshold, coil_threshold, logname, weight_rsa, weight_dis, weight_coil, weight_dist, stdev)

    # Save the results to a CSV
    print_results(ha_insert_position_data, logname)
    