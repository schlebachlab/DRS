# DRS
Scripts developed for building and curating the deep receptor scanning library by the Schlebach Lab

The GPCR gene library was constructed by implementing a Python script to process and align genomic and proteomic data. The raw sequence data were fetched from the Ensembl and Uniprot databases, and pertinent information was extracted through data parsing. An alignment process was performed, matching protein sequences from Uniprot with corresponding nucleotide sequences from Ensembl, utilizing gene names as identifiers. The protocol accounted for alternate isoforms, effectively treating each isoform as a unique entry.

To facilitate downstream experimental tracking, a unique barcode was generated for each sequence entry, ensuring that no barcode sequence contained any restriction enzyme sites. To maintain compatibility with molecular biology protocols, the script specifically identified and removed BsmBI restriction sites from the sequences.

The final step involved the output of processed data, which was then categorized into three distinct groups: the primary output of aligned and barcoded sequences, Uniprot entries not found in Ensembl, and sequences exclusive to Ensembl. This information was exported in CSV format to enable further analysis, and sequence data was also formatted into FASTA files for potential utilization in other bioinformatics tools.

Installation of environment:

Please use provided Conda environment.

conda env create -f gpcr_env.yml

Scripts were run on an Intel workstation running Almalinux 10. Each step required only a few seconds of CPU time.

Description of library assembly

gpcr_seq_routines_v7_bsmbi_filter.py

The script parses protein sequences from Uniprot’s database of human GPCRs, DNA sequences from the Ensembl CDS database, and gene symbols and gene names from the HGNC database of GPCR entries, translates transcript sequences into protein sequences and matches them to protein sequences from Uniprot and gene symbols from HGNC. This was done since we have no definitive nucleic acid sequence directly available for most Uniprot entries. The script also takes a list of Ensembl transcript IDs from GPCRdb for alternative isoforms of GPCRs and adds their nucleic acid sequences to the library if a ‘canonical’ transcript is also available. The script searches all the DNA sequences obtained for common restriction sites and outputs a table of GPCR sequences and the restriction sites they contain. The script is designed to assemble wild-type sequences, but does make silent mutations (preserving codon usage bias as much as possible) to remove BsmbI sites from the DNA sequences. Finally, the script outputs a CSV files containing information about the hits obtained. It outputs a separate CSV containing GPCR DNA sequences which had information in HGNC’s database and Ensembl’s CDS library but not in Uniprot, in case we want to add those to the library at a later date. Finally, the script outputs FASTA format files of the DNA sequences in their native forms, a separate file with the DNA sequences with silent mutations to remove BsmbI sites, and a FASTA format file with the DNA sequences of the library translated into protein sequences. Finally, the script generates unique molecular identifiers (UMIs) for sequencing the library. UMIs were designed to have a length of ten bases and were designed to be non-redundant by setting a minimum Hamming distance of 4, were designed to have a maximum homopolymer length of 2 and restricted to having a GC content between 35% and 65%. UMIs containing 135 common restriction sites were filtered, as were UMIs containing the sequence ‘ATG’. The set of all possible 10mers were generated, randomly shuffled, and filtered using the criteria listed above using a custom Python script. Once UMIs were generated they were associated with each DNA sequence in the library. Gene blocks for synthesis were assembled using a separate script which added 3’ adapters and 5’ components including the attB recombination site, UMI, and HA tag. The script identified 991 gene blocks (792 canonical GPCR sequences and 199 alternative isoform transcripts). We ran the protein sequence translated version of the library through TOPCONS (https://topcons.cbr.su.se/) to search for signal peptides. TOPCONS identified 137 sequences with signal peptides (83 canonical sequences and 54 alternative isoforms). These were eliminated from the library. We eliminated any remaining sequence if the original CDS (that is, not including the 5’ and 3’ handles) had a length greater than 3000 bases. We noticed some adhesion GPCRs remained in the library: (ADGRG2, ADGRA1, ADGRB3, ADGRG7, ADGRD1). Since these are well-known to contain signal peptides, we checked the literature and a separate databases such as nextprot.org, uniport, the actual shape of predicted TMDs in TOPCONS (sometimes the first one is actually a signal peptide) and the human protein atlas. After these manual checks, we determined that all of them except ADGRA1 likely had signal peptides and that these were mostly false negatives. For ADGRA1, we could not find evidence of a signal peptide and this entry remained in the library. After one last manual check to determine that all the alternative isoforms could be compared to a canonical sequence, we were left with 701 canonical GPCR sequences and 133 alternative isoform transcripts.

Input files:

Ensembl database
Homo_sapiens.GRCh38.cds.all.fa
This is a database of CDS transcripts from Ensembl for Homo sapiens. Downloaded from:
http://www.ensembl.org/info/data/ftp/index.html

CDS (Fasta format). Last modified  2023-04-22 04:25

Fasta-format sequences from Uniprot
uniprot_gpcrs.fasta
This is a database of all the protein sequences for GPCRs from humans. You can obtain this using the keyword for GPCRs (KW-0297). There are 3135 entries with that keyword in all of uniprot and 835 from humans.

https://www.uniprot.org/uniprotkb?dir=descend&facets=reviewed%3Atrue%2Cmodel_organism%3A9606&query=%28keyword%3AKW-0297%29&sort=organism_name

Once the table is displayed you can download it as a FASTA file.

List of alternate isoform transcripts from GPCRdb
isoform_transcripts_list.txt
These were obtained from GPCRdb. The file is simply a list of Ensembl transcript IDs in the ‘isoforms’ section.
https://gpcrdb.org/protein/isoforms

These were ultimately curated by Maria Marti-Solano, M. Madan Babu, and colleagues:
https://www.nature.com/articles/s41586-020-2888-2

The script uses these Ensembl IDs to fetch sequences from the CDS database and add them to the library if the canonical isoform is also present after various checks and filters. Note: I made the choice that a sequence must begin with an ATG codon to be included. If the 3’ end lacked a complete codon, the last 1-2 bases were truncated and a stop codon was added. We are, of course, flexible on these choices.

All the GPCRs from the GeneNames database.
https://www.genenames.org/data/genegroup/#!/group/139

These are organized into the different classes of GPCRs (rhodopsin-like, secretin, etc). I downloaded the text files (tsv format) and concatenated them together into one big file with the GPCR family name added as a new column, creating a new CSV:
GPCRs_genenames.csv
This includes HGNC symbol, HGNC name, and ensemble gene name. There were 1416 entries. I broke these up into these two somewhat redundant files:

gene_name_data.csv (this has all the data in a more simplified format). I use this to find gene names, GPCR classes, etc, if there is a corresponding uniprot sequence.
gene_names_all_ensembl.csv (this has only the data that has an esembl ID – I iterate through this list in a separate function in the script to find Ensembl gene IDs for GPCRs not in Uniprot).

List of common restriction sites
restriction_enzymes.csv

This is a list of the 135 common restriction sites. The script reads these in the format shown and creates regular expressions out of them.

gpcr_build_geneblocks_from_final_output_gly_labels.py

Inserts the gene blocks into our complete vector.

find_HA_insert_positions_v2b.py

Searches for optimal places to insert HA tags using predictited structural features from NetsurfP and SignalP

This script is designed to find the optimal positions for inserting an HA tag into a set of proteins. The HA tag is a short peptide sequence derived from the influenza hemagglutinin protein, often used in biochemistry and molecular biology to tag a protein of interest. The script attempts to choose insertion sites such that the HA tag’s insertion neither disrupts the protein's function nor overlaps with the signal peptide cleavage site, maximizing its accessibility for detection. The script imports and processes data from SignalP and NetSurfP predictions performed on sequences with and without the HA tags inserted at a series of sites downstream from the cleavage site, assesses potential HA tag insertion sites based on various predicted biophysical parameters, scores each site, and selects the optimal position for each protein. The results, along with detailed logs, are saved for further analysis. Adjustments to scoring weights and parameters can be made to refine the selection process.



