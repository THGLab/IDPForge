# AlphaFlex (AFX-IDPForge) Protocols

Scripts for building AFX-IDPForge ensembles are labeled in order (Step 1 - Step 4). 

AFX-IDPForge Python scripts are extensions of the scripts provided by the base installation of IDPForge, and as such must be run within the `IDPForge` Python environment created during its installation. See the base `README.md` file for installation instructions.

Alongside the scripts in the `Data_Inputs` directory are 3 files for testing:

1. `AlphaFlex_database_Jul2024` is standard JSON dictionary database where each key is a UniProt ID and the values correspond to IDR boundaries (`idrs`), mean PAEs between folded (`F`) and disordered (`D`) regions (`mean_pae`), and any interactions between folded domains where the mean PAE is less than 15 Angstroms is documented in interactions.

2. `AF2_9606_HUMAN_v4_num_residues.json` is a database that contains the total length of the amino-acid residues from the AlphaFold2 9606 Human v4 database.

3. `Test_Structures/O14653.pdb` is a sample Category 3 case that can be run a test for each step.

# Running AFX-IDPForge Scripts

For testing purposes, a Category 3 protein (`ID: O14653`) is chosen. 

Each script is designed to take in the outputs of the previous script, beginning with `Step_1_case_label.py`. High level descriptions of each file are listed below along with their usage commands. See each file for more specific details.

## `config.py`

Contains the parameters associated with each step script. Most are configured for the purposes of testing the script, but some notable adjustments can be made to the following:

1. `VERBOSE` (Show more console outputs for debugging)
2. `SUBSET_( MIN / MAX )_LENGTH` (Adjust residue size bins)
3. `SAMPLE_N_CONFS` (Number of output conformers of Step 3)
4. `STITCH_N_CONFORMERS`(Number of output stitched/minimized conformers of Step 4)

See `config.py` for more details on each tunable parameter.

## Step 1: Case Label `(Output Directory: Step_1_Labeling)`

Augments the `AlphaFlex_database_Jul2024.json` file with additional information (labels) for each IDR. The following information is added

* Range: Residue number range of the given domain.
* Type: Tail/Linker/Loop based on adjacent folded domain interactions. See `https://doi.org/10.1101/2025.11.24.690279` for more details on IDP/IDR categorization.
* Label: `Dx` where `x` is the ordinality of the domain (D1 is the 1st disordered domain in the protein).
* Flanking Domains: Folded domains adjacent to the given domain.

Using this information, an `idr_type_summary.txt` file will be given that outlines the distribution of each protein into AlphaFlex defined categories.

Usage: `python Step_1_case_label.py`

## Step 1B: Subset Label `(Output Directory: Step_1_Labeling)`

Creates an `id_lists` directory containing a list of proteins in each category within a specified length subset. The lists will contain the IDs of each protein for subsequent processing.

Usage: `python Step_1B_subset_label.py`

## Step 2: Template Creation `(Output Directory: Step_2_Templates)`

Creates a template for each individual IDR of proteins listed in the `.txt` files within `id_lists`. Proteins are skipped if its corresponding PDB file is not found within the `Data_Inputs/Test_Structures` directory or if a template is already found for that IDR in the corresponding `Step_2_Templates` output directory. 

Tails and Loops have templates made by `mk_ldr_template.py` which keeps all regions outside of the specified IDR frozen. 

Linkers have templates made by `mk_flex_template.py` which designates the 2 adjacent folded domains as separate objects and randomly shifts them within a certain distance of one another (mimicking the flexibility of non-interacting folded domains).

Usage: `python Step_2_mk_ldr_template.py`

## Step 3: IDR Conformer Generation `(Output Directory: Step_3_Raw_Conformers)`

Generates `X` conformers (Default `X` = 10) for each previously created template. Each IDR region is diffused upon individually, so proteins containing multiple IDRs will have multiple templates, leading to the creation of individual 10-conformer ensembles unique to each IDR. Conformers are minimized with their folded domains restrained to minimize structural shifts.

Generation of IDR conformers is done through the use of `sample_ldr.py`. 

Usage: `python Step_3_sample_conformer.py`.

## Step 4: Stitching and Minimization `(Output Directory: Step_4_Final_Models)`

Determines whether a protein has 1 or multiple IDRs in its chain. 

If a protein has multiple IDRs, it proceeds with the following workflow:

1. Determine the number and location of IDRs present in the protein.
2. Start with the first folded domain of the protein (F1) from the AlphaFold2 predicted structure.
3. Identify the first disordered domain of the protein (D1), and randomly sample a conformer from the previously
generated ensemble.
4. Align the middle 11 F1 residues of the AlphaFold2 predicted and IDPForge predicted structures.
5. Overwrite the original structure with the IDPForge predicted structure from those 11 residues onwards.
6. Calculate the radius of gyration (Rg) of the protein and reject any conformers that have backbones piercing
through 0.6 Rg.
7. Find and manually flip any D-amino acids in the structure.
8. Minimize the fully stitched structure.
9. Reject any structures with more than 5 major clashes per 1000 atoms (clashscore = 5).
10. Reject any structures whos longest folded domain has an RMSD higher than 1.5 A when aligned with the
AlphaFold2 structure.
11. Reject any structures with bond and chirality issues.

If a protein has only 1 IDR, it goes straight towards the minimization step (Step 8).

Usage: `python Step_4_ldr_stitch.py --id_file Pipeline_Outputs/Step_1_Labeling/id_lists/0-250AA/cat_3.txt`

# Resources
* AlphaFlex Manuscript (pre-print): https://doi.org/10.1101/2025.11.24.690279
* AlphaFlex Zenodo Repository: https://zenodo.org/records/17684898