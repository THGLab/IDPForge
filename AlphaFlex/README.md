# AlphaFlex (AFX-IDPForge) Protocols

Scripts for building AFX-IDPForge ensembles are labeled in order (Step 1 - Step 4).

AFX-IDPForge Python scripts are extensions of the scripts provided by the base installation of IDPForge, and as such must be run within the `IDPForge` Python environment created during its installation. See the base `README.md` file for installation instructions.

Alongside the scripts, the `Data_Inputs` directory contains the following resources for testing:

1. `AlphaFlex_database_Nov2025.json` is a JSON dictionary database where each key is a UniProt ID and the values contain IDR boundaries (`idrs`), mean PAEs between folded (`F`) and disordered (`D`) regions (`mean_pae`), and interactions between folded domains where the mean PAE is less than 15 Angstroms (`interactions`).

2. `AF2_9606_HUMAN_v4_num_residues.json` is a database that contains the total length of the amino-acid residues from the AlphaFold2 9606 Human v4 database.

3. `Test_Structures/O14653.pdb` is a sample Category 3 case that can be run as a test for each step.

4. `knot_screening.json` is a JSON dictionary confirming the knot presence/absence (`label`) and native knot type (`closure_polys`) for each protein containing a folded domain. This was pre-calculated using the predictions for all proteins containing one or more folded domains in the AlphaFold2 9606 Human v4 database as they are used as templates during conformer generation. Topology checks during structure validation aim to match the native knot type of the protein being generated, defaulting to `Unknot` in the event that one is not found.

# Running AFX-IDPForge Scripts

For testing purposes, a Category 3 protein (`ID: O14653`) is chosen.

Each script is designed to take in the outputs of the previous script, beginning with `Step_1_case_label.py`. To test the AFX-IDPForge pipeline, input the **Sample usage** command given at the bottom of each script description.


Descriptions of each file are listed below along with their usage commands. See each file for more specific details.

## `config.py`

Contains the parameters associated with each step script. Most are configured for the purposes of testing the pipeline, but some notable adjustments can be made to the following:

1. `VERBOSE` (Show more console outputs for debugging)
2. `SUBSET_( MIN / MAX )_LENGTH` (Protein length range for Step 1B)
3. `SUBSET_OUTPUT_NAME` (Base filename for Step 1B output files, default: `test_subset`)
4. `SUBSET_TAIL_COUNT` (Required Tail IDR count for Step 1B; None = unconstrained)
5. `SUBSET_LINKER_COUNT` (Required Linker IDR count for Step 1B; None = unconstrained)
6. `SUBSET_LOOP_COUNT` (Required Loop IDR count for Step 1B; None = unconstrained)
7. `SUBSET_EXACT_COUNT` (True = exact count matching, False = minimum; Step 1B)
8. `SUBSET_IDR_( MIN / MAX )_LENGTH` (IDR length range applied to all IDRs; None = no filter; Step 1B)
9. `SUBSET_MAX_SAMPLES` (Cap on output proteins for Step 1B; None = no cap)
10. `TEMPLATE_N_CONFS` (Number of template conformations generated in Step 2)
11. `SAMPLE_N_CONFS` (Number of output conformers of Step 3)
12. `SAMPLE_BATCH_SIZE` (Diffusion batch size for Step 3)
13. `SAMPLE_MAX_TOTAL_ATTEMPTS` (Maximum generation attempts per IDR in Step 3)
14. `STITCH_N_CONFORMERS` (Number of output stitched/minimized conformers of Step 4)
15. `STITCH_MAX_ATTEMPTS` (Maximum stitching attempts per protein in Step 4)
16. `CURRENT_BATCH_FOLDER` (Which subfolder under `Step_1_Labeling` Step 2 reads from, default: `custom_subsets`)

See `config.py` for more details on each tunable parameter.

## Step 1: Case Label `(Output Directory: Step_1_Labeling)`

Augments the `AlphaFlex_database_Nov2025.json` file with additional information (labels) for each IDR. The following information is added:

* Range: Residue number range of the given domain.
* Type: Tail/Linker/Loop based on adjacent folded domain interactions. See `https://doi.org/10.1101/2025.11.24.690279` for more details on IDP/IDR categorization.
* Label: `Dx` where `x` is the ordinality of the domain (D1 is the 1st disordered domain in the protein).
* Flanking Domains: Folded domains adjacent to the given domain.

Using this information, an `idr_type_summary.txt` file will be given that outlines the distribution of each protein into AlphaFlex defined categories.

Optional arguments: `--input_db`, `--length_ref`, `--output_dir`, `--verbose`

Sample usage: `python Step_1_case_label.py`

## Step 1B: Subset Filtering `(Output Directory: Step_1_Labeling)`

Creates a filtered list of protein IDs and writes it to `custom_subsets/`. The total protein length range filter is always active as a baseline subset filter. Advanced per-IDR filters activate when any of `--tail_count`, `--linker_count`, `--loop_count`, `--idr_min_len`, or `--idr_max_len` are specified. For the purposes of testing the pipeline, some of these advanced filters are activated. See `config.py` for more details.

**Basic mode** (length-only): Filters proteins by total residue count. All matching proteins are included regardless of IDR composition.

**Advanced mode** (when any IDR filter is specified): Additionally filters by per-type IDR counts and/or IDR length range. Counts are matched exactly by default; pass `--min_mode` for minimum-count matching. The IDR length range is applied to **all** IDRs in a protein, so every IDR must fall within the range for the protein to pass. When advanced filters are active, an IDR length histogram and histogram data table are also generated.

Output is written to `custom_subsets/` using the base name from `--output_name` (default: `test_subset`):
* `<output_name>.txt` — Filtered UniProt ID list
* `<output_name>_report.txt` — Per-protein report table with type counts and IDR details
* `<output_name>_histogram.png` — IDR length distribution (advanced mode only)
* `<output_name>_histogram_table.txt` — Histogram bin counts (advanced mode only)

Optional arguments: `--labeled_db`, `--length_ref`, `--output_root`, `--output_name`, `--min_len`, `--max_len`, `--tail_count`, `--linker_count`, `--loop_count`, `--exact` / `--min_mode` (mutually exclusive; `--min_mode` overrides `--exact` if both are passed), `--idr_min_len`, `--idr_max_len`, `--max_samples`, `--verbose`

Sample usage: `python Step_1B_subset_label.py`

## Step 2: Template Creation `(Output Directory: Step_2_Templates)`

Creates a template for each individual IDR of proteins listed in the `.txt` files within the Step 1B output directory. By default, reads from `Step_1_Labeling/custom_subsets/` (configurable via `CURRENT_BATCH_FOLDER` in `config.py`). Proteins are skipped if their corresponding PDB file is not found within the default `Test_structures` directory or if a template already exists for that IDR in the output directory. Supports resumable execution via `Step_2_progress.txt`.

Tails and Loops have templates made by `mk_ldr_template.py` which keeps all regions outside of the specified IDR frozen.

Linkers have templates made by `mk_flex_template.py` which designates the 2 adjacent folded domains as separate objects and randomly shifts them within a certain distance of one another (mimicking the flexibility of non-interacting folded domains).

Category 0 (full IDP) cases are logged to `idp_cases_to_run.json` for separate processing via `sample_idp.py`.

Optional arguments: `--start-index`, `--labeled_db`, `--id_lists_dir`, `--pdb_library`, `--output_dir`, `--n_confs`, `--timeout_static`, `--timeout_dynamic`, `--verbose`

Sample usage: `python Step_2_mk_ldr_template.py`

## Step 3: Conformer Generation `(Output Directory: Step_3_Raw_Conformers)`

Generates `X` validated conformers (Default `X` = 10) for each IDR template created in Step 2. Each IDR region is diffused upon individually, so proteins with multiple IDRs produce separate per-IDR conformer pools.

The pipeline loops through three phases until the target is met or max attempts are exhausted:

1. **Generate + Relax**: Calls `sample_ldr.py` to produce diffusion conformers, which are immediately relaxed via AMBER minimization (relax config loaded from `configs/sample.yml`).
2. **Repair**: Checks each relaxed structure for D-amino acids (chirality) and broken HIS ring bonds. Applies fixes and re-relaxes if any repairs were made.
3. **Validate**: Runs unified validation checking chirality, bond integrity, clash score (adaptive smart threshold), and backbone topology (knot detection). Passing structures are renamed to `N_validated.pdb`.

State is persisted to `.step3_state.json` after each round, enabling recovery from cluster preemption. Orphaned relaxed files from killed runs are automatically recovered on restart.

Supports HPC parallel execution via `--total_splits` and `--split_index` for deterministic sharding across jobs.

Required argument: `--id_file`
Optional arguments: `--total_splits`, `--split_index`, `--template_dir`, `--output_dir`, `--n_confs`, `--max_attempts`, `--batch_size`, `--device`, `--weights`, `--model_config`, `--ss_db`, `--verbose`

Sample usage: `python Step_3_sample_conformer.py --id_file Pipeline_Outputs/Step_1_Labeling/custom_subsets/test_subset.txt`

## Step 4: Stitching and Minimization `(Output Directory: Step_4_Final_Models)`

Assembles full-length protein models from the per-IDR conformer pools generated in Step 3. Behavior depends on the number of IDRs:

**Single IDR / Full IDP (Fast-Pass):** Collects all validated conformers from Step 3 and combines them directly into a multi-model ensemble PDB. No stitching or additional minimization is performed.

**Multiple IDRs (Monte Carlo Stitching):** Runs a stochastic assembly loop that repeats until the target conformer count is reached or max attempts are exhausted:

1. Determine the number and location of IDRs present in the protein.
2. Start with the first folded domain of the protein (F1) from the AlphaFold2 predicted structure.
3. Identify the first disordered domain of the protein (D1), and randomly sample a conformer from the previously generated ensemble.
4. Align the middle 11 F1 residues of the AlphaFold2 predicted and IDPForge predicted structures.
5. Overwrite the original structure with the IDPForge predicted structure from those 11 residues onwards. Repeat steps 1-5 until all folded and disordered domains are stitched into a full structure. 
6. Pre-relax repair: check and flip any D-amino acids in the stitched structure.
7. Minimize the fully stitched structure via AMBER (ff14SB) with harmonic restraints on folded domains, allowing IDRs and junction regions to relax freely. Discard if relaxation fails.
8. Post-relax repair: check for D-amino acids and broken HIS ring bonds introduced during relaxation. Apply fixes and re-relax if any repairs were made. Discard if re-relax fails.
9. Validate via unified checks: chirality, bond integrity, clash score (adaptive smart threshold), and backbone topology (knot detection). Discard any structures that fail.
10. Combine all valid conformers into a single multi-model ensemble PDB.

Output is organized as `Step_4_Final_Models/<Category>/<Length_Label>/<protein_id>/<protein_id>_ensemble_n<N>.pdb`.

Supports HPC parallel execution via `--total_splits` and `--split_index` for deterministic sharding, and `--workers` for local parallel processing.

Required argument: `--id_file`
Optional arguments: `--total_splits`, `--split_index`, `--workers`, `--labeled_db`, `--length_ref`, `--conformer_dir`, `--output_dir`, `--n_conformers`, `--max_attempts`, `--verbose`

Sample usage: `python Step_4_ldr_stitch.py --id_file Pipeline_Outputs/Step_1_Labeling/custom_subsets/test_subset.txt`

# Resources
* AlphaFlex Manuscript (pre-print): https://doi.org/10.1101/2025.11.24.690279
* AlphaFlex Zenodo Repository: https://zenodo.org/records/17684898
