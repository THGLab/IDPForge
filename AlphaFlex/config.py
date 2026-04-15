"""
Centralized Configuration for the AlphaFlex-IDPForge Pipeline (Steps 1-4).

Author: SDC, 12/18/2025
"""

import os
import sys

# =============================================================================
# LOGGING CONTROL
# =============================================================================
# True  = Print every step (debugging / small batches)
# False = Progress bar only (HPC / large batches)
VERBOSE = True

# =============================================================================
# GLOBAL PATHS
# =============================================================================
PYTHON_EXEC = sys.executable
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(PROJECT_ROOT)

INPUT_DATA_DIR = os.path.join(PROJECT_ROOT, "Data_Inputs")
MASTER_DB_PATH = os.path.join(INPUT_DATA_DIR, "AlphaFlex_database_Nov2025.json")
LENGTH_REF_PATH = os.path.join(INPUT_DATA_DIR, "AF2_9606_HUMAN_v4_num_residues.json")
PDB_LIBRARY_PATH = os.path.join(INPUT_DATA_DIR, "Test_Structures")
KNOT_SCREENING_PATH = os.path.join(INPUT_DATA_DIR, "knot_screening.json")

PIPELINE_OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "Pipeline_Outputs")

# =============================================================================
# STEP 1: CASE LABELING
# =============================================================================
STEP_1_DIR = os.path.join(PIPELINE_OUTPUT_ROOT, "Step_1_Labeling")
LABELED_DB_PATH = os.path.join(STEP_1_DIR, "Labeled_AlphaFlex_database_Nov2025.json")
SUMMARY_TEXT_PATH = os.path.join(STEP_1_DIR, "idr_type_summary.txt")

# =============================================================================
# STEP 1B: SUBSET FILTERING
# =============================================================================
ID_LISTS_OUTPUT_ROOT = STEP_1_DIR
SUBSET_OUTPUT_NAME = "test_subset"     # base filename for output files

# Basic filter (always active)
SUBSET_MIN_LENGTH = 0
SUBSET_MAX_LENGTH = 250

# Advanced filters (optional: None = inactive / unconstrained)
SUBSET_TAIL_COUNT = 2                  # required Tail IDR count (None = unconstrained)
SUBSET_LINKER_COUNT = 1                # required Linker IDR count (None = unconstrained)
SUBSET_LOOP_COUNT = 1                  # required Loop IDR count (None = unconstrained)
SUBSET_EXACT_COUNT = True              # True = exact match, False = minimum match
SUBSET_IDR_MIN_LENGTH = None           # min IDR length applied to all IDRs (None = no filter)
SUBSET_IDR_MAX_LENGTH = None           # max IDR length applied to all IDRs (None = no filter)
SUBSET_MAX_SAMPLES = None              # None = no cap; integer = random subsample

# =============================================================================
# STEP 2: TEMPLATE GENERATION
# =============================================================================
TEMPLATE_OUTPUT_DIR = os.path.join(PIPELINE_OUTPUT_ROOT, "Step_2_Templates")
IDP_CASES_LIST_PATH = os.path.join(TEMPLATE_OUTPUT_DIR, "idp_cases_to_run.json")

CURRENT_BATCH_FOLDER = "custom_subsets"
ID_LISTS_DIR = os.path.join(ID_LISTS_OUTPUT_ROOT, CURRENT_BATCH_FOLDER)

SCRIPT_STATIC_TEMPLATE = os.path.join(PARENT_DIR, "mk_ldr_template.py")
SCRIPT_FLEX_TEMPLATE = os.path.join(PARENT_DIR, "mk_flex_template.py")

TEMPLATE_N_CONFS = 200
TIMEOUT_STATIC_TEMPLATE = 60           # seconds
TIMEOUT_DYNAMIC_TEMPLATE = 1000        # seconds

# =============================================================================
# STEP 3: CONFORMER GENERATION
# =============================================================================
CONFORMER_POOL_DIR = os.path.join(PIPELINE_OUTPUT_ROOT, "Step_3_Raw_Conformers")

# Generation
SAMPLE_N_CONFS = 10                    # target validated conformers per IDR
SAMPLE_BATCH_SIZE = 6                  # diffusion batch size per generation round
SAMPLE_MAX_TOTAL_ATTEMPTS = 500        # max validation attempts before giving up
DEVICE = "cuda"                        # "cuda" or "cpu"

# Model paths
SCRIPT_SAMPLE_LDR = os.path.join(PARENT_DIR, "sample_ldr.py")
MODEL_WEIGHTS_PATH = os.path.join(PARENT_DIR, "weights", "mdl.ckpt")
MODEL_CONFIG_PATH = os.path.join(PARENT_DIR, "configs", "sample.yml")
SS_DB_PATH = os.path.join(PARENT_DIR, "data", "example_data.pkl")

# =============================================================================
# STEP 4: STITCHING & RELAXATION
# =============================================================================
STITCH_OUTPUT_ROOT = os.path.join(PIPELINE_OUTPUT_ROOT, "Step_4_Final_Models")

# Stitching
STITCH_N_CONFORMERS = 10               # target ensemble conformers per protein
STITCH_MAX_ATTEMPTS = 500              # max stitching attempts before giving up

# AMBER relaxation (folded domains restrained, IDRs free)
RELAX_STIFFNESS = 10.0                 # harmonic restraint strength on folded residues
RELAX_MAX_OUTER_ITER = 20              # outer iteration limit for relaxation
MINIMIZATION_MAX_ITER = 0              # L-BFGS iterations (0 = OpenMM default)
MINIMIZATION_TOLERANCE = 10.0          # energy convergence tolerance (kJ/mol/nm)

# Alignment / stitching geometry
ALIGNMENT_STUB_HALF_SIZE = 5           # half-window (residues) around junction midpoint for alignment
ALIGNMENT_JUNCTION_SIZE = 5            # fallback junction stub size if stub window is too small
MIN_CONFORMER_POOL_SIZE = 5            # warn if fewer conformers per IDR (stitching still proceeds)

# Adaptive clash scoring
STITCH_BASE_CLASH_THRESHOLD = 10.0     # starting clash score threshold
STITCH_CLASH_INCREMENT = 5.0           # threshold increase per escalation step
