"""Centralized configuration for the AlphaFlex-IDPForge pipeline (Steps 1-4)."""

import os
import sys

# =============================================================================
# LOGGING CONTROL
# =============================================================================
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
SUBSET_OUTPUT_NAME = "test_subset"     # base filename

# Basic filter
SUBSET_MIN_LENGTH = 0
SUBSET_MAX_LENGTH = 250

# Advanced filters
SUBSET_TAIL_COUNT = 2                  # Tail count
SUBSET_LINKER_COUNT = 1                # Linker count
SUBSET_LOOP_COUNT = 1                  # Loop count
SUBSET_EXACT_COUNT = True
SUBSET_IDR_MIN_LENGTH = None
SUBSET_IDR_MAX_LENGTH = None
SUBSET_MAX_SAMPLES = None

# =============================================================================
# STEP 2: TEMPLATE GENERATION
# =============================================================================
TEMPLATE_OUTPUT_DIR = os.path.join(PIPELINE_OUTPUT_ROOT, "Step_2_Templates")
IDP_CASES_LIST_PATH = os.path.join(TEMPLATE_OUTPUT_DIR, "idp_cases_to_run.json")

CURRENT_BATCH_FOLDER = "custom_subsets"
ID_LISTS_DIR = os.path.join(ID_LISTS_OUTPUT_ROOT, CURRENT_BATCH_FOLDER)

RUN_ID_FILE = os.path.join(ID_LISTS_DIR, SUBSET_OUTPUT_NAME + ".txt")

SCRIPT_STATIC_TEMPLATE = os.path.join(PARENT_DIR, "mk_ldr_template.py")
SCRIPT_FLEX_TEMPLATE = os.path.join(PARENT_DIR, "mk_flex_template.py")

TEMPLATE_N_CONFS = 200
TEMPLATE_SEED_SKEW = 0.5               # seed skew
TEMPLATE_MAX_RESIDUES = None           # graft-mode size cap
TEMPLATE_FOLD_PER_SIDE = 10            # fold residues per junction
TIMEOUT_STATIC_TEMPLATE = 60           # seconds
TIMEOUT_DYNAMIC_TEMPLATE = 1000        # seconds

TRUNCATE_TO_ADJACENT = False

# =============================================================================
# STEP 3: CONFORMER GENERATION
# =============================================================================
CONFORMER_POOL_DIR = os.path.join(PIPELINE_OUTPUT_ROOT, "Step_3_Raw_Conformers")

# Generation
SAMPLE_N_CONFS = 10                    # target per IDR
SAMPLE_BATCH_SIZE = 6                  # batch size
SAMPLE_MAX_TOTAL_ATTEMPTS = 500        # max attempts
DEVICE = "cuda"                        # "cuda" or "cpu"

# Curvature gates
SAMPLE_FOLD_CURV_RATIO = 0.5           # fold curvature gate
SAMPLE_FOLD_CURV_WINDOW = 15           # fold window
SAMPLE_JUNCTION_KAPPA = 0.12           # junction gate (A^-1)

# Model paths
SCRIPT_SAMPLE_LDR = os.path.join(PARENT_DIR, "sample_ldr.py")
SCRIPT_SAMPLE_IDP = os.path.join(PARENT_DIR, "sample_idp.py")
MODEL_WEIGHTS_PATH = os.path.join(PARENT_DIR, "weights", "mdl.ckpt")
MODEL_CONFIG_PATH = os.path.join(PARENT_DIR, "configs", "sample.yml")
SS_DB_PATH = os.path.join(PARENT_DIR, "data", "example_data.pkl")

# =============================================================================
# STEP 4: STITCHING & RELAXATION
# =============================================================================
STITCH_OUTPUT_ROOT = os.path.join(PIPELINE_OUTPUT_ROOT, "Step_4_Final_Models")

# Stitching
STITCH_N_CONFORMERS = 10               # target per protein
STITCH_MAX_ATTEMPTS = 500              # max attempts
STITCH_FOLD_CURV_RATIO = 0.5           # fold curvature gate
STITCH_FOLD_CURV_WINDOW = 15           # fold window

# AMBER relaxation
RELAX_STIFFNESS = 10.0                 # restraint strength
RELAX_MAX_OUTER_ITER = 20              # outer iterations
MINIMIZATION_MAX_ITER = 0              # L-BFGS iterations
MINIMIZATION_TOLERANCE = 10.0          # tolerance (kJ/mol/nm)

# Alignment geometry
ALIGNMENT_STUB_HALF_SIZE = 5           # stub half-window
ALIGNMENT_JUNCTION_SIZE = 5            # fallback stub size
MIN_CONFORMER_POOL_SIZE = 5            # pool-size warning

# Adaptive clash scoring
STITCH_BASE_CLASH_THRESHOLD = 10.0     # base threshold
STITCH_CLASH_INCREMENT = 5.0           # escalation step
