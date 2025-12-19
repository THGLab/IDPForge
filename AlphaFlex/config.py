"""
Centralized Configuration for the AlphaFlex-IDPForge Pipeline (Steps 1-4).

Author: SDC, 12/18/2025
"""

import os
import sys

# =============================================================================
# LOGGING CONTROL
# =============================================================================
# True  = Print every step (Good for debugging or small batches)
# False = Only print errors and progress summaries every 100 items
VERBOSE = True

# =============================================================================
# GLOBAL PATHS
# =============================================================================

# --- External Scripts Configuration ---
PYTHON_EXEC = sys.executable 

# Current Project Root Directory (Where this config file lives)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Define the Parent Directory (One level up from where this config file lives)
PARENT_DIR = os.path.dirname(PROJECT_ROOT)

# Input Data Directory (Where Step 0 outputs live)
INPUT_DATA_DIR = os.path.join(PROJECT_ROOT, "Data_Inputs")
MASTER_DB_PATH = os.path.join(INPUT_DATA_DIR, "AlphaFlex_database_Jul2024.json")
LENGTH_REF_PATH = os.path.join(INPUT_DATA_DIR, "AF2_9606_HUMAN_v4_num_residues.json")
PDB_LIBRARY_PATH = os.path.join(INPUT_DATA_DIR, "Test_Structures")

# Main Output Directory (All steps will write subfolders here)
PIPELINE_OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "Pipeline_Outputs")

# =============================================================================
# STEP 1: CASE LABELING SETTINGS
# =============================================================================

# Directory for Step 1 results
STEP_1_DIR = os.path.join(PIPELINE_OUTPUT_ROOT, "Step_1_Labeling")

# The crucial JSON that drives Steps 2-4
LABELED_DB_PATH = os.path.join(STEP_1_DIR, "Labeled_AlphaFlex_database_Jul2024.json")
SUMMARY_TEXT_PATH = os.path.join(STEP_1_DIR, "idr_type_summary.txt")

# =============================================================================
# STEP 1B: SUBSET SAMPLING SETTINGS
# =============================================================================

# Directory to save the generated ID lists
ID_LISTS_OUTPUT_ROOT = os.path.join(STEP_1_DIR, "id_lists") 

# Filtering Criteria
SUBSET_MIN_LENGTH = 0        # Lower bound of protein size (inclusive)
SUBSET_MAX_LENGTH = 250 # Upper bound of protein size (inclusive)
SUBSET_SAMPLE_SIZE = 100000  # Max proteins to pick per category

# =============================================================================
# STEP 2: TEMPLATE GENERATION SETTINGS
# =============================================================================

# Directory where .npz templates will be saved
TEMPLATE_OUTPUT_DIR = os.path.join(PIPELINE_OUTPUT_ROOT, "Step_2_Templates")
IDP_CASES_LIST_PATH = os.path.join(TEMPLATE_OUTPUT_DIR, "idp_cases_to_run.json")

# Input ID Lists for Step 2 (Points to the specific batch folder from Step 1B)
# Note: You must update the folder name manually if you change the range above!
CURRENT_BATCH_FOLDER = f"{SUBSET_MIN_LENGTH}-{SUBSET_MAX_LENGTH}AA"
ID_LISTS_DIR = os.path.join(ID_LISTS_OUTPUT_ROOT, CURRENT_BATCH_FOLDER)

# Point to scripts in the parent directory
SCRIPT_STATIC_TEMPLATE = os.path.join(PARENT_DIR, "mk_ldr_template.py")
SCRIPT_FLEX_TEMPLATE   = os.path.join(PARENT_DIR, "mk_flex_template.py")

# Generation Parameters
TEMPLATE_N_CONFS = 200          # Number of dummy conformers for the template
TIMEOUT_STATIC_TEMPLATE = 60    # Seconds
TIMEOUT_DYNAMIC_TEMPLATE = 1000 # Seconds

# =============================================================================
# STEP 3 INPUTS: MODEL WEIGHTS & DATABASES
# =============================================================================

# Directory where raw conformers will be saved
CONFORMER_POOL_DIR = os.path.join(PIPELINE_OUTPUT_ROOT, "Step_3_Raw_Conformers")

# Sampling Parameters
SAMPLE_N_CONFS = 10      # Target conformers per IDR
SAMPLE_BATCH_SIZE = 4    # Batch size (lower if GPU OOM)
DEVICE = "cuda"          # 'cuda' or 'cpu'

# Path to the external sampling script
SCRIPT_SAMPLE_LDR = os.path.join(PARENT_DIR, "sample_ldr.py")

# Model Weights & Configs
MODEL_WEIGHTS_PATH = os.path.join(PARENT_DIR, "weights", "mdl.ckpt")
MODEL_CONFIG_PATH  = os.path.join(PARENT_DIR, "configs", "sample.yml")

# Path to the Secondary Structure database
SS_DB_PATH = os.path.join(PARENT_DIR, "data", "example_data.pkl")

# Optional: Limit residue count to prevent OOM errors on GPU
MAX_RESIDUE_LIMIT = 1000 
ATTENTION_CHUNK_SIZE = 0  # 0 = Auto/None. Set to 32 or 64 if running out of GPU memory.

# =============================================================================
# 4. STEP 4: STITCHING & RELAXATION
# =============================================================================
import glob

# 1. PATHS
# -----------------------------------------------------------------------------
# Where to save the final models
STITCH_OUTPUT_ROOT = os.path.join(PIPELINE_OUTPUT_ROOT, "Step_4_Final_Models")

# Min number of conformers needed from Step 3 to proceed
MIN_CONFORMER_POOL_SIZE = 10 # Will not sample from conformer directories containing less than this value.

# How many final models to generate
STITCH_N_CONFORMERS = 10 # Target number of final models per IDR.

# Alignment settings
ALIGNMENT_STUB_HALF_SIZE = 5 # Stub will be the midpoint +/− this size. Default: 5 (Stub = 11 Residues)
ALIGNMENT_JUNCTION_SIZE = 10 # If the folded domain is smaller than this (< 11 residues), use the full domain (Default: 10 residues).

# Relaxation settings mapped into the dictionary structure Step 4 expects
RELAX_STIFFNESS = 10.0        # Folded domain stiffness
RELAX_MAX_OUTER_ITER = 20     # Max iterations for the outer loop
MINIMIZATION_MAX_ITER = 0     # Max iterations for the inner loop (0 = no limit)
MINIMIZATION_TOLERANCE = 10.0 # Tolerance for the inner loop minimization

# The dictionary Step 4 looks for:
AF2_RELAX_CONFIG = {
    'max_outer_iterations': RELAX_MAX_OUTER_ITER,
    'stiffness': RELAX_STIFFNESS,
    'exclude_residues': [], # Filled dynamically during run
    'max_iterations': MINIMIZATION_MAX_ITER,
    'tolerance': MINIMIZATION_TOLERANCE
}

# Clash detection logic
STITCH_MAX_ATTEMPTS = 1_000_000 # Max attempts to find a stitched model before giving up.
STITCH_CLASHSCORE = 5.0         # Base clash score threshold  
CLASH_RELAX_INCREMENT = 2.5     # Increase by this amount every 5000 stitch attempts.

# Output behavior
COMBINE_ENSEMBLE = True # If True, merges all PDBs into one multi-model file at the end