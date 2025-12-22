"""
Step 1B: Subset Sampling & ID List Generation
=============================================

Creates lists of protein IDs filtered by length and groups them into the 4 mutually exclusive 
categories defined in Step 1:
  - Cat 0: Full IDP
  - Cat 1: Tails (only)
  - Cat 2: Linkers (may have tails)
  - Cat 3: Loops (may have linkers/tails)

Output:
  data/id_lists/0-1000_AA/
     ├── cat_0.txt
     ├── cat_1.txt
     ├── cat_2.txt
     └── cat_3.txt
"""

import json
import os
import random
import sys
from collections import defaultdict

try:
    import config as cfg
except ImportError:
    print("CRITICAL ERROR: 'project_config.py' not found.")
    sys.exit(1)

def log(msg, force=False):
    if getattr(cfg, 'VERBOSE', True) or force:
        print(msg)

def main():
    log("--- Step 1B: Subset Generation ---", force=True)
    
    if not os.path.exists(cfg.LABELED_DB_PATH):
        print(f"ERROR: Labeled DB not found: {cfg.LABELED_DB_PATH}")
        sys.exit(1)

    log(f"Loading labeled DB...")
    with open(cfg.LABELED_DB_PATH, 'r') as f:
        master_db = json.load(f)

    log(f"Loading length reference...")
    with open(cfg.LENGTH_REF_PATH, 'r') as f:
        num_residues_db = json.load(f)

    # Output Setup
    range_label = f"{cfg.SUBSET_MIN_LENGTH}-{cfg.SUBSET_MAX_LENGTH}AA"
    output_dir = os.path.join(cfg.ID_LISTS_OUTPUT_ROOT, range_label)
    os.makedirs(output_dir, exist_ok=True)
    
    log(f"Target Range: {cfg.SUBSET_MIN_LENGTH} - {cfg.SUBSET_MAX_LENGTH} residues", force=True)
    log(f"Output Folder: {output_dir}", force=True)

    pools = defaultdict(list)
    count_skipped_len = 0
    count_skipped_no_cat = 0

    # Filtering
    for prot_id, data in master_db.items():
        length = num_residues_db.get(prot_id, 0)
        
        # Inclusive bounds check
        if not (cfg.SUBSET_MIN_LENGTH <= length <= cfg.SUBSET_MAX_LENGTH):
            count_skipped_len += 1
            continue

        cat = data.get('category')
        if cat is None or cat == -1:
            count_skipped_no_cat += 1
            continue
            
        pools[cat].append(prot_id)

    # Sampling
    log("\n--- Generating Lists ---", force=True)
    summary_counts = {}

    for cat_id in [0, 1, 2, 3]:
        id_list = pools[cat_id]
        total_available = len(id_list)
        
        filename = f"cat_{cat_id}.txt"
        filepath = os.path.join(output_dir, filename)

        if total_available == 0:
            log(f"  [ ] {filename:<12} : Empty (0 proteins)", force=True)
            open(filepath, 'w').close()
            continue
            
        sample_size = min(total_available, cfg.SUBSET_SAMPLE_SIZE)
        sampled_ids = random.sample(id_list, sample_size)
        sampled_ids.sort() 
        
        try:
            with open(filepath, 'w') as f:
                f.write("\n".join(sampled_ids))
            
            # Always print summary of files created
            log(f"  [\u2713] {filename:<12} : {sample_size} IDs (Pool: {total_available})", force=True)
            summary_counts[f"cat_{cat_id}"] = sample_size
            
        except IOError as e:
            print(f"  [!] Error writing {filename}: {e}")

    # Summary
    summary_json_path = os.path.join(output_dir, "batch_manifest.json")
    with open(summary_json_path, 'w') as f:
        json.dump({
            "length_range": range_label,
            "min_len": cfg.SUBSET_MIN_LENGTH,
            "max_len": cfg.SUBSET_MAX_LENGTH,
            "counts": summary_counts,
        }, f, indent=4)

    log("\n--- Step 1B Complete ---", force=True)

if __name__ == "__main__":
    main()