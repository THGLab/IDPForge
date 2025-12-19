"""
Step 1: Case Labeling & Classification

Classifies proteins into four hierarchical categories based on IDR topology:
  0. Category 0 (IDP):   Fully disordered protein (no folded domains).
  1. Category 1 (Tails): Folded domain(s) with N/C-terminal IDRs only.
  2. Category 2 (Linkers): Internal IDRs between non-interacting domains.
  3. Category 3 (Loops): Internal IDRs between interacting domains (or within one).

"""

import json
import os
import sys
from collections import defaultdict

# --- Load Centralized Configuration ---
try:
    import config as cfg
except ImportError:
    print("CRITICAL ERROR: 'project_config.py' not found.")
    sys.exit(1)

# =============================================================================
# LOGGING HELPER
# =============================================================================
def log(msg, force=False):
    """Prints message only if VERBOSE is True or force is True."""
    if getattr(cfg, 'VERBOSE', True) or force:
        print(msg)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("--- Step 1: IDR Classification & Labeling ---")

    # 1. Verify Inputs
    if not os.path.exists(cfg.MASTER_DB_PATH):
        print(f"ERROR: Untouched DB not found at {cfg.MASTER_DB_PATH}")
        sys.exit(1)
        
    if not os.path.exists(cfg.LENGTH_REF_PATH):
        print(f"ERROR: Length reference not found at {cfg.LENGTH_REF_PATH}")
        sys.exit(1)

    # 2. Load Data
    log(f"Loading master database...", force=True)
    with open(cfg.MASTER_DB_PATH, 'r') as f:
        master_db = json.load(f)

    log(f"Loading residue counts...", force=True)
    with open(cfg.LENGTH_REF_PATH, 'r') as f:
        num_residues_db = json.load(f)

    log(f"  -> Loaded {len(master_db)} proteins.", force=True)

    # 3. Initialize Counters
    processed_count = 0
    skipped_count = 0
    type_counts = defaultdict(int)
    
    # Category Counters
    count_cat_0 = 0 # Full IDP
    count_cat_1 = 0 # Tails
    count_cat_2 = 0 # Linkers
    count_cat_3 = 0 # Loops

    total_proteins = len(master_db)

    # 4. Main Classification Loop
    print(f"Starting classification of {total_proteins} proteins...")
    
    for i, (prot_id, data) in enumerate(master_db.items()):
        
        # --- Progress Heartbeat (Silent Mode) ---
        if not getattr(cfg, 'VERBOSE', True):
            if i > 0 and i % 5000 == 0:
                print(f"  ... Scanned {i}/{total_proteins} proteins ...")

        # --- Verbose Log ---
        log(f"Processing {prot_id}...")

        # Validation
        if prot_id not in num_residues_db:
            log(f"  [Skip] No length data.")
            skipped_count += 1
            continue

        if 'idrs' not in data or not data['idrs']:
            data['labeled_idrs'] = [] 
            data['category'] = None
            continue

        idrs = data['idrs']
        num_res = num_residues_db[prot_id]
        
        # Interaction map
        interactions_list = data.get('interactions', [])
        interaction_set = set(frozenset(pair) for pair in interactions_list)

        labeled_idrs = []
        
        # Flags
        is_full_idp = False
        has_tail = False
        has_linker = False
        has_loop = False

        # --- CASE A: Full-Protein IDP (Category 0) ---
        if len(idrs) == 1 and idrs[0][0] == 1 and idrs[0][1] == num_res:
            idr_type = "IDP"
            is_full_idp = True
            
            labeled_idrs.append({
                "range": idrs[0],
                "type": idr_type,
                "label": "D1",
                "flanking_domains": []
            })
            type_counts[idr_type] += 1
            
        # --- CASE B: Hybrid Proteins (Categories 1, 2, 3) ---
        else:
            f_domain_counter = 0
            if idrs[0][0] != 1: f_domain_counter += 1 

            for k, idr_range in enumerate(idrs):
                label = f'D{k+1}'
                start, end = idr_range
                
                is_n_term = (start == 1)
                is_c_term = (end == num_res)
                
                flanking = []
                idr_type = ""

                # 1. Tail IDRs
                if is_n_term:
                    idr_type = "Tail IDR"
                    flanking = [f'F{f_domain_counter + 1}']
                    has_tail = True
                elif is_c_term:
                    idr_type = "Tail IDR"
                    flanking = [f'F{f_domain_counter}']
                    has_tail = True
                
                # 2. Internal IDRs
                else:
                    domain_before = f'F{f_domain_counter}'
                    domain_after = f'F{f_domain_counter + 1}'
                    flanking = [domain_before, domain_after]

                    if frozenset(flanking) in interaction_set:
                        idr_type = "Loop IDR"
                        has_loop = True
                    else:
                        idr_type = "Linker IDR"
                        has_linker = True

                labeled_idrs.append({
                    "range": idr_range,
                    "type": idr_type,
                    "label": label,
                    "flanking_domains": flanking
                })
                type_counts[idr_type] += 1

                # Update F-domain counter
                is_last_idr = (k == len(idrs) - 1)
                if not is_last_idr:
                    if idrs[k+1][0] - end > 1: f_domain_counter += 1
                elif not is_c_term:
                    f_domain_counter += 1

        # --- APPLY HIERARCHY ---
        final_category = -1
        if is_full_idp:
            final_category = 0
            count_cat_0 += 1
        elif has_loop:
            final_category = 3
            count_cat_3 += 1
        elif has_linker:
            final_category = 2
            count_cat_2 += 1
        elif has_tail:
            final_category = 1
            count_cat_1 += 1

        # Save to Dict
        data['labeled_idrs'] = labeled_idrs
        data['category'] = final_category
        processed_count += 1
        
        log(f"  -> Assigned Category {final_category}")

    # 5. Save Output
    print(f"\nClassification complete. Saving to disk...")
    os.makedirs(cfg.STEP_1_DIR, exist_ok=True)

    try:
        with open(cfg.LABELED_DB_PATH, 'w') as f:
            json.dump(master_db, f, indent=4)
        print(f"[\u2713] Labeled DB saved: {cfg.LABELED_DB_PATH}")
    except IOError as e:
        print(f"[!] Error saving DB: {e}")

    # 6. Write Summary Report
    summary_path = cfg.SUMMARY_TEXT_PATH
    print(f"[\u2713] Writing summary: {summary_path}")

    total_categorized = count_cat_0 + count_cat_1 + count_cat_2 + count_cat_3

    summary_text = [
        "--- IDR Classification Summary ---",
        f"Source DB: {cfg.MASTER_DB_PATH}",
        "------------------------------------",
        f"Total Processed: {processed_count}",
        f"Total Skipped:   {skipped_count}",
        "------------------------------------",
        f"Individual IDRs Found:",
        f"  - Full IDP:    {type_counts['IDP']}",
        f"  - Tail IDR:    {type_counts['Tail IDR']}",
        f"  - Linker IDR:  {type_counts['Linker IDR']}",
        f"  - Loop IDR:    {type_counts['Loop IDR']}",
        "\n--- Hierarchical Categorization ---",
        "[Category 3] Loops:",
        f"  - {count_cat_3} proteins",
        "",
        "[Category 2] Linkers:",
        f"  - {count_cat_2} proteins",
        "",
        "[Category 1] Tails:",
        f"  - {count_cat_1} proteins",
        "",
        "[Category 0] Full IDPs:",
        f"  - {count_cat_0} proteins",
        "------------------------------------",
        f"Total Categorized: {total_categorized}"
    ]

    with open(summary_path, 'w') as f:
        f.write("\n".join(summary_text))

    print("\n" + "\n".join(summary_text) + "\n")

if __name__ == "__main__":
    main()