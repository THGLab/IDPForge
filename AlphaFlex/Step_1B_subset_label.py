"""
Step 1B: Subset Filtering & ID List Generation
===============================================

Creates a filtered list of protein IDs from the labeled database produced by
Step 1. Supports two levels of filtering:

  Basic (always active):
    - Total protein length range (inclusive)

  Advanced (optional, activate by specifying any of these):
    - Per-type IDR counts: --tail_count, --linker_count, --loop_count
    - Count matching mode: exact (default) or minimum (--min_mode)
    - IDR length range applied to ALL IDRs in the protein (inclusive)

Per-type count filters are independent. Only specified counts are enforced;
unspecified types (None) are unconstrained. For example:
  --tail_count 1 --linker_count 0
matches proteins with exactly 1 Tail IDR and 0 Linker IDRs, regardless of
how many Loop IDRs they have. With --min_mode, the same arguments match
proteins with at least 1 Tail and at most 0 Linkers.

When no advanced filters are specified, all proteins within the length range
are included (equivalent to the original Step 1B behavior).

Output is written to id_lists/custom_subsets/:
  custom_subsets/
     ├── <output_name>.txt                 Filtered UniProt ID list
     ├── <output_name>_report.txt          Detailed per-protein report table
     ├── <output_name>_histogram.png       IDR length distribution histogram
     └── <output_name>_histogram_table.txt Histogram bin counts as text

The report and histogram are only generated when advanced IDR filters are
active. The output filename is controlled by --output_name (default from
config.py: SUBSET_OUTPUT_NAME).

Defaults are read from config.py (SUBSET_* parameters) and can be overridden
via command-line arguments.
"""

import json
import os
import random
import sys
import argparse

try:
    import config as cfg
except ImportError:
    print("CRITICAL ERROR: 'config.py' not found.")
    sys.exit(1)


# =============================================================================
# IDR TYPE CONSTANTS
# =============================================================================
TYPE_TAIL = "Tail IDR"
TYPE_LINKER = "Linker IDR"
TYPE_LOOP = "Loop IDR"


def count_idr_types(labeled_idrs):
    """Count IDRs by type, ignoring Folded Domain entries."""
    counts = {TYPE_TAIL: 0, TYPE_LINKER: 0, TYPE_LOOP: 0}
    for idr in labeled_idrs:
        t = idr.get("type", "")
        if t in counts:
            counts[t] += 1
    return counts


def check_type_counts(actual_counts, tail, linker, loop, exact):
    """
    Check per-type count constraints.
    Only specified (non-None) counts are enforced.
    In exact mode: actual == required.
    In min mode:   actual >= required.
    """
    for required, type_key in [(tail, TYPE_TAIL),
                                (linker, TYPE_LINKER),
                                (loop, TYPE_LOOP)]:
        if required is None:
            continue
        actual = actual_counts[type_key]
        if exact and actual != required:
            return False
        if not exact and actual < required:
            return False
    return True


def has_advanced_filters(args):
    """Return True if any advanced IDR filter is active."""
    if args.tail_count is not None:
        return True
    if args.linker_count is not None:
        return True
    if args.loop_count is not None:
        return True
    if args.idr_min_len is not None:
        return True
    if args.idr_max_len is not None:
        return True
    return False


def main(args):
    def log(msg, force=False):
        if args.verbose or force:
            print(msg)

    min_p, max_p = args.min_len, args.max_len
    advanced = has_advanced_filters(args)
    mode_str = "Exact" if args.exact else "Minimum"

    log("--- Step 1B: Subset Generation ---", force=True)
    log(f"Matching Criteria:", force=True)
    log(f"  Protein Length: {min_p} <= length <= {max_p}", force=True)

    if advanced:
        log(f"  Advanced Filters: ON", force=True)
        log(f"  Count Mode:       {mode_str}", force=True)
        log(f"  Tail IDR Count:   {args.tail_count if args.tail_count is not None else 'any'}", force=True)
        log(f"  Linker IDR Count: {args.linker_count if args.linker_count is not None else 'any'}", force=True)
        log(f"  Loop IDR Count:   {args.loop_count if args.loop_count is not None else 'any'}", force=True)
        idr_min = args.idr_min_len if args.idr_min_len is not None else 0
        idr_max = args.idr_max_len if args.idr_max_len is not None else "inf"
        log(f"  IDR Length Range:  {idr_min}-{idr_max} (applied to all IDRs)", force=True)
    else:
        log(f"  Advanced Filters: OFF (length-only mode)", force=True)

    # 1. Load Databases
    if not os.path.exists(args.labeled_db):
        print(f"ERROR: Labeled DB not found: {args.labeled_db}")
        sys.exit(1)

    log("Loading labeled DB...")
    with open(args.labeled_db, 'r') as f:
        master_db = json.load(f)

    log("Loading length reference...")
    with open(args.length_ref, 'r') as f:
        num_residues_db = json.load(f)

    report_data = []

    # 2. Process Data
    for prot_id, data in master_db.items():

        # --- A. Protein Length Filter (always active) ---
        total_prot_len = num_residues_db.get(prot_id, 0)
        if not (min_p <= total_prot_len <= max_p):
            continue

        labeled_idrs = data.get('labeled_idrs', [])
        actual_counts = count_idr_types(labeled_idrs)
        idrs_only = [idr for idr in labeled_idrs
                     if idr.get("type", "") != "Folded Domain"]

        if advanced:
            # --- B. Per-Type IDR Count Filter ---
            if not check_type_counts(actual_counts, args.tail_count,
                                      args.linker_count, args.loop_count,
                                      args.exact):
                continue

            # --- C. IDR Length Filter (applied to ALL IDRs) ---
            idr_min = args.idr_min_len if args.idr_min_len is not None else 0
            idr_max = args.idr_max_len if args.idr_max_len is not None else float('inf')

            if not idrs_only:
                continue

            all_idrs_ok = True
            idr_details = []
            for idr in idrs_only:
                start, end = idr['range']
                idr_len = (end - start) + 1
                if not (idr_min <= idr_len <= idr_max):
                    all_idrs_ok = False
                    break
                idr_details.append({
                    "idr_type": idr['type'],
                    "range_str": f"{start}-{end}",
                    "idr_len": idr_len
                })

            if not all_idrs_ok:
                continue
        else:
            # Basic mode: collect IDR details for report but don't filter
            idr_details = []
            for idr in idrs_only:
                start, end = idr['range']
                idr_len = (end - start) + 1
                idr_details.append({
                    "idr_type": idr['type'],
                    "range_str": f"{start}-{end}",
                    "idr_len": idr_len
                })

        report_data.append({
            "prot_id": prot_id,
            "total_len": total_prot_len,
            "category": data.get('category', -1),
            "type_counts": actual_counts,
            "idrs": idr_details
        })

    # --- Sampling ---
    if args.max_samples and len(report_data) > args.max_samples:
        report_data = random.sample(report_data, args.max_samples)

    report_data.sort(key=lambda x: x["prot_id"])

    id_list = [entry["prot_id"] for entry in report_data]

    # --- Output ---
    output_dir = os.path.join(args.output_root, "custom_subsets")
    advanced_info_dir = os.path.join(args.output_root, "advanced_info")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(advanced_info_dir, exist_ok=True)

    output_name = args.output_name
    list_path = os.path.join(output_dir, f"{output_name}.txt")
    report_path = os.path.join(advanced_info_dir, f"{output_name}_report.txt")
    hist_path = os.path.join(advanced_info_dir, f"{output_name}_histogram.png")
    table_path = os.path.join(advanced_info_dir, f"{output_name}_histogram_table.txt")

    # =========================================================================
    # OUTPUT 1: The ID List
    # =========================================================================
    with open(list_path, 'w') as f:
        f.write("\n".join(id_list))

    # =========================================================================
    # OUTPUT 2: The Report Table
    # =========================================================================
    with open(report_path, 'w') as f:
        f.write(f" FILTRATION REPORT\n")
        f.write(f" Protein Length: {min_p}-{max_p}\n")
        if advanced:
            idr_min = args.idr_min_len if args.idr_min_len is not None else 0
            idr_max = args.idr_max_len if args.idr_max_len is not None else "inf"
            f.write(f" IDR Length: {idr_min}-{idr_max}\n")
            f.write(f" Count Mode: {mode_str}\n")
            f.write(f" Tail={args.tail_count}  Linker={args.linker_count}  Loop={args.loop_count}\n")
        f.write(f" Total Matches: {len(report_data)}\n")
        f.write("=" * 95 + "\n")
        header = (f"{'Protein ID':<15} | {'Total Len':<10} | {'Cat':<4} | {'Tails':<6} | "
                  f"{'Linkers':<8} | {'Loops':<6} | {'IDR Details'}")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for entry in report_data:
            c = entry['type_counts']
            idr_strs = [f"{d['idr_type']}({d['range_str']},{d['idr_len']}aa)"
                        for d in entry['idrs']]
            row = (f"{entry['prot_id']:<15} | {entry['total_len']:<10} | "
                   f"{entry['category']:<4} | "
                   f"{c[TYPE_TAIL]:<6} | {c[TYPE_LINKER]:<8} | "
                   f"{c[TYPE_LOOP]:<6} | {'; '.join(idr_strs)}")
            f.write(row + "\n")

    # =========================================================================
    # OUTPUT 3 & 4: Histogram (only when advanced IDR filters are active)
    # =========================================================================
    if advanced:
        import matplotlib.pyplot as plt
        import numpy as np

        length_distribution = []
        for entry in report_data:
            for idr in entry["idrs"]:
                length_distribution.append(idr["idr_len"])

        if length_distribution:
            max_val = max(length_distribution)
            bins = list(range(0, max_val + 100, 50))

            # Histogram PNG
            plt.figure(figsize=(10, 6))
            plt.hist(length_distribution, bins=bins, color='skyblue',
                     edgecolor='black', alpha=0.7)
            idr_min = args.idr_min_len if args.idr_min_len is not None else 0
            idr_max = args.idr_max_len if args.idr_max_len is not None else "inf"
            plt.title(f"IDR Lengths (Prot Len {min_p}-{max_p}, IDR Len {idr_min}-{idr_max})")
            plt.xlabel("IDR Length")
            plt.ylabel("Frequency")
            plt.xticks(bins)
            plt.grid(axis='y', alpha=0.5)
            plt.savefig(hist_path)
            plt.close()

            # Histogram Data Table
            counts, bin_edges = np.histogram(length_distribution, bins=bins)
            with open(table_path, 'w') as f:
                f.write(f" HISTOGRAM DATA ({min_p}-{max_p}, IDR Len {idr_min}-{idr_max})\n")
                f.write("-" * 28 + "\n")
                for i in range(len(counts)):
                    range_str = f"{int(bin_edges[i])}-{int(bin_edges[i+1])}"
                    f.write(f"{range_str:<15} | {counts[i]:<10}\n")

    # --- Summary ---
    log("\n--- Step 1B Complete ---", force=True)
    log(f"  Output: {list_path}", force=True)
    log(f"  Total Matches: {len(report_data)}", force=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 1B: Subset Filtering & ID List Generation")

    # --- Path arguments ---
    parser.add_argument("--labeled_db", default=cfg.LABELED_DB_PATH,
                        help=f"Path to labeled database JSON (default: {cfg.LABELED_DB_PATH}).")
    parser.add_argument("--length_ref", default=cfg.LENGTH_REF_PATH,
                        help=f"Path to residue-count reference JSON (default: {cfg.LENGTH_REF_PATH}).")
    parser.add_argument("--output_root", default=cfg.ID_LISTS_OUTPUT_ROOT,
                        help=f"Root directory for output files (default: {cfg.ID_LISTS_OUTPUT_ROOT}).")
    parser.add_argument("--output_name", default=cfg.SUBSET_OUTPUT_NAME,
                        help=f"Base name for output files (default: '{cfg.SUBSET_OUTPUT_NAME}').")

    # --- Protein length filter ---
    parser.add_argument("--min_len", type=int, default=cfg.SUBSET_MIN_LENGTH,
                        help=f"Minimum protein length, inclusive (default: {cfg.SUBSET_MIN_LENGTH}).")
    parser.add_argument("--max_len", type=int, default=cfg.SUBSET_MAX_LENGTH,
                        help=f"Maximum protein length, inclusive (default: {cfg.SUBSET_MAX_LENGTH}).")

    # --- Per-type IDR count filters (optional, None = unconstrained) ---
    parser.add_argument("--tail_count", type=int, default=cfg.SUBSET_TAIL_COUNT,
                        help=f"Required Tail IDR count; omit for unconstrained (default: {cfg.SUBSET_TAIL_COUNT}).")
    parser.add_argument("--linker_count", type=int, default=cfg.SUBSET_LINKER_COUNT,
                        help=f"Required Linker IDR count; omit for unconstrained (default: {cfg.SUBSET_LINKER_COUNT}).")
    parser.add_argument("--loop_count", type=int, default=cfg.SUBSET_LOOP_COUNT,
                        help=f"Required Loop IDR count; omit for unconstrained (default: {cfg.SUBSET_LOOP_COUNT}).")

    # --- Count matching mode ---
    parser.add_argument("--exact", action="store_true", default=cfg.SUBSET_EXACT_COUNT,
                        help=f"Exact count matching (default: {cfg.SUBSET_EXACT_COUNT}).")
    parser.add_argument("--min_mode", action="store_true", default=False,
                        help="Minimum count matching (overrides --exact).")

    # --- IDR length filter (optional, None = no filter) ---
    parser.add_argument("--idr_min_len", type=int, default=cfg.SUBSET_IDR_MIN_LENGTH,
                        help=f"Minimum IDR length, inclusive, applied to all IDRs (default: {cfg.SUBSET_IDR_MIN_LENGTH}).")
    parser.add_argument("--idr_max_len", type=int, default=cfg.SUBSET_IDR_MAX_LENGTH,
                        help=f"Maximum IDR length, inclusive, applied to all IDRs (default: {cfg.SUBSET_IDR_MAX_LENGTH}).")

    # --- Output cap ---
    parser.add_argument("--max_samples", type=int, default=cfg.SUBSET_MAX_SAMPLES,
                        help=f"Maximum output proteins; omit for no limit (default: {cfg.SUBSET_MAX_SAMPLES}).")

    # --- Logging ---
    parser.add_argument("--verbose", action="store_true", default=cfg.VERBOSE,
                        help="Enable detailed logging.")

    args = parser.parse_args()

    # --min_mode overrides --exact
    if args.min_mode:
        args.exact = False

    main(args)
