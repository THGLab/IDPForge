"""Step-2 truncation window: build a per-IDR template from the IDR plus its adjacent folded domain(s)."""
from utils.stitch import build_segment_map


def compute_idr_truncation_window(labeled_idrs, target_idr, all_residues_set):
    """Compute the truncation window for one IDR: the IDR plus its adjacent folded domain(s)."""
    if not all_residues_set:
        raise ValueError("compute_idr_truncation_window: empty all_residues_set")

    target_label = target_idr.get("label")
    if target_label is None:
        raise ValueError("compute_idr_truncation_window: target_idr has no 'label'")
    idr_lo, idr_hi = target_idr["range"]

    seg_map, _idr_set, _f_count = build_segment_map(list(labeled_idrs), set(all_residues_set))
    if target_label not in seg_map:
        raise KeyError(
            f"compute_idr_truncation_window: target IDR label {target_label!r} not in "
            f"segment map (keys: {sorted(seg_map.keys())})")

    # Resolve which folded domains flank this IDR
    flanks = list(target_idr.get("flanking_domains") or [])
    if not flanks:
        # Fall back to the immediate folded neighbours
        ordered = sorted(seg_map.keys(), key=lambda k: seg_map[k][0])
        idx = ordered.index(target_label)
        if idx > 0 and ordered[idx - 1].startswith("F"):
            flanks.append(ordered[idx - 1])
        if idx < len(ordered) - 1 and ordered[idx + 1].startswith("F"):
            flanks.append(ordered[idx + 1])
    # Keep only labels that resolve to a folded segment
    retained_labels = [f for f in flanks if f in seg_map and f.startswith("F") and seg_map[f]]

    # Contiguous span = IDR plus flanking folded domains
    span_residues = set(range(idr_lo, idr_hi + 1))
    for f in retained_labels:
        span_residues.update(seg_map[f])
    trunc_lo, trunc_hi = min(span_residues), max(span_residues)

    # Sanity: no other IDR may fall inside the window
    for other in labeled_idrs:
        if other.get("label") == target_label:
            continue
        o_lo, o_hi = other["range"]
        if o_lo <= trunc_hi and o_hi >= trunc_lo:
            raise ValueError(
                f"compute_idr_truncation_window: window {trunc_lo}-{trunc_hi} for "
                f"{target_label} overlaps another IDR {other.get('label')} "
                f"({o_lo}-{o_hi}); flanking_domains are likely mis-specified")

    obs_lo, obs_hi = min(all_residues_set), max(all_residues_set)
    is_noop = (trunc_lo <= obs_lo and trunc_hi >= obs_hi)

    offset = trunc_lo - 1
    return {
        "trunc_lo": int(trunc_lo),
        "trunc_hi": int(trunc_hi),
        "offset": int(offset),
        "idr_range_full": (int(idr_lo), int(idr_hi)),
        "idr_range_seq": (int(idr_lo - offset), int(idr_hi - offset)),
        "retained_labels": retained_labels,
        "is_noop": bool(is_noop),
    }
