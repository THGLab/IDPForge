#!/usr/bin/env python3
"""slice_pdb.py - extract a contiguous residue range from a PDB into a new PDB."""
import argparse
import sys


def slice_pdb_file(input_path, start, end, output_path, renumber=False):
    """Slice residues [start, end] (inclusive, original resSeq) out of a PDB."""
    if start > end:
        raise ValueError(f"start ({start}) > end ({end})")

    kept, resnums = [], []
    with open(input_path) as fh:
        for line in fh:
            if line[:6].strip() in ("ATOM", "HETATM"):
                try:
                    resseq = int(line[22:26])
                except ValueError:
                    continue
                if start <= resseq <= end:
                    if renumber:
                        line = line[:22] + f"{resseq - start + 1:>4d}" + line[26:]
                    kept.append(line)
                    resnums.append(resseq)

    if not kept:
        raise ValueError(f"no residues in range {start}-{end} found in {input_path}")

    with open(output_path, "w") as out:
        out.writelines(kept)
        out.write("TER\nEND\n")

    uniq = sorted(set(resnums))
    return {
        "n_residues": len(uniq),
        "first_resid": uniq[0],
        "last_resid": uniq[-1],
        "n_atoms": len(kept),
        "renumbered": bool(renumber),
    }


def main():
    ap = argparse.ArgumentParser(description="Slice a contiguous residue range out of a PDB.")
    ap.add_argument("input")
    ap.add_argument("start", type=int, help="First residue to keep (PDB resSeq, inclusive).")
    ap.add_argument("end", type=int, help="Last residue to keep (PDB resSeq, inclusive).")
    ap.add_argument("output")
    ap.add_argument("--renumber", action="store_true",
                    help="Renumber kept residues from 1 (default: preserve original numbers).")
    args = ap.parse_args()

    try:
        info = slice_pdb_file(args.input, args.start, args.end, args.output, renumber=args.renumber)
    except ValueError as e:
        sys.exit(f"ERROR: {e}")

    tag = f" -> renumbered 1-{info['n_residues']}" if info["renumbered"] else ""
    print(f"Wrote {args.output}: {info['n_residues']} residues "
          f"(orig {info['first_resid']}-{info['last_resid']}{tag}), "
          f"{info['n_atoms']} atom records.")


if __name__ == "__main__":
    main()
