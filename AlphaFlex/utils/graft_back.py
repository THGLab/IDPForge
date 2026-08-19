"""Graft-back reconstruction for size-capped single-IDR conformer pools."""
import os
import glob
import numpy as np


# ----------------------------------------------------------------------------- PDB IO
def parse_pdb_atoms(path):
    """Return list of dicts {line, resseq, name, xyz} for ATOM/HETATM records (model 1)."""
    atoms = []
    for ln in open(path):
        if ln.startswith(("ATOM", "HETATM")):
            atoms.append({
                "line": ln.rstrip("\n"),
                "resseq": int(ln[22:26]),
                "name": ln[12:16].strip(),
                "xyz": np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])]),
            })
        elif ln.startswith("ENDMDL"):
            break  # first model only
    return atoms


def _set_line(atom, xyz=None, resseq=None):
    """Rewrite a PDB ATOM line with new coords and/or resSeq (fixed-column)."""
    ln = atom["line"]
    if xyz is not None:
        ln = ln[:30] + f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}" + ln[54:]
    if resseq is not None:
        ln = ln[:22] + f"{resseq:>4d}" + ln[26:]
    return ln


def _write_grafted(out_lines, out_path):
    """Write sorted (resSeq, line) tuples to a PDB, dropping non-terminal OXT/OT1/OT2."""
    out_lines = sorted(out_lines, key=lambda x: x[0])
    cterm = out_lines[-1][0] if out_lines else None
    with open(out_path, "w") as f:
        for rs, ln in out_lines:
            if rs != cterm and ln[12:16].strip() in ("OXT", "OT1", "OT2"):
                continue
            f.write(ln + "\n")
        f.write("TER\nEND\n")


# --------------------------------------------------------------------------- geometry
def kabsch(P, Q):
    """Least-squares rigid transform mapping P onto Q. Returns (R, t): (R @ P.T).T + t ~ Q."""
    Pc = P - P.mean(0)
    Qc = Q - Q.mean(0)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = Q.mean(0) - R @ P.mean(0)
    return R, t


def _ca_by_resseq(atoms, resseqs):
    out = {}
    want = set(resseqs)
    for a in atoms:
        if a["name"] == "CA" and a["resseq"] in want:
            out[a["resseq"]] = a["xyz"]
    return out


# Kabsch anchor for the fragment/template splice
ANCHOR_RES = 2
KABSCH_ATOMS = ("N", "CA", "C")


def _anchor_points(conf, static_atoms, anchor_res):
    """Return (P_conf, Q_static, keys) over KABSCH_ATOMS of `anchor_res`, present in both structures."""
    want_r, want_n = set(anchor_res), set(KABSCH_ATOMS)
    c = {(a["resseq"], a["name"]): a["xyz"] for a in conf
         if a["resseq"] in want_r and a["name"] in want_n}
    s = {(a["resseq"], a["name"]): a["xyz"] for a in static_atoms
         if a["resseq"] in want_r and a["name"] in want_n}
    keys = sorted(set(c) & set(s))
    P = np.array([c[k] for k in keys]) if keys else np.zeros((0, 3))
    Q = np.array([s[k] for k in keys]) if keys else np.zeros((0, 3))
    return P, Q, keys


def _rmsd(P, Q, R, t):
    """RMSD of the superposed P onto Q: sqrt(mean over atoms of |d|^2)."""
    d = (R @ P.T).T + t - Q
    return float(np.sqrt((d ** 2).sum(1).mean()))


# ------------------------------------------------------------------------------ graft
def graft_conformer(conf_path, static_atoms, idr_ranges, fold_range, offset, out_path,
                    stub_size=15):
    """Complete one truncated single-IDR conformer to full length via a 3-step graft."""
    conf = parse_pdb_atoms(conf_path)
    for a in conf:
        a["resseq"] += offset  # -> native numbering

    def is_idr(r):
        return any(lo <= r <= hi for (lo, hi) in idr_ranges)

    flo, fhi = fold_range
    conf_fold_resseqs = {a["resseq"] for a in conf
                         if not is_idr(a["resseq"]) and flo <= a["resseq"] <= fhi}
    static_fold_resseqs = {a["resseq"] for a in static_atoms if flo <= a["resseq"] <= fhi}
    dropped = static_fold_resseqs - conf_fold_resseqs  # distal fold to restore
    if not conf_fold_resseqs:
        return None  # nothing to align against

    if not dropped:
        # Untruncated pool: copy verbatim
        _write_grafted([(a["resseq"], _set_line(a, resseq=a["resseq"])) for a in conf], out_path)
        return {"rms": 0.0, "n_stub": 0, "n_dropped": 0, "junction_gap": None, "seam_gap": None}

    # Step 1: truncation-boundary stub
    conf_fold = sorted(conf_fold_resseqs)
    dmin, dmax = min(dropped), max(dropped)
    if dmin > conf_fold[-1]:        # dropped fold above the fragment
        stub_res = conf_fold[-ANCHOR_RES:]
    elif dmax < conf_fold[0]:       # dropped fold below the fragment
        stub_res = conf_fold[:ANCHOR_RES]
    else:                           # dropped on both flanks
        stub_res = conf_fold

    # Peptide-plane anchor
    Q, P, shared = _anchor_points(conf, static_atoms, stub_res)   # Q conformer (fixed), P template (moving)
    if len(shared) < 3:
        return None
    R, t = kabsch(P, Q)                          # template -> conformer frame
    rms = _rmsd(P, Q, R, t)

    # Step 2: keep conformer verbatim, splice dropped template fold
    out_lines = [(a["resseq"], _set_line(a, resseq=a["resseq"])) for a in conf]
    for a in static_atoms:
        if a["resseq"] in dropped:
            out_lines.append((a["resseq"], _set_line(a, xyz=(R @ a["xyz"]) + t)))

    _write_grafted(out_lines, out_path)

    all_ca = _ca_by_resseq(
        [{"name": "CA", "resseq": rs, "xyz": np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])}
         for rs, ln in out_lines if ln[12:16].strip() == "CA"],
        [rs for rs, _ in out_lines])
    # Junction gap: IDR/fold boundary
    jgaps = [float(np.linalg.norm(all_ca[j] - all_ca[k]))
             for (lo, hi) in idr_ranges for j, k in ((lo - 1, lo), (hi + 1, hi))
             if j in all_ca and k in all_ca]
    # Seam gap: fragment/template boundary
    if dmin > conf_fold[-1]:
        b = conf_fold[-1]; seam = (b, b + 1)
    elif dmax < conf_fold[0]:
        b = conf_fold[0]; seam = (b, b - 1)
    else:
        seam = None
    seam_gap = (float(np.linalg.norm(all_ca[seam[0]] - all_ca[seam[1]]))
                if seam and seam[0] in all_ca and seam[1] in all_ca else None)
    return {"rms": rms, "n_stub": len(shared), "n_dropped": len(dropped),
            "junction_gap": max(jgaps) if jgaps else None, "seam_gap": seam_gap}


def graft_conformer_multi(conf_path, static_atoms, idr_ranges, fold_ranges, offset, out_path,
                          stub_size=15):
    """Complete a truncated multi-domain (linker) conformer by grafting each domain independently."""
    ranges = [tuple(r) for r in fold_ranges]
    if not ranges:
        return None
    if len(ranges) == 1:
        return graft_conformer(conf_path, static_atoms, idr_ranges, ranges[0], offset, out_path,
                               stub_size=stub_size)
    src = conf_path
    tmps, reports = [], []
    for k, fr in enumerate(ranges):
        last = (k == len(ranges) - 1)
        dst = out_path if last else "%s.d%d.tmp" % (out_path, k)
        rep = graft_conformer(src, static_atoms, idr_ranges, fr,
                              offset if k == 0 else 0, dst, stub_size=stub_size)
        if rep is None:
            for t in tmps:
                try:
                    os.remove(t)
                except OSError:
                    pass
            return None
        rep["fold_range"] = fr
        reports.append(rep)
        if not last:
            tmps.append(dst)
        src = dst
    for t in tmps:
        try:
            os.remove(t)
        except OSError:
            pass
    seams = [r["seam_gap"] for r in reports if r.get("seam_gap") is not None]
    jgaps = [r["junction_gap"] for r in reports if r.get("junction_gap") is not None]
    return {"rms": max(r["rms"] for r in reports),
            "n_stub": min(r["n_stub"] for r in reports),
            "n_dropped": sum(r["n_dropped"] for r in reports),
            "junction_gap": max(jgaps) if jgaps else None,
            "seam_gap": max(seams) if seams else None,
            "per_domain": reports}


def graft_continuation(base_atoms, cont_atoms, anchor_range, new_range, out_path,
                       anchor_res=None):
    """Splice a Step-2 continuation fragment onto a base model by superposing the shared fragment's facing end."""
    K = ANCHOR_RES if anchor_res is None else int(anchor_res)
    alo, ahi = anchor_range
    blo, bhi = new_range
    if blo > ahi:                                   # append: B above A
        stub_res = list(range(max(alo, ahi - K + 1), ahi + 1))
        junction = (ahi, blo)
    elif bhi < alo:                                 # prepend: B below A
        stub_res = list(range(alo, min(ahi, alo + K - 1) + 1))
        junction = (alo, bhi)
    else:                                           # B overlaps A -> not a continuation
        return None

    # cont is the moving set; base is fixed
    P, Q, shared = _anchor_points(cont_atoms, base_atoms, stub_res)   # P cont, Q base
    if len(shared) < 3:
        return None
    R, t = kabsch(P, Q)
    rms = _rmsd(P, Q, R, t)

    out = [(a["resseq"], _set_line(a)) for a in base_atoms]      # base verbatim
    for a in cont_atoms:
        if blo <= a["resseq"] <= bhi:                            # B -> base frame
            out.append((a["resseq"], _set_line(a, xyz=(R @ a["xyz"]) + t)))
    _write_grafted(out, out_path)

    ca = _ca_by_resseq(
        [{"name": "CA", "resseq": rs, "xyz": np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])}
         for rs, ln in out if ln[12:16].strip() == "CA"], [rs for rs, _ in out])
    j, k = junction
    jgap = (float(np.linalg.norm(ca[j] - ca[k]))
            if abs(j - k) == 1 and j in ca and k in ca else None)
    return {"rms": rms, "junction_gap": jgap, "n_stub": len(shared)}


def graft_continuation_chain(base_path, cont_specs, out_path, work_dir=None, anchor_res=None):
    """Chain a sequence of continuation chunks onto a base model, one splice at a time."""
    import os as _os, tempfile as _tmp, shutil as _sh
    wd = work_dir or _tmp.mkdtemp(prefix="_chain_")
    _os.makedirs(wd, exist_ok=True)
    reports, ok = [], True
    cur = base_path
    tmp_made = []
    for i, spec in enumerate(cont_specs):
        nxt = _os.path.join(wd, "_chain_%d.pdb" % i)
        rep = graft_continuation(parse_pdb_atoms(cur), parse_pdb_atoms(spec["conf_path"]),
                                 tuple(spec["anchor_range"]), tuple(spec["new_range"]),
                                 nxt, anchor_res=anchor_res)
        reports.append(rep)
        if rep is None:
            ok = False
            break
        cur = nxt
        tmp_made.append(nxt)
    if ok and cur != out_path:
        _sh.copyfile(cur, out_path)
    for t in tmp_made:
        if t != out_path and _os.path.exists(t):
            try:
                _os.remove(t)
            except OSError:
                pass
    return ok, reports


def graft_idrs_onto_fold(static_atoms, fold_range, idr_specs, out_path, stub_size=15):
    """Assemble a multi-tail full-length model from separately-generated IDR conformers."""
    flo, fhi = fold_range
    static_fold = {a["resseq"] for a in static_atoms if flo <= a["resseq"] <= fhi}

    idr_lines, frag_lines, frag_res = [], [], set()
    reports = []
    seams = []  # template/fragment boundaries
    for spec in sorted(idr_specs, key=lambda s: s["idr_range"][0]):
        lo, hi = spec["idr_range"]
        conf = parse_pdb_atoms(spec["conf_path"])
        for a in conf:
            a["resseq"] += spec["offset"]
        conf_fold = sorted({a["resseq"] for a in conf if flo <= a["resseq"] <= fhi
                            and not (lo <= a["resseq"] <= hi)})
        if not conf_fold:
            reports.append({"idr_range": (lo, hi), "rms": None, "junction_gap": None, "seam_gap": None})
            continue
        dropped = static_fold - set(conf_fold)
        # Boundary stub
        if dropped and min(dropped) > conf_fold[-1]:
            stub_res = conf_fold[-ANCHOR_RES:]; seams.append((conf_fold[-1], conf_fold[-1] + 1))
        elif dropped and max(dropped) < conf_fold[0]:
            stub_res = conf_fold[:ANCHOR_RES]; seams.append((conf_fold[0] - 1, conf_fold[0]))
        else:
            stub_res = conf_fold
        # Peptide-plane anchor; conformer is the moving set, template fixed
        P, Q, shared = _anchor_points(conf, static_atoms, stub_res)   # P conformer (moving), Q template (fixed)
        if len(shared) < 3:
            reports.append({"idr_range": (lo, hi), "rms": None, "junction_gap": None, "seam_gap": None})
            continue
        R, t = kabsch(P, Q)                           # conformer -> template frame
        rms = _rmsd(P, Q, R, t)
        for a in conf:
            ln = _set_line(a, xyz=(R @ a["xyz"]) + t, resseq=a["resseq"])
            if lo <= a["resseq"] <= hi:
                idr_lines.append((a["resseq"], ln))
            elif flo <= a["resseq"] <= fhi:  # keep conformer fold fragment
                frag_lines.append((a["resseq"], ln)); frag_res.add(a["resseq"])
        reports.append({"idr_range": (lo, hi), "rms": rms})

    # Template fills only the fold residues no fragment covers.
    tmpl_lines = [(a["resseq"], _set_line(a)) for a in static_atoms
                  if flo <= a["resseq"] <= fhi and a["resseq"] not in frag_res]
    out = idr_lines + frag_lines + tmpl_lines
    _write_grafted(out, out_path)

    ca = _ca_by_resseq(
        [{"name": "CA", "resseq": rs, "xyz": np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])}
         for rs, ln in out if ln[12:16].strip() == "CA"], [rs for rs, _ in out])
    for rep in reports:
        if rep.get("rms") is None:
            rep.setdefault("junction_gap", None); rep.setdefault("seam_gap", None); continue
        lo, hi = rep["idr_range"]
        jg = [float(np.linalg.norm(ca[j] - ca[k]))
              for j, k in ((lo - 1, lo), (hi + 1, hi)) if j in ca and k in ca]
        rep["junction_gap"] = max(jg) if jg else None
    # worst template/fragment seam
    seam_vals = [float(np.linalg.norm(ca[j] - ca[k])) for (j, k) in seams if j in ca and k in ca]
    for rep in reports:
        rep["seam_gap"] = max(seam_vals) if seam_vals else None
    return reports
