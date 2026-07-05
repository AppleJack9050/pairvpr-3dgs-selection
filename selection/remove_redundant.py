# Copyright 2024-present CSIRO.
# Licensed under CSIRO BSD-3-Clause-Clear (Non-Commercial)
#
# Remove redundant images: select a diverse, non-redundant subset from a precomputed
# similarity matrix using Facility Location (a submodular coverage objective).
#
# Input:  an N x N similarity matrix S with S[i, j] in [0, 1] (higher = more similar),
#         exactly as produced by similarity_matrix.py, plus the matching image_order.txt.
# Output: a subset of K images that best *covers* the dataset -- i.e. the K least-redundant,
#         most representative images. The complement (the N - K images left out) is the set
#         of "redundant" images that have a close-enough representative in the kept subset.
#
# Method (default): greedily maximize the Facility Location objective
#
#         f(S) = sum_{i in V} max_{j in S} sim(i, j)
#
# a monotone, submodular "coverage" function over the ground set V of all images. Each term
# rewards the subset for having *some* selected image that is similar to image i, so f(S) is
# large when every image is well represented by its nearest selected neighbour. Greedy
# maximization (Nemhauser et al.: a (1 - 1/e) approximation) spreads the selection across the
# dataset: once an image is picked, near-duplicates of it give almost no marginal gain, so
# they are skipped -- exactly the behaviour we want for redundancy removal.
#
# A second method, --method maxmin (farthest-point / k-center on dissimilarity 1 - S), is
# offered as a complement: it directly maximizes the *minimum* pairwise dissimilarity, i.e.
# the most mutually-uncorrelated subset.
#
# This script depends only on numpy (no torch / Pair-VPR), so it runs anywhere the matrix
# file can be read. It lives in the project root alongside similarity_matrix.py.

import os
import csv
import heapq
import shutil
import argparse
from glob import glob
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).parent.resolve()
DEFAULT_MATRIX = "results_simmatrix/similarity_matrix.npy"


# --------------------------------------------------------------------------------------
# Load + validate
# --------------------------------------------------------------------------------------
def resolve_matrix_arg(path: str) -> str:
    """Accept either a matrix FILE (.npy/.csv) or a results_simmatrix FOLDER.

    Given a folder, locate the similarity matrix inside it. Prefer the bare
    `similarity_matrix.npy`; fall back to a single `similarity_matrix*.npy` (e.g. the
    --method both `_pair`/`_global` variant) or, if no .npy exists, a single .csv. If several
    candidates are ambiguous, ask the user to pass one explicitly.
    """
    if not os.path.isdir(path):
        return path
    bare = os.path.join(path, "similarity_matrix.npy")
    if os.path.exists(bare):
        return bare
    for pattern in ("similarity_matrix*.npy", "similarity_matrix*.csv"):
        hits = sorted(glob(os.path.join(path, pattern)))
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise SystemExit(
                f"Multiple similarity matrices in folder '{path}':\n  " + "\n  ".join(hits) +
                f"\nPass one explicitly, e.g. --matrix {hits[0]}")
    raise FileNotFoundError(
        f"No similarity_matrix*.npy or .csv found in folder '{path}'. "
        f"Is this a results_simmatrix folder produced by similarity_matrix.py?")


def load_matrix(matrix_path: str) -> np.ndarray:
    """Load the N x N similarity matrix (.npy or .csv) written by similarity_matrix.py."""
    if not os.path.exists(matrix_path):
        # similarity_matrix.py --method both writes *_pair.npy / *_global.npy (no bare file);
        # point the user at whichever siblings actually exist so recovery is obvious.
        base, ext = os.path.splitext(matrix_path)
        existing = [base + suf + ext for suf in ("_pair", "_global")
                    if os.path.exists(base + suf + ext)]
        hint = ""
        if existing:
            hint = ("\nFound --method both outputs instead; pass one explicitly:\n  "
                    + "\n  ".join(f"--matrix {a}" for a in existing))
        raise FileNotFoundError(
            f"Similarity matrix not found at '{matrix_path}'.\n"
            f"Run similarity_matrix.py first, or pass --matrix /path/to/similarity_matrix.npy"
            f" (or the results_simmatrix folder)"
            + hint
        )
    if os.path.splitext(matrix_path)[1].lower() == ".csv":
        S = np.loadtxt(matrix_path, delimiter=",")
    else:
        S = np.load(matrix_path)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError(f"Expected a square 2-D matrix, got shape {S.shape}")
    return S.astype(np.float64)


def cross_check_csv(matrix_path: str, S: np.ndarray, max_n: int = 3000) -> None:
    """Confirm the sibling .csv holds the same matrix as the .npy, so whichever file the user
    inspects has the same values in the same row/column order. Best-effort, never fatal."""
    if os.path.splitext(matrix_path)[1].lower() != ".npy":
        return  # loaded from the csv itself -> nothing to compare against
    csv_path = os.path.splitext(matrix_path)[0] + ".csv"
    if not os.path.exists(csv_path):
        return
    name = os.path.basename(csv_path)
    if S.shape[0] > max_n:
        print(f"   [skip] .npy/{name} cross-check (n={S.shape[0]} > {max_n}; csv parse slow)")
        return
    try:
        C = np.loadtxt(csv_path, delimiter=",")
    except Exception as e:  # noqa: BLE001
        print(f"---WARNING could not parse {name} for cross-check ({e})")
        return
    if C.ndim != 2 or C.shape != S.shape:
        print(f"---WARNING {name} shape {getattr(C, 'shape', None)} != .npy {S.shape}; "
              f"they look like different runs. Using the .npy.")
        return
    maxdiff = float(np.abs(C - S).max())
    if maxdiff <= 2e-6:  # csv is written with fmt=%.6f, so rounding is <= 5e-7
        print(f"   [PASS] .npy and {name} agree (max diff {maxdiff:.1e}) "
              f"-> same matrix, same row/col order")
    else:
        print(f"---WARNING .npy and {name} differ by up to {maxdiff:.3g} (csv is saved at 6 "
              f"decimals; larger gaps mean different runs). Using the .npy.")


def default_order_path(matrix_path: str) -> str:
    """image_order.txt sits next to the matrix (same convention as similarity_matrix.py)."""
    return os.path.join(os.path.dirname(os.path.abspath(matrix_path)), "image_order.txt")


def load_names(order_path: str, n: int):
    """Return the per-row image names, or None if the order file is absent."""
    if order_path is None or not os.path.exists(order_path):
        return None
    with open(order_path) as f:
        names = [line.rstrip("\n") for line in f if line.strip() != ""]
    if len(names) != n:
        raise ValueError(
            f"image_order.txt has {len(names)} names but the matrix is {n}x{n}. "
            f"They must line up row-for-row -- pass the matching --image_order, or the "
            f"matrix and order file are from different runs."
        )
    return names


def validate_matrix(S: np.ndarray, sym_tol: float = 1e-3) -> np.ndarray:
    """Mirror similarity_matrix.py's self-check; symmetrize / clip defensively (with warnings)."""
    n = S.shape[0]
    if not np.isfinite(S).all():
        raise ValueError("Similarity matrix contains non-finite values (NaN/Inf).")

    lo, hi = float(S.min()), float(S.max())
    if lo < -1e-4 or hi > 1 + 1e-4:
        print(f"---WARNING matrix values fall outside [0, 1] (min={lo:.4f}, max={hi:.4f}); "
              f"clipping. Expected similarities in [0, 1] (higher = more similar).")
        S = np.clip(S, 0.0, 1.0)

    asym = float(np.abs(S - S.T).max()) if n > 1 else 0.0
    if asym > sym_tol:
        print(f"---WARNING matrix is not symmetric (max |S - S^T| = {asym:.4g}); "
              f"symmetrizing as (S + S^T) / 2.")
        S = 0.5 * (S + S.T)
    return S


def resolve_subset_size(args, n: int) -> int:
    """Turn --subset_size / --fraction into a concrete count, validated against N."""
    if (args.subset_size is None) == (args.fraction is None):
        raise SystemExit("Specify exactly one of --subset_size/-k (a count) or --fraction (0 < f < 1).")
    if args.fraction is not None:
        if not (0.0 < args.fraction < 1.0):
            raise SystemExit(f"--fraction must be in (0, 1), got {args.fraction}")
        k = int(round(args.fraction * n))
    else:
        k = args.subset_size
    k = max(1, min(k, n))
    return k


# --------------------------------------------------------------------------------------
# Facility Location (submodular coverage) -- lazy greedy (CELF)
# --------------------------------------------------------------------------------------
def facility_location_greedy(S: np.ndarray, k: int):
    """Maximize f(S) = sum_i max_{j in S} S[i, j] with the lazy-greedy / CELF algorithm.

    CELF (Leskovec et al. 2007) exploits submodularity -- marginal gains never increase -- to
    avoid recomputing every candidate each round, yet returns the *identical* set to the naive
    greedy. Returns (selected_idx, marginal_gains, objective_after_each_pick) in pick order.
    """
    n = S.shape[0]
    coverage = np.zeros(n, dtype=np.float64)   # coverage[i] = max_{j in S} S[i, j]
    cov_sum = 0.0                              # = f(current S)
    selected, gains, objective = [], [], []
    in_selected = np.zeros(n, dtype=bool)

    # Upper-bound heap of (-gain, idx, round_last_computed). The singleton gain f({c}) - f({})
    # equals the column sum, since coverage starts at 0 and S >= 0.
    init_gain = S.sum(axis=0)
    heap = [(-float(init_gain[c]), c, 0) for c in range(n)]
    heapq.heapify(heap)

    while len(selected) < k:
        neg_gain, c, last_round = heapq.heappop(heap)
        if in_selected[c]:
            continue
        if last_round == len(selected):
            # Gain was recomputed against the current coverage and is still the heap max,
            # so by submodularity it is the true best candidate -- commit it.
            g = -neg_gain
            selected.append(c)
            in_selected[c] = True
            coverage = np.maximum(coverage, S[:, c])
            cov_sum += g
            gains.append(g)
            objective.append(cov_sum)
        else:
            # Stale upper bound: recompute the real marginal gain and re-insert.
            g = float(np.maximum(coverage, S[:, c]).sum() - cov_sum)
            heapq.heappush(heap, (-g, c, len(selected)))

    return np.array(selected, dtype=int), np.array(gains), np.array(objective)


def facility_objective(S: np.ndarray, idx) -> float:
    """f(S) = sum_i max_{j in S} S[i, j] for a given selection (recomputed from scratch)."""
    idx = np.asarray(idx, dtype=int)
    if idx.size == 0:
        return 0.0
    return float(np.maximum.reduce(S[:, idx], axis=1).sum())


def facility_location_greedy_naive(S: np.ndarray, k: int) -> np.ndarray:
    """Plain O(k*N^2) greedy, used only by --verify to cross-check the CELF implementation."""
    n = S.shape[0]
    coverage = np.zeros(n, dtype=np.float64)
    selected, mask = [], np.ones(n, dtype=bool)
    for _ in range(k):
        gains = np.maximum(coverage[:, None], S).sum(axis=0) - coverage.sum()
        gains[~mask] = -np.inf
        c = int(np.argmax(gains))
        selected.append(c)
        mask[c] = False
        coverage = np.maximum(coverage, S[:, c])
    return np.array(selected, dtype=int)


# --------------------------------------------------------------------------------------
# Max-min dispersion (farthest-point / k-center) -- the "most uncorrelated" complement
# --------------------------------------------------------------------------------------
def maxmin_greedy(S: np.ndarray, k: int):
    """Greedy farthest-point traversal on dissimilarity d = 1 - S.

    Seeds with the most central image (largest total similarity) for a deterministic start,
    then repeatedly adds the image *least* similar to everything already chosen -- maximizing
    the minimum pairwise dissimilarity of the subset. Returns (selected_idx, min_dissim_curve).
    """
    n = S.shape[0]
    max_sim = np.full(n, -np.inf)              # max_sim[i] = max_{j in S} S[i, j]
    selected, min_dissim = [], []

    first = int(np.argmax(S.sum(axis=0)))
    selected.append(first)
    max_sim = np.maximum(max_sim, S[:, first])

    while len(selected) < k:
        cand = max_sim.copy()
        cand[selected] = np.inf                # never reselect (we take the argmin)
        c = int(np.argmin(cand))               # least similar to the current subset
        # The dissimilarity gained == 1 - (its similarity to the nearest selected image).
        min_dissim.append(1.0 - float(max_sim[c]))
        selected.append(c)
        max_sim = np.maximum(max_sim, S[:, c])

    return np.array(selected, dtype=int), np.array(min_dissim)


# --------------------------------------------------------------------------------------
# Reporting + output
# --------------------------------------------------------------------------------------
def offdiag_mean(sub: np.ndarray) -> float:
    """Mean of the strictly-upper-triangular (pairwise) similarities of a sub-matrix."""
    m = sub.shape[0]
    if m < 2:
        return float("nan")
    iu = np.triu_indices(m, k=1)
    return float(sub[iu].mean())


def report(S: np.ndarray, selected: np.ndarray, method: str, objective):
    """Diversity self-check: the kept subset should be less internally similar than the whole
    set, and (for facility) should cover the dataset well. Mirrors similarity_matrix.py style."""
    n, k = S.shape[0], len(selected)
    sub = S[np.ix_(selected, selected)]
    full_mean = offdiag_mean(S)
    sub_mean = offdiag_mean(sub)
    # Coverage = average best-similarity of every image to its nearest selected representative,
    # in [0, 1]; higher = the subset represents the dataset better. (f(S) / N.)
    coverage = float(np.maximum.reduce(S[:, selected], axis=1).mean()) if k else 0.0

    print("---Selection report:")
    print(f"   method                         : {method}")
    print(f"   kept / total                   : {k} / {n}  (removed {n - k})")
    print(f"   mean pairwise sim (kept subset): {sub_mean:.4f}")
    print(f"   mean pairwise sim (whole set)  : {full_mean:.4f}")
    if np.isfinite(sub_mean) and np.isfinite(full_mean):
        verdict = "more diverse (good)" if sub_mean <= full_mean else "LESS diverse (!)"
        print(f"   -> kept subset is {verdict}")
    print(f"   dataset coverage f(S)/N        : {coverage:.4f}  (1.0 = every image has an "
          f"identical pick)")
    if objective is not None and len(objective):
        print(f"   facility objective f(S)        : {float(objective[-1]):.4f}")


def save_outputs(out_dir: str, selected: np.ndarray, names, n: int,
                 gains, objective, method: str):
    os.makedirs(out_dir, exist_ok=True)
    selected = np.asarray(selected, dtype=int)
    kept = set(selected.tolist())
    removed = np.array([i for i in range(n) if i not in kept], dtype=int)  # original order

    idx_path = os.path.join(out_dir, "selected_indices.npy")
    np.save(idx_path, selected)
    print(f"---Saved {idx_path}  ({len(selected)} indices, in selection order)")

    if names is not None:
        keep_path = os.path.join(out_dir, "selected_images.txt")
        with open(keep_path, "w") as f:
            f.write("\n".join(names[i] for i in selected) + "\n")
        print(f"---Saved {keep_path}  ({len(selected)} kept, selection order)")

        rm_path = os.path.join(out_dir, "removed_images.txt")
        with open(rm_path, "w") as f:
            if len(removed):
                f.write("\n".join(names[i] for i in removed) + "\n")
        print(f"---Saved {rm_path}  ({len(removed)} removed/redundant)")

    # Per-pick CSV: rank, index, image, marginal gain, cumulative objective.
    csv_path = os.path.join(out_dir, "selection_report.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "index", "image", "marginal_gain", "cumulative_objective"])
        for rank, i in enumerate(selected):
            img = names[i] if names is not None else ""
            g = f"{float(gains[rank]):.6f}" if gains is not None and rank < len(gains) else ""
            o = f"{float(objective[rank]):.6f}" if objective is not None and rank < len(objective) else ""
            w.writerow([rank, int(i), img, g, o])
    print(f"---Saved {csv_path}")


def export_subset(selected, names, images_dir: str, export_dir: str, mode: str) -> int:
    """Materialize the kept images into export_dir as a ready-to-use sub-dataset.

    Names in image_order.txt are stored relative to step 1's --images_dir, so we need that
    same source root to find the actual files. The kept files are copied (default: real
    independent files) or sym-/hard-linked, preserving any relative sub-folders.
    """
    if names is None:
        raise SystemExit(
            "--export_dir needs image filenames, but image_order.txt was not found next to the "
            "matrix. Pass --image_order /path/to/image_order.txt.")
    os.makedirs(export_dir, exist_ok=True)
    n_ok = n_missing = 0
    for i in selected:
        rel = names[int(i)]
        src = rel if os.path.isabs(rel) else os.path.join(images_dir, rel)
        if not os.path.exists(src):
            print(f"---WARNING source image missing, skipping: {src}")
            n_missing += 1
            continue
        # Mirror the relative layout under export_dir (flat basename if the name is absolute).
        dst = os.path.join(export_dir, rel) if not os.path.isabs(rel) \
            else os.path.join(export_dir, os.path.basename(rel))
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        if os.path.lexists(dst):
            os.remove(dst)
        if mode == "symlink":
            os.symlink(os.path.abspath(src), dst)
        elif mode == "hardlink":
            os.link(src, dst)
        else:  # copy
            shutil.copy2(src, dst)
        n_ok += 1
    msg = f"---Exported {n_ok} images to {export_dir} (mode={mode})"
    if n_missing:
        msg += f"; {n_missing} source file(s) missing (see warnings above)"
    print(msg)
    return n_ok


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def get_args_parser(add_help: bool = True):
    p = argparse.ArgumentParser(
        "Remove redundant images via Facility Location on a similarity matrix", add_help=add_help)
    p.add_argument("--matrix", "--sim_matrix", "--sim-matrix", type=str, default=DEFAULT_MATRIX,
                   help="The N x N similarity matrix from similarity_matrix.py: either the "
                        ".npy/.csv file, or the whole results_simmatrix FOLDER (the matrix and "
                        "image_order.txt are auto-located inside it).")
    p.add_argument("--image_order", "--image-order", type=str, default=None,
                   help="Path to image_order.txt (default: alongside the matrix / in the folder).")
    p.add_argument("--subset_size", "-k", "--subset-size", type=int, default=None,
                   help="Number of images to KEEP (the diverse subset size).")
    p.add_argument("--fraction", type=float, default=None,
                   help="Alternative to -k: keep this fraction of the dataset (0 < f < 1).")
    p.add_argument("--method", choices=["facility", "maxmin"], default="facility",
                   help="facility = Facility Location coverage (recommended); "
                        "maxmin = farthest-point max-min dispersion (leaves the "
                        "marginal_gain/cumulative_objective columns blank in selection_report.csv).")
    p.add_argument("--output_dir", "--output-dir", type=str, default="results_subset")
    p.add_argument("--images_dir", "--images-dir", type=str, default=None,
                   help="Source folder the names in image_order.txt are relative to (e.g. the "
                        "step-1 --images_dir). Required to export the actual files with --export_dir.")
    p.add_argument("--export_dir", "--export-dir", type=str, default=None,
                   help="If set, also copy/link the SELECTED image files here -- a ready-to-use "
                        "sub-dataset (e.g. to re-run COLMAP on the pruned set).")
    p.add_argument("--export_mode", "--export-mode", choices=["copy", "symlink", "hardlink"],
                   default="copy",
                   help="How to materialize exported images: copy = real independent files "
                        "(default); symlink = links (instant, no disk); hardlink = real files "
                        "sharing disk blocks (no extra space, same filesystem only).")
    p.add_argument("--no_metadata", "--no-metadata", action="store_true",
                   help="Do NOT write the selected/removed lists, indices .npy, or report .csv "
                        "(keep only the exported images). Useful when you just want the sub-dataset.")
    p.add_argument("--verify", action="store_true",
                   help="Cross-check the lazy-greedy result against the naive greedy (small N).")
    return p


def main(args):
    # --matrix may be a .npy/.csv file OR a results_simmatrix folder; resolve to a file.
    matrix_path = resolve_matrix_arg(args.matrix)
    if matrix_path != args.matrix:
        print(f"---Input folder: {args.matrix}")
    print(f"---Matrix: {matrix_path}")
    S = load_matrix(matrix_path)
    S = validate_matrix(S)
    n = S.shape[0]
    cross_check_csv(matrix_path, S)  # guarantee the .npy and .csv hold the same matrix/order

    order_path = args.image_order or default_order_path(matrix_path)
    names = load_names(order_path, n)  # raises unless image_order.txt has exactly n rows
    if names is not None:
        # Make the row <-> filename alignment concrete and checkable at a glance.
        print(f"---Image order: {order_path}  ({n} names == matrix dim {n})")
        print(f"   row 0   -> {names[0]}")
        print(f"   row {n - 1:<4}-> {names[-1]}  (each row/col i of the matrix is image i)")
    else:
        print(f"---Image names: none found at {order_path}; using integer indices")

    k = resolve_subset_size(args, n)
    if k == n:
        print(f"---Note: subset size {k} == dataset size {n}; nothing is redundant to remove.")
    print(f"---Selecting {k} of {n} images, method='{args.method}'")

    if args.method == "facility":
        selected, gains, objective = facility_location_greedy(S, k)
        if args.verify:
            naive = facility_location_greedy_naive(S, k)
            f_lazy, f_naive = facility_objective(S, selected), facility_objective(S, naive)
            # CELF returns the *same objective* as naive greedy. The exact index order can
            # differ only on floating-point near-ties (e.g. the gain-zero tail when k ~ N),
            # which are algorithmically equivalent, so we compare the objective, not the order.
            tol = 1e-9 * max(1.0, abs(f_naive))
            ok = abs(f_lazy - f_naive) <= tol
            print(f"   [{'PASS' if ok else 'FAIL'}] lazy CELF objective == naive greedy "
                  f"(f_lazy={f_lazy:.6f}, f_naive={f_naive:.6f})")
            if not ok:
                raise SystemExit("Lazy and naive greedy reach different objectives -- a bug, aborting.")
    else:
        selected, _curve = maxmin_greedy(S, k)
        gains, objective = None, None

    report(S, selected, args.method, objective)
    if args.no_metadata:
        if not args.export_dir:
            print("---WARNING --no_metadata and no --export_dir: nothing will be written.")
        else:
            print("---Skipping list/index/report files (--no_metadata); exporting images only.")
    else:
        save_outputs(args.output_dir, selected, names, n, gains, objective, args.method)
    if args.export_dir:
        if not args.images_dir:
            raise SystemExit("--export_dir requires --images_dir (the source folder of the actual "
                             "image files; names in image_order.txt are relative to it).")
        export_subset(selected, names, args.images_dir, args.export_dir, args.export_mode)
    print("done")


if __name__ == "__main__":
    main(get_args_parser().parse_args())
