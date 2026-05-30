#!/usr/bin/env bash
# Generate a series of evenly-sized diverse subsets of a dataset from a precomputed Pair-VPR
# similarity matrix, using Facility Location (remove_redundant.py).
#
# It writes N_SUBSETS subset folders with sizes evenly spaced from START up to the full dataset
# size N (default 300 -> 589 in 10 steps). Each subset_<k>/ holds only an images/ folder with the
# k selected image files (real copies by default; COLMAP-ready). No metadata files are written.
#
# Because the facility-location greedy is deterministic and prefix-consistent, the subsets are
# NESTED:  subset_300 ⊂ subset_332 ⊂ ... ⊂ subset_589  (each larger set just adds images),
# which is ideal for an "reconstruction quality vs #images" ablation.
#
# The largest size equals N (the whole dataset), so that subset is simply the full set -- no
# real pruning happens there ("no need to split the final set").
#
# Usage:   bash bash/make_subsets.sh [OUTPUT_DIR] [options]
#   OUTPUT_DIR is the directory that will CONTAIN the generated subset_<k>/ folders
#   (subset_300/, subset_332/, ...). Run with -h/--help for all options. Every option also has
#   an UPPERCASE env-var form, e.g.:
#     bash bash/make_subsets.sh /data/sfm/subsets --start 250 -n 8 --mode copy
#     OUT_DIR=/data/sfm/subsets START=250 N_SUBSETS=8 bash bash/make_subsets.sh
set -euo pipefail

# ----------------------------------- configuration ------------------------------------------
SFM_DIR="${SFM_DIR:-/mnt/windows/2016-11-28_Howchin-AlphLake_Imagery-Files.beh/sfm}"
MATRIX="${MATRIX:-$SFM_DIR/results_simmatrix}"   # folder OR .npy/.csv (remove_redundant.py resolves it)
IMAGES_DIR="${IMAGES_DIR:-$SFM_DIR/images}"      # source image files
OUT_DIR="${OUT_DIR:-$SFM_DIR/subsets}"           # where the subset_<k>/ folders are written
METHOD="${METHOD:-facility}"                     # facility (recommended) | maxmin
EXPORT_MODE="${EXPORT_MODE:-copy}"               # copy (real files, default) | hardlink | symlink
START="${START:-300}"                            # smallest subset size
N_SUBSETS="${N_SUBSETS:-10}"                      # how many subsets to make

# ------------------------------- command-line overrides -------------------------------------
# CLI flags take precedence over the env-var defaults above. The output location (the dir that
# will contain subset_<k>/) can be given as a bare positional argument or via -o/--out.
usage() {
    cat <<EOF
Generate evenly-sized, nested diverse subsets from a Pair-VPR similarity matrix.

Usage: bash bash/make_subsets.sh [OUTPUT_DIR] [options]

  OUTPUT_DIR             directory that will CONTAIN the subset_<k>/ folders
                        (e.g. subset_300/, subset_332/, ...). Positional, or use -o/--out.

Options (each also settable via the UPPERCASE env var in brackets):
  -o, --out DIR         output location, holds the subset_<k>/ folders   [OUT_DIR]
  -m, --matrix PATH     similarity matrix folder or .npy/.csv            [MATRIX]
  -i, --images DIR      source image files                               [IMAGES_DIR]
      --start N         smallest subset size                             [START=$START]
  -n, --num N           number of subsets                                [N_SUBSETS=$N_SUBSETS]
      --method M        facility | maxmin                                [METHOD=$METHOD]
      --mode M          copy | hardlink | symlink                        [EXPORT_MODE=$EXPORT_MODE]
  -h, --help            show this help

Examples:
  bash bash/make_subsets.sh /data/sfm/subsets
  bash bash/make_subsets.sh --out /data/sfm/subsets --start 250 -n 8 --mode copy
EOF
}

require_val() { [[ $# -ge 2 ]] || { echo "ERROR: $1 requires a value" >&2; exit 2; }; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)                      usage; exit 0 ;;
        -o|--out|--output|--output-dir) require_val "$@"; OUT_DIR="$2"; shift 2 ;;
        -m|--matrix)                    require_val "$@"; MATRIX="$2"; shift 2 ;;
        -i|--images|--images-dir)       require_val "$@"; IMAGES_DIR="$2"; shift 2 ;;
        --start)                        require_val "$@"; START="$2"; shift 2 ;;
        -n|--num|--num-subsets)         require_val "$@"; N_SUBSETS="$2"; shift 2 ;;
        --method)                       require_val "$@"; METHOD="$2"; shift 2 ;;
        --mode|--export-mode)           require_val "$@"; EXPORT_MODE="$2"; shift 2 ;;
        --)                             shift; break ;;
        -*)                             echo "ERROR: unknown option '$1'" >&2; usage; exit 2 ;;
        *)                              OUT_DIR="$1"; shift ;;   # bare positional = output location
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # the bash/ folder
ROOT="$(dirname "$SCRIPT_DIR")"                              # repo root
RR="$ROOT/python/remove_redundant.py"                       # the tool lives in python/

# Pick a Python interpreter (remove_redundant.py is numpy-only, so the conda env is optional).
PY="${PYTHON:-python}"
command -v "$PY" >/dev/null 2>&1 || PY=python3

# Activate the project's conda env if present (override with CONDA_ENV=<name>); harmless to skip.
CONDA_ENV="${CONDA_ENV:-pairvpr}"
if command -v conda >/dev/null 2>&1 && [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV" 2>/dev/null || true
fi

# ---------------------------- locate image_order.txt and size N -----------------------------
if [[ -d "$MATRIX" ]]; then ORDER_FILE="$MATRIX/image_order.txt"
else ORDER_FILE="$(dirname "$MATRIX")/image_order.txt"; fi
if [[ ! -f "$ORDER_FILE" ]]; then
    echo "ERROR: image_order.txt not found at '$ORDER_FILE' (set MATRIX to the results_simmatrix folder)." >&2
    exit 1
fi
# Pass the path via argv (not string interpolation) so paths with quotes/backslashes are safe.
N="$("$PY" -c "import sys; print(sum(1 for l in open(sys.argv[1]) if l.strip()))" "$ORDER_FILE")"

# Evenly-spaced, de-duplicated, clamped sizes from START to N (pure Python, no numpy needed).
SIZES="$("$PY" -c "
import sys
s,e,m=int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3])
v=[e] if m<=1 else [round(s+i*(e-s)/(m-1)) for i in range(m)]
print(' '.join(map(str, sorted({min(max(x,1),e) for x in v}))))
" "$START" "$N" "$N_SUBSETS")"

echo "=== dataset N=$N images; subset sizes: $SIZES ==="
echo "    matrix : $MATRIX"
echo "    images : $IMAGES_DIR"
echo "    output : $OUT_DIR   (method=$METHOD, export=$EXPORT_MODE)"
mkdir -p "$OUT_DIR"

# --------------------------------- generate each subset -------------------------------------
for k in $SIZES; do
    out="$OUT_DIR/subset_$k"
    echo ""
    echo "######################## subset k=$k  ->  $out ########################"
    if [[ "$k" -ge "$N" ]]; then
        echo "    (k=$k == N=$N: this is the full dataset, no pruning)"
    fi
    rm -rf "$out"                                   # clean re-run; only removes our outputs
    "$PY" "$RR" \
        --matrix "$MATRIX" \
        --method "$METHOD" \
        -k "$k" \
        --images_dir "$IMAGES_DIR" \
        --export_dir "$out/images" \
        --export_mode "$EXPORT_MODE" \
        --no_metadata
done

# ------------------------------------- summary ----------------------------------------------
echo ""
echo "=== done: subsets under $OUT_DIR ==="
for k in $SIZES; do
    cnt=$(find "$OUT_DIR/subset_$k/images" -maxdepth 1 \( -type f -o -type l \) 2>/dev/null | wc -l)
    printf "  subset_%-4s -> %3s images   (%s)\n" "$k" "$cnt" "$OUT_DIR/subset_$k/images"
done
