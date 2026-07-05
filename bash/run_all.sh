#!/usr/bin/env bash
# End-to-end for ANY backend: build the similarity matrix, then run Facility Location selection
# for one or more subset sizes from that single matrix.
#
# Usage:
#   bash bash/run_all.sh <backend> <images_dir> [size ...]
#
#   <backend>  = pairvpr | edtformer | fol | unipr3d
#   <size ...> = subset sizes to select (default: 300 332 364)
#
# Outputs go under OUT_DIR (default <repo>/results_<backend>/), which is git-ignored:
#   results_<backend>/results_simmatrix/   the NxN matrix (+ image_order.txt)
#   results_<backend>/subset_<k>/          selected lists/report + images/ (symlinks) per size
#
# Because the facility-location greedy is deterministic and prefix-consistent, the subsets are
# NESTED (subset_300 subset_332 ...), an ideal "quality vs #images" ablation.  Override defaults
# with env vars: OUT_DIR, EXPORT_MODE (symlink|copy|hardlink), METHOD (facility|maxmin), CONDA_ENV.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BACKEND="${1:-}"
IMAGES="${2:-}"
if [[ -z "$BACKEND" || -z "$IMAGES" ]]; then
    echo "Usage: bash bash/run_all.sh <pairvpr|edtformer|fol|unipr3d> <images_dir> [size ...]" >&2
    exit 2
fi
shift 2
SIZES=("$@")
[[ ${#SIZES[@]} -eq 0 ]] && SIZES=(300 332 364)

OUT_DIR="${OUT_DIR:-$ROOT/results_$BACKEND}"
M="$OUT_DIR/results_simmatrix"
EXPORT_MODE="${EXPORT_MODE:-symlink}"
METHOD="${METHOD:-facility}"
mkdir -p "$OUT_DIR"

echo "=== Step 1: [$BACKEND] similarity matrix over: $IMAGES ==="
# Delegate to similarity_matrix.sh so each backend gets its recommended flags + conda activation.
bash "$ROOT/bash/similarity_matrix.sh" "$BACKEND" "$IMAGES" --output_dir "$M"

echo
echo "=== Step 2: Facility Location selection (${SIZES[*]}) ==="
for K in "${SIZES[@]}"; do
    echo "--- selecting $K ---"
    python "$ROOT/selection/remove_redundant.py" \
        --matrix "$M" \
        -k "$K" \
        --method "$METHOD" \
        --output_dir "$OUT_DIR/subset_$K" \
        --images_dir "$IMAGES" \
        --export_dir "$OUT_DIR/subset_$K/images" \
        --export_mode "$EXPORT_MODE"
done

echo
echo "=== Done. Outputs under $OUT_DIR ==="
echo "  matrix : $M/similarity_matrix.npy (+ image_order.txt)"
for K in "${SIZES[@]}"; do
    echo "  k=$K   : $OUT_DIR/subset_$K/  (selected_images.txt, selected_indices.npy, selection_report.csv, images/)"
done
