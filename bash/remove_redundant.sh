#!/usr/bin/env bash
# Select a diverse, non-redundant subset of images from a precomputed Pair-VPR similarity
# matrix using Facility Location (a submodular coverage objective). Runnable from anywhere.
# This step is pure-numpy (no torch / GPU), so the conda env is optional.
set -euo pipefail

# Repo root = parent of this bash/ folder; the shared selector lives in <root>/selection/.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Activate the project's conda env if present (override with CONDA_ENV=<name>); harmless to skip.
CONDA_ENV="${CONDA_ENV:-pairvpr}"
if command -v conda >/dev/null 2>&1 && [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV" 2>/dev/null || true
fi

# Keep the 50 most representative / least-redundant images from a matrix already computed
# by bash/similarity_matrix.sh. Use --fraction 0.2 instead of -k to keep a percentage.
#
# To also write the actual image FILES as a ready-to-use sub-dataset, set IMAGES_DIR (the
# step-1 source folder that image_order.txt names are relative to) and EXPORT_DIR below.
IMAGES_DIR="${IMAGES_DIR:-}"
EXPORT_DIR="${EXPORT_DIR:-}"
EXPORT_MODE="${EXPORT_MODE:-copy}"   # real files by default (symlink | hardlink to override)
export_args=()
[[ -n "$IMAGES_DIR" ]] && export_args+=(--images_dir "$IMAGES_DIR")
[[ -n "$EXPORT_DIR" ]] && export_args+=(--export_dir "$EXPORT_DIR" --export_mode "$EXPORT_MODE")

python "$ROOT/selection/remove_redundant.py" \
    --matrix results_simmatrix \
    --method facility \
    -k 50 \
    --output_dir results_subset \
    "${export_args[@]}"
