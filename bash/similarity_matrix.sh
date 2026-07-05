#!/usr/bin/env bash
# Compute an NxN similarity matrix over a folder of images with ANY of the four VPR backends
# (values in [0,1], higher = more similar). Runnable from anywhere.
#
# Usage:
#   bash bash/similarity_matrix.sh <backend> <images_dir> [extra args passed to the backend]
#
#   <backend> = pairvpr | edtformer | fol | unipr3d
#
# Each backend is dispatched to its script under backends/<backend>/ with that model's recommended
# defaults; any extra args you pass are forwarded verbatim (and override the defaults), e.g.
#   bash bash/similarity_matrix.sh fol /data/images --output_dir /data/results_simmatrix
#   bash bash/similarity_matrix.sh pairvpr /data/images --method global
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BACKEND="${1:-}"
IMAGES="${2:-}"
if [[ -z "$BACKEND" || -z "$IMAGES" ]]; then
    echo "Usage: bash bash/similarity_matrix.sh <pairvpr|edtformer|fol|unipr3d> <images_dir> [extra args]" >&2
    exit 2
fi
shift 2   # remaining "$@" is forwarded to the backend script

# Map backend -> script + that model's recommended flags (see each backend's header / the README).
case "$BACKEND" in
    pairvpr)
        SCRIPT="$ROOT/backends/pairvpr/similarity_matrix.py"
        DEFAULTS=(--method pair --save_csv) ;;
    edtformer)
        SCRIPT="$ROOT/backends/edtformer/edtformer_similarity.py"
        DEFAULTS=(--resize 322 322 --extract_batch 32 --save_csv) ;;
    fol)
        SCRIPT="$ROOT/backends/fol/fol_similarity.py"
        DEFAULTS=(--backbone vitl --resize 504 504 --extract_batch 16 --save_csv) ;;
    unipr3d)
        SCRIPT="$ROOT/backends/unipr3d/unipr3d_similarity.py"
        DEFAULTS=(--resize 392 518 --extract_batch 4 --save_csv) ;;
    *)
        echo "ERROR: unknown backend '$BACKEND' (expected pairvpr|edtformer|fol|unipr3d)" >&2
        exit 2 ;;
esac

# Activate the project's conda env (override with CONDA_ENV=<name>). All backends need torch.
CONDA_ENV="${CONDA_ENV:-pairvpr}"
if command -v conda >/dev/null 2>&1 && [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
echo "=== [$BACKEND] similarity matrix over: $IMAGES ==="
python "$SCRIPT" --images_dir "$IMAGES" "${DEFAULTS[@]}" "$@"
