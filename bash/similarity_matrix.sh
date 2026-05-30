#!/usr/bin/env bash
# Compute an NxN Pair-VPR similarity matrix over a folder of images (values in [0,1], higher = more similar).
# Runnable from anywhere. The script auto-finds Pair-VPR/ for the package, config and checkpoint.
set -euo pipefail

# Repo root = parent of this bash/ folder; the Python scripts live in <root>/python/.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Activate the project's conda env (override with CONDA_ENV=<name>).
CONDA_ENV="${CONDA_ENV:-pairvpr}"
if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

export CUDA_VISIBLE_DEVICES=0
python "$ROOT/python/similarity_matrix.py" --images_dir /YOURIMAGEFOLDER --method pair --output_dir results_simmatrix --save_csv
