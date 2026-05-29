#!/usr/bin/env bash
# Compute an NxN Pair-VPR similarity matrix over a folder of images (values in [0,1], higher = more similar).
# Run from the project root (this folder). The script auto-finds Pair-VPR/ for the package, config and checkpoint.
export CUDA_VISIBLE_DEVICES=0
python similarity_matrix.py --images_dir /YOURIMAGEFOLDER --method pair --output_dir results_simmatrix --save_csv
