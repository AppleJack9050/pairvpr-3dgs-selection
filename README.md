# Diverse image-subset selection — a multi-backend VPR toolkit

Select maximally-diverse, redundancy-pruned image subsets from a larger dataset (e.g. for a
"reconstruction quality vs #images" ablation, or to prune a set before COLMAP / SfM). The pipeline
is always the **same two steps**; only the embedding model in step 1 differs:

```
  images/ ──▶  [1] similarity matrix        ──▶  [2] Facility Location selection  ──▶  subset/
              backends/<model>/*.py               selection/remove_redundant.py
              (pairvpr | edtformer | fol |        (shared, numpy-only, model-agnostic)
               unipr3d)  →  NxN S in [0,1]
```

1. **Similarity matrix** — embed every image with a Visual Place Recognition (VPR) model and build
   an `N×N` matrix `S` with `S[i, j] ∈ [0, 1]` (higher = more similar; diagonal 1.0, symmetric).
   Four interchangeable **backends** produce this matrix in one common format.
2. **Selection** — `selection/remove_redundant.py` greedily maximizes the **Facility Location**
   submodular objective `f(S) = Σ_i max_{j∈S} S[i,j]` (CELF lazy-greedy, a `(1 − 1/e)`
   approximation) to keep the least-redundant, most representative subset of the requested size.
   It depends only on **numpy** and is identical for every backend — the matrix format is the
   contract between the two steps, so backends are freely swappable.

This repo unifies four previously-separate tools (Pair-VPR / EDTformer / FoL / UniPR-3D) into one
codebase with a **single shared selector**.

## Repository layout

```
elsevier_2026/
├── README.md                 # this file
├── requirements.txt          # install ALL backends at once
├── requirements/             # per-backend dependency files
│   ├── base.txt              #   torch, torchvision, numpy, Pillow, tqdm  (shared)
│   ├── pairvpr.txt  edtformer.txt  fol.txt  unipr3d.txt
├── selection/
│   └── remove_redundant.py   # THE shared, model-agnostic selector (Facility Location / max-min)
├── backends/
│   ├── pairvpr/similarity_matrix.py      # Pair-VPR (vitG): pair-classifier OR global-cosine
│   ├── edtformer/edtformer_similarity.py # EDTformer (DINOv2 ViT-B/14 + decoder): global-cosine
│   ├── fol/fol_similarity.py             # FoL "Focus on Local" (DINOv2 ViT-L): global-cosine
│   └── unipr3d/unipr3d_similarity.py     # UniPR-3D (VGGT + DINOv2 / SALAD): global-cosine
├── bash/
│   ├── run_all.sh            # end-to-end for any backend (matrix + selection over sizes)
│   ├── similarity_matrix.sh  # step 1 only, model-aware dispatcher
│   ├── remove_redundant.sh   # step 2 only
│   └── make_subsets.sh       # step 2b: a nested series of subsets (e.g. 300..N)
├── Pair-VPR/                 # git submodule (Pair-VPR model code)         [pairvpr]
├── EDTformer/                # you clone this (git-ignored)                [edtformer]
├── UniPR-3D/                 # you clone this (git-ignored)                [unipr3d]
└── models/                   # UniPR-3D checkpoint lands here (git-ignored)
```

## The four backends

| Backend | Model | Descriptor / similarity | Extra setup needed |
|---------|-------|-------------------------|--------------------|
| **pairvpr** | [Pair-VPR](https://csiro-robotics.github.io/Pair-VPR/) vitG (DINOv2-G) | 2nd-stage **pair classifier** (default) or global cosine | `Pair-VPR/` **submodule** + vitG checkpoint (~4.7 GB, kept outside the repo) |
| **edtformer** | [EDTformer](https://github.com/Tong-Jin01/EDTformer) (DINOv2 ViT-B/14 + decoder) | 4096-d global, cosine | clone `EDTformer/`; weights (~449 MB) auto-download from torch.hub |
| **fol** | [FoL](https://github.com/chenshunpeng/FoL) "Focus on Local" (DINOv2 ViT-L) | 8448-d global, cosine | **nothing** — model + backbone auto-download from torch.hub |
| **unipr3d** | [UniPR-3D](https://github.com/dtc111111/UniPR-3D) (VGGT + DINOv2 / SALAD + LoRA) | 17152-d global, cosine | clone `UniPR-3D/`; `single_model.ckpt` (~3.7 GB) auto-downloads from HF |

**Which to use?** `pairvpr --method pair` is the most accurate but O(N²) decoder passes (heavy;
fine for a few hundred images). The three global-cosine backends are O(N) extraction + one matmul,
so they scale to thousands of images cheaply — `fol` is the easiest to set up (no clone, no manual
checkpoint). All four write the identical matrix format, so you can compare them on the same set.

## Setup

**1. Python env + deps.** Python 3.11+ (verified on 3.13), PyTorch ≥ 2.7 with a CUDA build matching
your GPU. On an **RTX 5090 / Blackwell (sm_120)** you must use a **CUDA 12.8** torch build
(`torch 2.11.0+cu128`), or kernels fail with *"no kernel image is available"*:

```bash
conda create -n pairvpr python=3.13 && conda activate pairvpr
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt                 # all backends
# ...or just one backend, e.g.:  pip install -r requirements/fol.txt
```

The `bash/` wrappers auto-activate the `pairvpr` conda env (override with `CONDA_ENV=<name>`). The
selection step (`selection/remove_redundant.py`) is numpy-only, so any env works for it.

**2. Per-backend model code + weights.**

- **pairvpr** — init the submodule and download the vitG checkpoint (kept outside the repo):
  ```bash
  git submodule update --init Pair-VPR
  wget -O /mnt/windows/model/pairvpr_models/pairvpr-vitG.pth \
    https://huggingface.co/CSIRORobotics/Pair-VPR/resolve/main/pairvpr-vitG.pth
  ```
  Override the checkpoint path with `--trained_ckpt`. (This repo ships a small fix to
  `Pair-VPR/pairvpr/models/tools/blocks.py` so it runs without xformers.)
- **edtformer** — clone the model code (weights auto-download on first run):
  ```bash
  git clone https://github.com/Tong-Jin01/EDTformer.git EDTformer
  ```
- **fol** — nothing to do. On first run it fetches `chenshunpeng/FoL` + `facebookresearch/dinov2`
  into `~/.cache/torch/hub` (needs internet once, then runs offline).
- **unipr3d** — clone the model code (checkpoint auto-downloads into `models/`):
  ```bash
  git clone https://github.com/dtc111111/UniPR-3D.git UniPR-3D
  ```

`EDTformer/` and `UniPR-3D/` are git-ignored — they are external clones, not committed here.

## Usage

### Quick start — end to end (any backend)

`bash/run_all.sh <backend> <images_dir> [sizes…]` builds the matrix once, then selects each size:

```bash
bash bash/run_all.sh fol      /path/to/images                 # sizes 300 332 364 (default)
bash bash/run_all.sh pairvpr  /path/to/images  50 100 200
bash bash/run_all.sh unipr3d  /path/to/images
```

Outputs land under `results_<backend>/` (git-ignored):
`results_<backend>/results_simmatrix/` (the matrix) and `results_<backend>/subset_<k>/` (per size:
`selected_images.txt`, `selected_indices.npy`, `removed_images.txt`, `selection_report.csv`, and
`images/` symlinks). Override `OUT_DIR`, `EXPORT_MODE` (`symlink|copy|hardlink`), `METHOD`.

### The two steps by hand

```bash
# Step 1 — similarity matrix (model-aware wrapper; forwards extra flags to the backend)
bash bash/similarity_matrix.sh fol /path/to/images --output_dir results_simmatrix
#   ...or call the backend script directly:
python backends/fol/fol_similarity.py --images_dir /path/to/images --save_csv

# Step 2 — selection from that matrix (numpy-only; the pairvpr env is optional here)
python selection/remove_redundant.py --matrix results_simmatrix -k 300 \
    --images_dir /path/to/images --export_dir subset_300/images --export_mode symlink
```

`--matrix` accepts the whole `results_simmatrix` **folder** (it finds `similarity_matrix.npy` and
`image_order.txt` inside) or a specific `.npy`/`.csv`. Use `--fraction 0.2` instead of `-k` to keep
a percentage; `--method maxmin` for farthest-point (max-min dispersion) instead of coverage;
`--verify` to cross-check the lazy greedy against the naive greedy on small sets.

### A nested series of subsets

`bash/make_subsets.sh` loops the selector over evenly-spaced sizes (default 10 subsets from 300 up
to the full size N) and exports each as its own COLMAP-ready `images/` folder. Because the greedy is
deterministic and prefix-consistent, the subsets are **nested** (`subset_300 ⊂ … ⊂ subset_N`):

```bash
MATRIX=results_simmatrix IMAGES_DIR=/path/to/images bash bash/make_subsets.sh /out/subsets
bash bash/make_subsets.sh -h    # all options (--start, -n, --method, --mode, …)
```

## Output format (the backend↔selector contract)

Every backend writes, into `--output_dir`:

- `similarity_matrix.npy` — float32 `N×N` matrix in `[0, 1]` (`pairvpr --method both` writes
  `_pair` / `_global` variants instead of a bare file).
- `image_order.txt` — image paths (relative to `--images_dir`), one per line, in matrix row/column
  order. `--save_csv` also writes `similarity_matrix.csv`.

The selector maps matrix row/column *i* to line *i* of `image_order.txt`; it errors unless the
counts match and prints the first/last filenames so you can eyeball the alignment. Load a matrix
back with:

```python
import numpy as np
S = np.load("results_simmatrix/similarity_matrix.npy")
names = open("results_simmatrix/image_order.txt").read().splitlines()
```

## How selection works

`f(S) = Σ_i max_{j∈S} sim(i, j)` is a monotone submodular **coverage** function: each image
contributes its similarity to the most similar *selected* image, so `f(S)` is large only when every
image is well represented by some pick. Greedy maximization spreads the selection out — once an
image is chosen, near-duplicates of it yield almost no marginal gain and are skipped, which is
exactly "remove redundant images". `--method maxmin` instead maximizes the *minimum* pairwise
dissimilarity (the most mutually-uncorrelated subset). The selector prints a diversity self-check:
the kept subset's mean pairwise similarity should be **lower** than the whole set's.

## Notes & troubleshooting

- **`no kernel image is available` on GPU** → your torch CUDA build doesn't match the GPU. For the
  RTX 5090 reinstall torch from the cu128 index (see Setup).
- **GPU out of memory** → lower `--extract_batch` (and, for `pairvpr`, `--pair_batch`). `pairvpr`
  supports `--store_fp16` to halve dense-map RAM.
- **First run needs internet** for `fol` / `edtformer` (torch.hub) and `unipr3d` (HF checkpoint);
  afterwards the caches are warm and they run offline.
- **Scaling** → the `pairvpr` pair method is O(N²) decoder passes; for thousands of images use a
  global-cosine backend (`fol` / `edtformer` / `unipr3d`, or `pairvpr --method global`). Selection
  itself is cheap (CELF is ~O(N²) once, then near-linear per pick).

> This repository combines four formerly-independent tools. EDTformer originally bundled its own
> `apricot`-based selection in a single `select_diverse_subset.py`; it is now
> `backends/edtformer/edtformer_similarity.py` (matrix only) + the shared
> `selection/remove_redundant.py`, matching the other three backends (same result — both maximize
> the same Facility Location objective with lazy greedy).
