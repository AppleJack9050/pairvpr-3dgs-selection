# Pair-VPR Pairwise Similarity Matrix

Compute an **N×N similarity matrix** over a whole folder of images using
[Pair-VPR](https://csiro-robotics.github.io/Pair-VPR/) (vitG model). Entry `S[i, j]` is the
similarity between image *i* and image *j*, normalized to **[0, 1]** where **higher = more
similar**. The diagonal is 1.0 and the matrix is symmetric.

## Folder layout

```
elsevier_2026/
├── python/                  # the Python tools
│   ├── similarity_matrix.py #   step 1: build the NxN similarity matrix
│   └── remove_redundant.py  #   step 2: pick a diverse, non-redundant subset (numpy-only)
├── bash/                    # convenience wrappers (run these, or call python/ directly)
│   ├── similarity_matrix.sh #   step 1 wrapper
│   ├── remove_redundant.sh  #   step 2 wrapper
│   └── make_subsets.sh      #   step 2b: generate a series of nested subsets (e.g. 300..N)
├── requirements.txt
├── README.md                # this file
└── Pair-VPR/                # the Pair-VPR repo (git submodule)
    └── pairvpr/             #   model + config package (imported by the script)
```

The `pairvpr-vitG.pth` checkpoint is kept **outside** the repo (default
`/mnt/windows/model/pairvpr_models/pairvpr-vitG.pth`); override with `--trained_ckpt`.

Run everything **from the project root** (e.g. `python python/similarity_matrix.py …`).
`python/similarity_matrix.py` finds the sibling `Pair-VPR/` checkout one level up, adds it to
`sys.path`, and auto-locates the config and checkpoint there, so no path arguments are needed.
The `bash/` wrappers locate the repo root from their own path, so they work from anywhere.

## Requirements

- **Python 3.11+** (verified on 3.13).
- **PyTorch ≥ 2.7** with a CUDA build that matches your GPU. A GPU is strongly recommended
  (the vitG model is heavy; CPU works but is slow).
  - **RTX 5090 / Blackwell (sm_120):** you must use a **CUDA 12.8** build of torch
    (this environment uses `torch 2.11.0+cu128`). Older cu126/cu121 builds fail with
    *"no kernel image is available for execution on the device"*.
- torchvision, numpy, Pillow, omegaconf, tqdm (see [requirements.txt](requirements.txt)).
- `xformers` and `faiss` are **not** required — the model uses a pure-PyTorch attention
  fallback, and the matrix is built with plain torch.

## Setup

**1. Create the conda env, install PyTorch for your GPU, then the rest of the deps.**
For the RTX 5090:

```bash
conda create -n pairvpr python=3.13
conda activate pairvpr
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

The `bash/similarity_matrix.sh` wrapper auto-activates the `pairvpr` env (override the
env name with `CONDA_ENV=<name>`).

**2. Download the Pair-VPR vitG checkpoint (~4.7 GB).** It is kept **outside** the
repo; the script defaults to `/mnt/windows/model/pairvpr_models/pairvpr-vitG.pth`:

```bash
wget -O /mnt/windows/model/pairvpr_models/pairvpr-vitG.pth \
  https://huggingface.co/CSIRORobotics/Pair-VPR/resolve/main/pairvpr-vitG.pth
```

**3. DINOv2 backbone.** On first run the model fetches `dinov2_vitg14_reg` via `torch.hub`
(cached under `~/.cache/torch/hub`). This needs internet the first time; afterwards it runs
offline.

> Note: this repo includes a small fix to `Pair-VPR/pairvpr/models/tools/blocks.py` so the
> model runs **without** xformers (the upstream non-xformers cross-attention fallback was
> broken). No action needed — it's already applied.

## Usage

From the project root:

```bash
conda activate pairvpr

# Default: Pair-VPR pair-classifier similarity, auto-selects GPU
python python/similarity_matrix.py --images_dir /path/to/your/dataset --save_csv

# Fast alternative: cosine of global descriptors
python python/similarity_matrix.py --images_dir /path/to/your/dataset --method global

# Compute both matrices
python python/similarity_matrix.py --images_dir /path/to/your/dataset --method both --save_csv
```

Or edit the image folder in [bash/similarity_matrix.sh](bash/similarity_matrix.sh) and run
`bash bash/similarity_matrix.sh` — the wrapper activates the `pairvpr` conda env for you.

### Key options

| Flag | Default | Description |
|------|---------|-------------|
| `--images_dir` | *(required)* | Folder of images to score (searched recursively). |
| `--method` | `pair` | `pair` (Pair-VPR classifier), `global` (descriptor cosine), or `both`. |
| `--output_dir` | `results_simmatrix` | Where outputs are written. |
| `--trained_ckpt` | `/mnt/windows/model/pairvpr_models/pairvpr-vitG.pth` | Checkpoint path. |
| `--config-file` | `Pair-VPR/pairvpr/configs/pairvpr_performance.yaml` | Pair-VPR config (vitG). |
| `--device` | `auto` | `auto` / `cuda` / `cpu`. |
| `--extract_batch` | `32` | Image batch size for feature extraction (lower if GPU OOM). |
| `--pair_batch` | `64` | Pair-decoder chunk size (lower if GPU OOM). |
| `--store_fp16` | off | Store dense feature maps as fp16 to halve RAM. |
| `--save_csv` | off | Also write the matrix as CSV. |
| `--no-force-diagonal` | off | Keep the computed self-similarity instead of forcing the diagonal to 1.0. |
| `--max_images` | `2000` | Safety guard; raise with `--allow_large` for big sets. |

Supported image types: `.jpg .jpeg .png .bmp .tif .tiff .webp`.

## Output

Written to `--output_dir` (default `results_simmatrix/`):

- `similarity_matrix.npy` — float32 N×N matrix in [0, 1] (`_pair` / `_global` suffix when `--method both`).
- `image_order.txt` — image paths, one per line, in the same order as the matrix rows/columns.
- `similarity_matrix.csv` — only with `--save_csv`.

Load it back with:

```python
import numpy as np
S = np.load("results_simmatrix/similarity_matrix.npy")
names = open("results_simmatrix/image_order.txt").read().splitlines()
```

## Removing redundant images (diverse subset selection)

Once you have a similarity matrix, [python/remove_redundant.py](python/remove_redundant.py) selects a
**size-K subset of the least-redundant, most representative images** and treats the rest as
redundant. It maximizes the **Facility Location** objective — a monotone *submodular* coverage
function

```
f(S) = Σ_i  max_{j ∈ S} sim(i, j)
```

with the lazy-greedy (CELF) algorithm. Each image *i* contributes its similarity to the most
similar *selected* image, so `f(S)` is large only when every image is well represented by some
pick. Greedy maximization spreads the selection out: once an image is chosen, near-duplicates of
it yield almost no marginal gain and are skipped — exactly what "remove redundant images" needs.
It depends only on **numpy** (no torch/GPU), so the `pairvpr` env is optional for this step.

```bash
# Keep the 50 most representative / least-redundant images.
# --matrix accepts the whole results_simmatrix FOLDER (the .npy and image_order.txt are found
# inside it); passing the .npy or .csv file directly also works.
python python/remove_redundant.py --matrix results_simmatrix -k 50

# Or keep a fraction of the dataset
python python/remove_redundant.py --fraction 0.2

# Alternative objective: farthest-point max-min dispersion (most mutually-uncorrelated)
python python/remove_redundant.py -k 50 --method maxmin

# Cross-check the lazy greedy against the naive greedy (small sets)
python python/remove_redundant.py -k 50 --verify

# Materialize the subset as an actual folder of images (a ready-to-use sub-dataset).
# image_order.txt stores names relative to step 1's --images_dir, so point --images_dir at it.
python python/remove_redundant.py \
    --matrix /data/sfm/results_simmatrix/similarity_matrix.npy \
    -k 120 \
    --images_dir /data/sfm/images \
    --export_dir /data/sfm/subset/images        # 120 copied images you can run COLMAP on
```

Or run [bash/remove_redundant.sh](bash/remove_redundant.sh) after editing the size near the bottom.

### Key options

| Flag | Default | Description |
|------|---------|-------------|
| `--matrix` | `results_simmatrix/similarity_matrix.npy` | The N×N matrix from step 1: the `.npy`/`.csv` file **or** the whole `results_simmatrix` folder (auto-locates the matrix + `image_order.txt`). |
| `--image_order` | *(next to the matrix)* | `image_order.txt` mapping rows → image paths. |
| `--subset_size` / `-k` | *(one of these)* | Number of images to **keep**. |
| `--fraction` | *(one of these)* | Keep this fraction of the dataset (0 < f < 1) instead of `-k`. |
| `--method` | `facility` | `facility` (Facility Location coverage, recommended) or `maxmin` (max-min dispersion). |
| `--output_dir` | `results_subset` | Where the subset files (lists/CSV) are written. |
| `--images_dir` | *(none)* | Source folder the names in `image_order.txt` are relative to; needed for `--export_dir`. |
| `--export_dir` | *(none)* | If set, copy/link the **selected image files** here — a ready-to-use sub-dataset. |
| `--export_mode` | `copy` | `copy` (real independent files), `symlink` (links, no disk), or `hardlink` (real files, no extra disk, same filesystem). |
| `--no_metadata` | off | Skip the lists/index/report files — write only the exported images (the sub-dataset). |
| `--verify` | off | Assert the lazy-greedy objective equals the naive greedy (small sets). |

Outputs (written to `--output_dir`, default `results_subset/`):

- `selected_indices.npy` — int row indices of the kept images, in selection order (most
  representative first).
- `selected_images.txt` — the kept image paths, selection order (only when `image_order.txt` is found).
- `removed_images.txt` — the redundant images that were dropped.
- `selection_report.csv` — per-pick `rank, index, image, marginal_gain, cumulative_objective`
  (the last two columns are populated only for `--method facility`; blank for `maxmin`).
- With `--export_dir`: the actual selected image files (real copies by default; `--export_mode
  symlink`/`hardlink` to override), mirroring their relative paths, ready to feed straight back
  into COLMAP / your SfM pipeline.

**Order consistency.** Row/column *i* of the matrix is image *i* of `image_order.txt`. The
script enforces this — it errors unless `image_order.txt` has exactly N lines, prints the row 0
and row N−1 filenames so you can eyeball the alignment, and (when a sibling `.csv` is present)
verifies the `.npy` and `.csv` hold the same matrix in the same order before selecting.

> If step 1 was run with `--method both`, it writes `similarity_matrix_pair.npy` and
> `similarity_matrix_global.npy` (no bare `similarity_matrix.npy`), so pass the one you want
> explicitly, e.g. `--matrix results_simmatrix/similarity_matrix_pair.npy`.

### Generating a series of subsets (`bash/make_subsets.sh`)

To produce several subsets at once — e.g. for a "reconstruction quality vs #images" ablation —
[bash/make_subsets.sh](bash/make_subsets.sh) loops `remove_redundant.py` over evenly-spaced sizes
and exports each as its own COLMAP-ready folder. Defaults: 10 subsets from 300 up to the full
dataset size N, facility method, real-file **copy** export.

```bash
bash bash/make_subsets.sh                                 # 300..N in 10 steps, into <sfm>/subsets/

# Choose WHERE the subset_<k>/ folders go (positional arg, or -o/--out):
bash bash/make_subsets.sh /data/sfm/subsets
bash bash/make_subsets.sh --out /data/sfm/subsets --start 250 -n 8 --mode copy
bash bash/make_subsets.sh -h                              # all options
```

The first positional argument (or `-o/--out`) is the **output location** — the directory that
will contain the `subset_<k>/` folders. Other flags: `-m/--matrix`, `-i/--images`, `--start`,
`-n/--num`, `--method`, `--mode`. Each also has an UPPERCASE env-var form (`OUT_DIR`, `MATRIX`,
`IMAGES_DIR`, `START`, `N_SUBSETS`, `METHOD`, `EXPORT_MODE`, plus `SFM_DIR`); CLI flags win over
env vars. It writes `<out>/subset_<k>/`, each containing **only an `images/` folder** with the
selected image files (it passes `--no_metadata`, so no lists/index/report files are written),
and prints a size summary. The largest size equals N, so that subset is just the full dataset
(no pruning).

> **Disk note:** copy mode writes real, independent files, so the nested subsets duplicate
> image data — the default 10 subsets of a 589 × ~5 MB dataset total roughly 20 GB. If you want
> real files without the duplication (and the output sits on the same filesystem as the source),
> use `--mode hardlink`; for zero-disk links, `--mode symlink`.

Because the facility-location greedy is **deterministic and prefix-consistent**, the subsets are
**nested**: `subset_300 ⊂ subset_332 ⊂ … ⊂ subset_589` — each larger set only *adds* images, so
the series is a clean controlled experiment.

The script prints a diversity self-check: the kept subset's mean pairwise similarity should be
**lower** than the whole set's (more diverse), alongside the dataset coverage `f(S)/N ∈ [0, 1]`.
For thousands of images prefer `--method global` in step 1 so the O(N²) matrix stays affordable;
selection itself is cheap (CELF is roughly O(N²) once, then near-linear per pick).

## How it works

1. **Extract** each image's dense feature map and global descriptor once
   (`model(img, None, mode="global")`).
2. **Pair method (default):** for every ordered pair, run Pair-VPR's second-stage
   cross-attention classifier to get a logit `L[i,j]`, then fuse both directions and squash
   to [0, 1]: `S = sigmoid((L + Lᵀ) / 2)`. Higher logit = more similar (the classifier is
   trained with `BCEWithLogitsLoss`, target=1 for "same place").
   **Global method:** cosine similarity of the L2-normalized descriptors, mapped via
   `(cos + 1) / 2`.

The script runs a built-in self-check after computing the matrix (square, finite, in [0, 1],
symmetric, diagonal handling).

## Notes & troubleshooting

- **`no kernel image is available` on GPU** → your torch CUDA build doesn't match the GPU.
  For the RTX 5090, reinstall from the cu128 index (see Setup).
- **GPU out of memory** → lower `--extract_batch` and/or `--pair_batch`. Dense maps are kept
  on CPU and streamed to the GPU in chunks; `--store_fp16` halves their RAM footprint.
- **Cost** → the pair method is O(N²) decoder passes. It's fast for a few hundred images on a
  modern GPU; for thousands of images prefer `--method global` (O(N) extraction + one matmul).
- **CPU run** → add `--device cpu`. Correct but slow for the vitG model; fine for small tests.
