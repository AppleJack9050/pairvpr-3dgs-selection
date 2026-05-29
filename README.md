# Pair-VPR Pairwise Similarity Matrix

Compute an **N×N similarity matrix** over a whole folder of images using
[Pair-VPR](https://csiro-robotics.github.io/Pair-VPR/) (vitG model). Entry `S[i, j]` is the
similarity between image *i* and image *j*, normalized to **[0, 1]** where **higher = more
similar**. The diagonal is 1.0 and the matrix is symmetric.

## Folder layout

```
elsevier_2026/
├── similarity_matrix.py     # the script (run this)
├── similarity_matrix.sh     # convenience wrapper
├── requirements.txt
├── README.md                # this file
└── Pair-VPR/                # the Pair-VPR repo checkout
    ├── pairvpr/             #   model + config package (imported by the script)
    └── trained_models/      #   put pairvpr-vitG.pth here
```

`similarity_matrix.py` lives in the project root (not inside `Pair-VPR/`). It adds `Pair-VPR/`
to `sys.path` and auto-locates the config and checkpoint there, so you can run it from this
folder with no path arguments.

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

**1. Install PyTorch for your GPU, then the rest of the deps.** For the RTX 5090:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

**2. Download the Pair-VPR vitG checkpoint (~4.7 GB)** into `Pair-VPR/trained_models/`:

```bash
wget -O Pair-VPR/trained_models/pairvpr-vitG.pth \
  https://huggingface.co/CSIRORobotics/Pair-VPR/resolve/main/pairvpr-vitG.pth
```

**3. DINOv2 backbone.** On first run the model fetches `dinov2_vitg14_reg` via `torch.hub`
(cached under `~/.cache/torch/hub`). This needs internet the first time; afterwards it runs
offline.

> Note: this repo includes a small fix to `Pair-VPR/pairvpr/models/tools/blocks.py` so the
> model runs **without** xformers (the upstream non-xformers cross-attention fallback was
> broken). No action needed — it's already applied.

## Usage

From this folder:

```bash
# Default: Pair-VPR pair-classifier similarity, auto-selects GPU
python similarity_matrix.py --images_dir /path/to/your/dataset --save_csv

# Fast alternative: cosine of global descriptors
python similarity_matrix.py --images_dir /path/to/your/dataset --method global

# Compute both matrices
python similarity_matrix.py --images_dir /path/to/your/dataset --method both --save_csv
```

Or edit the image folder in [similarity_matrix.sh](similarity_matrix.sh) and run `bash similarity_matrix.sh`.

### Key options

| Flag | Default | Description |
|------|---------|-------------|
| `--images_dir` | *(required)* | Folder of images to score (searched recursively). |
| `--method` | `pair` | `pair` (Pair-VPR classifier), `global` (descriptor cosine), or `both`. |
| `--output_dir` | `results_simmatrix` | Where outputs are written. |
| `--trained_ckpt` | `Pair-VPR/trained_models/pairvpr-vitG.pth` | Checkpoint path. |
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
