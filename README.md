# Towards Efficient 3D Gaussian Splatting Reconstruction with PairVPR-Based Image Selection: A Glacier UAV Case Study

Official code for the paper **"Towards Efficient 3D Gaussian Splatting Reconstruction with
PairVPR-Based Image Selection: A Glacier UAV Case Study."**

Sicheng Zhao¹²,  Arjun Pakrashi²,  Soumyabrata Dev¹²³ (corresponding author)

¹ ADAPT SFI Research Centre, Dublin, Ireland  ·  ² School of Computer Science, University College
Dublin, Ireland  ·  ³ School of Computer Science and Statistics, Trinity College Dublin, Ireland

📧 `sicheng.zhao@ucdconnect.ie` · `devs@tcd.ie`

> **Status:** manuscript submitted to *Elsevier Applied Computing and Geosciences* (gold
> open-access). DOI and final links will be added on acceptance — see [Citation](#citation).

---

## TL;DR

UAV surveys capture hundreds of highly overlapping images, most of which are redundant for 3D
reconstruction. This repository **selects a small, diverse, non-redundant subset** of the images so
that **3D Gaussian Splatting (3DGS)** reconstruction runs substantially faster — with little loss of
quality at moderate budgets, and a graceful trade-off under more aggressive pruning. Selection is a
two-step pipeline:

```
  images/ ──▶ [1] pairwise similarity  ──▶ [2] Facility-Location selection ──▶ subset/ ──▶ COLMAP + 3DGS
             a VPR model → NxN matrix       greedy submodular coverage         (external reconstruction)
             K_ij ∈ [0,1]  (Sec. 3.2)       (Algorithm 1, Sec. 3.3)
```

The paper's proposed selector uses **PairVPR** pairwise similarity; the repository also implements
the three descriptor-based baselines from the ablation (FoL, UniPR-3D, EDTformer) behind a common
interface, so every method is reproducible with one command.

## Abstract

Large-scale 3D reconstruction from UAV imagery is essential for remote sensing and environmental
monitoring, yet modern neural rendering methods such as 3D Gaussian Splatting (3DGS) are
computationally intensive due to redundant views. We present a scalable image subset-selection
framework that integrates transformer-based Visual Place Recognition (PairVPR) with a
facility-location selection strategy to identify and remove redundant views while maintaining
spatial coverage. On six standard benchmark scenes (five Mip-NeRF360 scenes and the Tanks\&Temples
*Truck* scene), under a fixed image budget PairVPR-based selection attains the best average PSNR,
SSIM, and LPIPS among all evaluated VPR selectors. In a case study on a real-world glacier UAV
survey of 589 images, pruning to a compact subset reduces overall runtime by more than half at the
most aggressive setting while incurring only a modest loss in reconstruction quality, and PairVPR
again attains the best reconstruction quality among the evaluated selectors.

## Method in one paragraph

For every unordered image pair, PairVPR produces a symmetric similarity `K_ij ∈ [0,1]` (higher =
more overlapping; `K_ii ≈ 1`). Subset selection is cast as **facility location** — pick the size-`k`
subset `S` that maximizes the monotone submodular coverage objective

```
f(S) = Σ_i  max_{j∈S} K_ij
```

so that every image in the collection has a close representative in `S`. The greedy algorithm
(Algorithm 1 in the paper) gives a `(1 − 1/e) ≈ 0.63` approximation, starts from the empty set (its
first pick is automatically the collection medoid), and prunes near-duplicates because once a region
of viewpoint space is covered, further views there add almost no marginal gain. The selected subset
is then reconstructed with a standard **COLMAP (SfM) → 3DGS** pipeline.

## What this repository provides

This repo covers **the image-selection stage** (the paper's contribution). Reconstruction (COLMAP +
3DGS) uses the standard external tools with the exact settings in
[Implementation details](#implementation-details). Selection is split into two reusable steps and
supports **four interchangeable VPR backends** — the proposed method plus the three ablation
baselines:

| Backend (this repo) | Role in the paper | Model | Similarity |
|---------------------|-------------------|-------|------------|
| **`pairvpr`** | **proposed method** | [Pair-VPR](https://csiro-robotics.github.io/Pair-VPR/) **PairVPR-p (ViT-G)**, frozen | 2nd-stage **pair classifier** `K_ij∈[0,1]` (default) |
| `fol` | ablation baseline | [FoL](https://github.com/chenshunpeng/FoL) "Focus on Local" (DINOv2 ViT-L) | global-descriptor cosine |
| `unipr3d` | ablation baseline | [UniPR-3D](https://github.com/dtc111111/UniPR-3D) (VGGT + DINOv2 / SALAD) | global-descriptor cosine |
| `edtformer` | ablation baseline | [EDTformer](https://github.com/Tong-Jin01/EDTformer) (DINOv2 ViT-B/14 + decoder) | global-descriptor cosine |

All four write the **same matrix format**, and all feed the **same** facility-location selector, so
the paper's selection-method ablation varies *only* the similarity model — exactly as reproduced here.
PairVPR-p (ViT-G) is used as a frozen, off-the-shelf pairwise scorer from the official HuggingFace
release ([CSIRORobotics/Pair-VPR](https://huggingface.co/CSIRORobotics/Pair-VPR)); no PairVPR weights
are fine-tuned.

## Repository structure

```
.
├── selection/
│   └── remove_redundant.py     # Algorithm 1: greedy facility-location selection (numpy only)
├── backends/                   # step 1 — one similarity script per VPR model
│   ├── pairvpr/similarity_matrix.py       # PairVPR (proposed): --method pair | global
│   ├── fol/fol_similarity.py              # FoL (baseline)
│   ├── unipr3d/unipr3d_similarity.py      # UniPR-3D (baseline)
│   └── edtformer/edtformer_similarity.py  # EDTformer (baseline)
├── bash/
│   ├── run_all.sh              # end-to-end: matrix + selection over several budgets k
│   ├── similarity_matrix.sh    # step 1 only (model-aware dispatcher)
│   ├── remove_redundant.sh     # step 2 only
│   └── make_subsets.sh         # a nested series of subsets (e.g. 300..N) for the budget sweep
├── requirements/               # per-backend dependency files (+ union requirements.txt)
├── Pair-VPR/                   # git submodule (PairVPR model code)                [pairvpr]
├── EDTformer/                  # you clone this (git-ignored)                      [edtformer]
├── UniPR-3D/                   # you clone this (git-ignored)                      [unipr3d]
└── models/                     # UniPR-3D checkpoint lands here (git-ignored)
```

## Setup

**1. Python environment.** Python 3.11+ (verified on 3.13), PyTorch ≥ 2.7 with a CUDA build matching
your GPU. On an **RTX 5090 / Blackwell (sm_120)** install torch from the **CUDA 12.8** wheel index
first, or kernels fail with *"no kernel image is available"*:

```bash
conda create -n pairvpr python=3.13 && conda activate pairvpr
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt            # all backends
# ...or a single backend, e.g. the proposed method only:
pip install -r requirements/pairvpr.txt
```

**2. Model code + weights** (only for the backend you use):

- **pairvpr (proposed)** — init the submodule and download the PairVPR-p (ViT-G) checkpoint
  (~4.7 GB, kept outside the repo; override the path with `--trained_ckpt`):
  ```bash
  git submodule update --init Pair-VPR
  wget -O /mnt/windows/model/pairvpr_models/pairvpr-vitG.pth \
    https://huggingface.co/CSIRORobotics/Pair-VPR/resolve/main/pairvpr-vitG.pth
  ```
- **fol** — nothing to do (model + DINOv2 backbone auto-download from torch.hub on first run).
- **unipr3d** — `git clone https://github.com/dtc111111/UniPR-3D.git UniPR-3D` (checkpoint
  auto-downloads into `models/`).
- **edtformer** — `git clone https://github.com/Tong-Jin01/EDTformer.git EDTformer` (weights
  auto-download from the v1.0.0 torch.hub release).

## Reproducing the paper

### Step 1 + 2 — select a diverse subset

The one-shot wrapper builds the similarity matrix once and then selects each budget `k`:

```bash
# <backend> ∈ {pairvpr, fol, unipr3d, edtformer};  proposed method = pairvpr
bash bash/run_all.sh pairvpr /path/to/glacier/images 300 332 364 396 428 461 493 525 557
```

or run the two steps by hand:

```bash
# Step 1 — pairwise similarity matrix (writes results_simmatrix/similarity_matrix.npy + image_order.txt)
python backends/pairvpr/similarity_matrix.py --images_dir /path/to/images --method pair --save_csv

# Step 2 — facility-location selection for a budget k (Algorithm 1; numpy only, no GPU needed)
python selection/remove_redundant.py --matrix results_simmatrix -k 396 \
    --images_dir /path/to/images --export_dir subset_396/images --export_mode symlink
```

Each `subset_<k>/` gets `selected_images.txt`, `selected_indices.npy`, `removed_images.txt`,
`selection_report.csv`, and an `images/` folder ready for COLMAP. To generate the whole nested
budget sweep at once, see `bash/make_subsets.sh`.

> **Held-out protocol (paper).** For the glacier study the fixed 74-image test set (every 8th image)
> is always retained and excluded from 3DGS training, so a budget `k` corresponds to `k − 74`
> training views; benchmark scenes use `k = 100` training views with every 8th image held out. The
> selector here chooses the subset from the similarity matrix; the train/test split is applied around
> it per this protocol.

### Step 3 — reconstruct (external: COLMAP + 3DGS)

Run each selected `subset_<k>/images` through a standard **COLMAP SfM → 3DGS** pipeline
([graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)) using
the settings in [Implementation details](#implementation-details), then evaluate PSNR / SSIM / LPIPS
on the held-out test views and read SfM reprojection RMSE from COLMAP. This stage is not vendored
here; the numbers in the paper come from that off-the-shelf pipeline with fixed settings.

## Datasets

- **Glacier UAV survey (case study).** 589 images from a DJI FC330 (pinhole + radial/tangential
  distortion), native `3992×2992` (GSD 3.49 mm/px at 11.6 m altitude, ≈ `1.92×10³` m²). Images are
  undistorted and downscaled to a maximum of `2000×1498`. Fixed test set = every 8th image (74
  images). Evaluated budgets `k ∈ {300, 332, 364, 396, 428, 461, 493, 525, 557}` plus the full set
  (589). *Availability: see the paper / contact the authors.*
- **Mip-NeRF360** (benchmark) — five outdoor scenes: Garden, Bicycle, Stump, Treehill, Flowers.
- **Tanks & Temples** (benchmark) — the *Truck* scene.

## Implementation details

Exact settings used for every reconstruction (identical across all runs; from the paper's
implementation-details table):

| Component | Configuration |
|-----------|---------------|
| **COLMAP** — feature extraction | SIFT, max 8192 features |
| COLMAP — peak threshold | 0.01 |
| COLMAP — matching | exhaustive matcher |
| COLMAP — bundle adjustment | default |
| **3DGS** — mode | default |
| 3DGS — position learning rate | `1.6×10⁻⁴` |
| 3DGS — training iterations | 30k |
| 3DGS — Gaussian count | 10k |
| 3DGS — spherical harmonics | degree 3 |
| 3DGS — densification | 500 |
| **Environment** | NVIDIA RTX 5090 (32 GB) · Ubuntu 24.04.2 · CUDA 12.9 / cuDNN 9.1.2 · PyTorch 2.8.0 |

## Results (headline)

**Benchmark scenes — best VPR selector at a fixed budget `k = 100`** (mean over 3 seeds; full
per-scene numbers are in the paper). PairVPR gives the best average on all three metrics:

| Selector | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|----------|:------:|:------:|:-------:|
| **PairVPR (ours)** | **20.75** | **0.606** | **0.372** |
| EDTformer | 20.47 | 0.602 | 0.380 |
| FoL | 19.77 | 0.590 | 0.380 |
| UniPR-3D | 19.62 | 0.580 | 0.391 |

**Glacier quality–efficiency sweep** (PairVPR selector). Quality saturates by `k ≈ 525` (on par with
the full set) while runtime falls sharply — `k = 396` keeps near-peak quality at ~40% lower runtime;
end-to-end runtime drops from ~77 min (589) to ~33 min (300). Full per-budget tables (PSNR/SSIM/
LPIPS/RMSE and SfM/3DGS/VRAM) are in the paper's glacier quality and efficiency tables.

**Glacier selection-method ablation** (mean ± std over 4 runs; PairVPR is best on every metric at
every budget):

| Method | k=300 PSNR↑ | SSIM↑ | LPIPS↓ | k=332 PSNR↑ | SSIM↑ | LPIPS↓ | k=364 PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **PairVPR (ours)** | **20.54** | **0.621** | **0.462** | **20.98** | **0.630** | **0.445** | **21.20** | **0.625** | **0.458** |
| FoL | 19.89 | 0.594 | 0.489 | 20.01 | 0.610 | 0.481 | 19.72 | 0.595 | 0.488 |
| UniPR-3D | 19.84 | 0.593 | 0.495 | 19.94 | 0.606 | 0.482 | 19.99 | 0.617 | 0.474 |
| EDTformer | 19.54 | 0.588 | 0.495 | 19.98 | 0.607 | 0.483 | 19.76 | 0.612 | 0.481 |

## Citation

If you use this code or find the work useful, please cite the paper (preprint — update on
publication):

```bibtex
@article{zhao2026pairvpr,
  title   = {Towards Efficient 3D Gaussian Splatting Reconstruction with PairVPR-Based
             Image Selection: A Glacier UAV Case Study},
  author  = {Zhao, Sicheng and Pakrashi, Arjun and Dev, Soumyabrata},
  journal = {Applied Computing and Geosciences},
  note    = {Manuscript submitted; under review},
  year    = {2026}
}
```

## Acknowledgements & licensing

This project builds on the released code and weights of **Pair-VPR**, **FoL**, **UniPR-3D**, and
**EDTformer**; please cite and comply with each upstream project's own licence. In particular, the
bundled **Pair-VPR** submodule is licensed **CSIRO BSD-3-Clause-Clear (Non-Commercial)** (see
`Pair-VPR/LICENSE`) — the proposed `pairvpr` backend inherits that non-commercial restriction.
Reconstruction uses **COLMAP** and **3D Gaussian Splatting**, which carry their own licences. No
licence is asserted here over the third-party model code or weights.
