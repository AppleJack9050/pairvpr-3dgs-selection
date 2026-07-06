# Efficient 3D Gaussian Splatting Reconstruction via PairVPR-Based Image Selection

With the spirit of reproducible research, this repository contains all the codes required to produce the image-selection results in the manuscript:

> Sicheng Zhao, Arjun Pakrashi, and Soumyabrata Dev, Towards Efficient 3D Gaussian Splatting Reconstruction with PairVPR-Based Image Selection: A Glacier UAV Case Study, *Applied Computing and Geosciences*, 2026.

Please cite the above paper if you intend to use whole/part of the code. This code is only for academic and research purposes.

```
@article{zhao2026pairvpr,
title = {Towards Efficient 3D Gaussian Splatting Reconstruction with PairVPR-Based Image Selection: A Glacier UAV Case Study},
journal = {Applied Computing and Geosciences},
year = {2026},
note = {Manuscript submitted; under review},
author = {Sicheng Zhao and Arjun Pakrashi and Soumyabrata Dev},
}
```

## Code Organization
All codes are written in `python` and `bash`. The method has two steps: compute a pairwise image-similarity matrix with a Visual Place Recognition model (`backends/`), then select a diverse, non-redundant subset by greedy facility location (`selection/`). PairVPR is the proposed method; FoL, UniPR-3D, and EDTformer are the ablation baselines. The selected subset is reconstructed with a standard COLMAP + 3D Gaussian Splatting pipeline.

### Code
+ `backends/pairvpr/similarity_matrix.py`: PairVPR (proposed) pairwise-similarity matrix
+ `backends/{fol,unipr3d,edtformer}/*_similarity.py`: baseline global-descriptor similarity matrices
+ `selection/remove_redundant.py`: greedy facility-location subset selection (Algorithm 1 in the paper)
+ `bash/run_all.sh`: end-to-end wrapper — similarity matrix, then selection over several budgets `k`
+ `requirements/`: per-backend Python dependencies

### Usage
```
bash bash/run_all.sh pairvpr /path/to/images 300 332 364 396 428 461 493 525 557
```
Setup notes: `pip install -r requirements.txt`; for `pairvpr` run `git submodule update --init Pair-VPR` and download the ViT-G checkpoint from HuggingFace `CSIRORobotics/Pair-VPR`; for `edtformer`/`unipr3d` clone the model repo into `EDTformer/`/`UniPR-3D/`; `fol` needs nothing extra.

## Results
Evaluated on six benchmark scenes (five Mip-NeRF360 + Tanks&Temples *Truck*) and a real-world glacier UAV survey (589 images). PairVPR-based selection is the best VPR selector in both settings.

**Benchmark scenes — average over 6 scenes at a fixed budget k = 100** (best in **bold**):

| Selector | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|----------|:------:|:------:|:-------:|
| **PairVPR (ours)** | **20.75** | **0.606** | **0.372** |
| EDTformer | 20.47 | 0.602 | 0.380 |
| FoL | 19.77 | 0.590 | 0.380 |
| UniPR-3D | 19.62 | 0.580 | 0.391 |

**Glacier — reconstruction quality vs. subset size** (PairVPR selector):

| Images (k) | SfM RMSE (px) ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|:----------:|:---------------:|:------:|:------:|:-------:|
| 589 (full) | 0.079 | 21.80 | 0.653 | 0.440 |
| 557 | 0.080 | 21.69 | 0.644 | 0.450 |
| 525 | 0.079 | 21.86 | 0.654 | 0.441 |
| 493 | 0.082 | 21.47 | 0.638 | 0.450 |
| 461 | 0.081 | 21.55 | 0.634 | 0.453 |
| 428 | 0.082 | 21.43 | 0.635 | 0.456 |
| 396 | 0.081 | 21.58 | 0.640 | 0.456 |
| 364 | 0.084 | 21.23 | 0.626 | 0.457 |
| 332 | 0.084 | 21.08 | 0.629 | 0.445 |
| 300 | 0.090 | 20.51 | 0.622 | 0.461 |

**Glacier — computational efficiency vs. subset size** (minutes; peak VRAM in GB):

| Images (k) | SfM ↓ | 3DGS ↓ | VRAM ↓ |
|:----------:|:-----:|:------:|:------:|
| 589 (full) | 59.2 | 17.8 | 16.4 |
| 557 | 55.3 | 17.5 | 16.0 |
| 525 | 48.1 | 17.6 | 15.6 |
| 493 | 43.3 | 17.5 | 15.0 |
| 461 | 34.8 | 17.4 | 14.5 |
| 428 | 28.7 | 17.4 | 13.9 |
| 396 | 28.8 | 17.4 | 13.2 |
| 364 | 24.0 | 17.4 | 12.3 |
| 332 | 18.0 | 17.3 | 11.5 |
| 300 | 15.7 | 17.5 | 11.1 |

**Glacier — selection-method ablation** (PSNR ↑, mean over 4 runs; PairVPR is also best on SSIM and LPIPS at every budget):

| Method | k = 300 | k = 332 | k = 364 |
|--------|:-------:|:-------:|:-------:|
| **PairVPR (ours)** | **20.54** | **20.98** | **21.20** |
| FoL | 19.89 | 20.01 | 19.72 |
| UniPR-3D | 19.84 | 19.94 | 19.99 |
| EDTformer | 19.54 | 19.98 | 19.76 |

Moderate pruning (e.g. k = 396) cuts total runtime by ~40% with near-peak quality; quality saturates around k = 525. Full per-scene and per-budget tables are in the paper.
