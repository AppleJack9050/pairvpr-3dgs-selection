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
All codes are written in `python` and `bash`. The method is a two-step image-selection pipeline: compute a pairwise image-similarity matrix with a Visual Place Recognition (VPR) model (`backends/`), then select a diverse, non-redundant subset by greedy facility location (`selection/`). PairVPR is the proposed method; FoL, UniPR-3D, and EDTformer are the baselines used in the ablation. The selected subset is then reconstructed with a standard COLMAP + 3D Gaussian Splatting pipeline (external).

### Code
The scripts to reproduce the image-selection results in the paper are as follows:
+ `backends/pairvpr/similarity_matrix.py`: PairVPR (proposed) pairwise-similarity matrix
+ `backends/fol/fol_similarity.py`: FoL baseline global-descriptor similarity matrix
+ `backends/unipr3d/unipr3d_similarity.py`: UniPR-3D baseline global-descriptor similarity matrix
+ `backends/edtformer/edtformer_similarity.py`: EDTformer baseline global-descriptor similarity matrix
+ `selection/remove_redundant.py`: greedy facility-location subset selection (Algorithm 1 in the paper)
+ `bash/run_all.sh`: end-to-end wrapper — similarity matrix, then selection over several budgets `k`
+ `bash/similarity_matrix.sh`, `bash/remove_redundant.sh`, `bash/make_subsets.sh`: individual-step wrappers
+ `requirements/`: per-backend Python dependencies (`base`, `pairvpr`, `fol`, `unipr3d`, `edtformer`)

### Setup
+ Install dependencies: `pip install -r requirements.txt` (all backends), or `pip install -r requirements/<backend>.txt` for one. On an NVIDIA RTX 5090 (Blackwell) install torch from the CUDA 12.8 wheel index first.
+ `pairvpr` (proposed): `git submodule update --init Pair-VPR`, then download the ViT-G checkpoint from HuggingFace `CSIRORobotics/Pair-VPR` (Pair-VPR is licensed CSIRO BSD-3-Clause-Clear, non-commercial).
+ `edtformer` / `unipr3d`: `git clone` the model repo into `EDTformer/` / `UniPR-3D/` (weights/checkpoints auto-download on first run).
+ `fol`: nothing extra — the model and DINOv2 backbone auto-download via `torch.hub`.

### Usage
Compute the similarity matrix once and select the paper's subset budgets in one command (`<backend>` = `pairvpr` | `fol` | `unipr3d` | `edtformer`):
```
bash bash/run_all.sh pairvpr /path/to/images 300 332 364 396 428 461 493 525 557
```
Each `subset_<k>/` contains the selected image list, indices, a per-pick selection report, and an `images/` folder ready to feed into COLMAP + 3D Gaussian Splatting.

### Results
We evaluate on a real-world glacier UAV survey (589 images) and on six standard benchmark scenes (five Mip-NeRF360 scenes and the Tanks&Temples *Truck* scene). Under a fixed image budget, PairVPR-based selection attains the best average PSNR, SSIM, and LPIPS among the evaluated VPR selectors; on the glacier survey, moderate pruning reduces total runtime by roughly 40% (over 50% under more aggressive pruning) with only a modest loss in reconstruction quality. The full per-scene and per-budget tables are reported in the paper.
