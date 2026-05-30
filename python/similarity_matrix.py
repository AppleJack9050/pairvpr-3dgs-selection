# Copyright 2024-present CSIRO.
# Licensed under CSIRO BSD-3-Clause-Clear (Non-Commercial)
#
# Compute a pairwise similarity matrix over a whole folder of images using Pair-VPR.
#
# Given a directory of images, this script produces an N x N matrix S where S[i, j]
# is the Pair-VPR similarity between image i and image j, normalized to [0, 1]
# (higher = more similar). The default method uses Pair-VPR's distinctive second-stage
# pair classifier (a cross-attention decoder trained with BCEWithLogitsLoss, target=1
# for "same place"), so a higher logit means more similar; we pass it through a sigmoid.
# A fast "global" method (cosine of the global descriptors) is also available.
#
# This script lives in <repo>/python/ (alongside the sibling Pair-VPR/ checkout at <repo>/) and
# adds Pair-VPR/ to sys.path so the `pairvpr.*` package imports resolve.


import os
import sys
import time
import argparse
from pathlib import Path
from glob import glob

# The Pair-VPR checkout sits at <repo>/Pair-VPR; this file is at <repo>/python/, so the project
# root is the parent of this file's directory.
BASE_DIR = Path(__file__).resolve().parent.parent
PAIRVPR_DIR = BASE_DIR / "Pair-VPR"
sys.path.insert(0, str(PAIRVPR_DIR))

import numpy as np
import torch
from omegaconf import OmegaConf
import torchvision.transforms as T
from PIL import Image, ImageFile
from tqdm import tqdm

from pairvpr.models.pairvpr import PairVPRNet
from pairvpr.models.tools.pos_embed import interpolate_pos_embed
from pairvpr.configs import pairvpr_speed

# Match the rest of the repo: tolerate slightly corrupt / truncated images.
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ImageNet stats, identical to pairvpr/eval/get_datasets.py
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

CKPT_URL = "https://huggingface.co/CSIRORobotics/Pair-VPR/resolve/main/pairvpr-vitG.pth"

DEFAULT_CKPT = "/mnt/windows/model/pairvpr_models/pairvpr-vitG.pth"
DEFAULT_CONFIG = str(PAIRVPR_DIR / "pairvpr" / "configs" / "pairvpr_performance.yaml")


# --------------------------------------------------------------------------------------
# Config + model
# --------------------------------------------------------------------------------------
def resolve_path(path: str) -> str:
    """Accept paths that are absolute, relative to CWD, or relative to the Pair-VPR checkout."""
    for candidate in (path, PAIRVPR_DIR / path, BASE_DIR / path):
        if os.path.exists(candidate):
            return str(candidate)
    return path


def get_cfg(config_file: str):
    """Merge the loaded config onto the speed defaults, exactly like eval.py."""
    default_cfg = OmegaConf.create(pairvpr_speed)
    cfg = OmegaConf.load(resolve_path(config_file))
    return OmegaConf.merge(default_cfg, cfg)


def load_model(cfg, ckpt_path: str, device: torch.device) -> PairVPRNet:
    """Mirror eval.py main(): build PairVPRNet, interpolate pos-embed, load (strict=False)."""
    ckpt_path = resolve_path(ckpt_path)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Pair-VPR checkpoint not found at '{ckpt_path}'.\n"
            f"Download the vitG checkpoint first, e.g.:\n"
            f"    wget -O {DEFAULT_CKPT} {CKPT_URL}"
        )
    ckpt = torch.load(ckpt_path, map_location=device)
    model = PairVPRNet(cfg).to(device)
    interpolate_pos_embed(cfg, model, ckpt)  # mutates ckpt in place, must run BEFORE load
    model.load_state_dict(ckpt, strict=False)
    return model.eval()


# --------------------------------------------------------------------------------------
# Image discovery + dataset
# --------------------------------------------------------------------------------------
def discover_images(images_dir: str, recursive: bool = True) -> list:
    """Find images of any common type, deterministically sorted, as absolute paths."""
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"images_dir '{images_dir}' is not a directory")
    pattern = "**/*" if recursive else "*"
    paths = set()
    for p in glob(os.path.join(images_dir, pattern), recursive=recursive):
        if os.path.isfile(p) and p.lower().endswith(IMG_EXTS):
            paths.add(os.path.abspath(p))
    paths = sorted(paths)
    if len(paths) == 0:
        raise FileNotFoundError(
            f"No images found in '{images_dir}' (looked for {IMG_EXTS}, recursive={recursive})"
        )
    return paths


def validate_images(paths: list) -> list:
    """Drop unreadable files up front so the matrix stays square and aligned with names."""
    good = []
    for p in paths:
        try:
            with Image.open(p) as im:
                im.verify()  # cheap header check
            good.append(p)
        except Exception as e:  # noqa: BLE001
            print(f"---WARNING skipping unreadable image: {p} ({e})")
    if len(good) == 0:
        raise RuntimeError("All discovered images failed to load.")
    return good


def build_transform(img_res: int) -> T.Compose:
    """Exactly the inference transform from pairvpr/eval/get_datasets.py."""
    return T.Compose([
        T.Resize((img_res, img_res), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class ImageFolderDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths: list, transform: T.Compose):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img = Image.open(self.image_paths[index]).convert("RGB")
        return self.transform(img), index


# --------------------------------------------------------------------------------------
# Phase A: feature extraction
# --------------------------------------------------------------------------------------
@torch.no_grad()
def extract_features(model, image_paths, transform, device, args, want_maps: bool):
    """Returns (dense_maps [N,P,D] or None, global_desc [N,Dg]). Scattered by index."""
    dataset = ImageFolderDataset(image_paths, transform)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.extract_batch, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    n = len(dataset)
    maps_cpu = None
    desc_cpu = None
    map_dtype = torch.float16 if args.store_fp16 else torch.float32

    start = time.time()
    for imgs, idxs in tqdm(loader, desc="extract"):
        maps, descriptors = model(imgs.to(device), None, mode="global")
        descriptors = descriptors.detach().float().cpu()
        if desc_cpu is None:
            desc_cpu = torch.empty((n, descriptors.shape[1]), dtype=torch.float32)
        desc_cpu[idxs] = descriptors
        if want_maps:
            maps = maps.detach().to(map_dtype).cpu()
            if maps_cpu is None:
                maps_cpu = torch.empty((n, maps.shape[1], maps.shape[2]), dtype=map_dtype)
            maps_cpu[idxs] = maps

    elapsed = time.time() - start
    print(f"---Extracted {n} images in {elapsed:.1f}s ({elapsed / n:.3f}s/image)")
    if want_maps:
        per_img_mb = maps_cpu.element_size() * maps_cpu[0].numel() / 1024 ** 2
        print(f"---Dense maps: {tuple(maps_cpu.shape)} {maps_cpu.dtype} (~{per_img_mb:.2f} MB/image)")
    return maps_cpu, desc_cpu


# --------------------------------------------------------------------------------------
# Phase B: similarity matrices
# --------------------------------------------------------------------------------------
@torch.no_grad()
def similarity_pair(model, dense_maps, device, pair_batch: int) -> torch.Tensor:
    """Pair-VPR second-stage similarity. S = sigmoid((L + L^T)/2), in [0,1], symmetric."""
    n = dense_maps.shape[0]
    if n == 1:
        return torch.ones((1, 1), dtype=torch.float32)
    L = torch.empty((n, n), dtype=torch.float32)
    for i in tqdm(range(n), desc="pair-score"):
        feat_i = dense_maps[i].unsqueeze(0).float().to(device)  # (1, P, D)
        for j0 in range(0, n, pair_batch):
            j1 = min(j0 + pair_batch, n)
            feat_j = dense_maps[j0:j1].float().to(device)       # (chunk, P, D)
            logits = model(feat_i.expand(j1 - j0, -1, -1), feat_j, "pairvpr")
            L[i, j0:j1] = logits.squeeze(1).float().cpu()
    # Bidirectional fusion (matches eval.py's scoresa + scoresb), then sigmoid -> [0,1]
    S = torch.sigmoid((L + L.t()) / 2.0)
    return S.clamp_(0.0, 1.0)


def similarity_global(global_desc: torch.Tensor) -> torch.Tensor:
    """Cosine similarity of the L2-normalized global descriptors, mapped to [0,1]."""
    desc = global_desc.float()
    cos = desc @ desc.t()                 # descriptors are already L2-normalized
    return ((cos + 1.0) / 2.0).clamp_(0.0, 1.0)


# --------------------------------------------------------------------------------------
# Output + self-check
# --------------------------------------------------------------------------------------
def save_outputs(S: torch.Tensor, names: list, out_dir: str, tag: str, save_csv: bool):
    os.makedirs(out_dir, exist_ok=True)
    mat = S.numpy().astype(np.float32)
    npy_path = os.path.join(out_dir, f"similarity_matrix{tag}.npy")
    np.save(npy_path, mat)
    order_path = os.path.join(out_dir, "image_order.txt")
    with open(order_path, "w") as f:
        f.write("\n".join(names) + "\n")
    print(f"---Saved {npy_path}  shape={mat.shape}")
    print(f"---Saved {order_path}  ({len(names)} images, row/col order)")
    if save_csv:
        csv_path = os.path.join(out_dir, f"similarity_matrix{tag}.csv")
        np.savetxt(csv_path, mat, delimiter=",", fmt="%.6f")
        print(f"---Saved {csv_path}")


def smoke_self_check(S: torch.Tensor, forced_diag: bool, tol: float = 1e-4):
    n = S.shape[0]
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"   [{'PASS' if cond else 'FAIL'}] {name}")

    print("---Self-check:")
    check(f"square ({n}x{n})", S.ndim == 2 and S.shape[0] == S.shape[1])
    check("all finite", bool(torch.isfinite(S).all()))
    check("in [0,1]", bool(S.min() >= -tol and S.max() <= 1 + tol))
    check("symmetric", bool(torch.allclose(S, S.t(), atol=tol)))
    if forced_diag:
        check("diagonal == 1.0", bool(torch.allclose(S.diagonal(), torch.ones(n), atol=tol)))
    elif n > 1:
        row_max = S.max(dim=1).values
        check("diagonal is row-max", bool(torch.all(S.diagonal() >= row_max - tol)))
    return ok


def report_correlation(S_pair: torch.Tensor, S_global: torch.Tensor):
    """Positive correlation confirms the pair score isn't inverted."""
    iu = torch.triu_indices(S_pair.shape[0], S_pair.shape[1], offset=1)
    a = S_pair[iu[0], iu[1]].numpy()
    b = S_global[iu[0], iu[1]].numpy()
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        print("---Correlation pair vs global: n/a (too few / constant pairs)")
        return
    pearson = float(np.corrcoef(a, b)[0, 1])
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    spearman = float(np.corrcoef(ra, rb)[0, 1])
    print(f"---Correlation pair vs global: Pearson={pearson:.3f}  Spearman={spearman:.3f} "
          f"(expect positive)")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def get_args_parser(add_help: bool = True):
    p = argparse.ArgumentParser("Pair-VPR pairwise similarity matrix", add_help=add_help)
    p.add_argument("--images_dir", "--images-dir", required=True, type=str,
                   help="Folder of images to score (all pairs).")
    p.add_argument("--trained_ckpt", "--trained-ckpt", type=str, default=DEFAULT_CKPT)
    p.add_argument("--config-file", "--config_file", type=str, default=DEFAULT_CONFIG)
    p.add_argument("--output_dir", "--output-dir", type=str, default="results_simmatrix")
    p.add_argument("--method", choices=["pair", "global", "both"], default="pair")
    p.add_argument("--extract_batch", type=int, default=32)
    p.add_argument("--pair_batch", type=int, default=64,
                   help="Column chunk size for the pair decoder.")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--store_fp16", action="store_true",
                   help="Store CPU dense maps as fp16 to halve RAM.")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--no-force-diagonal", dest="force_diagonal", action="store_false",
                   help="Keep the computed self-similarity instead of forcing diagonal to 1.")
    p.set_defaults(force_diagonal=True)
    p.add_argument("--save_csv", action="store_true")
    p.add_argument("--no-recursive", dest="recursive", action="store_false")
    p.set_defaults(recursive=True)
    p.add_argument("--max_images", type=int, default=2000,
                   help="Abort if more images are found (use --allow_large to override).")
    p.add_argument("--allow_large", action="store_true")
    return p


def pick_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("---Warning: no GPU available, Pair-VPR (vitG) will be very slow on CPU")
    return torch.device("cpu")


def main(args):
    device = pick_device(args.device)
    print(f"---Device: {device}")

    image_paths = discover_images(args.images_dir, recursive=args.recursive)
    image_paths = validate_images(image_paths)
    n = len(image_paths)
    if n > args.max_images and not args.allow_large:
        raise SystemExit(
            f"Found {n} images (> --max_images {args.max_images}). The pair method is O(N^2); "
            f"re-run with --allow_large to proceed, or use --method global for large sets."
        )
    print(f"---Scoring {n} images, method='{args.method}'")

    cfg = get_cfg(args.config_file)
    model = load_model(cfg, args.trained_ckpt, device)

    want_maps = args.method in ("pair", "both")
    dense_maps, global_desc = extract_features(
        model, image_paths, build_transform(cfg.augmentation.img_res), device, args, want_maps
    )

    names = [os.path.relpath(p, os.path.abspath(args.images_dir)) for p in image_paths]

    S_pair = S_global = None
    if args.method in ("pair", "both"):
        S_pair = similarity_pair(model, dense_maps, device, args.pair_batch)
        if args.force_diagonal:
            S_pair.fill_diagonal_(1.0)
    if args.method in ("global", "both"):
        S_global = similarity_global(global_desc)
        if args.force_diagonal:
            S_global.fill_diagonal_(1.0)

    if args.method == "both":
        save_outputs(S_pair, names, args.output_dir, "_pair", args.save_csv)
        save_outputs(S_global, names, args.output_dir, "_global", args.save_csv)
        print("--- pair matrix:");   smoke_self_check(S_pair, args.force_diagonal)
        print("--- global matrix:"); smoke_self_check(S_global, args.force_diagonal)
        report_correlation(S_pair, S_global)
    else:
        S = S_pair if args.method == "pair" else S_global
        save_outputs(S, names, args.output_dir, "", args.save_csv)
        smoke_self_check(S, args.force_diagonal)

    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("done")


if __name__ == "__main__":
    main(get_args_parser().parse_args())
