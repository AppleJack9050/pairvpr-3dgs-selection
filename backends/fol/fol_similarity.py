# Compute a pairwise similarity matrix over a whole folder of images using FoL.
#
# FoL ("Focus on Local", AAAI 2025) is a DINOv2-backbone Visual Place Recognition model.
# Given a directory of images, this script produces an N x N matrix S where S[i, j] is the
# cosine similarity between the FoL *global descriptors* of image i and image j, normalized to
# [0, 1] (higher = more similar). The output (similarity_matrix.npy + image_order.txt) is in the
# exact format consumed by remove_redundant.py, which then prunes redundancy via Facility Location.
#
# The model is pulled straight from torch.hub:
#     torch.hub.load("chenshunpeng/FoL", "FoL", pretrained=True, backbone="vitl", trust_repo=True)
# Two downloads happen on the first run (cached under ~/.cache/torch/hub):
#   1. chenshunpeng/FoL          + the FoL_large/base.pth weights (from HF shunpeng/FoL)
#   2. facebookresearch/dinov2   + the DINOv2 ViT weights  (FoLNet builds its backbone from hub)
# so an internet connection (or a warm cache) is required once.
#
# This is the FoL counterpart of backends/pairvpr/similarity_matrix.py (which used Pair-VPR);
# the image-handling, output format and self-checks are deliberately kept identical.

import os
import time
import argparse
from pathlib import Path
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageFile
from tqdm import tqdm

# Tolerate slightly corrupt / truncated images (same as the Pair-VPR script).
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ImageNet stats -- FoL's datasets.py normalizes with exactly these.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

# FoL hub entrypoint + the two supported backbones (see hubconf.py).
FOL_REPO = "chenshunpeng/FoL"
FOL_ENTRY = "FoL"
BACKBONES = ("vitl", "vitb")  # vitl -> FoL_large.pth (dinov2_vitl14); vitb -> FoL_base.pth


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
def load_fol_model(backbone: str, device: torch.device):
    """Load FoLNet (pretrained) from torch.hub and put it in eval mode on `device`.

    Mirrors the README: `torch.hub.load("chenshunpeng/FoL", "FoL", pretrained=True,
    backbone="vitl", trust_repo=True)`. The weights come from HF (shunpeng/FoL) and the DINOv2
    backbone is fetched separately from facebookresearch/dinov2 -- both cached on first use.
    """
    try:
        model = torch.hub.load(
            FOL_REPO, FOL_ENTRY, pretrained=True, backbone=backbone,
            trust_repo=True, map_location="cpu",
        )
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            f"Failed to load FoL from torch.hub ({type(e).__name__}: {e}).\n"
            f"This step needs internet (or a warm ~/.cache/torch/hub) to fetch BOTH "
            f"'{FOL_REPO}' (+ the FoL weights from HF shunpeng/FoL) AND "
            f"'facebookresearch/dinov2' (the backbone). If you are offline, pre-warm the hub "
            f"cache on a connected machine, or copy ~/.cache/torch/hub over."
        )
    return model.eval().to(device)


# --------------------------------------------------------------------------------------
# Image discovery + dataset  (identical to similarity_matrix.py)
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


def build_transform(resize) -> T.Compose:
    """FoL's eval transform: hard-resize to (H, W) then ImageNet-normalize.

    `resize` is (H, W); FoL's eval default is 504x504 (parser.py --resize). 322x322 is the
    training size -- pass --resize 322 322 to match that instead.
    """
    h, w = int(resize[0]), int(resize[1])
    return T.Compose([
        T.Resize((h, w), interpolation=T.InterpolationMode.BILINEAR),
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
# Feature extraction
# --------------------------------------------------------------------------------------
@torch.no_grad()
def extract_features(model, image_paths, transform, device, args) -> torch.Tensor:
    """Return the FoL global descriptors as a CPU float32 tensor [N, D], scattered by index.

    FoLNet.forward(x, test=True) returns a 7-tuple; outputs[0] is the global descriptor (already
    L2-normalized inside the FoL aggregator). The descriptor dim D is read at runtime (8448 for
    both ViT-L and ViT-B), never hardcoded. We re-normalize defensively (a no-op on unit vectors).
    """
    dataset = ImageFolderDataset(image_paths, transform)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.extract_batch, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    n = len(dataset)
    desc_cpu = None
    norm_reported = False

    start = time.time()
    for imgs, idxs in tqdm(loader, desc="extract"):
        outputs = model(imgs.to(device), test=True)
        desc = outputs[0].detach().float()                 # [B, D], unit-norm from FoL
        if not norm_reported:
            norms = desc.norm(dim=1)
            print(f"---FoL global descriptor: dim={desc.shape[1]}, "
                  f"L2-norm mean={float(norms.mean()):.4f} min={float(norms.min()):.4f} "
                  f"max={float(norms.max()):.4f}  (expect ~1.0 -> already L2-normalized)")
            norm_reported = True
        desc = F.normalize(desc, dim=1).cpu()              # idempotent guard against fp drift
        if desc_cpu is None:
            desc_cpu = torch.empty((n, desc.shape[1]), dtype=torch.float32)
        desc_cpu[idxs] = desc

    elapsed = time.time() - start
    print(f"---Extracted {n} images in {elapsed:.1f}s ({elapsed / max(n, 1):.3f}s/image)")
    return desc_cpu


# --------------------------------------------------------------------------------------
# Similarity  (identical to similarity_matrix.py's global method)
# --------------------------------------------------------------------------------------
def similarity_global(global_desc: torch.Tensor) -> torch.Tensor:
    """Cosine similarity of the L2-normalized global descriptors, mapped to [0,1]."""
    desc = global_desc.float()
    cos = desc @ desc.t()                 # descriptors are already L2-normalized
    return ((cos + 1.0) / 2.0).clamp_(0.0, 1.0)


# --------------------------------------------------------------------------------------
# Output + self-check  (identical to similarity_matrix.py)
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


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def get_args_parser(add_help: bool = True):
    p = argparse.ArgumentParser("FoL pairwise (global-cosine) similarity matrix", add_help=add_help)
    p.add_argument("--images_dir", "--images-dir", required=True, type=str,
                   help="Folder of images to score (all pairs).")
    p.add_argument("--output_dir", "--output-dir", type=str, default="results_simmatrix")
    p.add_argument("--backbone", choices=list(BACKBONES), default="vitl",
                   help="vitl = FoL_large (dinov2_vitl14, recommended); vitb = FoL_base (lighter).")
    p.add_argument("--resize", type=int, nargs=2, default=[504, 504], metavar=("H", "W"),
                   help="Hard-resize images to H W before the model (FoL eval default 504 504; "
                        "training size is 322 322).")
    p.add_argument("--extract_batch", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--no-force-diagonal", dest="force_diagonal", action="store_false",
                   help="Keep the computed self-similarity instead of forcing the diagonal to 1.")
    p.set_defaults(force_diagonal=True)
    p.add_argument("--save_csv", action="store_true")
    p.add_argument("--no-recursive", dest="recursive", action="store_false")
    p.set_defaults(recursive=True)
    p.add_argument("--max_images", type=int, default=20000,
                   help="Abort if more images are found (use --allow_large to override). Global "
                        "cosine is cheap, so this cap is generous.")
    p.add_argument("--allow_large", action="store_true")
    return p


def pick_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("---Warning: no GPU available, FoL (DINOv2 ViT) will be slow on CPU")
    return torch.device("cpu")


def main(args):
    device = pick_device(args.device)
    print(f"---Device: {device}")

    image_paths = discover_images(args.images_dir, recursive=args.recursive)
    image_paths = validate_images(image_paths)
    n = len(image_paths)
    if n > args.max_images and not args.allow_large:
        raise SystemExit(
            f"Found {n} images (> --max_images {args.max_images}). Re-run with --allow_large to "
            f"proceed (the N x N matrix is {n}x{n})."
        )
    print(f"---Scoring {n} images with FoL backbone='{args.backbone}', "
          f"resize={tuple(args.resize)}, method='global' (cosine)")

    model = load_fol_model(args.backbone, device)
    global_desc = extract_features(model, image_paths, build_transform(args.resize), device, args)

    names = [os.path.relpath(p, os.path.abspath(args.images_dir)) for p in image_paths]

    S = similarity_global(global_desc)
    if args.force_diagonal:
        S.fill_diagonal_(1.0)

    save_outputs(S, names, args.output_dir, "", args.save_csv)
    smoke_self_check(S, args.force_diagonal)

    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("done")


if __name__ == "__main__":
    main(get_args_parser().parse_args())
