# Compute a pairwise similarity matrix over a whole folder of images using EDTformer.
#
# EDTformer (https://github.com/Tong-Jin01/EDTformer) is a DINOv2 ViT-B/14 backbone + decoder
# transformer Visual Place Recognition model that produces a 4096-d, L2-normalized global
# descriptor per image. Given a directory of images, this script produces an N x N matrix S where
# S[i, j] is the cosine similarity between the EDTformer descriptors of image i and image j,
# normalized to [0, 1] (higher = more similar). The output (similarity_matrix.npy +
# image_order.txt) is in the exact format consumed by remove_redundant.py, which then prunes
# redundancy via Facility Location.
#
# This is the EDTformer counterpart of backends/fol/fol_similarity.py; the image-discovery, output
# format and self-checks are deliberately kept identical -- only the model loading, the input
# transform and the forward pass differ. It supersedes the original stand-alone
# `select_diverse_subset.py`, which bundled its own apricot-based Facility Location selection: in
# the unified layout EDTformer emits a standard matrix and the shared numpy selector
# (selection/remove_redundant.py) does the selection, exactly like the other three backends.
#
# Model specifics (verified against the EDTformer repo):
#   * The model is `network.VPRNet()` from the cloned repo (put on sys.path via --repo_dir). It is
#     the same class the repo wraps in torch.nn.DataParallel; we load the released checkpoint into
#     the DataParallel wrapper (its keys are "module.*") and then keep the unwrapped `.module`.
#   * The release weights (EDTformer.pth, ~449 MB) are fetched once from the v1.0.0 GitHub release
#     via torch.hub and cached under ~/.cache/torch/hub/. The clone only supplies the model *code*.
#   * Input transform (datasets_ws.py eval): ToTensor (-> [0, 1]) -> ImageNet-normalize -> resize
#     to (H, W). NOTE the order: EDTformer normalizes BEFORE resizing (322x322 is the parser.py
#     default). We replicate that order exactly so the descriptors match the repo's own eval.

import os
import sys
import time
import argparse
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageFile
from tqdm import tqdm

# Tolerate slightly corrupt / truncated images (same as the FoL / UniPR-3D scripts).
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ImageNet stats -- EDTformer's datasets_ws.py normalizes with exactly these.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

# This file is at <repo>/backends/edtformer/, so the repo root is two levels up. The cloned model
# repo (EDTformer/) lives at the repo root (matching the .gitignore entry), shared alongside the
# other backends' external checkouts.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_REPO_DIR = str(REPO_ROOT / "EDTformer")

# EDTformer release weights (torch.hub v1.0.0). Cached after first download.
WEIGHTS_URL = "https://github.com/Tong-Jin01/EDTformer/releases/download/v1.0.0/EDTformer.pth"

# EDTformer's eval input size (parser.py --resize default).
DEFAULT_RESIZE = [322, 322]


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
def load_edtformer_model(repo_dir: str, device: torch.device):
    """Instantiate network.VPRNet from the cloned EDTformer repo and load the release weights.

    Defensive: the repo is not pip-installable, so we put `repo_dir` on sys.path and import the
    model code from the actual clone. The DINOv2 ViT-B/14 backbone is built inside VPRNet; the
    released checkpoint (EDTformer.pth) is pulled from the GitHub release via torch.hub.
    """
    if not os.path.isdir(repo_dir):
        raise SystemExit(
            f"--repo_dir '{repo_dir}' is not a directory. Clone the model repo first:\n"
            f"  git clone https://github.com/Tong-Jin01/EDTformer.git {repo_dir}")
    repo_dir = str(Path(repo_dir).resolve())
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    try:
        import network  # noqa: from the EDTformer repo (provides VPRNet)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            f"Could not import `network` from '{repo_dir}' ({type(e).__name__}: {e}).\n"
            f"Check that the EDTformer repo is cloned there and its deps (timm) are installed. If "
            f"the module layout changed upstream, inspect {os.path.join(repo_dir, 'network.py')}.")

    model = network.VPRNet()
    model = torch.nn.DataParallel(model)
    try:
        state = torch.hub.load_state_dict_from_url(WEIGHTS_URL, map_location="cpu")
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            f"Failed to download EDTformer weights from {WEIGHTS_URL} "
            f"({type(e).__name__}: {e}). This needs internet (or a warm ~/.cache/torch/hub) once.")
    model.load_state_dict(state["model_state_dict"])
    return model.module.to(device).eval()


# --------------------------------------------------------------------------------------
# Image discovery + dataset  (identical to fol_similarity.py, bar the transform order)
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


class ImageFolderDataset(torch.utils.data.Dataset):
    """EDTformer eval preprocessing: ToTensor -> ImageNet-normalize -> resize to (H, W).

    NOTE the order matches the repo (datasets_ws.py): normalization happens BEFORE the resize,
    and the resize uses bilinear + antialias. `resize` is (H, W); the eval default is 322x322.
    """

    def __init__(self, image_paths: list, resize):
        self.image_paths = image_paths
        self.resize = [int(resize[0]), int(resize[1])]
        self._to_norm = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img = Image.open(self.image_paths[index]).convert("RGB")
        img = self._to_norm(img)
        img = TF.resize(img, self.resize, antialias=True)
        return img, index


# --------------------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------------------
@torch.no_grad()
def extract_features(model, image_paths, resize, device, args) -> torch.Tensor:
    """Return the EDTformer global descriptors as a CPU float32 tensor [N, D], scattered by index.

    VPRNet(x) returns the 4096-d L2-normalized global descriptor. The dim D is read at runtime,
    never hardcoded. We re-normalize defensively (autocast/half can perturb the unit norm a hair).
    """
    dataset = ImageFolderDataset(image_paths, resize)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.extract_batch, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    n = len(dataset)
    desc_cpu = None
    norm_reported = False
    use_amp = (device.type == "cuda")

    start = time.time()
    for imgs, idxs in tqdm(loader, desc="extract"):
        imgs = imgs.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            desc = model(imgs)
        desc = desc.detach().float()                       # [B, D]
        if not norm_reported:
            norms = desc.norm(dim=1)
            print(f"---EDTformer global descriptor: dim={desc.shape[1]}, "
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
# Similarity  (identical to fol_similarity.py's global method)
# --------------------------------------------------------------------------------------
def similarity_global(global_desc: torch.Tensor) -> torch.Tensor:
    """Cosine similarity of the L2-normalized global descriptors, mapped to [0,1]."""
    desc = global_desc.float()
    cos = desc @ desc.t()                 # descriptors are L2-normalized in extract_features
    return ((cos + 1.0) / 2.0).clamp_(0.0, 1.0)


# --------------------------------------------------------------------------------------
# Output + self-check  (identical to fol_similarity.py)
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
    p = argparse.ArgumentParser("EDTformer pairwise (global-cosine) similarity matrix", add_help=add_help)
    p.add_argument("--images_dir", "--images-dir", required=True, type=str,
                   help="Folder of images to score (all pairs).")
    p.add_argument("--output_dir", "--output-dir", type=str, default="results_simmatrix")
    p.add_argument("--repo_dir", "--repo-dir", type=str, default=DEFAULT_REPO_DIR,
                   help="Path to the cloned EDTformer repo (put on sys.path to import `network`).")
    p.add_argument("--resize", type=int, nargs=2, default=DEFAULT_RESIZE, metavar=("H", "W"),
                   help="Hard-resize images to H W before the model (EDTformer eval default 322 322).")
    p.add_argument("--extract_batch", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
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
    print("---Warning: no GPU available, EDTformer (DINOv2 ViT-B/14) will be slow on CPU")
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
    print(f"---Scoring {n} images with EDTformer, resize={tuple(args.resize)}, method='global' (cosine)")

    model = load_edtformer_model(args.repo_dir, device)
    global_desc = extract_features(model, image_paths, args.resize, device, args)

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
