# Compute a pairwise similarity matrix over a whole folder of images using UniPR-3D.
#
# UniPR-3D (https://github.com/dtc111111/UniPR-3D, weights on HF dtc111/UniPR-3D) is a Visual
# Place Recognition model built on VGGT + DINOv2 with LoRA adapters and SALAD aggregation. Given a
# directory of images, this script produces an N x N matrix S where S[i, j] is the cosine
# similarity between the UniPR-3D *global descriptors* of image i and image j, normalized to [0, 1]
# (higher = more similar). The output (similarity_matrix.npy + image_order.txt) is in the exact
# format consumed by remove_redundant.py, which then prunes redundancy via Facility Location.
#
# This is the UniPR-3D counterpart of backends/fol/fol_similarity.py (which used FoL); the
# image-handling, output format and self-checks are deliberately kept identical -- only the model
# loading, the input transform and the forward pass differ.
#
# Model specifics (verified against the cloned repo):
#   * The single-frame model is `VGGTPR_LoRA` (vggt/models/vggtpr_lora.py). We instantiate the core
#     nn.Module directly (the `VPRModel` Lightning wrapper pulls in faiss / pytorch-metric-learning,
#     which we don't need for forward-only feature extraction).
#   * with_geo_features=True + with_dinov2_features=True -> a 17152-d descriptor (geo 8448 + dino
#     8704). The dim is read at runtime, never hardcoded.
#   * The descriptor is NOT globally unit-norm: SALAD L2-normalizes each sub-block, so the full
#     descriptor has a constant (input-independent) norm. Inner-product retrieval is therefore
#     monotonically equivalent to cosine; we L2-normalize the whole descriptor so cosine maps
#     cleanly into [0, 1] for the Facility Location matrix (no ranking information is lost).
#   * Input transform (eval_lora.py): Resize to (392, 518) [H, W] bilinear, ToTensor (-> [0, 1]),
#     Normalize(mean=0, std=1) -- identity, because the aggregator normalizes internally.
#   * Each image is embedded INDEPENDENTLY: we feed [B, 1, 3, H, W] (sequence length S=1) so there
#     is no cross-image attention. (Feeding [N, 3, H, W] and relying on the model's auto-unsqueeze
#     would make B=1, S=N and let all images cross-attend -- WRONG for per-image descriptors.)
#
# Needs the cloned repo on sys.path (--repo_dir) and `loralib` installed; the VGGT backbone weights
# are bundled in the checkpoint (no separate download). xformers is NOT required -- the vendored
# attention uses torch SDPA.

import os
import sys
import time
import argparse
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageFile
from tqdm import tqdm

# Tolerate slightly corrupt / truncated images (same as the FoL script).
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

# This file is at <repo>/backends/unipr3d/, so the repo root is two levels up. The cloned model
# repo (UniPR-3D/) and the checkpoint (models/single_model.ckpt) live at the repo root, shared
# across backends and matching the .gitignore entries.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DEFAULT_REPO_DIR = os.path.join(REPO_ROOT, "UniPR-3D")
DEFAULT_CHECKPOINT = os.path.join(REPO_ROOT, "models", "single_model.ckpt")
HF_REPO_ID = "dtc111/UniPR-3D"
HF_CKPT_FILE = "single_model.ckpt"

# UniPR-3D's eval input size (eval_lora.py --image_size default).
DEFAULT_RESIZE = [392, 518]


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
def _load_pretrain(path: str) -> dict:
    """Repo's load_pretrain (vpr_vggt_lora.py) verbatim: unwrap a Lightning checkpoint to a plain
    state_dict whose keys match VGGTPR_LoRA (aggregator.* / geo_salad_head.* / dino_salad_head.*)."""
    weights = torch.load(path, map_location="cpu", weights_only=False)
    if "model" in weights:
        weights = weights["model"]
    if "state_dict" in weights:
        weights = weights["state_dict"]
    if len(weights) > 0 and next(iter(weights.keys())).startswith("model."):
        weights = {k.replace("model.", ""): v for k, v in weights.items()}
    return weights


def ensure_checkpoint(checkpoint: str) -> str:
    """Return a local checkpoint path, downloading single_model.ckpt from HF on first use."""
    if os.path.exists(checkpoint):
        return checkpoint
    print(f"---Checkpoint not found at {checkpoint}; downloading {HF_CKPT_FILE} from "
          f"HF {HF_REPO_ID} (~3.7 GB, one-time)")
    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"huggingface_hub is required to auto-download the checkpoint ({e}). "
                         f"pip install huggingface_hub, or pass --checkpoint /path/to/single_model.ckpt")
    local_dir = os.path.dirname(checkpoint) or "."
    os.makedirs(local_dir, exist_ok=True)
    path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_CKPT_FILE, local_dir=local_dir)
    print(f"---Downloaded checkpoint to {path}")
    return path


def load_unipr3d_model(checkpoint: str, repo_dir: str, device: torch.device):
    """Build VGGTPR_LoRA (single-frame, 17152-d) and load the pretrained weights onto `device`.

    Defensive: the repo is not pip-installable, so we put `repo_dir` on sys.path and import the
    model from the actual clone. Errors point at the exact files to inspect if upstream changed.
    """
    if not os.path.isdir(repo_dir):
        raise SystemExit(
            f"--repo_dir '{repo_dir}' is not a directory. Clone the model repo first:\n"
            f"  git clone https://github.com/dtc111111/UniPR-3D.git {repo_dir}")
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    try:
        from vggt.models.vggtpr_lora import VGGTPR_LoRA
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            f"Could not import VGGTPR_LoRA from '{repo_dir}' "
            f"({type(e).__name__}: {e}).\n"
            f"Check that the repo is cloned there and that 'loralib' (and 'einops') are installed "
            f"(pip install loralib). If the class moved upstream, inspect "
            f"{os.path.join(repo_dir, 'vggt', 'models', 'vggtpr_lora.py')} and adjust the import.")

    # Same toggles eval_lora.py uses for the 17152-d single-frame descriptor.
    try:
        model = VGGTPR_LoRA(
            with_geo_features=True, with_dinov2_features=True,
            with_camera_pose=False, camera_pose_type="yaw",
            lora_rank=8, lora_alpha=16, lora_dropout=0.1,
            lora_frame_attn=True, lora_global_attn=True, lora_patch_embed=False,
        )
    except TypeError as e:  # noqa: BLE001
        raise SystemExit(
            f"VGGTPR_LoRA(...) constructor differs from the expected signature ({e}). Open "
            f"{os.path.join(repo_dir, 'vggt', 'models', 'vggtpr_lora.py')} and align the kwargs.")

    checkpoint = ensure_checkpoint(checkpoint)
    print(f"---Loading checkpoint: {checkpoint}")
    weights = _load_pretrain(checkpoint)
    missing, unexpected = model.load_state_dict(weights, strict=False)
    lora_like = lambda k: k.endswith((".lora_A", ".lora_B", ".lora_dropout"))  # noqa: E731
    real_missing = [k for k in missing if not lora_like(k)]
    print(f"---load_state_dict: {len(real_missing)} non-LoRA missing, {len(unexpected)} unexpected "
          f"(strict=False)")
    if real_missing:
        print(f"   WARNING missing non-LoRA keys, e.g.: {real_missing[:8]}")
    if unexpected:
        print(f"   note unexpected keys, e.g.: {list(unexpected)[:8]}")
    return model.eval().to(device)


# --------------------------------------------------------------------------------------
# Image discovery + dataset  (identical to fol_similarity.py)
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
    """UniPR-3D's eval transform: hard-resize to (H, W), ToTensor (-> [0, 1]), identity-normalize.

    Normalize(mean=0, std=1) is a no-op kept for parity with eval_lora.py -- the aggregator
    normalizes internally, so the model expects raw [0, 1] pixels.
    """
    h, w = int(resize[0]), int(resize[1])
    return T.Compose([
        T.Resize((h, w), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
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
    """Return the UniPR-3D global descriptors as a CPU float32 tensor [N, D], scattered by index.

    Each image is embedded independently: we reshape [B, 3, H, W] -> [B, 1, 3, H, W] (S=1) so there
    is no cross-image attention. The 17152-d descriptor (out['salad_pred']) has a constant norm; we
    L2-normalize it so the downstream cosine similarity is in [-1, 1] (-> [0, 1]). D is read at
    runtime, never hardcoded.
    """
    dataset = ImageFolderDataset(image_paths, transform)
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
        imgs = imgs.to(device).unsqueeze(1)                # [B, 1, 3, H, W] -> single-frame
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            out = model(imgs, camera_pose=None)            # with_camera_pose=False -> pose ignored
        desc = out["salad_pred"].detach().float()          # [B, D]
        if not norm_reported:
            norms = desc.norm(dim=1)
            print(f"---UniPR-3D global descriptor: dim={desc.shape[1]}, "
                  f"raw L2-norm mean={float(norms.mean()):.4f} min={float(norms.min()):.4f} "
                  f"max={float(norms.max()):.4f}  (constant by construction; we L2-normalize for "
                  f"cosine)")
            norm_reported = True
        desc = F.normalize(desc, dim=1).cpu()              # unit vectors -> cosine = dot product
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
    p = argparse.ArgumentParser("UniPR-3D pairwise (global-cosine) similarity matrix", add_help=add_help)
    p.add_argument("--images_dir", "--images-dir", required=True, type=str,
                   help="Folder of images to score (all pairs).")
    p.add_argument("--output_dir", "--output-dir", type=str, default="results_simmatrix")
    p.add_argument("--checkpoint", "--ckpt", type=str, default=DEFAULT_CHECKPOINT,
                   help="UniPR-3D single-frame checkpoint (.ckpt). Auto-downloaded from "
                        f"HF {HF_REPO_ID} if absent.")
    p.add_argument("--repo_dir", "--repo-dir", type=str, default=DEFAULT_REPO_DIR,
                   help="Path to the cloned UniPR-3D repo (put on sys.path to import the model).")
    p.add_argument("--resize", type=int, nargs=2, default=DEFAULT_RESIZE, metavar=("H", "W"),
                   help="Hard-resize images to H W before the model (UniPR-3D eval default 392 518).")
    p.add_argument("--extract_batch", type=int, default=4,
                   help="Batch size for feature extraction (VGGT+DINOv2 at 392x518 is heavy; raise "
                        "if VRAM allows).")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--no-force-diagonal", dest="force_diagonal", action="store_false",
                   help="Keep the computed self-similarity instead of forcing the diagonal to 1.")
    p.set_defaults(force_diagonal=True)
    p.add_argument("--save_csv", action="store_true")
    p.add_argument("--no-recursive", dest="recursive", action="store_false")
    p.set_defaults(recursive=True)
    p.add_argument("--max_images", type=int, default=20000,
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
    print("---Warning: no GPU available, UniPR-3D (VGGT + DINOv2 ViT) will be very slow on CPU")
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
    print(f"---Scoring {n} images with UniPR-3D, resize={tuple(args.resize)}, method='global' (cosine)")

    model = load_unipr3d_model(args.checkpoint, args.repo_dir, device)
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
