from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

def _index(folder):
    result = {}
    for p in Path(folder).rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            result[p.stem.lower()] = p
    return result

def _pairs_from_dir(image_dir, mask_dir):
    images, masks = _index(image_dir), _index(mask_dir)
    return [(images[k], masks[k]) for k in sorted(images.keys() & masks.keys())]


def find_pairs(root, split):
    root = Path(root)
    # Standard SOS Kaggle/Zenodo arrangement: images/train + masks/train
    candidates = [
        (root/"images"/split, root/"masks"/split),
        (root/split/"images", root/split/"masks"),
        (root/"SOS"/"images"/split, root/"SOS"/"masks"/split),
    ]
    for image_dir, mask_dir in candidates:
        if image_dir.exists() and mask_dir.exists():
            pairs = _pairs_from_dir(image_dir, mask_dir)
            if pairs:
                return pairs

    split_root = root / split
    if split_root.exists():
        for image_dir in sorted({p for p in split_root.rglob("*") if p.is_dir() and p.name in {"image", "images"}}):
            for mask_name in ("label", "labels", "mask", "masks"):
                mask_dir = image_dir.parent / mask_name
                if mask_dir.exists():
                    pairs = _pairs_from_dir(image_dir, mask_dir)
                    if pairs:
                        return pairs

    # Search recursively for directory names ending in images/<split> and masks/<split>
    image_dirs = list(root.rglob(f"images/{split}"))
    mask_dirs = list(root.rglob(f"masks/{split}"))
    for image_dir in image_dirs:
        for mask_dir in mask_dirs:
            if image_dir.parent.parent == mask_dir.parent.parent:
                pairs = _pairs_from_dir(image_dir, mask_dir)
                if pairs:
                    return pairs

    raise FileNotFoundError(
        f"No paired {split} images/masks found under {root}. Expected images/{split} and masks/{split}."
    )

def load_gray(path, size=256, mask=False):
    image = Image.open(path).convert("L")
    resample = Image.Resampling.NEAREST if mask else Image.Resampling.BILINEAR
    image = image.resize((size, size), resample)
    arr = np.asarray(image, dtype=np.float32)
    if mask:
        return (arr > 0).astype(np.float32)
    # Robust per-image min/max scaling handles common 8-bit and 16-bit SAR exports.
    lo, hi = np.percentile(arr, [1, 99])
    arr = np.clip((arr-lo)/(hi-lo+1e-8), 0, 1).astype(np.float32)
    return arr

class SARSegmentationDataset(Dataset):
    def __init__(self, root, split="train", size=256, augment=False):
        self.pairs = find_pairs(root, split)
        self.size, self.augment = size, augment
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        ip, mp = self.pairs[idx]
        x, y = load_gray(ip, self.size), load_gray(mp, self.size, mask=True)
        if self.augment:
            if np.random.rand() < .5: x, y = np.fliplr(x).copy(), np.fliplr(y).copy()
            if np.random.rand() < .5: x, y = np.flipud(x).copy(), np.flipud(y).copy()
        return torch.from_numpy(x[None]), torch.from_numpy(y[None])
