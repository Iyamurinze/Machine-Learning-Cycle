"""Data acquisition for the malaria cell classification pipeline.

Downloads the NIH Malaria Cell Images dataset (27,558 segmented red blood
cell images, balanced across Parasitized / Uninfected), then produces a
stratified train/test split on disk.

The dataset is served directly by the U.S. National Library of Medicine, so
no Kaggle account or API token is required:
    https://lhncbc.nlm.nih.gov/LHC-research/LHC-projects/image-processing/malaria-datasheet.html

Usage:
    python src/data_acquisition.py
    python src/data_acquisition.py --test-size 0.2 --seed 42
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen

DATA_URL = "https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ZIP_PATH = RAW_DIR / "cell_images.zip"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

CLASSES = ("Parasitized", "Uninfected")


def download(url: str = DATA_URL, dest: Path = ZIP_PATH, force: bool = False) -> Path:
    """Stream the dataset archive to disk, showing progress."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        print(f"[skip] archive already present: {dest} "
              f"({dest.stat().st_size / 1e6:.0f} MB)")
        return dest

    print(f"[download] {url}")
    with urlopen(url) as response:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1 << 20  # 1 MB

        with open(dest, "wb") as handle:
            while chunk := response.read(chunk_size):
                handle.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {downloaded / 1e6:7.0f} / {total / 1e6:.0f} MB "
                          f"({pct:5.1f}%)", end="", flush=True)
        print()

    print(f"[done] saved to {dest}")
    return dest


def extract(archive: Path = ZIP_PATH, dest: Path = RAW_DIR) -> Path:
    """Extract the archive and return the directory holding the class folders."""
    marker = dest / "cell_images"

    if marker.exists():
        print(f"[skip] already extracted: {marker}")
    else:
        print(f"[extract] {archive} -> {dest}")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
        print("[done] extracted")

    # The archive nests a duplicate cell_images/ inside itself in some
    # mirrors; walk down until we find the directory that holds the classes.
    candidate = marker
    for _ in range(3):
        if all((candidate / c).is_dir() for c in CLASSES):
            return candidate
        nested = candidate / "cell_images"
        if nested.is_dir():
            candidate = nested
        else:
            break

    raise FileNotFoundError(
        f"Could not locate {CLASSES} class folders under {marker}. "
        "The archive layout may have changed."
    )


def split(source: Path, test_size: float = 0.2, seed: int = 42) -> dict[str, int]:
    """Stratified split into data/train/<class> and data/test/<class>.

    Splitting per class keeps the 50/50 balance intact in both partitions,
    which matters because every downstream metric (precision, recall, F1)
    is reported against a balanced test set.
    """
    rng = random.Random(seed)
    counts: dict[str, int] = {}

    for split_dir in (TRAIN_DIR, TEST_DIR):
        for cls in CLASSES:
            target = split_dir / cls
            if target.exists():
                shutil.rmtree(target)
            target.mkdir(parents=True, exist_ok=True)

    for cls in CLASSES:
        # .png only — the archive ships stray Thumbs.db files that would
        # otherwise be counted as images and crash the loader later.
        images = sorted(p for p in (source / cls).iterdir()
                        if p.suffix.lower() == ".png")
        rng.shuffle(images)

        cut = int(len(images) * (1 - test_size))
        for name, subset in (("train", images[:cut]), ("test", images[cut:])):
            dest_dir = (TRAIN_DIR if name == "train" else TEST_DIR) / cls
            for path in subset:
                shutil.copy2(path, dest_dir / path.name)
            counts[f"{name}/{cls}"] = len(subset)

    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="fraction held out for testing (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="shuffle seed for a reproducible split")
    parser.add_argument("--force", action="store_true",
                        help="re-download even if the archive exists")
    args = parser.parse_args()

    archive = download(force=args.force)
    source = extract(archive)
    counts = split(source, test_size=args.test_size, seed=args.seed)

    print("\n[split] images written")
    for key in sorted(counts):
        print(f"  {key:28s} {counts[key]:6,d}")
    print(f"  {'TOTAL':28s} {sum(counts.values()):6,d}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
