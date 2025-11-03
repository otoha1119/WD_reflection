from __future__ import annotations

from pathlib import Path
import sys
import cv2

from app.model.reflection_removal_model import ReflectionRemovalModel


def process_images(images_dir: Path, mask_dir: Path, result_dir: Path) -> None:
    if not images_dir.is_dir():
        raise NotADirectoryError(f"Images directory not found: {images_dir}")
    if not mask_dir.is_dir():
        raise NotADirectoryError(f"Mask directory not found: {mask_dir}")
    result_dir.mkdir(parents=True, exist_ok=True)
    rr_model = ReflectionRemovalModel()
    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            continue
        stem = img_path.stem
        mask_path = mask_dir / f"{stem}_mask.png"
        if not mask_path.exists():
            print(f"Warning: mask not found for {img_path.name}, skipping.", file=sys.stderr)
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            print(f"Warning: failed to load image or mask for {img_path.name}, skipping.", file=sys.stderr)
            continue
        result = rr_model.remove_reflections(img, mask)
        out_path = result_dir / f"{stem}_reflection_removed.png"
        cv2.imwrite(str(out_path), result)
        print(f"Reflection removed image saved: {out_path}")


def main() -> None:
    """Entry point for command–line execution.

    Removes reflections from all images in the default ``data/images``
    directory and writes results to ``result``.
    """
    images_dir = Path("data/images")
    mask_dir = Path("mask")
    result_dir = Path("result")
    process_images(images_dir, mask_dir, result_dir)


if __name__ == "__main__":
    main()