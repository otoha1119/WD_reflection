from __future__ import annotations

from pathlib import Path
import sys
import cv2
from typing import Optional

from typing import Optional

# Models are imported inside the functions depending on the
# ``use_mvc_mask`` flag.  Do not import them here to avoid loading
# unnecessary modules when not needed.


def process_images(
    images_dir: Path,
    mask_dir: Path,
    use_mvc_mask: bool = False,
    save_intermediate: bool = False,
) -> None:
    if not images_dir.is_dir():
        raise NotADirectoryError(f"Images directory not found: {images_dir}")
    mask_dir.mkdir(parents=True, exist_ok=True)
    # Dynamically import models from either the legacy mvc_mask
    # package or the local app.model package based on the flag.
    if use_mvc_mask:
        try:
            from mvc_mask.model.pretreatment_model import PreprocessingModel  # type: ignore
            from mvc_mask.model.waterdrop_mask_model import WaterdropMaskModel  # type: ignore
            from mvc_mask.model.box_mask_model import BoxMaskModel  # type: ignore
        except Exception as e:
            raise ImportError("Failed to import mvc_mask models: " + str(e))
    else:
        from app.model.pretreatment_model import PreprocessingModel  # type: ignore
        from app.model.waterdrop_mask_model import WaterdropMaskModel  # type: ignore
        from app.model.box_mask_model import BoxMaskModel  # type: ignore
    pre = PreprocessingModel()
    wd_model = WaterdropMaskModel()
    bx_model = BoxMaskModel()
    # Prepare debug directory if needed
    debug_dir: Optional[Path] = None
    if save_intermediate:
        debug_dir = mask_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
    # Iterate through image files
    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"Warning: failed to load {img_path}, skipping.", file=sys.stderr)
            continue
        # Preprocess the image (gamma + CLAHE) for water–drop detection
        _, processed = pre.apply_full_preprocessing(img)
        # Compute water–drop mask (ignore visualisation)
        wd_mask, _ = wd_model.generate_mask(processed)
        # Compute crate mask on the original image
        bx_mask, _ = bx_model.generate_mask(img)
        # Intersect to keep only reflections within the crate
        final_mask = cv2.bitwise_and(wd_mask, bx_mask)
        # Save final mask
        out_name = f"{img_path.stem}_mask.png"
        out_path = mask_dir / out_name
        cv2.imwrite(str(out_path), final_mask)
        # Save intermediate masks if requested
        if save_intermediate and debug_dir is not None:
            cv2.imwrite(str(debug_dir / f"{img_path.stem}_box.png"), bx_mask)
            cv2.imwrite(str(debug_dir / f"{img_path.stem}_waterdrop.png"), wd_mask)
        print(f"Mask saved: {out_path}")


def main(
    use_mvc_mask: bool = False,
    save_intermediate: bool = False,
) -> None:
    images_dir = Path("data/images")
    mask_dir = Path("mask")
    process_images(
        images_dir,
        mask_dir,
        use_mvc_mask=use_mvc_mask,
        save_intermediate=save_intermediate,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate reflection masks for all images.")
    parser.add_argument(
        "--use-mvc-mask",
        action="store_true",
        help="Use models from mvc_mask instead of app.model",
    )
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Save individual crate and water–drop masks in addition to final masks",
    )
    args = parser.parse_args()
    main(use_mvc_mask=args.use_mvc_mask, save_intermediate=args.save_intermediate)