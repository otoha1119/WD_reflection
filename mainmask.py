from __future__ import annotations

import sys
from pathlib import Path

import cv2

from mvc_mask.model.box_mask_model import BoxMaskModel
from mvc_mask.model.pretreatment_model import PreprocessingModel
from mvc_mask.model.waterdrop_mask_model import WaterdropMaskModel
from mvc_mask.view.mask_view import (
    save_mask,
    save_polygon_overlay,
    save_contour_overlay,
)


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
# Directory containing input images.  It is assumed to reside within
# the repository root.  Modify this path to suit your environment.
INPUT_DIR = Path("data/images")

# Directory where binary masks will be saved.  Crate masks and water–drop
# masks are saved here with different suffixes.  This directory will
# be created if it does not exist.
MASK_OUTPUT_DIR = Path("mask")

# Directory where overlay visualisations will be saved.  Overlays
# include crate outlines and water–drop contours drawn on the original
# images.  This directory will be created if it does not exist.
OVERLAY_OUTPUT_DIR = Path("mask_vis")

# Supported image file extensions (case insensitive)
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def process_image(
    image_path: Path,
    box_model: BoxMaskModel,
    preproc_model: PreprocessingModel,
    water_model: WaterdropMaskModel,
) -> None:
    
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"[WARN] Could not read image: {image_path}")
        return
    # Generate crate mask and polygon
    crate_mask, crate_poly = box_model.generate_mask(img)
    # Save crate mask
    mask_name = image_path.stem + "_box_mask.png"
    #save_mask(crate_mask, MASK_OUTPUT_DIR / mask_name)
    # Save crate overlay
    overlay_name = image_path.stem + "_box_overlay.png"
    save_polygon_overlay(img, crate_poly, OVERLAY_OUTPUT_DIR / overlay_name)
    # Pre–process image for water–drop detection
    _, processed = preproc_model.apply_full_preprocessing(img)
    # Generate water–drop mask and visualisation
    water_mask, _ = water_model.generate_mask(processed)
    # Restrict detections to within the crate mask
    water_mask = cv2.bitwise_and(water_mask, crate_mask)
    # Save water–drop mask
    water_mask_name = image_path.stem + "_waterdrop_mask.png"
    save_mask(water_mask, MASK_OUTPUT_DIR / water_mask_name)
    # Save water–drop overlay
    water_overlay_name = image_path.stem + "_waterdrop_overlay.png"
    save_contour_overlay(img, water_mask, OVERLAY_OUTPUT_DIR / water_overlay_name)
    print(f"[OK] Processed {image_path.name}")


def main() -> int:
    # Check input directory
    if not INPUT_DIR.exists() or not INPUT_DIR.is_dir():
        print(f"[ERROR] Input directory not found: {INPUT_DIR}")
        return 1
    # Instantiate models
    box_model = BoxMaskModel()
    preproc_model = PreprocessingModel()
    water_model = WaterdropMaskModel()
    # Ensure output directories exist
    MASK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Iterate over files in the input directory
    for item in sorted(INPUT_DIR.iterdir()):
        # Skip directories and unsupported files
        if item.is_dir() or item.suffix.lower() not in SUPPORTED_EXTS:
            continue
        process_image(item, box_model, preproc_model, water_model)
    print("All images processed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())