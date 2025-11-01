"""
solution.py
This script performs simple image processing to generate a binary mask for
a green plastic crate (container) seen on a production line.  The goal
is to isolate the container from the rest of the scene without using
any machine‑learning based techniques.  The pipeline implemented here
relies on colour thresholding and morphological operations provided by
OpenCV.

Overview of the steps:

1. **Load the input image** from the fixed path ``data/images/1.png``.
2. **Convert the image to the HSV colour space**.  The crate is dark
   green while the surrounding conveyor and walls are grey or metal.
   A broad green hue range is defined and pixels within this range are
   selected to build an initial binary mask.  The chosen range has
   been tuned empirically so that it includes both saturated and
   desaturated greens while excluding most of the background.
3. **Clean up the raw mask** with morphological operations:
   - A *closing* operation (dilation followed by erosion) using a
     relatively large kernel fills in the gaps between the slats of the
     crate.
   - A *hole filling* step ensures that any remaining holes inside the
     crate region are filled.  A common approach is to flood‑fill from
     the background and then invert the result.
   - A *opening* operation (erosion followed by dilation) removes
     spurious small blobs that do not belong to the crate.
4. **Find the largest connected component** in the cleaned mask and
   assume it corresponds to the crate.  This step helps to avoid
   interference from any residual conveyor parts that may have passed
   the colour threshold.
5. **Save the resulting mask** alongside the input image.  The mask
   will be a single‑channel PNG image where crate pixels are white
   (value 255) and the background is black (value 0).

Although this approach does not perfectly model the projective shape of
the crate, it provides a reasonable, heuristic solution that does not
require any heavy AI or training data.  The parameters (HSV range,
morphological kernel sizes, etc.) can be tuned further based on the
lighting conditions and crate colour variations in your environment.
"""

import os
import cv2
import numpy as np


def generate_crate_mask(image: np.ndarray) -> np.ndarray:
    """Generate a binary mask isolating the green crate in the input image.

    Parameters
    ----------
    image : numpy.ndarray
        BGR image loaded with OpenCV.

    Returns
    -------
    numpy.ndarray
        Single‑channel 8‑bit image where pixels belonging to the crate
        are 255 and others are 0.
    """
    # Convert the image to HSV.  HSV separates colour (hue) from
    # brightness, which makes it easier to specify colour ranges.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define a range of green hues.  Dark green crates tend to have
    # hues between roughly 40 and 100 degrees in OpenCV's 0‑180 scale.
    # Saturation and value thresholds are kept broad to include both
    # saturated and desaturated greens under varying lighting.
    lower_green = np.array([40, 40, 20])
    upper_green = np.array([100, 255, 200])
    raw_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Use morphological closing to bridge the gaps between the slats of
    # the crate.  A rectangular kernel of size 25×25 works well for
    # crates of the resolution provided (around 1500–2000 pixels in
    # height).  If your resolution is markedly different you may
    # consider scaling this size accordingly.
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    closed = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel_close)

    # Fill any remaining holes inside the closed mask.  Flood‑fill from
    # the border and then invert to obtain the foreground mask.
    h, w = closed.shape
    flood_filled = closed.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood_filled, mask, (0, 0), 255)
    flood_filled_inv = cv2.bitwise_not(flood_filled)
    filled_mask = closed | flood_filled_inv

    # Remove small spurious regions with a morphological opening.  A
    # smaller kernel is sufficient here since the main blobs have
    # already been consolidated.
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    cleaned_mask = cv2.morphologyEx(filled_mask, cv2.MORPH_OPEN, kernel_open)

    # Identify the largest connected component.  This assumes that the
    # crate occupies the largest greenish region in the scene.  Any
    # smaller green patches (e.g. stray tape or labels) will be
    # discarded.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned_mask)
    if num_labels <= 1:
        # No components detected; return the cleaned mask as is.
        return cleaned_mask
    # Skip the first label (background) and find the label with the
    # maximum area.
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))
    mask_out = np.zeros_like(cleaned_mask)
    mask_out[labels == largest_label] = 255

    return mask_out


def main():
    """Main entry point for the script.

    Loads the image from ``data/images/1.png``, computes the crate
    mask and writes the result to ``data/images/1_mask.png``.
    """
    # Compose the input and output paths.  The input directory is
    # assumed to exist relative to the working directory.
    input_path = os.path.join("data", "images", "1.png")
    output_path = os.path.join("data", "images", "1_mask.png")

    # Load the input image.  If the file cannot be found an
    # informative error will be raised.
    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"Input image not found: {input_path}")

    # Generate the mask using the helper function.
    mask = generate_crate_mask(image)

    # Write the mask to disk.  Ensure the parent directory exists.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, mask)


if __name__ == "__main__":
    main()