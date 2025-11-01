"""
mask_view.py
============

The view component of the MVC architecture contains functions for
saving and visualising masks produced by the model layer.  It does
not perform any image processing itself; instead it deals with
presentation concerns such as drawing outlines and writing files to
disk.  These utilities are intended to be called from the controller.

Two kinds of overlays are supported:

* **Polygon overlay**: for the crate mask, the model returns a list
  of polygon vertices outlining the detected crate.  The view can
  draw this polygon on top of the original image.
* **Contour overlay**: for waterdrop masks, the model returns a
  binary mask of specular highlights.  The view can draw the
  contours of these regions on top of the original image.

All functions accept and return ``numpy.ndarray`` images and
``pathlib.Path`` objects; any necessary directories are created
automatically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Iterable, Tuple

import cv2
import numpy as np


def ensure_dir(path: Path) -> None:
    """Ensure that the directory for the given path exists."""
    path.parent.mkdir(parents=True, exist_ok=True)


def save_mask(mask: np.ndarray, out_path: Path) -> None:
    """Save a binary mask to disk.

    Parameters
    ----------
    mask : numpy.ndarray
        Single–channel uint8 image containing 0 and 255 values.
    out_path : pathlib.Path
        Destination file path.  Parent directories are created if
        necessary.
    """
    ensure_dir(out_path)
    cv2.imwrite(str(out_path), mask)


def save_polygon_overlay(
    image: np.ndarray,
    polygon: Optional[Iterable[Tuple[int, int]]],
    out_path: Path,
    colour: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 4,
) -> None:
    """Draw a polygon on an image and save the result.

    Parameters
    ----------
    image : numpy.ndarray
        Input BGR image on which to draw.
    polygon : iterable of (x, y) pairs or None
        Sequence of vertex coordinates.  If ``None``, no polygon is
        drawn and the input image is saved as is.
    out_path : pathlib.Path
        Path to save the resulting image.
    colour : tuple[int, int, int]
        BGR colour for the outline.  Defaults to red.
    thickness : int
        Line thickness in pixels.  Defaults to 4.
    """
    ensure_dir(out_path)
    vis = image.copy()
    if polygon is not None and len(polygon) >= 2:
        poly_np = np.array(list(polygon), dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [poly_np], isClosed=True, color=colour, thickness=thickness, lineType=cv2.LINE_AA)
    cv2.imwrite(str(out_path), vis)


def save_contour_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    out_path: Path,
    colour: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 1,
) -> None:
    """Draw contours of a binary mask on an image and save it.

    This is useful for visualising waterdrop masks.  The contours
    correspond to connected components in ``mask``.

    Parameters
    ----------
    image : numpy.ndarray
        Input BGR image.
    mask : numpy.ndarray
        Single–channel uint8 mask with non–zero values indicating
        regions of interest.
    out_path : pathlib.Path
        Destination file path.
    colour : tuple[int, int, int]
        BGR colour for the contours.  Defaults to green.
    thickness : int
        Contour line thickness.  Defaults to 1.
    """
    ensure_dir(out_path)
    vis = image.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, colour, thickness)
    cv2.imwrite(str(out_path), vis)