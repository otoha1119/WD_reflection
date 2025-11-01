"""
box_mask_model.py
===================

This module contains the :class:`BoxMaskModel` class, which
implements an algorithm to locate a green plastic crate within a
production line image and produce a corresponding binary mask.

The implementation originates from ``GenerateBoxMask.py`` but has
been refactored into a class so that it can be used as part of an
MVC architecture.  The core steps of the algorithm are:

1. **Colour thresholding in HSV space** to extract dark green
   regions that correspond to the crate surface while suppressing
   bright highlights.
2. **Morphological operations** to clean up the mask and remove
   noise.  Closing operations fill holes between the crate slats,
   while opening removes small spurious blobs.
3. **Long horizontal edge suppression** using a Hough transform to
   detect the conveyor rails and subtract them from the mask.
4. **Connected component filtering** to choose the region that best
   resembles the crate (based on area, aspect ratio and position).
5. **Convex hull and polygon approximation** to derive a neat
   quadrilateral representing the crate, and **polygon fill** to
   produce the final binary mask.

The resulting mask can be combined with the original image to draw
only the outline for visualisation purposes.  See the accompanying
view module for saving routines.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Optional, Tuple, List


class BoxMaskModel:
    """Model for generating a mask of a crate from an input image.

    Parameters
    ----------
    hue_range : tuple[int, int], optional
        Lower and upper bounds for the hue component (0–180).  By
        default this targets a broad range of greens.
    sat_range : tuple[int, int], optional
        Lower and upper bounds for the saturation component.  Low
        saturation values are excluded to avoid grey backgrounds.
    val_range : tuple[int, int], optional
        Lower and upper bounds for the value (brightness) component.  A
        fairly low upper bound prevents very bright highlights from
        being included.
    highlight_threshold : int, optional
        Pixels above this V threshold are considered specular highlights
        and are removed from the colour mask.  Defaults to 150.
    min_area_ratio : float, optional
        Minimum area (as a fraction of the image area) for a
        candidate region to be considered as the crate.  Defaults to
        0.04 (i.e., 4 % of the image area).
    aspect_ratio_range : tuple[float, float], optional
        Acceptable range of aspect ratios (width/height) for the
        bounding box of a candidate region.  Defaults to (1.1, 5.0).
    position_bias : float, optional
        Vertical position bias term used to rank candidate regions.
        The score for a region is multiplied by ``1 - |(cy/h) -
        position_bias|``.  By default the crate is expected to be
        located around 65 % down the image.
    """

    def __init__(
        self,
        hue_range: Tuple[int, int] = (45, 95),
        sat_range: Tuple[int, int] = (60, 255),
        val_range: Tuple[int, int] = (15, 140),
        highlight_threshold: int = 150,
        min_area_ratio: float = 0.04,
        aspect_ratio_range: Tuple[float, float] = (1.1, 5.0),
        position_bias: float = 0.65,
    ) -> None:
        self.hue_range = hue_range
        self.sat_range = sat_range
        self.val_range = val_range
        self.highlight_threshold = highlight_threshold
        self.min_area_ratio = min_area_ratio
        self.aspect_ratio_range = aspect_ratio_range
        self.position_bias = position_bias

    # ------------------------------------------------------------------
    # Private helper methods
    # ------------------------------------------------------------------
    @staticmethod
    def _remove_long_horizontal_edges(
        img_shape: Tuple[int, int],
        edges: np.ndarray,
        min_len: int = 300,
        gap: int = 30,
        thick: int = 22,
        angle_threshold: float = 12.0,
    ) -> np.ndarray:
        """Detect and dilate long horizontal edges to form a mask.

        These edges correspond to the conveyor rails and should be
        subtracted from the crate mask.  The Hough transform is
        restricted to detect nearly horizontal lines; detected lines are
        drawn thickly and then dilated to ensure coverage.

        Parameters
        ----------
        img_shape : tuple[int, int]
            Height and width of the image (h, w).
        edges : numpy.ndarray
            Binary edge map (e.g., from :func:`cv2.Canny`).
        min_len : int
            Minimum length of a line to be considered a rail.
        gap : int
            Maximum gap between line segments to merge them.
        thick : int
            Thickness used when drawing the detected rails.
        angle_threshold : float
            Maximum deviation from horizontal (in degrees) for a line to
            be considered a rail.

        Returns
        -------
        numpy.ndarray
            Binary image where rail pixels are 255 and others are 0.
        """
        h, w = img_shape[:2]
        rail = np.zeros((h, w), dtype=np.uint8)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=120,
                                minLineLength=min_len, maxLineGap=gap)
        if lines is None:
            return rail
        for (x1, y1, x2, y2) in lines[:, 0]:
            # Compute absolute angle w.r.t. horizontal
            ang = abs(np.degrees(np.arctan2((y2 - y1), (x2 - x1))))
            if ang < angle_threshold:
                cv2.line(rail, (x1, y1), (x2, y2), 255, thick)
        # Dilate to bridge small gaps and ensure coverage
        rail = cv2.dilate(
            rail,
            cv2.getStructuringElement(cv2.MORPH_RECT, (19, 19)),
            iterations=1,
        )
        return rail

    def _colour_threshold(self, img: np.ndarray) -> np.ndarray:
        """Apply HSV colour thresholding to extract green crate regions.

        Parameters
        ----------
        img : numpy.ndarray
            BGR image loaded with OpenCV.

        Returns
        -------
        numpy.ndarray
            Binary mask highlighting candidate crate pixels.
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([self.hue_range[0], self.sat_range[0], self.val_range[0]], dtype=np.uint8)
        upper = np.array([self.hue_range[1], self.sat_range[1], self.val_range[1]], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        # Remove very bright highlights (specular reflections) from the mask
        v = hsv[:, :, 2]
        _, hi = cv2.threshold(v, self.highlight_threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(hi))
        return mask

    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Perform simple pre–filtering on the input image.

        Currently a bilateral filter is applied to suppress noise while
        preserving edges.

        Parameters
        ----------
        img : numpy.ndarray
            Input BGR image.

        Returns
        -------
        numpy.ndarray
            Filtered image.
        """
        return cv2.bilateralFilter(img, d=7, sigmaColor=40, sigmaSpace=7)

    def _select_best_candidate(
        self,
        mask_candidates: np.ndarray,
        area_threshold: float,
        aspect_ratio_range: Tuple[float, float],
        pos_bias: float,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Select the most plausible crate region among connected components.

        Parameters
        ----------
        mask_candidates : numpy.ndarray
            Binary image containing candidate pixels.
        area_threshold : float
            Minimum absolute area (in pixels) of a region to be considered.
        aspect_ratio_range : tuple[float, float]
            Acceptable range of bounding box aspect ratios.
        pos_bias : float
            Vertical position bias (0–1).  Regions closer to this
            relative y position receive a higher score.

        Returns
        -------
        tuple
            A tuple containing (best_contour, polygon_points).  If no
            suitable region is found, best_contour is ``None`` and the
            polygon is ``None``.
        """
        h, w = mask_candidates.shape
        contours, _ = cv2.findContours(mask_candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None
        best = None
        best_score = -1.0
        for c in contours:
            area = cv2.contourArea(c)
            if area < area_threshold:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            aspect_ratio = bw / (bh + 1e-6)
            if not (aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]):
                continue
            # Compute a position–biased score; crates are usually in the
            # lower middle of the image.  Regions closer to `pos_bias`*h
            # are scored higher.
            cy = y + bh / 2.0
            pos_factor = 1.0 - abs((cy / h) - pos_bias)
            score = area * pos_factor
            if score > best_score:
                best_score = score
                best = c
        if best is None:
            # Fallback: choose the largest contour
            best = max(contours, key=cv2.contourArea)
        # Compute convex hull and approximate polygon
        hull = cv2.convexHull(best)
        perim = cv2.arcLength(hull, True)
        epsilon = 0.015 * perim
        approx = cv2.approxPolyDP(hull, epsilon, True)
        # If too many points remain, simplify further
        if len(approx) > 8:
            approx = cv2.approxPolyDP(hull, 0.03 * perim, True)
        # The polygon is returned as a 2D array of shape (N, 2)
        poly_points = approx.reshape(-1, 2) if approx is not None else None
        return best, poly_points

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate_mask(
        self, img: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Compute the crate mask and corresponding polygon.

        Parameters
        ----------
        img : numpy.ndarray
            Input BGR image for which the crate should be masked.

        Returns
        -------
        tuple
            Two elements: (mask, polygon)
            - ``mask`` is a single–channel uint8 array with pixel
              values 0 (background) or 255 (crate).
            - ``polygon`` is an ``(M, 2)`` array of vertices outlining
              the crate.  It may be ``None`` if no suitable region is
              found.
        """
        # Pre–filter to suppress noise
        img_filtered = self._preprocess_image(img)
        # Colour threshold to isolate crate candidates
        mask_colour = self._colour_threshold(img_filtered)
        # Morphological closing and opening to tidy up the mask
        mask_colour = cv2.morphologyEx(
            mask_colour,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
            iterations=1,
        )
        mask_colour = cv2.morphologyEx(
            mask_colour,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            iterations=1,
        )
        # Remove long horizontal rails
        gray = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 60, 160)
        rails = self._remove_long_horizontal_edges(img_filtered.shape, edges)
        mask_candidates = cv2.bitwise_and(mask_colour, cv2.bitwise_not(rails))
        # Compute area threshold in pixels
        h, w = mask_candidates.shape
        area_thresh_px = self.min_area_ratio * h * w
        # Select the best contour and polygon
        contour, polygon = self._select_best_candidate(
            mask_candidates,
            area_threshold=area_thresh_px,
            aspect_ratio_range=self.aspect_ratio_range,
            pos_bias=self.position_bias,
        )
        # If we have a contour and polygon, fill it
        mask_out = np.zeros((h, w), dtype=np.uint8)
        if polygon is not None and len(polygon) >= 3:
            cv2.fillPoly(mask_out, [polygon.astype(np.int32)], 255)
            # Smooth the edges with a small closing
            mask_out = cv2.morphologyEx(
                mask_out,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
                iterations=1,
            )
        return mask_out, polygon
