from __future__ import annotations

from typing import Optional, Tuple, List

import cv2
import numpy as np


class BoxMaskModel:
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
        """Detect and dilate long horizontal edges to form a rail mask.

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
            ang = abs(np.degrees(np.arctan2((y2 - y1), (x2 - x1))))
            if ang < angle_threshold:
                cv2.line(rail, (x1, y1), (x2, y2), 255, thick)
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
        v = hsv[:, :, 2]
        _, hi = cv2.threshold(v, self.highlight_threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(hi))
        return mask

    @staticmethod
    def _preprocess_image(img: np.ndarray) -> np.ndarray:
        """Apply a bilateral filter to suppress noise while preserving edges."""
        return cv2.bilateralFilter(img, d=7, sigmaColor=40, sigmaSpace=7)

    def _select_best_candidate(
        self,
        mask_candidates: np.ndarray,
        area_threshold: float,
        aspect_ratio_range: Tuple[float, float],
        pos_bias: float,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
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
            cy = y + bh / 2.0
            pos_factor = 1.0 - abs((cy / h) - pos_bias)
            score = area * pos_factor
            if score > best_score:
                best_score = score
                best = c
        if best is None:
            best = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(best)
        perim = cv2.arcLength(hull, True)
        epsilon = 0.015 * perim
        approx = cv2.approxPolyDP(hull, epsilon, True)
        if len(approx) > 8:
            approx = cv2.approxPolyDP(hull, 0.03 * perim, True)
        poly_points = approx.reshape(-1, 2) if approx is not None else None
        return best, poly_points

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate_mask(
        self, img: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        img_filtered = self._preprocess_image(img)
        mask_colour = self._colour_threshold(img_filtered)
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
        gray = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 60, 160)
        rails = self._remove_long_horizontal_edges(img_filtered.shape, edges)
        mask_candidates = cv2.bitwise_and(mask_colour, cv2.bitwise_not(rails))
        h, w = mask_candidates.shape
        area_thresh_px = self.min_area_ratio * h * w
        _, polygon = self._select_best_candidate(
            mask_candidates,
            area_threshold=area_thresh_px,
            aspect_ratio_range=self.aspect_ratio_range,
            pos_bias=self.position_bias,
        )
        mask_out = np.zeros((h, w), dtype=np.uint8)
        if polygon is not None and len(polygon) >= 3:
            cv2.fillPoly(mask_out, [polygon.astype(np.int32)], 255)
            mask_out = cv2.morphologyEx(
                mask_out,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
                iterations=1,
            )
        return mask_out, polygon