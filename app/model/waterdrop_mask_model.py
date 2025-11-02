"""
waterdrop_mask_model
====================

This module provides a :class:`WaterdropMaskModel` class that
encapsulates the water–drop/specular reflection detection algorithm
used in the original programming assignment.  The goal of this model
is to identify small, bright specular highlights on a container
surface caused by water droplets or glare, and output a binary mask
of those regions.

The algorithm integrates several detection cues:

* **HSV thresholding** to isolate bright, low–saturation pixels
  characteristic of specular reflections.
* **Morphological top–hat filtering** to pick up small bright spots.
* **Local z–score thresholding** to detect locally bright regions
  despite global brightness variations.
* **Difference–of–Gaussian (DoG) filtering** to highlight streaks or
  diffused glints.

These cues are OR–ed together to form an initial candidate mask.
Subsequent morphological operations clean up the mask and a mild
watershed segmentation splits touching droplets.  Finally, small
components are removed.

The model exposes a single public method :meth:`generate_mask` which
takes a pre–processed BGR image and returns both the final binary
mask and a visualisation image where droplet contours are drawn on
top of the input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass
class WaterdropMaskModel:
    """Detect specular highlights (water droplets) in an image.

    The parameters below mirror those in the original
    ``GenerateWaterdropMask.py`` script.  They can be adjusted to
    tune the sensitivity of each detection branch but reasonable
    defaults are provided for typical crate images.
    """

    # HSV threshold parameters
    v_k: float = 1.5
    v_floor: float = 150.0
    s_max: float = 80.0
    # Top–hat filtering
    tophat_se_size: int = 9
    tophat_k: float = 1.1
    tophat_floor: float = 10.0
    # Local z–score
    z_win: int = 31
    z_thr: float = 2.1
    # DoG filtering
    dog_sigma1: float = 1.2
    dog_sigma2: float = 2.0
    dog_percentile: float = 98.3
    # Morphological post–processing
    open_k: int = 3
    close_k: int = 3
    min_area: int = 20
    # Watershed splitting
    ws_min_peak_dist: int = 2
    ws_rel_peak: float = 0.20

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------
    @staticmethod
    def _to_gray(img: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    @staticmethod
    def _area_filter(mask: np.ndarray, min_area: int) -> np.ndarray:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out = np.zeros_like(mask)
        for i in range(1, num):  # skip background
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                out[labels == i] = 255
        return out

    @staticmethod
    def _overlay_and_contours(img_bgr: np.ndarray, mask_bin: np.ndarray) -> np.ndarray:
        vis = img_bgr.copy()
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0, 255, 0), 1)
        return vis

    def _split_touching(self, mask_bin: np.ndarray) -> np.ndarray:
        """Separate touching droplets using a watershed segmentation.

        This method replicates the weak watershed used in the original
        script.  It computes the distance transform of the mask,
        identifies local maxima as markers, and applies watershed on a
        colour version of the mask.  The resulting segmentation
        separates touching blobs while attempting not to erase small
        droplets entirely.  Only the mask of split regions is
        returned; the original mask must be OR–ed with this result.
        """
        if mask_bin.max() == 0:
            return mask_bin
        dist = cv2.distanceTransform(mask_bin, cv2.DIST_L2, 5)
        dmax = float(dist.max())
        if dmax < 1e-3:
            return mask_bin
        k = 2 * int(self.ws_min_peak_dist) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        local_max = (dist == cv2.dilate(dist, kernel)).astype(np.uint8)
        peaks = np.logical_and(local_max > 0, dist >= self.ws_rel_peak * dmax).astype(np.uint8)
        num, markers = cv2.connectedComponents(peaks.astype(np.uint8))
        if num <= 1:
            return mask_bin
        markers = markers + 1
        markers[mask_bin == 0] = 0
        colour = cv2.cvtColor(mask_bin, cv2.COLOR_GRAY2BGR)
        cv2.watershed(colour, markers)
        split = (markers > 1).astype(np.uint8) * 255
        return split

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------
    def generate_mask(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a water–drop/reflection mask from a BGR image.

        Parameters
        ----------
        img : numpy.ndarray
            Input BGR image (uint8).  It is recommended to apply
            gamma correction and CLAHE before calling this method but
            not strictly necessary.

        Returns
        -------
        tuple
            ``(mask, vis)`` where ``mask`` is a binary image (255
            where droplets were detected) and ``vis`` overlays the
            mask contours on the input for debugging.
        """
        if img is None or img.ndim != 3 or img.dtype != np.uint8:
            raise ValueError("Input must be a BGR uint8 image")
        # HSV branch
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v_f32 = v.astype(np.float32)
        v_mean, v_std = float(np.mean(v_f32)), float(np.std(v_f32))
        v_thr_std = v_mean + self.v_k * v_std
        v_thr_pct = np.percentile(v_f32, self.dog_percentile)  # note: using dog_percentile here is a quirk of original script
        v_thr = max(self.v_floor, v_thr_std, v_thr_pct)
        specular_mask = ((v_f32 > v_thr) & (s.astype(np.float32) < self.s_max)).astype(np.uint8) * 255
        # Top–hat branch
        gray = self._to_gray(img)
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.tophat_se_size, self.tophat_se_size))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se)
        th_mean, th_std = float(np.mean(tophat)), float(np.std(tophat))
        th_thr = max(th_mean + self.tophat_k * th_std, self.tophat_floor)
        _, tophat_mask = cv2.threshold(tophat, int(th_thr), 255, cv2.THRESH_BINARY)
        # Local z–score branch
        g32 = gray.astype(np.float32)
        mean = cv2.boxFilter(g32, ddepth=-1, ksize=(self.z_win, self.z_win), normalize=True)
        mean_sq = cv2.boxFilter(g32 * g32, ddepth=-1, ksize=(self.z_win, self.z_win), normalize=True)
        var = np.clip(mean_sq - mean * mean, 1e-6, None)
        zmap = (g32 - mean) / np.sqrt(var)
        zmask = (zmap > self.z_thr).astype(np.uint8) * 255
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        zmask = cv2.morphologyEx(zmask, cv2.MORPH_CLOSE, k5)
        zmask = cv2.morphologyEx(zmask, cv2.MORPH_OPEN, k3)
        # DoG branch
        g1 = cv2.GaussianBlur(gray, (0, 0), self.dog_sigma1)
        g2 = cv2.GaussianBlur(gray, (0, 0), self.dog_sigma2)
        dog = cv2.subtract(g1, g2).astype(np.float32)
        p = np.percentile(dog, self.dog_percentile)
        dogmask = (dog >= p).astype(np.uint8) * 255
        dogmask = cv2.morphologyEx(dogmask, cv2.MORPH_CLOSE, k5)
        # Combine all branches
        combined = cv2.bitwise_or(specular_mask, tophat_mask)
        combined = cv2.bitwise_or(combined, zmask)
        combined = cv2.bitwise_or(combined, dogmask)
        # Fallback if no detections
        if combined.max() == 0:
            combined = cv2.max(specular_mask, tophat_mask)
        if combined.max() == 0:
            combined = cv2.max(zmask, dogmask)
        # Open then close
        open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.open_k, self.open_k))
        close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.close_k, self.close_k))
        mask_oc = cv2.morphologyEx(combined, cv2.MORPH_OPEN, open_k)
        mask_oc = cv2.morphologyEx(mask_oc, cv2.MORPH_CLOSE, close_k)
        # Watershed splitting
        mask_ws = self._split_touching(mask_oc)
        # Merge split and original
        mask_merged = cv2.bitwise_or(mask_oc, mask_ws)
        # Area filter
        mask_final = self._area_filter(mask_merged, self.min_area)
        vis = self._overlay_and_contours(img, mask_final)
        return mask_final, vis