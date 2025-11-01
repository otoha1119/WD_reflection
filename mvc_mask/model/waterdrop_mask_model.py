"""
waterdrop_mask_model.py
=======================

This module provides a :class:`WaterdropMaskModel` class that
encapsulates the water–drop/specular reflection detection algorithm
formerly implemented in ``GenerateWaterdropMask.py``.  The goal of
this model is to identify small, bright specular highlights on a
container surface which are caused by water droplets or glare, and
output a binary mask of those regions.

The algorithm integrates several detection cues:

* **HSV thresholding** to isolate bright, low–saturation pixels
  characteristic of specular reflections.
* **Morphological top–hat filtering** to pick up small bright spots.
* **Local z–score thresholding** to detect locally bright regions
  despite global brightness variations.
* **Difference–of–Gaussian (DoG)** filtering to highlight streaks or
  diffused glints.

These cues are OR–ed together to form an initial candidate mask.
Subsequent morphological operations clean up the mask and a mild
watershed segmentation splits touching droplets.  Finally, small
components are removed.

The model exposes a single public method :meth:`generate_mask` which
takes a (preferably pre–processed) BGR image and returns both the
final binary mask and a visualisation image where droplet contours
are drawn on top of the input.

This model does not perform any file I/O; saving of masks and
visualisations is delegated to the view layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass
class WaterdropMaskModel:
    """Detect specular highlights (water droplets) in an image.

    The parameters below control the sensitivity of the various
    detection branches.  They are exposed as dataclass fields so that
    they can easily be tuned from the controller or via unit tests.
    """

    # HSV threshold parameters
    # Further lower the specular detection threshold to pick up even
    # darker reflections.  v_k controls how many standard deviations
    # above the mean the V threshold lies, and v_floor caps the
    # minimum value of the threshold.  s_max is the maximum allowed
    # saturation for a pixel to be considered specular.  These
    # defaults were tuned experimentally on crate images with dim
    # water streaks: setting v_k < 1.0 and v_floor around the 97th
    # percentile of the background brightness helps reveal dark
    # droplets.
    v_k: float = 0.8
    # Raise the floor slightly to suppress extremely dark regions
    v_floor: float = 70
    s_max: float = 130
    # Percentile for the V channel when computing the specular
    # threshold.  Instead of solely relying on the mean+std method,
    # the threshold is taken as the larger of ``v_floor`` and the
    # ``v_percentile``–th percentile of the V channel.  This makes
    # the algorithm adapt to images with very dark global brightness,
    # ensuring that the brightest few percent of pixels are always
    # considered.  Typical values are between 96 and 99.
    v_percentile: float = 98.0
    # Top–hat filtering parameters
    # Use a smaller structuring element and a more permissive
    # threshold (lower k and floor) to capture weak highlights.
    tophat_se_size: int = 9
    tophat_k: float = 0.9
    tophat_floor: float = 4
    # Local z–score parameters
    # Keep a moderate window but reduce the z–score threshold so that
    # subtle local brightening stands out.  Smaller windows risk
    # over‑detecting grid noise, so we keep 21 but lower the threshold.
    z_win: int = 21
    z_thr: float = 2.0
    # DoG parameters
    # Increase the spread between sigma1 and sigma2 and lower the
    # percentile threshold to bring out broad, faint streaks.
    dog_sigma1: float = 1.0
    dog_sigma2: float = 3.0
    # Raise percentile to tighten DoG detection so it focuses on the most
    # prominent diffused highlights rather than large swaths of texture.
    dog_percentile: float = 97.0
    # Morphological post–processing
    # Slightly lower the minimum area to retain small droplets,
    # while keeping the closing kernel modest to avoid filling holes.
    open_k: int = 3
    close_k: int = 5
    # Increase minimum area to filter out small specks that are likely
    # noise.  Water droplets tend to occupy multiple pixels after
    # pre–processing, whereas single–pixel detections are often grid
    # artefacts.
    min_area: int = 20
    # Watershed parameters
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
        for i in range(1, num):
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
            Input BGR image.  It is recommended to pre–process the
            image (e.g., via gamma correction and CLAHE) before
            passing it to this method, but it is not strictly
            necessary.

        Returns
        -------
        tuple
            ``(mask, vis)`` where ``mask`` is a single–channel
            uint8 image (255 for detected droplets, 0 otherwise) and
            ``vis`` is a visualisation where droplet contours are
            drawn on top of the input image.
        """
        # Ensure input is BGR and uint8
        if img is None or img.ndim != 3 or img.dtype != np.uint8:
            raise ValueError("Input must be a BGR uint8 image")
        # Convert to HSV for the specular branch
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        # Specular (HSV) branch
        v_f32 = v.astype(np.float32)
        v_mean, v_std = float(np.mean(v_f32)), float(np.std(v_f32))
        # Compute two candidate thresholds: one based on mean+std and
        # one based on a percentile.  Take the maximum to ensure that
        # we only consider pixels that are both brighter than the
        # global background and within the top fraction of the
        # brightness distribution.  Finally cap by v_floor.
        v_thr_std = v_mean + self.v_k * v_std
        v_thr_pct = np.percentile(v_f32, self.v_percentile)
        v_thr = max(float(self.v_floor), max(v_thr_std, v_thr_pct))
        specular_mask = ((v_f32 >= v_thr) & (s.astype(np.float32) <= self.s_max)).astype(np.uint8) * 255
        # Top–hat branch
        gray = self._to_gray(img)
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.tophat_se_size, self.tophat_se_size))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se)
        th_mean, th_std = float(np.mean(tophat)), float(np.std(tophat))
        th_thr = max(th_mean + self.tophat_k * th_std, float(self.tophat_floor))
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
        # Threshold using percentile
        p = np.percentile(dog, self.dog_percentile)
        dogmask = (dog >= p).astype(np.uint8) * 255
        dogmask = cv2.morphologyEx(dogmask, cv2.MORPH_CLOSE, k5)
        # Combine all branches via OR
        combined_mask = cv2.bitwise_or(specular_mask, tophat_mask)
        combined_mask = cv2.bitwise_or(combined_mask, zmask)
        combined_mask = cv2.bitwise_or(combined_mask, dogmask)
        # Non–zero safeguard: if everything is zero, fall back to
        # whichever branch yields something
        if combined_mask.max() == 0:
            combined_mask = cv2.max(specular_mask, tophat_mask)
        if combined_mask.max() == 0:
            combined_mask = cv2.max(zmask, dogmask)
        # Post–processing: open then close to tidy up
        open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.open_k, self.open_k))
        close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.close_k, self.close_k))
        mask_oc = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, open_k)
        mask_oc = cv2.morphologyEx(mask_oc, cv2.MORPH_CLOSE, close_k)
        # Watershed splitting to separate touching droplets
        mask_ws = self._split_touching(mask_oc)
        # Combine split and non–split masks to recover missed parts
        mask_merged = cv2.bitwise_or(mask_oc, mask_ws)
        # Filter out tiny components
        mask_final = self._area_filter(mask_merged, self.min_area)
        # Create visualisation
        vis = self._overlay_and_contours(img, mask_final)
        return mask_final, vis