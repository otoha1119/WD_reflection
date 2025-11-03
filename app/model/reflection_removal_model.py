from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class ReflectionRemovalModel:
    small_thresh: int = 300
    dilate_kernel: int = 5
    dilate_iter: int = 1
    telea_radius: float = 4.0
    ns_radius: float = 8.0
    sigma_space: float = 2.5
    sigma_color: float = 25.0

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def dilate_mask(self, mask: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.dilate_kernel, self.dilate_kernel))
        dilated = cv2.dilate(mask, kernel, iterations=self.dilate_iter)
        return dilated

    def classify_components(self, mask: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        small_masks: List[np.ndarray] = []
        large_masks: List[np.ndarray] = []
        for i in range(1, num):  # skip background
            area = stats[i, cv2.CC_STAT_AREA]
            comp_mask = (labels == i).astype(np.uint8) * 255
            if area < self.small_thresh:
                small_masks.append(comp_mask)
            else:
                large_masks.append(comp_mask)
        return small_masks, large_masks

    def inpaint_regions(self, img: np.ndarray, comp_masks: List[np.ndarray], method: int, radius: float) -> np.ndarray:
        result = img.copy()
        for cmask in comp_masks:
            result = cv2.inpaint(result, cmask, radius, method)
        return result

    def joint_bilateral_on_mask(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        # Convert mask to 0/1
        m = (mask > 0).astype(np.uint8)
        try:
            from cv2.ximgproc import jointBilateralFilter  # type: ignore
            guide = img
            blurred = jointBilateralFilter(guide, img, d=5, sigmaColor=self.sigma_color, sigmaSpace=self.sigma_space)
        except Exception:
            # Fallback if ximgproc is not available
            blurred = cv2.bilateralFilter(img, d=5, sigmaColor=self.sigma_color, sigmaSpace=self.sigma_space)
        out = img.copy()
        out[m > 0] = blurred[m > 0]
        return out

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def remove_reflections(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        # Dilate the mask to include halo pixels
        dilated = self.dilate_mask(mask)
        # Classify connected components by area
        small_masks, large_masks = self.classify_components(dilated)
        # Apply inpainting on a working copy
        result = img.copy()
        if small_masks:
            result = self.inpaint_regions(result, small_masks, cv2.INPAINT_TELEA, self.telea_radius)
        if large_masks:
            result = self.inpaint_regions(result, large_masks, cv2.INPAINT_NS, self.ns_radius)
        # Light blending on all dilated mask pixels
        blended = self.joint_bilateral_on_mask(result, dilated)
        return blended