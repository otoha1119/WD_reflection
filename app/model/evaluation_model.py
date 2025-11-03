"""
evaluation_model
================

This module defines the :class:`EvaluationModel` class which
implements various metrics for evaluating the quality of reflection
removal.  The model computes four key indicators:

1. **HLR** (Highlight Reduction Ratio): Percentage reduction in
   high-intensity pixels within the box region.

2. **Cov** (Coverage/Completeness): Percentage of mask pixels where
   highlights remain after processing (lower is better).

3. **Brightness Reduction Rate**: Average reduction in V-channel
   (brightness) values within the mask region.

4. **Local Variance Improvement**: Improvement in texture naturalness
   measured by local standard deviation changes.

All metrics are computed only within the crate (box) region to avoid
bias from unchanged background areas.  The model accepts pairs of
before/after images along with a box mask and produces a dictionary
of metric values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import cv2
import numpy as np


@dataclass
class EvaluationModel:
    """Model for evaluating reflection removal quality.

    Parameters
    ----------
    highlight_threshold : int, optional
        Threshold for considering a pixel as a highlight (V-channel).
        Pixels with V > this value are counted as highlights.
        Default is 200.
    local_window_size : int, optional
        Window size for computing local standard deviation.
        Default is 16 (i.e., 16x16 blocks).
    """

    highlight_threshold: int = 200
    local_window_size: int = 16

    @staticmethod
    def _extract_v_channel(img: np.ndarray) -> np.ndarray:
        """Extract V (brightness) channel from BGR image."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return hsv[:, :, 2]

    def _compute_local_std(self, img: np.ndarray) -> np.ndarray:
        """Compute local standard deviation map."""
        img_f32 = img.astype(np.float32)
        ksize = (self.local_window_size, self.local_window_size)
        mean = cv2.boxFilter(img_f32, ddepth=-1, ksize=ksize, normalize=True)
        mean_sq = cv2.boxFilter(img_f32 * img_f32, ddepth=-1, ksize=ksize, normalize=True)
        var = np.clip(mean_sq - mean * mean, 0, None)
        std = np.sqrt(var)
        return std

    def compute_hlr(self, v_before: np.ndarray, v_after: np.ndarray, box_mask: np.ndarray) -> float:
        """Compute Highlight Reduction Ratio (HLR)."""
        box_pixels = box_mask > 0
        highlights_before = np.sum((v_before > self.highlight_threshold) & box_pixels)
        if highlights_before == 0:
            return 0.0
        highlights_after = np.sum((v_after > self.highlight_threshold) & box_pixels)
        hlr = ((highlights_before - highlights_after) / highlights_before) * 100.0
        return float(hlr)

    def compute_cov(self, v_after: np.ndarray, mask: np.ndarray, box_mask: np.ndarray) -> float:
        """Compute Coverage (Cov) - residual highlight ratio."""
        combined = cv2.bitwise_and(mask, box_mask)
        mask_pixels = combined > 0
        total_mask_pixels = np.sum(mask_pixels)
        if total_mask_pixels == 0:
            return 0.0
        residual_highlights = np.sum((v_after > self.highlight_threshold) & mask_pixels)
        cov = (residual_highlights / total_mask_pixels) * 100.0
        return float(cov)

    def compute_brightness_reduction(
        self, v_before: np.ndarray, v_after: np.ndarray, mask: np.ndarray, box_mask: np.ndarray
    ) -> float:
        """Compute average brightness reduction rate in mask region."""
        combined = cv2.bitwise_and(mask, box_mask)
        mask_pixels = combined > 0
        if np.sum(mask_pixels) == 0:
            return 0.0
        mean_before = float(np.mean(v_before[mask_pixels]))
        mean_after = float(np.mean(v_after[mask_pixels]))
        if mean_before < 1e-6:
            return 0.0
        reduction = ((mean_before - mean_after) / mean_before) * 100.0
        return float(reduction)

    def compute_variance_improvement(
        self, gray_before: np.ndarray, gray_after: np.ndarray, box_mask: np.ndarray
    ) -> float:
        """Compute local variance improvement rate."""
        box_pixels = box_mask > 0
        std_before = self._compute_local_std(gray_before)
        std_after = self._compute_local_std(gray_after)
        global_std_before = float(np.std(gray_before[box_pixels]))
        global_std_after = float(np.std(gray_after[box_pixels]))
        if global_std_before < 1e-6:
            return 0.0
        local_deviation_before = float(np.mean(np.abs(std_before[box_pixels] - global_std_before)))
        local_deviation_after = float(np.mean(np.abs(std_after[box_pixels] - global_std_after)))
        if local_deviation_before < 1e-6:
            return 0.0
        improvement = ((local_deviation_before - local_deviation_after) / local_deviation_before) * 100.0
        return float(improvement)

    def evaluate(
        self,
        img_before: np.ndarray,
        img_after: np.ndarray,
        mask: np.ndarray,
        box_mask: np.ndarray,
    ) -> Dict[str, float]:
        """Compute all evaluation metrics for a single image pair."""
        v_before = self._extract_v_channel(img_before)
        v_after = self._extract_v_channel(img_after)
        gray_before = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
        gray_after = cv2.cvtColor(img_after, cv2.COLOR_BGR2GRAY)
        
        hlr = self.compute_hlr(v_before, v_after, box_mask)
        cov = self.compute_cov(v_after, mask, box_mask)
        brightness_reduction = self.compute_brightness_reduction(v_before, v_after, mask, box_mask)
        variance_improvement = self.compute_variance_improvement(gray_before, gray_after, box_mask)
        
        return {
            'HLR': hlr,
            'Cov': cov,
            'Brightness_Reduction': brightness_reduction,
            'Variance_Improvement': variance_improvement,
        }