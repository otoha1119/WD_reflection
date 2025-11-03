"""
evaluation_model
================

This module defines the :class:`EvaluationModel` class for assessing the
effectiveness of the reflection removal pipeline.  Since the evaluation
is unsupervised (there are no ground–truth specular–free images), we
derive two quantitative metrics from each input/output pair:

1. **Highlight reduction ratio**:  The proportion of bright
   specular pixels that remain after processing compared to the
   proportion in the original image.  Specular pixels are
   identified dynamically for each image by thresholding the V
   (brightness) channel of the HSV colour space at a high
   percentile (e.g., the 98th percentile within the crate region)
   and requiring low saturation.  The ratio ``R_out / R_in``
   quantifies how much of the original highlight area is still
   present; ``1 - (R_out / R_in)`` gives the fraction of specular
   highlights removed.

2. **Coefficient of variation (CoV) ratio**:  The CoV is defined
   as the standard deviation divided by the mean of the greyscale
   intensities within the crate region.  Bright specular
   reflections produce large variations in intensity.  After
   reflection removal, the CoV should decrease as the pixel
   distribution becomes more uniform.  The ratio ``CoV_out / CoV_in``
   measures this change.

An instance of :class:`EvaluationModel` takes optional parameters
controlling the highlight detection (percentile for the V threshold
and maximum allowable saturation) and supports evaluation on a
single pair of images or on batches via the controller.  The
``evaluate`` method accepts paths to the input (original) and
output (reflection removed) images along with a crate mask and
returns a dictionary of metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

from .box_mask_model import BoxMaskModel


@dataclass
class EvaluationModel:
    """Evaluate reflection removal quality on paired images.

    This implementation follows common literature on reflection removal
    where the percentage of highlights remaining (PHR) and the
    coefficient of variation (CoV) are computed on the entire crate
    region.  Specular highlights are identified solely by brightness
    thresholds computed from the crate region, without relying on
    predetermined waterdrop masks.  A dynamic threshold is used to
    accommodate dark images: the final V threshold is the maximum of
    a high percentile and a mean+``v_k``·std term.

    Parameters
    ----------
    v_percentile : float, optional
        Percentile used to determine the brightness threshold for
        candidate highlight pixels.  Values above this percentile in
        the V channel (0–255) within the crate region are considered
        highlights.  Defaults to 98.0.
    s_threshold : int, optional
        Maximum saturation value for a pixel to be considered a
        highlight.  A value of 255 (default) disables the saturation
        filter so that only brightness is used.  If set below 255,
        highlights must satisfy ``S <= s_threshold``.
    v_k : float, optional
        Number of standard deviations above the mean used for an
        adaptive brightness threshold.  The final V threshold is
        ``max(percentile_threshold, mean + v_k * std)``.  Larger
        values make the threshold stricter.  Defaults to 1.0.
    min_pixels : int, optional
        Minimum number of pixels required in the crate mask to
        compute meaningful statistics.  Images with fewer crate
        pixels will yield ``NaN`` for all metrics.
    """

    v_percentile: float = 98.0
    s_threshold: int = 255
    v_k: float = 1.0
    min_pixels: int = 100

    def __post_init__(self) -> None:
        # Instantiate a BoxMaskModel for crate detection.  The
        # defaults mirror those used in the assignment scripts.
        self.box_model = BoxMaskModel()

    def _compute_highlight_mask(self, hsv: np.ndarray, crate_mask: np.ndarray) -> np.ndarray:
        """Compute a binary mask of highlight pixels within the crate.

        Highlights are defined as pixels whose V channel exceeds an
        adaptive brightness threshold determined from the distribution of
        V values within the crate.  The threshold is the maximum of
        ``v_percentile`` of V values and ``mean + v_k * std``.
        If ``s_threshold`` is less than 255, a low-saturation constraint
        is also applied.

        Parameters
        ----------
        hsv : numpy.ndarray
            The input image in HSV colour space.
        crate_mask : numpy.ndarray
            A binary mask (0/255) indicating the crate region.

        Returns
        -------
        numpy.ndarray
            A binary mask (dtype=bool) of highlights within the crate.
        """
        # Extract V channel values within the crate region
        v_vals = hsv[:, :, 2][crate_mask > 0].astype(np.float32)
        if v_vals.size == 0:
            return np.zeros_like(crate_mask, dtype=bool)
        # Compute percentile and mean/std thresholds
        perc_thr = np.percentile(v_vals, self.v_percentile)
        mean_val = float(v_vals.mean())
        std_val = float(v_vals.std())
        dyn_thr = mean_val + self.v_k * std_val
        v_thr = max(perc_thr, dyn_thr)
        s_channel = hsv[:, :, 1]
        # Identify highlight pixels: high V and optionally low saturation
        highlight = (
            (hsv[:, :, 2] >= v_thr) & (s_channel <= self.s_threshold) & (crate_mask > 0)
        )
        return highlight

    def _compute_cov(self, img: np.ndarray, crate_mask: np.ndarray) -> float:
        """Compute the coefficient of variation within the crate region.

        The CoV is the standard deviation divided by the mean of the
        greyscale pixel intensities.  A small epsilon is added to the
        denominator to avoid division by zero.

        Parameters
        ----------
        img : numpy.ndarray
            Input BGR image.
        crate_mask : numpy.ndarray
            Binary mask of the crate region.

        Returns
        -------
        float
            The coefficient of variation.  If the crate region is
            empty, returns ``float('nan')``.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        pixels = gray[crate_mask > 0]
        if pixels.size == 0:
            return float("nan")
        mean_val = float(pixels.mean())
        std_val = float(pixels.std())
        cov = std_val / (mean_val + 1e-5)
        return cov

    def evaluate(
        self, input_img: np.ndarray, output_img: np.ndarray, crate_mask: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate highlight reduction and CoV change.

        Parameters
        ----------
        input_img : numpy.ndarray
            The original image before reflection removal (BGR).
        output_img : numpy.ndarray
            The image after reflection removal (BGR).
        crate_mask : numpy.ndarray
            Binary mask of the crate region (values 0 or 255).

        Returns
        -------
        dict[str, float]
            A dictionary with the following keys:

            - ``ratio_highlight``:  R_out / R_in, where R_in and
              R_out are the number of highlight pixels before and
              after removal.  ``float('nan')`` if no crate pixels or
              if no highlights are detected.
            - ``reduction_highlight``: 1 - ratio_highlight.  ``float('nan')`` if ratio_highlight is NaN.
            - ``cov_in``: CoV of the input image within the crate.
            - ``cov_out``: CoV of the output image within the crate.
            - ``ratio_cov``: cov_out / cov_in.  ``float('nan')`` if cov_in is zero or NaN.
            - ``reduction_cov``: 1 - ratio_cov.  ``float('nan')`` if ratio_cov is NaN.
        """
        # Ensure crate mask is binary 0/255
        if crate_mask.dtype != np.uint8:
            crate_mask = crate_mask.astype(np.uint8)
        # Check if there are enough pixels to evaluate
        if np.count_nonzero(crate_mask) < self.min_pixels:
            return {
                "ratio_highlight": float("nan"),
                "reduction_highlight": float("nan"),
                "cov_in": float("nan"),
                "cov_out": float("nan"),
                "ratio_cov": float("nan"),
                "reduction_cov": float("nan"),
            }
        # Convert both images to HSV
        hsv_in = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)
        hsv_out = cv2.cvtColor(output_img, cv2.COLOR_BGR2HSV)
        # Compute dynamic V threshold based on input crate region
        v_vals = hsv_in[:, :, 2][crate_mask > 0].astype(np.float32)
        perc_thr = np.percentile(v_vals, self.v_percentile)
        mean_val = float(v_vals.mean())
        std_val = float(v_vals.std())
        dyn_thr = mean_val + self.v_k * std_val
        v_thr = max(perc_thr, dyn_thr)
        s_thr = self.s_threshold
        # Compute highlight masks for input and output using the same threshold
        highlight_in = (
            (hsv_in[:, :, 2] >= v_thr) & (hsv_in[:, :, 1] <= s_thr) & (crate_mask > 0)
        )
        highlight_out = (
            (hsv_out[:, :, 2] >= v_thr) & (hsv_out[:, :, 1] <= s_thr) & (crate_mask > 0)
        )
        # Compute counts
        r_in = float(highlight_in.sum())
        r_out = float(highlight_out.sum())
        if r_in > 0:
            ratio_highlight = r_out / r_in
            reduction_highlight = 1.0 - ratio_highlight
        else:
            ratio_highlight = float("nan")
            reduction_highlight = float("nan")
        # Compute CoV
        cov_in = self._compute_cov(input_img, crate_mask)
        cov_out = self._compute_cov(output_img, crate_mask)
        if np.isfinite(cov_in) and cov_in > 0:
            ratio_cov = cov_out / cov_in
            reduction_cov = 1.0 - ratio_cov
        else:
            ratio_cov = float("nan")
            reduction_cov = float("nan")
        return {
            "ratio_highlight": ratio_highlight,
            "reduction_highlight": reduction_highlight,
            "cov_in": cov_in,
            "cov_out": cov_out,
            "ratio_cov": ratio_cov,
            "reduction_cov": reduction_cov,
        }

    def generate_crate_mask(self, img: np.ndarray) -> np.ndarray:
        """Generate a crate mask for the input image.

        This convenience method simply forwards to the underlying
        :class:`BoxMaskModel`.  It returns a binary mask (0/255).

        Parameters
        ----------
        img : numpy.ndarray
            Input BGR image.

        Returns
        -------
        numpy.ndarray
            Binary mask of the crate region.  If no region is found,
            returns an array of zeros.
        """
        mask, _ = self.box_model.generate_mask(img)
        return mask