"""
reflection_removal_model
=========================

This module defines the :class:`ReflectionRemovalModel` which
encapsulates the logic for removing specular reflections (e.g. water
droplets) from crate images given a precomputed binary mask.  The
algorithm follows the assignment specification: a mask is generated
first, then OpenCV's inpainting methods are used to fill in the
highlighted pixels.  Connected components are classified by size to
select between Telea's fast marching and the Navier–Stokes method,
and a joint bilateral filter is applied to blend the inpainted
regions with the surrounding texture.

The model exposes a single public method :meth:`remove_reflections`
which takes an image and its corresponding mask and returns the
corrected image.  All helper routines are encapsulated as instance
methods so that parameters can be tuned via the constructor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class ReflectionRemovalModel:
    """Remove specular reflections from an image using inpainting.

    Parameters
    ----------
    small_thresh : int, optional
        Area threshold (in pixels) to distinguish small and large
        connected components.  Components with an area strictly
        smaller than this value are inpainted using Telea's method;
        larger ones use Navier–Stokes.  Default is 300.
    dilate_kernel : int, optional
        Size of the structuring element (diameter in pixels) used
        when dilating the mask.  Dilation ensures that bright rims
        around droplets are included.  Default is 5.
    dilate_iter : int, optional
        Number of dilation iterations.  Default is 1.
    telea_radius : float, optional
        Inpainting radius for Telea's method.  Default is 4.0.
    ns_radius : float, optional
        Inpainting radius for the Navier–Stokes method.  Default is
        8.0.
    sigma_space : float, optional
        Spatial sigma for the joint bilateral filter applied to the
        inpainted regions.  Larger values blend over a wider area.
        Default is 2.5.
    sigma_color : float, optional
        Colour sigma for the joint bilateral filter.  Larger values
        allow more colour variation in the smoothing.  Default is
        25.0.
    """

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
        """Dilate a binary mask using an elliptical kernel.

        Parameters
        ----------
        mask : numpy.ndarray
            Single–channel binary mask (values 0 or 255).

        Returns
        -------
        numpy.ndarray
            Dilated mask with the same shape as ``mask``.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.dilate_kernel, self.dilate_kernel))
        dilated = cv2.dilate(mask, kernel, iterations=self.dilate_iter)
        return dilated

    def classify_components(self, mask: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Split a mask into small and large connected components.

        Parameters
        ----------
        mask : numpy.ndarray
            Dilated binary mask.

        Returns
        -------
        tuple
            ``(small_masks, large_masks)`` where each list contains
            binary masks (0 or 255) for the individual connected
            components.  Pixels belonging to other components are zero.
        """
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
        """Inpaint the regions defined by ``comp_masks`` on a copy of ``img``.

        Parameters
        ----------
        img : numpy.ndarray
            BGR image to inpaint.  This array is not modified in
            place; a copy is updated and returned.
        comp_masks : list of numpy.ndarray
            List of binary masks (0/255) corresponding to each
            connected component to be inpainted.
        method : int
            OpenCV inpainting flag (``cv2.INPAINT_TELEA`` or
            ``cv2.INPAINT_NS``).
        radius : float
            Inpainting radius controlling the neighbourhood size.

        Returns
        -------
        numpy.ndarray
            Image with the specified regions inpainted.
        """
        result = img.copy()
        for cmask in comp_masks:
            result = cv2.inpaint(result, cmask, radius, method)
        return result

    def joint_bilateral_on_mask(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply a joint bilateral filter only within the masked region.

        A joint bilateral filter smooths an image while respecting
        edges.  Here we use the inpainted image itself as both source
        and guide.  The filter is only applied to pixels where
        ``mask > 0``; unmasked pixels remain unchanged.

        Parameters
        ----------
        img : numpy.ndarray
            Inpainted BGR image.
        mask : numpy.ndarray
            Binary mask indicating where to apply the filter (0/255).

        Returns
        -------
        numpy.ndarray
            Blended image after bilateral filtering on the mask.
        """
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
        """Remove specular reflections from ``img`` using ``mask``.

        The input ``mask`` should already be a binary image with
        values 0 or 255 marking pixels to be corrected.  The
        algorithm dilates the mask, splits connected components by
        area, inpaints small and large components using Telea and
        Navier–Stokes respectively, then applies a joint bilateral
        filter to blend the result.  A new image is returned.

        Parameters
        ----------
        img : numpy.ndarray
            Original BGR image from which reflections should be
            removed.  This array is not modified in place.
        mask : numpy.ndarray
            Binary mask (0/255) indicating reflection pixels.  The
            mask should correspond to ``img`` in size and
            orientation.

        Returns
        -------
        numpy.ndarray
            Image with reflections removed and blended.
        """
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