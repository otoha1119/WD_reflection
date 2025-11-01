"""
pretreatment_model.py
=====================

This module defines the :class:`PreprocessingModel` class which
encapsulates the image pre–processing operations originally
implemented in ``Pretreatment.py``.  The purpose of this model
is to apply tonal adjustments to an input image prior to further
analysis.  In particular it provides gamma correction and CLAHE
(Contrast Limited Adaptive Histogram Equalisation) functionality as
well as a convenience routine to perform both steps in sequence.

Unlike the script–oriented predecessor, the model here does not
perform any I/O or user interaction; it simply exposes pure
functions that operate on ``numpy.ndarray`` images.  Any file
reading/writing or visualisation concerns are deferred to the view
layer of the MVC architecture.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


class PreprocessingModel:
    """Model for applying simple image pre–processing operations.

    The two primary transformations provided are gamma correction
    (which brightens or darkens an image depending on the gamma value)
    and CLAHE (which enhances local contrast).  The default
    parameters are chosen to slightly brighten dark regions and
    enhance contrast without over–amplifying noise.

    Methods in this class accept and return images as BGR numpy
    arrays.  They do not modify the input in place.
    """

    def __init__(self, gamma: float = 0.7, clip_limit: float = 1.0, tile_size: int = 8) -> None:
        """Initialise the pre–processor with default parameters.

        Parameters
        ----------
        gamma : float
            Gamma correction exponent.  Values less than 1 will
            brighten the image; values greater than 1 will darken it.
            Defaults to ``0.7`` which gently brightens dark regions.
        clip_limit : float
            Limit for contrast clipping in CLAHE.  Higher values
            produce more contrast.  Defaults to ``1.0``.
        tile_size : int
            Size of the tiles used by CLAHE.  Images are divided
            into ``tile_size × tile_size`` regions for local
            equalisation.  Defaults to ``8``.
        """
        self.gamma = gamma
        self.clip_limit = clip_limit
        self.tile_size = tile_size

    # ------------------------------------------------------------------
    # Individual processing steps
    # ------------------------------------------------------------------
    def apply_gamma_correction(self, image: np.ndarray, gamma: float | None = None) -> np.ndarray:
        """Apply gamma correction to an image.

        Parameters
        ----------
        image : numpy.ndarray
            Input BGR image (uint8).  Values are expected in [0, 255].
        gamma : float, optional
            Gamma exponent to use.  If ``None``, the instance's
            configured gamma is used.

        Returns
        -------
        numpy.ndarray
            Gamma corrected BGR image.
        """
        g = self.gamma if gamma is None else gamma
        # Normalise to [0,1], apply exponent, scale back to [0,255]
        norm = image.astype(np.float32) / 255.0
        corrected = np.power(norm, g)
        return np.clip(corrected * 255.0, 0, 255).astype(np.uint8)

    def apply_clahe(self, image: np.ndarray, clip_limit: float | None = None, tile_size: int | None = None) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation).

        CLAHE operates on the L channel of the LAB colour space to
        enhance local contrast without introducing excessive noise.

        Parameters
        ----------
        image : numpy.ndarray
            Input BGR image (uint8).
        clip_limit : float, optional
            Contrast clip limit.  If ``None``, the instance's
            configured clip limit is used.
        tile_size : int, optional
            Size of the grid for CLAHE.  If ``None``, the instance's
            configured tile size is used.

        Returns
        -------
        numpy.ndarray
            CLAHE–enhanced BGR image.
        """
        cl = self.clip_limit if clip_limit is None else clip_limit
        ts = self.tile_size if tile_size is None else tile_size
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=cl, tileGridSize=(ts, ts))
        l_equalised = clahe.apply(l_channel)
        lab_eq = cv2.merge((l_equalised, a_channel, b_channel))
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    def apply_full_preprocessing(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply gamma correction followed by CLAHE.

        This convenience method performs the standard two–step
        pre–processing pipeline used in the water–drop detection
        algorithm.  It returns both the intermediate gamma corrected
        image and the final fully processed result.

        Parameters
        ----------
        image : numpy.ndarray
            Input BGR image.

        Returns
        -------
        tuple
            ``(gamma_corrected, fully_processed)`` where both
            elements are BGR images.
        """
        gamma_img = self.apply_gamma_correction(image)
        processed = self.apply_clahe(gamma_img)
        return gamma_img, processed