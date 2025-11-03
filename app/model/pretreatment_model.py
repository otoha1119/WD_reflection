from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


class PreprocessingModel:

    def __init__(self, gamma: float = 0.7, clip_limit: float = 1.0, tile_size: int = 8) -> None:
        self.gamma = gamma
        self.clip_limit = clip_limit
        self.tile_size = tile_size

    # ------------------------------------------------------------------
    # Individual processing steps
    # ------------------------------------------------------------------
    def apply_gamma_correction(self, image: np.ndarray, gamma: float | None = None) -> np.ndarray:
        g = self.gamma if gamma is None else gamma
        norm = image.astype(np.float32) / 255.0
        corrected = np.power(norm, g)
        return np.clip(corrected * 255.0, 0, 255).astype(np.uint8)

    def apply_clahe(self, image: np.ndarray, clip_limit: float | None = None, tile_size: int | None = None) -> np.ndarray:
        cl = self.clip_limit if clip_limit is None else clip_limit
        ts = self.tile_size if tile_size is None else tile_size
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=cl, tileGridSize=(ts, ts))
        l_equalised = clahe.apply(l_channel)
        lab_eq = cv2.merge((l_equalised, a_channel, b_channel))
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    def apply_full_preprocessing(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        gamma_img = self.apply_gamma_correction(image)
        processed = self.apply_clahe(gamma_img)
        return gamma_img, processed