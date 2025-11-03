# app/model/evaluation_model.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import cv2
import numpy as np

from app.model.box_mask_model import BoxMaskModel


@dataclass
class EvalParams:
    v_percentile: float = 98.0   # 箱内Vのパーセンタイル（高Vの基準）
    s_threshold: float = 80.0    # 低Sの上限（S < しきい）
    eps: float = 1e-6            # ゼロ除算回避


class EvaluationModel:
    """
    入力(BGR)と出力(BGR)のペアから箱領域内で:
      - ハイライト残存率 (R_out/R_in) と除去率 (1 - それ)
      - CoV (σ/μ) の before→after と比・減少率
      - before/after のハイライト割合 (%)
    を算出する。
    """
    def __init__(self, params: EvalParams | None = None):
        self.params = params or EvalParams()
        self.box_model = BoxMaskModel()

    @staticmethod
    def _to_hsv(img_bgr: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    @staticmethod
    def _cov(gray: np.ndarray, mask: np.ndarray) -> float:
        vals = gray[mask > 0].astype(np.float32)
        if vals.size == 0:
            return 0.0
        mu = float(vals.mean())
        if abs(mu) < 1e-6:
            return 0.0
        sigma = float(vals.std(ddof=0))
        return sigma / mu

    def _highlight_binary(
        self,
        img_bgr: np.ndarray,
        box_mask: np.ndarray,
    ) -> Tuple[np.ndarray, float, float]:
        hsv = self._to_hsv(img_bgr)
        h, s, v = cv2.split(hsv)

        v_box = v[box_mask > 0]
        if v_box.size == 0:
            return np.zeros_like(v, np.uint8), 0.0, float(self.params.s_threshold)

        v_thr = float(np.percentile(v_box, self.params.v_percentile))
        s_thr = float(self.params.s_threshold)

        highlight = ((v.astype(np.float32) >= v_thr) &
                     (s.astype(np.float32) < s_thr) &
                     (box_mask > 0)).astype(np.uint8) * 255
        return highlight, v_thr, s_thr

    def compute_metrics(self, img_bgr_in: np.ndarray, img_bgr_out: np.ndarray) -> Dict[str, float]:
        # 箱マスクは入力から生成（Git版の流儀）
        box_mask, _ = self.box_model.generate_mask(img_bgr_in)
        box_area = int((box_mask > 0).sum())

        # 入出力のハイライト2値
        hl_in, v_thr, s_thr = self._highlight_binary(img_bgr_in, box_mask)
        hl_out, _, _ = self._highlight_binary(img_bgr_out, box_mask)

        highlight_in_count = int((hl_in > 0).sum())
        highlight_out_count = int((hl_out > 0).sum())

        highlight_in_pct = (highlight_in_count / max(box_area, 1)) * 100.0
        highlight_out_pct = (highlight_out_count / max(box_area, 1)) * 100.0

        if highlight_in_count == 0:
            ratio_highlight = 0.0
            reduction_highlight = 0.0 if highlight_out_count > 0 else 1.0
        else:
            ratio_highlight = highlight_out_count / highlight_in_count
            reduction_highlight = 1.0 - ratio_highlight

        gray_in = cv2.cvtColor(img_bgr_in, cv2.COLOR_BGR2GRAY)
        gray_out = cv2.cvtColor(img_bgr_out, cv2.COLOR_BGR2GRAY)
        cov_in = self._cov(gray_in, box_mask)
        cov_out = self._cov(gray_out, box_mask)

        ratio_cov = cov_out / max(cov_in, self.params.eps)
        reduction_cov = 1.0 - ratio_cov

        return dict(
            v_thr=v_thr,
            s_thr=s_thr,
            box_area=box_area,
            highlight_in_count=highlight_in_count,
            highlight_out_count=highlight_out_count,
            highlight_in_pct=highlight_in_pct,
            highlight_out_pct=highlight_out_pct,
            ratio_highlight=ratio_highlight,
            reduction_highlight=reduction_highlight,
            cov_in=cov_in,
            cov_out=cov_out,
            ratio_cov=ratio_cov,
            reduction_cov=reduction_cov,
        )
