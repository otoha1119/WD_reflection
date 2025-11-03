# app/controller/evaluation_controller.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple

import csv
import glob
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

from app.model.evaluation_model import EvaluationModel, EvalParams


def _stem_to_output(result_dir: Path, stem: str) -> Path | None:
    """result/<stem>_reflection_removed.* を探す。"""
    patterns = [
        result_dir / f"{stem}_reflection_removed.png",
        result_dir / f"{stem}_reflection_removed.jpg",
        result_dir / f"{stem}_reflection_removed.jpeg",
    ]
    for p in patterns:
        if p.exists():
            return p
    return None


def _read_bgr(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return img


def process_images(
    images_dir: Path,
    result_dir: Path,
    eval_dir: Path,                 # ← これを使う
    v_percentile: float = 98.0,
    s_threshold: float = 80.0,
    save_plots: bool = True,
) -> Path:
    """評価を実行し、CSV と（必要なら）可視化を eval_dir に保存して CSV のパスを返す。"""
    eval_dir.mkdir(parents=True, exist_ok=True)
    model = EvaluationModel(EvalParams(v_percentile=v_percentile, s_threshold=s_threshold))

    # 入力画像の一覧
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    image_paths: List[Path] = []
    for ext in exts:
        image_paths.extend(Path(images_dir).glob(ext))
    image_paths = sorted(image_paths, key=lambda p: p.stem)

    rows: List[Dict[str, float | str]] = []
    for ip in image_paths:
        outp = _stem_to_output(result_dir, ip.stem)
        if outp is None:
            # 出力が無ければスキップ
            continue

        img_in = _read_bgr(ip)
        img_out = _read_bgr(outp)
        if img_in is None or img_out is None:
            continue

        m = model.compute_metrics(img_in, img_out)
        rows.append({
            "filename": ip.name,
            **m,
        })

    # CSV 保存
    csv_path = eval_dir / "evaluation_results.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

        if save_plots:
            _save_plots(rows, eval_dir)
    else:
        # 空でもCSVを作っておく
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "note"])
            writer.writerow(["(none)", "no pairs found"])

    return csv_path


def _save_plots(rows: List[Dict[str, float | str]], eval_dir: Path) -> None:
    """簡単なヒスト・箱ひげ図を保存（比や%は0-1, 0-100が混在しないよう注意）"""
    import pandas as pd

    df = pd.DataFrame(rows)

    # ハイライト残存率（R_out/R_in）
    plt.figure()
    df["ratio_highlight"].plot(kind="hist", bins=20)
    plt.xlabel("Highlight ratio (R_out / R_in)")
    plt.ylabel("count")
    plt.title("Histogram of highlight ratio")
    plt.tight_layout()
    plt.savefig(eval_dir / "hist_ratio_highlight.png")
    plt.close()

    # CoV 比（out/in）
    plt.figure()
    df["ratio_cov"].plot(kind="hist", bins=20)
    plt.xlabel("CoV ratio (out / in)")
    plt.ylabel("count")
    plt.title("Histogram of CoV ratio")
    plt.tight_layout()
    plt.savefig(eval_dir / "hist_ratio_cov.png")
    plt.close()

    # 箱ひげ図（3指標）
    plt.figure()
    df[["ratio_highlight", "ratio_cov", "reduction_highlight"]].plot(kind="box")
    plt.title("Box plot (ratios)")
    plt.tight_layout()
    plt.savefig(eval_dir / "boxplot_metrics.png")
    plt.close()
