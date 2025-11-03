#!/usr/bin/env python3
# main_Evaluation.py
from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd

from app.controller.evaluation_controller import process_images


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate reflection removal without ground-truth.")
    p.add_argument("--images", type=str, default="data/images", help="input images directory")
    p.add_argument("--result", type=str, default="result", help="reflection-removed images directory")
    p.add_argument("--mask", type=str, default="mask", help="(unused here; kept for consistency)")
    p.add_argument("--eval", type=str, default="Evaluation", help="output evaluation directory")
    p.add_argument("--v-percentile", type=float, default=98.0, help="V percentile for highlight threshold (in-box)")
    p.add_argument("--s-threshold", type=float, default=80.0, help="S threshold (S < thr) for highlight")
    p.add_argument("--no-plots", action="store_true", help="disable plot image generation")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    images_dir = Path(args.images)
    result_dir = Path(args.result)
    eval_dir = Path(args.eval)

    csv_path = process_images(
        images_dir=images_dir,
        result_dir=result_dir,
        eval_dir=eval_dir,
        v_percentile=args.v_percentile,
        s_threshold=args.s_threshold,
        save_plots=(not args.no_plots),
    )

    # ---- コンソールに「簡単な前後の平均」を出す ----
    try:
        df = pd.read_csv(csv_path)
        if not df.empty and "filename" in df.columns:
            avg_hi_in = df["highlight_in_pct"].mean()
            avg_hi_out = df["highlight_out_pct"].mean()
            avg_cov_in = df["cov_in"].mean()
            avg_cov_out = df["cov_out"].mean()

            avg_red_hl = (df["reduction_highlight"].mean() * 100.0) if "reduction_highlight" in df.columns else None
            avg_red_cov = (df["reduction_cov"].mean() * 100.0) if "reduction_cov" in df.columns else None

            print("\n=== Simple Summary (Box 内) ===")
            print(f"Highlight: {avg_hi_in:.2f}%  →  {avg_hi_out:.2f}%"
                  + (f"  (reduction mean: {avg_red_hl:.2f}%)" if avg_red_hl is not None else ""))
            print(f"CoV      : {avg_cov_in:.4f}  →  {avg_cov_out:.4f}"
                  + (f"  (reduction mean: {avg_red_cov:.2f}%)" if avg_red_cov is not None else ""))
            print(f"\nCSV: {csv_path}\nPlots: {eval_dir} (disabled by --no-plots)")  # 目印
        else:
            print(f"\nNo pairs found. CSV: {csv_path}")
    except Exception as e:
        print(f"[WARN] could not summarise CSV: {e}")


if __name__ == "__main__":
    main()
