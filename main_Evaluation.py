#!/usr/bin/env python3
"""
main_Evaluation
================

Entry point for the evaluation stage of the reflection removal project.
This script loads paired images (original and reflection–removed) from
the specified directories, computes per–image metrics using the
evaluation controller and outputs a CSV file and optional plots
summarising the results.

Usage (default directories)::

    python3 main_Evaluation.py

Custom directories and options::

    python3 main_Evaluation.py \
        --images data/images \
        --result result \
        --mask mask \
        --eval Evaluation \
        --v-percentile 98.0 \
        --s-threshold 80 \
        --no-plots

This script is deliberately kept thin: the heavy lifting is performed
by :mod:`app.controller.evaluation_controller` and
 :mod:`app.model.evaluation_model`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.controller.evaluation_controller import process_images


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate reflection removal results.")
    parser.add_argument(
        "--images", default="data/images", help="Directory with original input images."
    )
    parser.add_argument(
        "--mask", default="mask", help="Directory with final masks (unused in evaluation)."
    )
    parser.add_argument(
        "--result", default="result", help="Directory with reflection removed images."
    )
    parser.add_argument(
        "--eval", default="Evaluation", help="Output directory for evaluation results."
    )
    parser.add_argument(
        "--v-percentile", type=float, default=98.0,
        help="Percentile for highlight detection threshold (default 98.0)."
    )
    parser.add_argument(
        "--s-threshold", type=int, default=255,
        help="Maximum saturation for highlight detection (default 255).  A value of 255 disables the saturation filter."
    )
    parser.add_argument(
        "--v-k", type=float, default=1.0,
        help=(
            "Number of standard deviations added to the mean when computing "
            "the adaptive brightness threshold.  The final threshold is "
            "max(percentile, mean + v_k*std)."
        ),
    )
    parser.add_argument(
        "--no-plots", action="store_true", help="Disable saving of histogram and boxplot images."
    )
    args = parser.parse_args()
    images_dir = Path(args.images)
    mask_dir = Path(args.mask)
    result_dir = Path(args.result)
    eval_dir = Path(args.eval)
    # Run evaluation
    process_images(
        images_dir=images_dir,
        mask_dir=mask_dir,
        result_dir=result_dir,
        evaluation_dir=eval_dir,
        save_plots=not args.no_plots,
        v_percentile=args.v_percentile,
        s_threshold=args.s_threshold,
        v_k=args.v_k,
    )


if __name__ == "__main__":
    main()