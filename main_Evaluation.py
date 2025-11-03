"""
main_Evaluation.py
==================

This top-level script delegates evaluation to the controller in
``app.controller.evaluation_controller``.  It provides a simple
command-line entry point for computing reflection removal quality
metrics across all processed images.

The script compares original images in ``data/images`` with
processed images in ``result``, using masks from ``mask`` and
limiting evaluation to the crate (box) region.  Results are printed
to the terminal and distribution plots are saved to the
``Evaluation`` directory.

Usage
-----

    python3 main_Evaluation.py

Prerequisites
-------------

Before running evaluation, ensure the following steps have been
completed:

1. Masks have been generated (via ``main_Mask.py``)
2. Reflections have been removed (via ``main_Reflection_Removal.py``)

Outputs
-------

The script produces:

- **Terminal output**: Mean values and statistics for all metrics
  (HLR, Cov, Brightness Reduction, Variance Improvement)
- **Distribution plots**: Saved to ``Evaluation/`` directory showing
  histograms for each metric across all images

Metrics Computed
----------------

1. **HLR (Highlight Reduction Ratio)**: Percentage reduction in
   high-intensity pixels within the crate region.

2. **Cov (Coverage)**: Percentage of mask pixels where highlights
   remain after processing (lower is better).

3. **Brightness Reduction**: Average reduction in V-channel
   brightness within the reflection mask.

4. **Variance Improvement**: Improvement in texture uniformity
   measured by local standard deviation changes.

All metrics are computed only within the crate region to avoid bias
from unchanged background areas.
"""

from __future__ import annotations

from app.controller.evaluation_controller import main as controller_main


def main() -> None:
    """Run evaluation on all processed images."""
    print("Starting reflection removal evaluation...")
    print("This will compare original images with processed results.")
    print()
    
    controller_main()
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()