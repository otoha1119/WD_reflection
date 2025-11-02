"""
main_Mask.py
=============

This top–level script delegates mask generation to the controller in
``app.controller.mask_controller``.  It exists to provide a simple
command–line entry point at the project root, preserving the
interface specified in the assignment.  All processing logic is
implemented in the controller.

Usage
-----

    python3 main_Mask.py [--use-mvc-mask]

The optional ``--use-mvc-mask`` flag causes the controller to use
the legacy models from the ``mvc_mask`` package instead of those in
``app.model``.
"""

from __future__ import annotations

import argparse

from app.controller.mask_controller import main as controller_main

# ----------------------------------------------------------------------
# Configuration flags
#
# By default the project uses the local ``app.model`` implementations
# for crate and water–drop detection.  These modules faithfully
# replicate the behaviour of ``GenerateBoxMask.py`` and
# ``GenerateWaterdropMask.py`` from the provided repository.  If you
# wish to fall back to the legacy ``mvc_mask`` models (which employ
# slightly different detection thresholds), set ``USE_MVC_MASK`` to
# ``True`` or provide the ``--use-mvc-mask`` flag on the command
# line.
#
# The ``SAVE_INTERMEDIATE_MASKS`` flag controls whether the
# controller saves intermediate masks for debugging.  When set to
# ``True`` the crate–only and water–drop–only masks are stored in
# ``mask/debug`` in addition to the final combined mask.  The full
# detection pipeline is always executed regardless of this flag;
# the flag only affects which images are written to disk.
USE_MVC_MASK: bool = False
SAVE_INTERMEDIATE_MASKS: bool = False


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reflection masks for all images.")
    parser.add_argument(
        "--use-mvc-mask",
        action="store_true",
        help="Override USE_MVC_MASK flag to use legacy mvc_mask models",
    )
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Override SAVE_INTERMEDIATE_MASKS flag to save crate and water–drop masks",
    )
    args = parser.parse_args()
    # Determine effective flags: CLI overrides module–level defaults
    use_mvc = USE_MVC_MASK
    if args.use_mvc_mask:
        use_mvc = True
    save_int = SAVE_INTERMEDIATE_MASKS
    if args.save_intermediate:
        save_int = True
    controller_main(use_mvc_mask=use_mvc, save_intermediate=save_int)


if __name__ == "__main__":
    main()