from __future__ import annotations

import argparse

from app.controller.mask_controller import main as controller_main

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