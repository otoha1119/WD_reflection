"""
main_Reflection_Removal.py
==========================

This top–level script delegates reflection removal to
``app.controller.reflection_controller``.  It preserves the entry
point specified in the assignment while the actual logic resides
inside the controller.

Usage
-----

    python3 main_Reflection_Removal.py

Masks must have been generated beforehand (e.g. via
``main_Mask.py``) and reside in the ``mask`` directory.  Cleaned
images are written to ``result``.
"""

from __future__ import annotations

from app.controller.reflection_controller import main as controller_main


def main() -> None:
    controller_main()


if __name__ == "__main__":
    main()