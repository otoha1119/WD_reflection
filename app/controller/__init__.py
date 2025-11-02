"""
app.controller package
======================

This package contains controller modules that orchestrate the flow
between models and views.  Controllers handle I/O, call the
appropriate models, and produce the final outputs (e.g., saving
images to disk).  They do not implement algorithmic details, which
are delegated to the models.
"""

__all__ = ["mask_controller", "reflection_controller"]