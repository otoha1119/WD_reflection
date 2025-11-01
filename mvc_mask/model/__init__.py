"""Model package for the masking application.

This package exposes classes that implement the core image processing
algorithms used in this assignment.  The primary class of interest
here is :class:`BoxMaskModel`, which encapsulates the crate masking
algorithm derived from ``GenerateBoxMask.py``.  Additional models are
provided for future extension, including water drop detection and
image pre–processing.  These other models are unused in the current
task but have been ported into a class structure to adhere to the
MVC separation of concerns.

Modules
=======
box_mask_model
    Crate masking algorithm.
waterdrop_mask_model
    Water drop candidate mask generation (unused for now).
pretreatment_model
    Image pre–processing functions (gamma correction and CLAHE).
"""
