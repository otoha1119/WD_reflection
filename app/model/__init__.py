"""
app.model package
=================

This package groups the data models used by the crate reflection
removal project.  Models encapsulate algorithms for pre–processing
images, detecting water droplets (mask generation), and removing
reflections via inpainting.  Each model is designed to be
stateless aside from tunable parameters exposed via dataclasses.

Available models:

- :class:`app.model.pretreatment_model.PreprocessingModel`
    Apply gamma correction and CLAHE to an image.

- :class:`app.model.waterdrop_mask_model.WaterdropMaskModel`
    Detect water droplets and produce a binary mask.

- :class:`app.model.reflection_removal_model.ReflectionRemovalModel`
    Remove reflections from an image given a binary mask.

- :class:`app.model.box_mask_model.BoxMaskModel`
    Detect the crate region and produce a binary mask.

- :class:`app.model.evaluation_model.EvaluationModel`
    Evaluate reflection removal quality using multiple metrics.
"""

from .pretreatment_model import PreprocessingModel
from .waterdrop_mask_model import WaterdropMaskModel
from .reflection_removal_model import ReflectionRemovalModel
from .box_mask_model import BoxMaskModel
from .evaluation_model import EvaluationModel

__all__ = [
    "PreprocessingModel",
    "WaterdropMaskModel",
    "ReflectionRemovalModel",
    "BoxMaskModel",
    "EvaluationModel",
]