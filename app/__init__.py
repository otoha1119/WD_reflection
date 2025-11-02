"""
app package
===========

This package contains the core functionality for the crate reflection
removal project.  It is organised following a simple MVC pattern:

- :mod:`app.model` holds the computational models used by the
  controllers.

Controllers (entry points) reside in the project root as
``main_Mask.py`` and ``main_Reflection_Removal.py``.
"""

from . import model  # re-export model submodule

__all__ = ["model"]