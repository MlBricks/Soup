"""SOUP — adaptable state/observer-memory/fusion architecture for MLBricks.

Canonical usage::

    from mlbricks import soup
    model = soup(...)

The ``mlbricks.soup`` module is callable, allowing the compact constructor
without replacing MLBricks' own top-level ``mlbricks/__init__.py``.
"""

from __future__ import annotations

import inspect
import sys
import types

from .core import SOUP

__all__ = ["SOUP"]
__version__ = "0.1.0a0"


class _CallableSOUPModule(types.ModuleType):
    def __call__(self, *args, **kwargs):
        return SOUP(*args, **kwargs)


_module = sys.modules[__name__]
_module.__class__ = _CallableSOUPModule
_module.__signature__ = inspect.signature(SOUP)
