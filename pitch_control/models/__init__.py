"""
Pitch Control Models
===================

High-performance implementations of various pitch control models.
"""

from .spearman import SpearmanModel, SpearmanConfig
from .base import ModelConfig

__all__ = ["SpearmanModel", "SpearmanConfig", "ModelConfig"]
