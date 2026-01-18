"""API module for prediction service."""

from .server import app, get_predictor

__all__ = ["app", "get_predictor"]
