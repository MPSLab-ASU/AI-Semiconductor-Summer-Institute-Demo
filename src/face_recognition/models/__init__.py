"""
Model utilities for face recognition
"""
from .create_model import (
    create_facenet_like_model,
    create_mobilenet_based_model,
    save_example_model
)

__all__ = [
    'create_facenet_like_model',
    'create_mobilenet_based_model',
    'save_example_model'
]
