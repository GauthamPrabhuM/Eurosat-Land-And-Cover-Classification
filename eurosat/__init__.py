"""EuroSAT package entrypoints"""

from .train import EuroSATClassifier, EuroSATDataset, get_transforms
from .evaluate import ModelEvaluator, compare_models

__all__ = [
    'EuroSATClassifier', 'EuroSATDataset', 'get_transforms',
    'ModelEvaluator', 'compare_models'
]
