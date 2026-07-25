"""Predictive metric re-exports."""

from src.tasks.node_classification import evaluate_node_classification, train_node_classifier
from src.tasks.link_prediction import evaluate_link_prediction

__all__ = [
    "evaluate_node_classification",
    "train_node_classifier",
    "evaluate_link_prediction",
]
