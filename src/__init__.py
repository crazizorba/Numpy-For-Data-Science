"""
Package src - All reusable functions and classes for the project.
Import this package to use everything with short names.
"""

# Data Processing
from .data_processing import (
    load_data,
    handle_unknown,
    one_hot_encode,
    standardize,
    min_max_scale,
    train_val_split,
    feature_engineering,
    handle_outliers
)

# Visualization
from .visualization import (
    plot_heatmap,
    plot_bar,
    plot_roc,
    plot_confusion_matrix,
    plot_feature_importance
)

# Models
from .models import LogisticRegression
from .models import (
    accuracy,
    precision,
    recall,
    f1_score,
    roc_auc
)

__all__ = [
    # data_processing
    "load_data", "handle_unknown","handle_outliers", "one_hot_encode", "standardize",
    "min_max_scale", "train_val_split", "feature_engineering",
    # visualization
    "plot_heatmap", "plot_bar", "plot_roc", "plot_confusion_matrix",
    "plot_feature_importance",
    # models
    "LogisticRegression", "accuracy", "precision", "recall", "f1_score", "roc_auc"
]