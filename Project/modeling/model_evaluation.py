"""Scores a fitted model on held-out rows and on its training rows.

No function here fits anything, and every score is a float rounded to
config.METRIC_DECIMALS places.
"""
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, precision_score,
                             r2_score, recall_score, roc_auc_score)

from config import config


def _rounded(value):
    return round(float(value), config.METRIC_DECIMALS)


def evaluate_classifier(model, train_features, train_target, held_out_features,
                        held_out_target):
    """Score the inflated-order model: the nine scores."""
    train_probability = model.predict_proba(train_features)[:, 1]
    held_out_probability = model.predict_proba(held_out_features)[:, 1]
    held_out_predicted = model.predict(held_out_features)
    train_roc_auc = roc_auc_score(train_target, train_probability)
    held_out_roc_auc = roc_auc_score(held_out_target, held_out_probability)
    return {
        "held_out_roc_auc": _rounded(held_out_roc_auc),
        "train_roc_auc": _rounded(train_roc_auc),
        "generalisation_gap": _rounded(train_roc_auc - held_out_roc_auc),
        "accuracy": _rounded(accuracy_score(held_out_target, held_out_predicted)),
        "precision": _rounded(precision_score(held_out_target, held_out_predicted,
                                              zero_division=0)),
        "recall": _rounded(recall_score(held_out_target, held_out_predicted, zero_division=0)),
        "f1": _rounded(f1_score(held_out_target, held_out_predicted, zero_division=0)),
        "mean_predicted_probability": _rounded(np.mean(held_out_probability)),
        "positive_share": _rounded(np.mean(np.asarray(held_out_target, dtype=float))),
    }


def evaluate_regressor(model, train_features, train_target, held_out_features,
                       held_out_target):
    """Score the demand model and the always-the-average baseline: the seven scores."""
    train_truth = np.asarray(train_target, dtype=float)
    held_out_truth = np.asarray(held_out_target, dtype=float)
    train_forecast = np.asarray(model.predict(train_features), dtype=float)
    held_out_forecast = np.asarray(model.predict(held_out_features), dtype=float)
    train_r2 = r2_score(train_truth, train_forecast)
    held_out_r2 = r2_score(held_out_truth, held_out_forecast)
    return {
        "held_out_r2": _rounded(held_out_r2),
        "train_r2": _rounded(train_r2),
        "generalisation_gap": _rounded(train_r2 - held_out_r2),
        "mean_absolute_error": _rounded(mean_absolute_error(held_out_truth, held_out_forecast)),
        "average_estimate_mean_absolute_error": _rounded(
            np.mean(np.abs(held_out_truth - np.mean(train_truth)))),
        "lowest_prediction": _rounded(np.min(held_out_forecast)),
        "highest_prediction": _rounded(np.max(held_out_forecast)),
    }


def evaluate_text_classifier(model, train_text, train_target, held_out_text, held_out_target):
    """Score the request model across every request type: the six entries."""
    train_predicted = model.predict(train_text)
    held_out_predicted = model.predict(held_out_text)
    train_macro_f1 = f1_score(train_target, train_predicted, average="macro")
    held_out_macro_f1 = f1_score(held_out_target, held_out_predicted, average="macro")
    classes = [str(label) for label in model.classes_]
    per_class = f1_score(held_out_target, held_out_predicted, labels=list(model.classes_),
                         average=None, zero_division=0)
    return {
        "held_out_macro_f1": _rounded(held_out_macro_f1),
        "train_macro_f1": _rounded(train_macro_f1),
        "generalisation_gap": _rounded(train_macro_f1 - held_out_macro_f1),
        "accuracy": _rounded(accuracy_score(held_out_target, held_out_predicted)),
        "weighted_f1": _rounded(f1_score(held_out_target, held_out_predicted,
                                         average="weighted")),
        "per_class_f1": {label: _rounded(score) for label, score in zip(classes, per_class)},
    }