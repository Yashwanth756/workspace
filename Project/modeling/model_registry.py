"""Saves the models, their feature schemas and their scores, and loads the
models and schemas back. Nothing else in the project writes to output/models/."""
import json
import os

import joblib

from config import config


def _make_folder_for(file_path):
    folder = os.path.dirname(file_path)
    if folder:
        os.makedirs(folder, exist_ok=True)


def save_model(model, file_path):
    """Save a fitted model, preparation included."""
    _make_folder_for(file_path)
    joblib.dump(model, file_path)


def save_feature_schema(feature_names, file_path):
    """Save the columns a model was fitted on, in order, as a JSON list."""
    _make_folder_for(file_path)
    with open(file_path, "w", encoding="utf-8") as schema_file:
        json.dump([str(name) for name in feature_names], schema_file, indent=2)


def save_metrics(metrics):
    """Save every model's record to config.MODEL_METRICS_FILE, indented, keys sorted."""
    _make_folder_for(config.MODEL_METRICS_FILE)
    with open(config.MODEL_METRICS_FILE, "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2, sort_keys=True)


def load_model(file_path):
    """Bring a saved model back into memory, exactly as saved."""
    return joblib.load(file_path)


def load_feature_schema(file_path):
    """Bring a model's input columns back, in the order they were saved."""
    with open(file_path, encoding="utf-8") as schema_file:
        return list(json.load(schema_file))