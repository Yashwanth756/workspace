"""Turns a cleaned table into model inputs, and builds the preparation saved
with a model and applied alike in fitting and scoring."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from data_pipeline.common_cleaning import _is_gap


def build_feature_frame(dataframe, numeric_columns, categorical_columns):
    """Numeric columns then categorical columns, in the order given; rows and index kept."""
    features = pd.DataFrame(index=dataframe.index)
    for column_name in numeric_columns:
        features[column_name] = np.array(
            [np.nan if _is_gap(value) else float(value) for value in dataframe[column_name]],
            dtype="float64")
    for column_name in categorical_columns:
        features[column_name] = pd.Series(
            [np.nan if _is_gap(value) else str(value) for value in dataframe[column_name]],
            index=dataframe.index, dtype=object)
    return features[list(numeric_columns) + list(categorical_columns)]


def build_model_pipeline(estimator, numeric_columns, categorical_columns):
    """The input preparation followed by the estimator, unfitted.

    Numbers pass through as they are (gaps are left for the estimator, which
    handles them); categories have their gaps filled and are one-hot encoded,
    a value never seen in fitting being ignored.
    """
    categories = Pipeline([
        ("fill", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preparation = ColumnTransformer(
        [("numbers", "passthrough", list(numeric_columns)),
         ("categories", categories, list(categorical_columns))],
        verbose_feature_names_out=False).set_output(transform="pandas")
    return Pipeline([("prepare", preparation), ("model", estimator)])