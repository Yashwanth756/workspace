"""The only data preparation module that reads or writes CSV files.

Raw files are read without inferring any column's type; cleaned files are
read with each column's type inferred.
"""
import os

import pandas as pd

from config import config


def load_raw_dataset(file_path):
    """Read one raw file as text, guessing nothing.

    Every column holds text with no leading or trailing whitespace, and a value
    that matches one of config.NULL_TOKENS, ignoring case, is NaN; a value that
    merely contains such a token is kept.
    """
    dataframe = pd.read_csv(file_path, dtype=str, keep_default_na=False)
    null_tokens = {token.strip().lower() for token in config.NULL_TOKENS}
    for column_name in dataframe.columns:
        stripped = dataframe[column_name].astype(str).str.strip()
        is_null = stripped.str.lower().isin(null_tokens)
        dataframe[column_name] = stripped.astype(object).where(~is_null)
    return dataframe


def load_processed_dataset(file_path):
    """Read a cleaned table: types follow the contents, only an empty cell is missing."""
    return pd.read_csv(file_path, keep_default_na=False, na_values=[""], low_memory=False)


def save_processed_dataset(dataframe, file_path):
    """Save a cleaned table without an index column, creating its folder if needed."""
    folder = os.path.dirname(file_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    dataframe.to_csv(file_path, index=False)