"""The cleaning run for the customer emails, and the repairs the live emails share with it."""
import html
import re

import numpy as np
import pandas as pd

from config import config
from data_pipeline.common_cleaning import (_is_gap, _refuse, convert_whole_numbers,
                                           normalize_category_values, remove_duplicate_records,
                                           remove_rows_missing_target)
from data_pipeline.dataset_loader import load_raw_dataset, save_processed_dataset


def convert_received_times(dataframe):
    """Write every time as the full weekday, a space and the time: 'Friday 10:17'."""
    result = dataframe.copy()
    column_name = config.EMAIL_RECEIVED_AT_COLUMN
    times, unread = [], []
    for value in result[column_name]:
        if _is_gap(value):
            times.append(np.nan)
            continue
        match = re.fullmatch(config.RECEIVED_AT_PATTERN, str(value).strip())
        weekday = None if match is None else match.group(config.RECEIVED_WEEKDAY_PART).lower()
        if match is None or weekday not in config.WEEKDAY_MAPPING:
            unread.append(value)
            times.append(np.nan)
            continue
        times.append(config.WEEKDAY_MAPPING[weekday] + " "
                     + match.group(config.RECEIVED_TIME_PART))
    if unread:
        _refuse(column_name, unread, "times that cannot be read")
    result[column_name] = pd.Series(times, index=result.index, dtype=object)
    return result


def convert_markup_to_plain_text(email_texts):
    """Plain text: no markup, escapes resolved, no-break spaces as spaces, one space between words."""
    def plain(value):
        if _is_gap(value):
            return np.nan
        text = re.sub(config.MARKUP_PATTERN, " ", str(value))
        text = html.unescape(text).replace("\xa0", " ")
        return re.sub(config.WHITESPACE_RUN_PATTERN, " ", text).strip()

    return pd.Series([plain(value) for value in email_texts], index=email_texts.index,
                     name=email_texts.name, dtype=object)


def _restore_capitals(text):
    if _is_gap(text):
        return np.nan
    if any(character.isalpha() for character in text) and text == text.upper():
        return text.lower()
    return text


def clean_email_text(dataframe):
    """Leave every email as plain words; one written wholly in capitals in lower case."""
    result = dataframe.copy()
    plain = convert_markup_to_plain_text(result[config.EMAIL_TEXT_COLUMN])
    result[config.EMAIL_TEXT_COLUMN] = pd.Series([_restore_capitals(text) for text in plain],
                                                 index=result.index, dtype=object)
    return result


def clean_email_dataset(input_file_path, output_file_path):
    """Produce an emails table the request model can learn from."""
    raw = load_raw_dataset(input_file_path)
    raw_columns = list(raw.columns)
    dataframe = remove_duplicate_records(raw)
    dataframe = normalize_category_values(dataframe, config.EMAIL_VALUE_MAPPINGS)
    dataframe = convert_whole_numbers(dataframe, config.EMAIL_WHOLE_NUMBER_PATTERNS)
    dataframe = convert_received_times(dataframe)
    dataframe = remove_rows_missing_target(dataframe, config.EMAIL_TARGET_COLUMN)
    dataframe = clean_email_text(dataframe)
    dataframe[config.EMAIL_LENGTH_COLUMN] = pd.array(
        [pd.NA if _is_gap(text) else len(text) for text in dataframe[config.EMAIL_TEXT_COLUMN]],
        dtype="Int64")
    enough_words = np.array([not _is_gap(text) and len(str(text).split())
                             >= config.MINIMUM_EMAIL_WORDS
                             for text in dataframe[config.EMAIL_TEXT_COLUMN]], dtype=bool)
    dataframe = dataframe.loc[enough_words].reset_index(drop=True)[raw_columns]
    save_processed_dataset(dataframe, output_file_path)
    return dataframe