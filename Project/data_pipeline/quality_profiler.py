"""Measures how dirty each raw extract was and how clean its cleaned table is,
and writes that evidence to output/data_quality_report.json.

Every count taken as received is taken once for each record, over its first
copy, and a gap is never counted as a defect. rows_removed is the sum of
duplicate_rows, rows_missing_target and the extract's own removal count.
"""
import json
import os
import re

from config import config
from data_pipeline.common_cleaning import (_is_gap, convert_whole_numbers,
                                           remove_duplicate_records)
from data_pipeline.email_cleaning import convert_markup_to_plain_text

CANONICAL_WEEKDAYS = set(config.WEEKDAY_MAPPING.values())


def _gap_counts(dataframe):
    counts = dataframe.isna().sum()
    return {str(column_name): int(count) for column_name, count in counts.items() if count > 0}


def _present(values):
    return [value for value in values if not _is_gap(value)]


def _count_not_matching(values, pattern):
    return sum(1 for value in _present(values) if re.fullmatch(pattern, str(value)) is None)


def _numbers_with_unit_or_sign(first_copies, patterns):
    return {column_name: _count_not_matching(first_copies[column_name],
                                             config.DECIMAL_NUMBER_PATTERN)
            for column_name in patterns if column_name in first_copies.columns}


def _read_number(value, pattern):
    if _is_gap(value):
        return None
    match = re.fullmatch(pattern, str(value).strip())
    return None if match is None else float(match.group(1))


def _below_minimum(first_copies, minimum_values, patterns):
    return {column_name: sum(1 for value in first_copies[column_name]
                             if (reading := _read_number(value, patterns[column_name])) is not None
                             and reading < minimum)
            for column_name, minimum in minimum_values.items()}


def _weekdays_not_canonical(values):
    return sum(1 for value in _present(values) if str(value) not in CANONICAL_WEEKDAYS)


def _target_shares(clean_frame, target_column):
    shares = clean_frame[target_column].value_counts(normalize=True)
    return {str(answer): round(float(share), config.TARGET_SHARE_DECIMALS)
            for answer, share in sorted(shares.items(), key=lambda item: str(item[0]))}


def profile_dataset(raw_frame, clean_frame, target_column):
    """Account for every row an extract arrived with."""
    first_copies = remove_duplicate_records(raw_frame)
    return {
        "rows_received": int(len(raw_frame)),
        "rows_kept": int(len(clean_frame)),
        "rows_removed": int(len(raw_frame) - len(clean_frame)),
        "duplicate_rows": int(len(raw_frame) - len(first_copies)),
        "rows_missing_target": int(first_copies[target_column].isna().sum()),
        "columns": int(clean_frame.shape[1]),
        "missing_cells_as_received": _gap_counts(first_copies),
        "missing_cells_after_cleaning": _gap_counts(clean_frame),
    }


def profile_order_details(raw_frame, clean_frame):
    """Record the defects peculiar to the orders."""
    first_copies = remove_duplicate_records(raw_frame)
    sequenced = convert_whole_numbers(
        first_copies, {config.RECEIVED_SEQUENCE_COLUMN:
                       config.ORDER_WHOLE_NUMBER_PATTERNS[config.RECEIVED_SEQUENCE_COLUMN]})
    earliest = sequenced.groupby(list(config.ORDER_KEY_COLUMNS))[
        config.RECEIVED_SEQUENCE_COLUMN].transform("min")
    later = (sequenced[config.RECEIVED_SEQUENCE_COLUMN] > earliest).fillna(False).astype(bool)
    answered = first_copies[config.ORDER_TARGET_COLUMN].notna()

    units_written = {config.CONSUMER_UNITS_WORD: 0, config.CASES_WORD: 0,
                     config.PALLETS_WORD: 0}
    for quantity, route in zip(first_copies[config.ORDERED_QUANTITY_COLUMN],
                               first_copies[config.ORDER_ROUTE_COLUMN]):
        if _is_gap(quantity) or route not in config.QUANTITY_NOTATIONS:
            continue
        match = re.fullmatch(config.QUANTITY_NOTATIONS[route], str(quantity))
        if match is not None:
            unit = match.group(config.QUANTITY_UNIT_PART)
            units_written[unit] = units_written.get(unit, 0) + 1

    number_patterns = {**config.ORDER_WHOLE_NUMBER_PATTERNS,
                       **config.ORDER_DECIMAL_NUMBER_PATTERNS}
    counts_above = {}
    for count_column, total_column in config.ORDER_COUNT_TOTALS.items():
        counts_above[count_column] = sum(
            1 for count, total in zip(first_copies[count_column], first_copies[total_column])
            if (count_read := _read_number(count, number_patterns[count_column])) is not None
            and (total_read := _read_number(total, number_patterns[total_column])) is not None
            and count_read > total_read)

    return {
        "answered_second_arrivals_removed": int((later & answered).sum()),
        "account_codes_not_written_canonically": _count_not_matching(
            first_copies[config.ACCOUNT_CODE_COLUMN], config.CANONICAL_ACCOUNT_CODE_PATTERN),
        "quantities_by_unit_as_written": units_written,
        "pack_descriptions_not_written_canonically": _count_not_matching(
            first_copies[config.PACK_DESCRIPTION_COLUMN],
            config.CANONICAL_PACK_DESCRIPTION_PATTERN),
        "numbers_written_with_a_unit_or_percent_sign": _numbers_with_unit_or_sign(
            first_copies, number_patterns),
        "values_below_minimum_as_received": _below_minimum(
            first_copies, config.ORDER_MINIMUM_VALUES, number_patterns),
        "counts_above_their_total_as_received": counts_above,
        "weekdays_not_written_canonically": _weekdays_not_canonical(
            first_copies[config.ORDER_WEEKDAY_COLUMN]),
        "target_shares": _target_shares(clean_frame, config.ORDER_TARGET_COLUMN),
    }


def profile_demand_details(raw_frame, clean_frame):
    """Record the defects peculiar to the customer weeks."""
    first_copies = remove_duplicate_records(raw_frame)
    number_patterns = {**config.DEMAND_WHOLE_NUMBER_PATTERNS,
                       **config.DEMAND_DECIMAL_NUMBER_PATTERNS}
    target = config.DEMAND_TARGET_COLUMN
    target_minimum = config.DEMAND_MINIMUM_VALUES[target]
    return {
        "pack_descriptions_not_written_canonically": _count_not_matching(
            first_copies[config.PACK_DESCRIPTION_COLUMN],
            config.CANONICAL_PACK_DESCRIPTION_PATTERN),
        "numbers_written_with_a_unit_or_percent_sign": _numbers_with_unit_or_sign(
            first_copies, number_patterns),
        "values_below_minimum_as_received": _below_minimum(
            first_copies, config.DEMAND_MINIMUM_VALUES, number_patterns),
        "rows_with_impossible_target": sum(
            1 for value in first_copies[target]
            if (reading := _read_number(value, number_patterns[target])) is not None
            and reading < target_minimum),
        "weekdays_not_written_canonically": _weekdays_not_canonical(
            first_copies[config.DEMAND_WEEKDAY_COLUMN]),
        "median_next_week_cases": float(clean_frame[target].astype(float).median()),
    }


def profile_email_details(raw_frame, clean_frame):
    """Record the defects peculiar to free text."""
    first_copies = remove_duplicate_records(raw_frame)
    texts = first_copies[config.EMAIL_TEXT_COLUMN]
    plain = convert_markup_to_plain_text(texts)

    weekdays_off = 0
    for value in _present(first_copies[config.EMAIL_RECEIVED_AT_COLUMN]):
        match = re.fullmatch(config.RECEIVED_AT_PATTERN, str(value))
        if match is not None and match.group(config.RECEIVED_WEEKDAY_PART) not in CANONICAL_WEEKDAYS:
            weekdays_off += 1

    lengths_off = 0
    for length, text in zip(first_copies[config.EMAIL_LENGTH_COLUMN], plain):
        if _is_gap(length) or _is_gap(text):
            continue
        reading = _read_number(length, r"^(-?[0-9]+)$")
        if reading is None or int(reading) != len(text):
            lengths_off += 1

    answered = first_copies[config.EMAIL_TARGET_COLUMN].notna().tolist()
    too_short = sum(1 for has_answer, text in zip(answered, plain)
                    if has_answer and (_is_gap(text)
                                       or len(str(text).split()) < config.MINIMUM_EMAIL_WORDS))
    return {
        "weekdays_not_written_canonically": weekdays_off,
        "emails_with_markup": sum(1 for text in _present(texts)
                                  if re.search(config.MARKUP_PATTERN, str(text))),
        "emails_with_encoding_artifacts": sum(
            1 for text in _present(texts) if re.search(config.ENCODING_ARTIFACT_PATTERN, str(text))),
        "emails_in_capitals": sum(1 for text in _present(plain)
                                  if any(character.isalpha() for character in text)
                                  and text == text.upper()),
        "email_lengths_disagreeing_with_plain_text_as_received": lengths_off,
        "rows_with_too_few_words": too_short,
        "shortest_email_words": int(min(len(str(text).split())
                                        for text in clean_frame[config.EMAIL_TEXT_COLUMN])),
        "target_shares": _target_shares(clean_frame, config.EMAIL_TARGET_COLUMN),
    }


def write_quality_report(profiles):
    """Write the profiles to config.DATA_QUALITY_REPORT_FILE as indented JSON, keys sorted."""
    folder = os.path.dirname(config.DATA_QUALITY_REPORT_FILE)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(config.DATA_QUALITY_REPORT_FILE, "w", encoding="utf-8") as report_file:
        json.dump(profiles, report_file, indent=2, sort_keys=True)