"""Repairs shared by the cleaning runs, the profiler and the shortage data tools.

Every function returns a repaired copy and leaves its input untouched; none
reads or writes a file. A value in an unrecognised notation is refused with a
ValueError naming the column and the values.
"""
import re

import numpy as np
import pandas as pd

from config import config


def _is_gap(value):
    if value is None or value is pd.NA:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _refuse(column_name, values, what):
    shown = sorted({str(value) for value in values})
    raise ValueError(f"{column_name}: {what}: {shown}")


def remove_duplicate_records(dataframe):
    """Drop rows identical in every column, keeping the first; renumber from 0."""
    result = dataframe.loc[~dataframe.duplicated(keep="first")].copy()
    return result.reset_index(drop=True)


def keep_first_arrival_of_each_order(dataframe):
    """One row per purchase order and product: the earliest in the order log."""
    line_column = config.ORDER_LINE_ID_COLUMN
    key_columns = list(config.ORDER_KEY_COLUMNS)
    sequence_column = config.RECEIVED_SEQUENCE_COLUMN
    incomplete = dataframe[key_columns + [sequence_column]].isna().any(axis=1)
    if incomplete.any():
        _refuse(line_column, dataframe.loc[incomplete, line_column],
                "order lines leave their purchase order, product or place in the order log empty")
    shared = dataframe[sequence_column].duplicated(keep=False)
    if shared.any():
        _refuse(line_column, dataframe.loc[shared, line_column],
                "order lines share a place in the order log")
    ordered = dataframe.sort_values(sequence_column, kind="stable")
    first = ordered.drop_duplicates(subset=key_columns, keep="first")
    return first.reset_index(drop=True)


def normalize_category_values(dataframe, value_mappings):
    """Fold every spelling and case of a category value onto its canonical form."""
    result = dataframe.copy()
    for column_name, value_mapping in value_mappings.items():
        if column_name not in result.columns:
            continue
        values = result[column_name]
        known = [_is_gap(value) or str(value).strip().lower() in value_mapping
                 for value in values]
        if not all(known):
            _refuse(column_name, values[[not flag for flag in known]],
                    "values the mapping does not recognise")
        result[column_name] = pd.Series(
            [np.nan if _is_gap(value) else value_mapping[str(value).strip().lower()]
             for value in values], index=result.index, dtype=object)
    return result


def normalize_customer_names(customer_names):
    """One spelling of each name: single spaces and no legal suffix."""
    def tidy(value):
        if _is_gap(value):
            return np.nan
        name = re.sub(config.WHITESPACE_RUN_PATTERN, " ", str(value)).strip()
        return re.sub(config.CUSTOMER_NAME_LEGAL_SUFFIX_PATTERN, "", name).strip()

    return pd.Series([tidy(value) for value in customer_names], index=customer_names.index,
                     name=customer_names.name, dtype=object)


def remove_rows_missing_target(dataframe, target_column):
    """Remove exactly the rows with no answer; renumber from 0."""
    return dataframe.loc[dataframe[target_column].notna()].reset_index(drop=True)


def read_pack_description(pack_descriptions):
    """The units in a case and the millilitres in a unit of every description."""
    units, millilitres, unread = [], [], []
    for value in pack_descriptions:
        if _is_gap(value):
            units.append(pd.NA)
            millilitres.append(pd.NA)
            continue
        for pattern in config.PACK_DESCRIPTION_NOTATIONS:
            match = re.fullmatch(pattern, str(value).strip())
            if match is not None:
                units.append(int(match.group(config.UNITS_PER_CASE_PART)))
                millilitres.append(int(match.group(config.UNIT_MILLILITRES_PART)))
                break
        else:
            unread.append(value)
            units.append(pd.NA)
            millilitres.append(pd.NA)
    if unread:
        _refuse(pack_descriptions.name, unread, "descriptions in none of the notations")
    return pd.DataFrame({config.UNITS_PER_CASE_PART: pd.array(units, dtype="Int64"),
                         config.UNIT_MILLILITRES_PART: pd.array(millilitres, dtype="Int64")},
                        index=pack_descriptions.index)


def convert_pack_descriptions(dataframe):
    """Write every pack description as config.CANONICAL_PACK_DESCRIPTION_FORMAT."""
    result = dataframe.copy()
    column_name = config.PACK_DESCRIPTION_COLUMN
    parts = read_pack_description(result[column_name].rename(column_name))
    result[column_name] = pd.Series(
        [np.nan if _is_gap(units) else config.CANONICAL_PACK_DESCRIPTION_FORMAT.format(
            units_per_case=int(units), unit_millilitres=int(millilitres))
         for units, millilitres in zip(parts[config.UNITS_PER_CASE_PART],
                                       parts[config.UNIT_MILLILITRES_PART])],
        index=result.index, dtype=object)
    return result


def convert_account_code(account_codes, order_routes):
    """Read every account code through the notation of the route that wrote it."""
    canonical, unread_codes, unknown_routes = [], [], []
    for code, route in zip(account_codes, order_routes):
        if _is_gap(code):
            canonical.append(np.nan)
            continue
        if _is_gap(route) or route not in config.ACCOUNT_CODE_NOTATIONS:
            unknown_routes.append(route)
            canonical.append(np.nan)
            continue
        match = re.fullmatch(config.ACCOUNT_CODE_NOTATIONS[route], str(code).strip())
        if match is None:
            unread_codes.append(code)
            canonical.append(np.nan)
            continue
        canonical.append(config.CANONICAL_ACCOUNT_CODE_FORMAT.format(
            account_number=int(match.group(config.ACCOUNT_NUMBER_PART))))
    if unknown_routes:
        _refuse(account_codes.name, unknown_routes, "routes with no account code notation")
    if unread_codes:
        _refuse(account_codes.name, unread_codes, "codes their own route does not write")
    return pd.Series(canonical, index=account_codes.index, name=account_codes.name, dtype=object)


def convert_quantity_to_cases(quantities, order_routes, units_per_case, cases_per_pallet):
    """Read every quantity, in whatever unit its route writes, as whole cases."""
    cases, unread, unknown_routes, no_size, not_whole = [], [], [], [], []
    for quantity, route, units, pallet in zip(quantities, order_routes, units_per_case,
                                              cases_per_pallet):
        if _is_gap(quantity):
            cases.append(pd.NA)
            continue
        if _is_gap(route) or route not in config.QUANTITY_NOTATIONS:
            unknown_routes.append(route)
            cases.append(pd.NA)
            continue
        match = re.fullmatch(config.QUANTITY_NOTATIONS[route], str(quantity).strip())
        if match is None:
            unread.append(quantity)
            cases.append(pd.NA)
            continue
        amount = int(match.group(config.QUANTITY_AMOUNT_PART))
        unit = match.group(config.QUANTITY_UNIT_PART)
        if unit == config.CASES_WORD:
            cases.append(amount)
        elif unit == config.CONSUMER_UNITS_WORD:
            if _is_gap(units):
                no_size.append(quantity)
                cases.append(pd.NA)
            elif amount % int(units) != 0:
                not_whole.append(quantity)
                cases.append(pd.NA)
            else:
                cases.append(amount // int(units))
        else:
            if _is_gap(pallet):
                no_size.append(quantity)
                cases.append(pd.NA)
            else:
                cases.append(amount * int(pallet))
    if unknown_routes:
        _refuse(quantities.name, unknown_routes, "routes with no quantity notation")
    if unread:
        _refuse(quantities.name, unread, "quantities their own route does not write")
    if no_size:
        _refuse(quantities.name, no_size, "units or pallets with no size to convert them")
    if not_whole:
        _refuse(quantities.name, not_whole, "units that are not a whole number of cases")
    return pd.Series(pd.array(cases, dtype="Int64"), index=quantities.index, name=quantities.name)


def _read_numbers(dataframe, number_patterns, converter, dtype):
    result = dataframe.copy()
    for column_name, pattern in number_patterns.items():
        if column_name not in result.columns:
            continue
        readings, unread = [], []
        for value in result[column_name]:
            if _is_gap(value):
                readings.append(pd.NA)
                continue
            match = re.fullmatch(pattern, str(value).strip())
            if match is None:
                unread.append(value)
                readings.append(pd.NA)
            else:
                readings.append(converter(match.group(1)))
        if unread:
            _refuse(column_name, unread, "values the column's pattern does not read")
        result[column_name] = pd.array(readings, dtype=dtype)
    return result


def convert_whole_numbers(dataframe, number_patterns):
    """Read each named column as whole numbers through its pattern; gaps stay gaps."""
    return _read_numbers(dataframe, number_patterns, int, "Int64")


def convert_decimal_numbers(dataframe, number_patterns):
    """Read each named column as decimal numbers through its pattern; gaps stay gaps."""
    return _read_numbers(dataframe, number_patterns, float, "Float64")


def treat_values_below_minimum_as_missing(dataframe, minimum_values):
    """A value below its column's minimum becomes missing; every row is kept."""
    result = dataframe.copy()
    for column_name, minimum in minimum_values.items():
        if column_name not in result.columns:
            continue
        too_low = (result[column_name] < minimum).fillna(False).astype(bool)
        result.loc[too_low, column_name] = pd.NA
    return result


def treat_counts_above_total_as_missing(dataframe):
    """For each count of config.ORDER_COUNT_TOTALS, a count above its total becomes missing."""
    result = dataframe.copy()
    for count_column, total_column in config.ORDER_COUNT_TOTALS.items():
        if count_column not in result.columns or total_column not in result.columns:
            continue
        too_high = (result[count_column] > result[total_column]).fillna(False).astype(bool)
        result.loc[too_high, count_column] = pd.NA
    return result