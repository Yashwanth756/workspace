"""The cleaning every set of order lines goes through, the history's and the
live queue's alike, and the cleaning run for the order history extracts."""
import pandas as pd

from config import config
from data_pipeline.common_cleaning import (_refuse, convert_account_code,
                                           convert_decimal_numbers, convert_pack_descriptions,
                                           convert_quantity_to_cases, convert_whole_numbers,
                                           keep_first_arrival_of_each_order,
                                           normalize_category_values, normalize_customer_names,
                                           read_pack_description, remove_duplicate_records,
                                           remove_rows_missing_target,
                                           treat_counts_above_total_as_missing,
                                           treat_values_below_minimum_as_missing)
from data_pipeline.dataset_loader import load_raw_dataset, save_processed_dataset


def clean_order_records(frame, value_mappings, whole_number_patterns, decimal_number_patterns,
                        minimum_values):
    """Clean a set of order lines, whatever route wrote each one.

    Every value of every line is read first, so a value no notation reads is
    refused even on a line cleaning later removes; then each order sent by two
    routes is kept once, as it first arrived.
    """
    dataframe = remove_duplicate_records(frame)
    dataframe = normalize_category_values(dataframe, value_mappings)
    dataframe[config.CUSTOMER_NAME_COLUMN] = normalize_customer_names(
        dataframe[config.CUSTOMER_NAME_COLUMN])
    dataframe = convert_whole_numbers(dataframe, whole_number_patterns)
    dataframe = convert_decimal_numbers(dataframe, decimal_number_patterns)

    pack_parts = read_pack_description(
        dataframe[config.PACK_DESCRIPTION_COLUMN].rename(config.PACK_DESCRIPTION_COLUMN))
    routes = dataframe[config.ORDER_ROUTE_COLUMN]
    dataframe[config.ACCOUNT_CODE_COLUMN] = convert_account_code(
        dataframe[config.ACCOUNT_CODE_COLUMN].rename(config.ACCOUNT_CODE_COLUMN), routes)
    dataframe[config.ORDERED_QUANTITY_COLUMN] = convert_quantity_to_cases(
        dataframe[config.ORDERED_QUANTITY_COLUMN].rename(config.ORDERED_QUANTITY_COLUMN),
        routes, pack_parts[config.UNITS_PER_CASE_PART],
        dataframe[config.CASES_PER_PALLET_COLUMN])
    dataframe = convert_pack_descriptions(dataframe)

    dataframe = treat_values_below_minimum_as_missing(dataframe, minimum_values)
    dataframe = treat_counts_above_total_as_missing(dataframe)

    accounts_per_order = (dataframe.dropna(subset=[config.ACCOUNT_CODE_COLUMN])
                          .groupby(list(config.ORDER_KEY_COLUMNS))[config.ACCOUNT_CODE_COLUMN]
                          .nunique())
    clashing = accounts_per_order[accounts_per_order > 1]
    if len(clashing) > 0:
        _refuse(config.ORDER_REFERENCE_COLUMN,
                [" ".join(map(str, key)) for key in clashing.index],
                "purchase orders and products arriving from two accounts")
    return keep_first_arrival_of_each_order(dataframe)


def clean_order_dataset(input_file_path, output_file_path):
    """Produce an order table the inflated-order model can be fitted on."""
    raw = load_raw_dataset(input_file_path)
    raw_columns = list(raw.columns)
    folded = normalize_category_values(raw, config.ORDER_TARGET_VALUE_MAPPINGS)
    cleaned = clean_order_records(folded, config.ORDER_VALUE_MAPPINGS,
                                  config.ORDER_WHOLE_NUMBER_PATTERNS,
                                  config.ORDER_DECIMAL_NUMBER_PATTERNS,
                                  config.ORDER_MINIMUM_VALUES)
    cleaned = remove_rows_missing_target(cleaned, config.ORDER_TARGET_COLUMN)[raw_columns]
    save_processed_dataset(cleaned, output_file_path)
    return cleaned