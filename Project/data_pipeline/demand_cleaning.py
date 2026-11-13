"""The cleaning run for the weekly demand extracts."""
from config import config
from data_pipeline.common_cleaning import (convert_decimal_numbers, convert_pack_descriptions,
                                           convert_whole_numbers, normalize_category_values,
                                           remove_duplicate_records, remove_rows_missing_target,
                                           treat_values_below_minimum_as_missing)
from data_pipeline.dataset_loader import load_raw_dataset, save_processed_dataset


def clean_demand_dataset(input_file_path, output_file_path):
    """Produce a demand table the demand model can be fitted on.

    Every value of every row is read before any row is removed. A reading below
    its minimum becomes missing; next_week_cases below nought therefore becomes
    missing, and a week with no answer is removed.
    """
    raw = load_raw_dataset(input_file_path)
    raw_columns = list(raw.columns)
    dataframe = remove_duplicate_records(raw)
    dataframe = normalize_category_values(dataframe, config.DEMAND_VALUE_MAPPINGS)
    dataframe = convert_whole_numbers(dataframe, config.DEMAND_WHOLE_NUMBER_PATTERNS)
    dataframe = convert_decimal_numbers(dataframe, config.DEMAND_DECIMAL_NUMBER_PATTERNS)
    dataframe = convert_pack_descriptions(dataframe)
    dataframe = treat_values_below_minimum_as_missing(dataframe, config.DEMAND_MINIMUM_VALUES)
    dataframe = remove_rows_missing_target(dataframe, config.DEMAND_TARGET_COLUMN)[raw_columns]
    save_processed_dataset(dataframe, output_file_path)
    return dataframe