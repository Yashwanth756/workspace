"""Cleans all five raw extracts and writes the quality report.

Run from the Project folder:  python3 run_data_pipeline.py
"""
import pathlib

from config import config
from data_pipeline.dataset_loader import load_processed_dataset, load_raw_dataset
from data_pipeline.demand_cleaning import clean_demand_dataset
from data_pipeline.email_cleaning import clean_email_dataset
from data_pipeline.order_cleaning import clean_order_dataset
from data_pipeline.quality_profiler import (profile_dataset, profile_demand_details,
                                            profile_email_details, profile_order_details,
                                            write_quality_report)

DATASET_JOBS = [
    (config.RAW_ORDER_HISTORY_TRAIN_FILE, config.CLEAN_ORDER_HISTORY_TRAIN_FILE,
     clean_order_dataset, config.ORDER_TARGET_COLUMN, profile_order_details),
    (config.RAW_ORDER_HISTORY_EVAL_FILE, config.CLEAN_ORDER_HISTORY_EVAL_FILE,
     clean_order_dataset, config.ORDER_TARGET_COLUMN, profile_order_details),
    (config.RAW_WEEKLY_DEMAND_TRAIN_FILE, config.CLEAN_WEEKLY_DEMAND_TRAIN_FILE,
     clean_demand_dataset, config.DEMAND_TARGET_COLUMN, profile_demand_details),
    (config.RAW_WEEKLY_DEMAND_EVAL_FILE, config.CLEAN_WEEKLY_DEMAND_EVAL_FILE,
     clean_demand_dataset, config.DEMAND_TARGET_COLUMN, profile_demand_details),
    (config.RAW_CUSTOMER_EMAILS_FILE, config.CLEAN_CUSTOMER_EMAILS_FILE,
     clean_email_dataset, config.EMAIL_TARGET_COLUMN, profile_email_details),
]


def main():
    profiles = {}
    for raw_path, clean_path, clean_function, target_column, profile_details in DATASET_JOBS:
        clean_function(raw_path, clean_path)
        raw_frame = load_raw_dataset(raw_path)
        clean_frame = load_processed_dataset(clean_path)
        profile = {**profile_dataset(raw_frame, clean_frame, target_column),
                   **profile_details(raw_frame, clean_frame)}
        name = pathlib.Path(raw_path).name.removesuffix(".csv")
        profiles[name] = profile
        print(f"{name:<30}{profile['rows_received']:>8} received -> {profile['rows_kept']:>6} "
              f"kept  ({profile['rows_removed']} removed)")
    write_quality_report(profiles)
    print()
    print(f"quality report written to {config.DATA_QUALITY_REPORT_FILE}")


if __name__ == "__main__":
    main()