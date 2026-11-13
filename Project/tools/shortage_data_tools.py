"""Serves the live queue: one product's profile, its open orders read through each
route's notation, its shipment history, its customers and their open emails.

Every function works on data its caller has loaded: none reads a file or changes
the table it is given.
"""
import re

import pandas as pd

from config import config
from data_pipeline.common_cleaning import (_is_gap, _refuse, convert_whole_numbers,
                                           normalize_category_values, remove_duplicate_records)
from data_pipeline.email_cleaning import clean_email_text, convert_received_times
from data_pipeline.order_cleaning import clean_order_records


def _is_canonical(account_code):
    return (not _is_gap(account_code)
            and re.fullmatch(config.CANONICAL_ACCOUNT_CODE_PATTERN, str(account_code)) is not None)


def fetch_shortage_profile(shortage_products, product_code):
    """One product's own facts: its row, its six counts as integers."""
    rows = shortage_products.loc[shortage_products[config.PRODUCT_CODE_COLUMN] == product_code]
    if len(rows) == 0:
        raise ValueError(config.PRODUCT_NOT_IN_QUEUE_MESSAGE.format(product_code=product_code))
    if len(rows) > 1:
        raise ValueError(f"Product {product_code} is held {len(rows)} times in the queue")
    blank = [column_name for column_name in rows.columns if _is_gap(rows.iloc[0][column_name])]
    if blank:
        raise ValueError(f"Product {product_code} leaves blank: {blank}")
    counts = convert_whole_numbers(rows, config.SHORTAGE_PRODUCT_WHOLE_NUMBER_PATTERNS)
    profile = {}
    for column_name in rows.columns:
        value = counts.iloc[0][column_name]
        if column_name in config.SHORTAGE_PRODUCT_WHOLE_NUMBER_PATTERNS:
            profile[column_name] = int(value)
        else:
            profile[column_name] = str(value)
    return profile


def fetch_open_orders(open_orders, product_code):
    """One product's open orders, each read through its route's notation, an order
    sent twice kept once as it first arrived."""
    product_lines = open_orders.loc[open_orders[config.PRODUCT_CODE_COLUMN] == product_code]
    if len(product_lines) == 0:
        raise ValueError(f"Product {product_code} has no open orders")
    product_lines = remove_duplicate_records(product_lines)
    line_ids = product_lines[config.ORDER_LINE_ID_COLUMN]
    if line_ids.isna().any() or line_ids.duplicated().any():
        shared = line_ids[line_ids.isna() | line_ids.duplicated(keep=False)]
        _refuse(config.ORDER_LINE_ID_COLUMN, shared,
                f"order lines of {product_code} without an order line id of their own")
    cleaned = clean_order_records(product_lines, config.OPEN_ORDER_VALUE_MAPPINGS,
                                  config.OPEN_ORDER_WHOLE_NUMBER_PATTERNS,
                                  config.OPEN_ORDER_DECIMAL_NUMBER_PATTERNS,
                                  config.OPEN_ORDER_MINIMUM_VALUES)
    incomplete = (cleaned[config.ACCOUNT_CODE_COLUMN].isna()
                  | cleaned[config.ORDERED_QUANTITY_COLUMN].isna())
    if incomplete.any():
        _refuse(config.ORDER_LINE_ID_COLUMN, cleaned.loc[incomplete, config.ORDER_LINE_ID_COLUMN],
                "orders that leave their account or their quantity blank")
    return cleaned.reset_index(drop=True)


def fetch_shipment_history(shipment_history, product_code):
    """One product's shipments, weeks_ago and cases_shipped read as whole numbers."""
    shipments = shipment_history.loc[
        shipment_history[config.PRODUCT_CODE_COLUMN] == product_code].reset_index(drop=True)
    references = shipments[config.SHIPMENT_REFERENCE_COLUMN]
    not_canonical = [code for code in shipments[config.ACCOUNT_CODE_COLUMN]
                     if not _is_canonical(code)]
    if not_canonical:
        _refuse(config.ACCOUNT_CODE_COLUMN, not_canonical, "shipments to codes that are not canonical")
    if references.isna().any():
        _refuse(config.SHIPMENT_REFERENCE_COLUMN,
                shipments.loc[references.isna(), config.ACCOUNT_CODE_COLUMN],
                "shipments with no reference, to")
    if references.duplicated().any():
        _refuse(config.SHIPMENT_REFERENCE_COLUMN, references[references.duplicated(keep=False)],
                "references recorded more than once")
    incomplete = (shipments[config.WEEKS_AGO_COLUMN].isna()
                  | shipments[config.CASES_SHIPPED_COLUMN].isna())
    if incomplete.any():
        _refuse(config.SHIPMENT_REFERENCE_COLUMN, references[incomplete],
                "shipments that leave their week or their cases blank")
    return convert_whole_numbers(shipments, config.SHIPMENT_WHOLE_NUMBER_PATTERNS)


def fetch_customer_accounts(customer_master, account_codes):
    """Each customer's name and channel, keyed by its code in code order."""
    canonical_channels = set(config.CHANNEL_MAPPING.values())
    accounts, problems = {}, []
    for account_code in sorted(account_codes):
        rows = customer_master.loc[customer_master[config.ACCOUNT_CODE_COLUMN] == account_code]
        if len(rows) != 1:
            problems.append(account_code)
            continue
        row = rows.iloc[0]
        if (_is_gap(row[config.CUSTOMER_NAME_COLUMN])
                or row[config.ACCOUNT_STATUS_COLUMN] != config.ACCOUNT_STATUS_ACTIVE
                or row[config.CHANNEL_COLUMN] not in canonical_channels):
            problems.append(account_code)
            continue
        accounts[account_code] = {config.CUSTOMER_NAME_COLUMN: str(row[config.CUSTOMER_NAME_COLUMN]),
                                  config.CHANNEL_COLUMN: str(row[config.CHANNEL_COLUMN])}
    if problems:
        _refuse(config.ACCOUNT_CODE_COLUMN, problems,
                "accounts the customer master does not hold once, active, named and in one of "
                "the eight channels")
    return accounts


def fetch_open_emails(open_customer_emails, product_code, account_codes):
    """The open emails of the given customers that name the product or no product,
    repaired as the training emails are, customers in code order and each
    customer's in the order it sent them."""
    named = open_customer_emails[config.PRODUCT_CODE_COLUMN]
    bearing = open_customer_emails.loc[[_is_gap(value) or value == product_code for value in named]]
    not_canonical = [code for code in bearing[config.ACCOUNT_CODE_COLUMN] if not _is_canonical(code)]
    if not_canonical:
        _refuse(config.ACCOUNT_CODE_COLUMN, not_canonical,
                f"emails about {product_code} from codes that are not canonical")
    emails = bearing.loc[bearing[config.ACCOUNT_CODE_COLUMN].isin(list(account_codes))].copy()
    if len(emails) == 0:
        return []
    ids = emails[config.EMAIL_ID_COLUMN]

    blank = [str(email_id) for email_id, sender, received, subject in zip(
        ids, emails[config.SENDER_ROLE_COLUMN], emails[config.EMAIL_RECEIVED_AT_COLUMN],
        emails[config.EMAIL_SUBJECT_COLUMN]) if _is_gap(sender) or _is_gap(received)
        or _is_gap(subject)]
    if blank:
        _refuse(config.EMAIL_ID_COLUMN, blank, "emails that leave their sender, time or subject blank")

    emails = normalize_category_values(emails, config.OPEN_EMAIL_VALUE_MAPPINGS)
    emails = convert_whole_numbers(emails, config.EMAIL_WHOLE_NUMBER_PATTERNS)
    emails = convert_received_times(emails)
    emails = clean_email_text(emails)

    no_text = [str(email_id) for email_id, text in zip(ids, emails[config.EMAIL_TEXT_COLUMN])
               if _is_gap(text) or str(text) == ""]
    if no_text:
        _refuse(config.EMAIL_ID_COLUMN, no_text, "emails with no text once cleaned")
    places = emails[[config.ACCOUNT_CODE_COLUMN, config.EMAIL_SEQUENCE_COLUMN]]
    unplaced = places[config.EMAIL_SEQUENCE_COLUMN].isna() | places.duplicated(keep=False)
    if unplaced.any():
        _refuse(config.EMAIL_ID_COLUMN, emails.loc[unplaced, config.EMAIL_ID_COLUMN],
                "emails that do not each carry their own place among their customer's emails")

    emails = emails.sort_values([config.ACCOUNT_CODE_COLUMN, config.EMAIL_SEQUENCE_COLUMN],
                                kind="stable")
    result = []
    for _, row in emails.iterrows():
        entry = {}
        for field_name in config.OPEN_EMAIL_FIELDS:
            value = row[field_name]
            entry[field_name] = (int(value) if field_name == config.EMAIL_SEQUENCE_COLUMN
                                 else str(value))
        result.append(entry)
    return result