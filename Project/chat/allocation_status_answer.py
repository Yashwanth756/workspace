"""Answers a question about one account's or one product's allocation from the
records, in the way the asker's role allows. No language model call is made."""
import re

from sqlalchemy import true

from chat.allocation_assistant import read_account_code
from config import config
from persistence.models import AllocationReview
from persistence.repository import fetch_review_records


def _is_blank(value):
    return value is None or (isinstance(value, float) and value != value) or str(value).strip() == ""


def _spaceless(text):
    return re.sub(r"\s+", "", str(text)).upper()


def _account_line(review, account_code):
    for line in review["allocation_lines"]:
        if line["account_code"] == account_code:
            return line
    return None


def _answer_adviser(session, customer_master, account_code, postcode, product_code):
    if _is_blank(account_code):
        return config.CHAT_ACCOUNT_NOT_NAMED_MESSAGE
    canonical = read_account_code(account_code)
    refusal = config.CHAT_NO_ALLOCATION_TEMPLATE.format(
        account=canonical or re.sub(config.RESPONSE_DECORATION_EDGE_PATTERN, "",
                                    str(account_code).strip()))
    if canonical is None or _is_blank(postcode):
        return refusal
    rows = customer_master.loc[customer_master[config.ACCOUNT_CODE_COLUMN] == canonical]
    if len(rows) != 1:
        return refusal
    recorded = rows.iloc[0][config.DELIVERY_POSTCODE_COLUMN]
    if _is_blank(recorded) or _spaceless(recorded) != _spaceless(postcode):
        return refusal

    condition = (AllocationReview.product_code == product_code if product_code
                 else true())
    lines = [config.CHAT_ACCOUNT_HEADING_TEMPLATE.format(
        account_code=canonical, customer_name=rows.iloc[0][config.CUSTOMER_NAME_COLUMN])]
    for review in fetch_review_records(session, condition):
        line = _account_line(review, canonical)
        if line is None:
            continue
        details = {"product_code": review["product_code"],
                   "product_name": review["product_name"],
                   "cases_ordered": line["cases_ordered"]}
        if review["status"] == config.REVIEW_STATUS_PENDING:
            lines.append(config.CHAT_ORDER_UNDER_REVIEW_TEMPLATE.format(**details))
        elif review["status"] == config.REVIEW_STATUS_ESCALATED:
            lines.append(config.CHAT_ORDER_ESCALATED_TEMPLATE.format(**details))
        else:
            lines.append(config.CHAT_ORDER_RELEASED_TEMPLATE.format(
                cases_confirmed=(review["cases_confirmed"] or {}).get(canonical, 0), **details))
    lines.append(config.CHAT_OTHER_ORDERS_TEXT)
    return "\n".join(lines)


def _cases_text(review, line):
    if review["status"] == config.REVIEW_STATUS_PENDING:
        open_releases = [f"{release} {line['cases_by_release'][release]:,}"
                         for release in review["decisions_open"]
                         if release in line["cases_by_release"]]
        return "; ".join(open_releases) if open_releases else config.CHAT_NO_RELEASE_OPEN_TEXT
    return config.CHAT_CONFIRMED_CASES_TEMPLATE.format(
        cases_confirmed=(review["cases_confirmed"] or {}).get(line["account_code"], 0))


def _answer_desk(session, account_code, product_code):
    account = None
    if not _is_blank(account_code):
        account = (read_account_code(account_code)
                   or re.sub(config.RESPONSE_DECORATION_EDGE_PATTERN, "",
                             str(account_code).strip()))
    product = None if _is_blank(product_code) else product_code
    subjects = []
    if product:
        subjects.append(config.CHAT_SUBJECT_PRODUCT_TEMPLATE.format(product_code=product))
    if account:
        subjects.append(config.CHAT_SUBJECT_ACCOUNT_TEMPLATE.format(account_code=account))
    if not subjects:
        return config.CHAT_SUBJECT_NOT_NAMED_MESSAGE
    subject = " and ".join(subjects)

    condition = AllocationReview.product_code == product if product else true()
    covering = []
    for review in fetch_review_records(session, condition):
        if account is None or _account_line(review, account) is not None:
            covering.append(review)
    if not covering:
        return config.CHAT_NO_REVIEW_TEMPLATE.format(subject=subject)

    lines = [config.CHAT_REVIEWS_HEADING_TEMPLATE.format(subject=subject)]
    for review in covering:
        lines.append(config.CHAT_REVIEW_TEMPLATE.format(
            review_id=review["review_id"], product_code=review["product_code"],
            product_name=review["product_name"], policy_document=review["policy_document"],
            planner_id=review["planner_id"],
            status=config.REVIEW_STATUS_LABELS[review["status"]],
            cases_available=review["cases_available"], cases_eligible=review["cases_eligible"]))
        if review["status"] == config.REVIEW_STATUS_PENDING:
            lines.append(config.CHAT_AWAITING_DECISION_TEMPLATE.format(
                decisions_open=", ".join(review["decisions_open"]),
                recommended_decision=review["recommended_decision"]))
        else:
            lines.append(config.CHAT_DECIDED_TEMPLATE.format(
                decided_by=review["decided_by"], decided_by_role=review["decided_by_role"],
                decision=review["decision"], decision_reason=review["decision_reason"],
                reserve_cases=review["reserve_cases"], cases_in_stock=review["cases_in_stock"]))
        for line in review["allocation_lines"]:
            if account is not None and line["account_code"] != account:
                continue
            lines.append(config.CHAT_REVIEW_LINE_TEMPLATE.format(
                customer_name=line["customer_name"], account_code=line["account_code"],
                channel=line["channel"], cases_ordered=line["cases_ordered"],
                eligible_cases=line["eligible_cases"], cases_text=_cases_text(review, line)))
    return "\n".join(lines)


def answer_allocation_status(session, signed_in_user, customer_master, account_code, postcode,
                             product_code):
    """Answer a question about one allocation from the records, across the identity line."""
    if signed_in_user["role"] == config.ROLE_ADVISER:
        return _answer_adviser(session, customer_master, account_code, postcode, product_code)
    return _answer_desk(session, account_code, product_code)