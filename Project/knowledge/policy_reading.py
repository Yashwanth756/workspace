"""Reads one product category's four rules out of its own allocation policy.

The rules are read from every passage of the category's own policy and from no
other document. Passages overlap, so the same sentence met in two passages is
one statement; a policy that states a rule two ways, or not at all, is refused.
"""
import re
from fractions import Fraction

from config import config
from knowledge.vector_index import read_document_chunks


def _statements(passages, pattern):
    """Each distinct sentence the pattern matches, with its captured value."""
    found = {}
    for passage in passages:
        for match in re.finditer(pattern, passage):
            found[match.group(0)] = match.group(1) if match.groups() else None
    return found


def _refuse(document_name, problem):
    raise ValueError(f"{document_name}: {problem}")


def _one_statement(document_name, statements, rule_name):
    if len(statements) == 0:
        _refuse(document_name, f"states no {rule_name}")
    if len(statements) > 1:
        _refuse(document_name, f"states {len(statements)} different {rule_name} rules: "
                               f"{sorted(statements)}")
    return next(iter(statements.items()))


def _one_of(document_name, stated, denied, rule_name):
    if stated and denied:
        _refuse(document_name, f"states a {rule_name} and also that there is none")
    if not stated and not denied:
        _refuse(document_name, f"states neither a {rule_name} nor that there is none")
    if len(denied) > 0:
        return next(iter(denied)), None
    return _one_statement(document_name, stated, rule_name)


def read_allocation_policy(collection, product_category):
    """The rules a product's cases are divided under, read from its own policy."""
    if product_category not in config.CATEGORY_POLICY_DOCUMENTS:
        raise ValueError(config.NO_POLICY_CONFIGURED_MESSAGE.format(
            product_category=product_category))
    document_name = config.CATEGORY_POLICY_DOCUMENTS[product_category]
    passages = [re.sub(config.WHITESPACE_RUN_PATTERN, " ", passage)
                for passage in read_document_chunks(collection, document_name)]

    history_sentence, weeks = _one_statement(
        document_name, _statements(passages, config.HISTORY_SENTENCE_PATTERN), "weeks of history")
    cap_sentence, cap = _one_statement(
        document_name, _statements(passages, config.CAP_SENTENCE_PATTERN), "cap")
    protected_sentence, channels_text = _one_of(
        document_name, _statements(passages, config.PROTECTED_SENTENCE_PATTERN),
        _statements(passages, config.NO_PROTECTION_SENTENCE_PATTERN), "protected channel")
    reserve_sentence, percent = _one_of(
        document_name, _statements(passages, config.RESERVE_SENTENCE_PATTERN),
        _statements(passages, config.NO_RESERVE_SENTENCE_PATTERN), "reserve")

    protected_channels = ([] if channels_text is None
                          else channels_text.split(config.PROTECTED_CHANNEL_SEPARATOR))
    known_channels = set(config.CHANNEL_MAPPING.values())
    unknown = [channel for channel in protected_channels if channel not in known_channels]
    if unknown:
        _refuse(document_name, f"protects channels that are not Wrenfield's: {unknown}")

    return {
        "document_name": document_name,
        "history_weeks": int(weeks),
        "cap": Fraction(cap),
        "protected_channels": protected_channels,
        "reserve_percent": None if percent is None else int(percent),
        "rule_sentences": {"history": history_sentence, "cap": cap_sentence,
                           "protected": protected_sentence, "reserve": reserve_sentence},
    }