"""Reads the passage an agent quotes from the one document that states it."""
from config import config
from knowledge.vector_index import find_nearest_passage


def _policy_document(product_category):
    if product_category not in config.CATEGORY_POLICY_DOCUMENTS:
        raise ValueError(config.NO_POLICY_CONFIGURED_MESSAGE.format(
            product_category=product_category))
    return config.CATEGORY_POLICY_DOCUMENTS[product_category]


def read_planner_checks(collection, request_type):
    """What the planner checks for a request, from the request type's own guide."""
    if request_type not in config.REQUEST_GUIDE_DOCUMENTS:
        raise ValueError(f"No request guide is configured for {request_type}")
    return find_nearest_passage(collection, config.PLANNER_CHECKS_QUESTION,
                                config.REQUEST_GUIDE_DOCUMENTS[request_type])


def read_customer_advice(collection, product_category):
    """The category policy's advice to customers while the product is on allocation."""
    return find_nearest_passage(collection, config.CUSTOMER_ADVICE_QUESTION,
                                _policy_document(product_category))


def read_sales_briefing_guidance(collection, product_category):
    """How the category policy says the sales team is briefed."""
    return find_nearest_passage(collection, config.SALES_BRIEFING_QUESTION,
                                _policy_document(product_category))