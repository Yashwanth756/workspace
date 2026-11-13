"""Turns a model's raw output into the facts the business acts on."""
from config import config
from data_pipeline.feature_builder import build_feature_frame


def _band(score):
    if score >= config.INFLATED_ORDER_BAND_HIGH_THRESHOLD:
        return config.INFLATED_ORDER_BAND_HIGH
    if score >= config.INFLATED_ORDER_BAND_MEDIUM_THRESHOLD:
        return config.INFLATED_ORDER_BAND_MEDIUM
    return config.INFLATED_ORDER_BAND_LOW


def build_inflated_order_facts(inflated_order_model, inflated_order_feature_names,
                               product_orders):
    """Score and band every order, in the table's order, matched by position."""
    features = build_feature_frame(product_orders, config.INFLATED_ORDER_NUMERIC_FEATURES,
                                   config.INFLATED_ORDER_CATEGORICAL_FEATURES)
    features = features[list(inflated_order_feature_names)]
    probabilities = inflated_order_model.predict_proba(features)[:, 1]
    facts = []
    for line_id, account_code, probability in zip(
            product_orders[config.ORDER_LINE_ID_COLUMN].tolist(),
            product_orders[config.ACCOUNT_CODE_COLUMN].tolist(), probabilities):
        score = round(float(probability), config.INFLATED_ORDER_SCORE_DECIMALS)
        facts.append({"order_line_id": str(line_id), "account_code": str(account_code),
                      "inflated_order_score": score, "inflated_order_band": _band(score)})
    return facts


def forecast_customer_demand(demand_model, demand_feature_names, product_orders):
    """Each customer's forecast for next week, from its first order line in the order log."""
    ordered = product_orders.sort_values(config.RECEIVED_SEQUENCE_COLUMN, kind="stable")
    first_lines = ordered.drop_duplicates(config.ACCOUNT_CODE_COLUMN, keep="first")
    first_lines = first_lines.sort_values(config.ACCOUNT_CODE_COLUMN, kind="stable")
    features = build_feature_frame(first_lines, config.DEMAND_NUMERIC_FEATURES,
                                   config.DEMAND_CATEGORICAL_FEATURES)
    features = features[list(demand_feature_names)]
    forecasts = demand_model.predict(features)
    return {str(account_code): float(forecast) for account_code, forecast
            in zip(first_lines[config.ACCOUNT_CODE_COLUMN].tolist(), forecasts)}


def build_request_facts(request_model, open_emails):
    """The request every email makes and each customer's most urgent request."""
    if len(open_emails) == 0:
        return {"email_requests": [], "customer_requests": {}}
    texts = [open_email[config.EMAIL_TEXT_COLUMN] for open_email in open_emails]
    read_types = [str(label) for label in request_model.predict(texts)]
    unranked = sorted(set(read_types) - set(config.REQUEST_URGENCY_ORDER))
    if unranked:
        raise ValueError(f"{config.EMAIL_TARGET_COLUMN}: requests the configuration does not "
                         f"rank: {unranked}")
    email_requests = [dict(open_email, **{config.EMAIL_TARGET_COLUMN: request_type})
                      for open_email, request_type in zip(open_emails, read_types)]
    customer_requests = {}
    for email_request in email_requests:
        account_code = email_request[config.ACCOUNT_CODE_COLUMN]
        request_type = email_request[config.EMAIL_TARGET_COLUMN]
        held = customer_requests.get(account_code)
        if held is None or (config.REQUEST_URGENCY_ORDER.index(request_type)
                            < config.REQUEST_URGENCY_ORDER.index(held)):
            customer_requests[account_code] = request_type
    return {"email_requests": email_requests,
            "customer_requests": dict(sorted(customer_requests.items()))}


def compute_forecast_shortfall(customer_forecasts, cases_available_next_week):
    """The forecasts summed, rounded once (an exact half to the even number), less
    next week's cases; below nought is kept as it is."""
    return int(round(sum(customer_forecasts.values()))) - int(cases_available_next_week)