"""The allocation detail page: one product, its division, customers, orders and emails,
and the form that opens its review."""
import altair as alt
import pandas as pd
import streamlit as st

from config import config
from dashboard.api_client import USER_KEY, call_service
from dashboard.shortage_queue import SELECTED_PRODUCT_KEY

REVIEW_WAIT_MESSAGE = ("The agents are reading the shortage and writing the review: "
                       f"{config.LLM_CALLS_BEFORE_DECISION} language model calls, which can "
                       "take a minute or two.")


def _badge(label, colour):
    return f":{colour}-badge[{label}]"


def _cover_text(cover_percent):
    return (config.COVER_NOT_WORKED_OUT_TEXT if cover_percent is None
            else f"{cover_percent:.2f}%")


def _division_graph(lines, release_name, release, cases_available):
    """The cases available, split into what each group gets, as a Graphviz diagram."""
    protected = [line for line in lines if line["protected"]]
    others = [line for line in lines if not line["protected"]]
    nodes = {"available": f"{cases_available:,} cases available\\nthis week"}
    edges = []
    if protected:
        cases = sum(line["cases_by_release"][release_name] for line in protected)
        nodes["protected"] = f"Protected customers\\n{cases:,} cases to {len(protected)} customers"
        edges.append(("protected", cases))
    if others:
        cases = sum(line["cases_by_release"][release_name] for line in others)
        nodes["shared"] = f"Shared by usual demand\\n{cases:,} cases to {len(others)} customers"
        edges.append(("shared", cases))
    if release["reserve_cases"]:
        nodes["reserve"] = f"Held in reserve\\n{release['reserve_cases']:,} cases"
        edges.append(("reserve", release["reserve_cases"]))
    if release["cases_in_stock"]:
        nodes["stock"] = f"Left in stock\\n{release['cases_in_stock']:,} cases"
        edges.append(("stock", release["cases_in_stock"]))
    dot = ["digraph division {", "rankdir=LR;",
           f'node [shape=box, style="rounded,filled", fontname="Helvetica", '
           f'fontcolor="{config.DIVISION_TEXT_COLOUR}"];',
           f'edge [color="{config.DIVISION_EDGE_COLOUR}", fontname="Helvetica", '
           f'fontcolor="{config.DIVISION_EDGE_COLOUR}"];']
    for key, label in nodes.items():
        dot.append(f'{key} [label="{label}", fillcolor="{config.DIVISION_FILL_COLOURS[key]}", '
                   f'color="{config.DIVISION_LINE_COLOURS[key]}"];')
    for key, cases in edges:
        dot.append(f'available -> {key} [label="{cases:,}"];')
    dot.append("}")
    return "\n".join(dot)


def _division_chart(lines, release_name):
    ordered = sorted(lines, key=lambda line: (not line["protected"], line["customer_name"]))
    names = [line["customer_name"] + (" (protected)" if line["protected"] else "")
             for line in ordered]
    series_gets, series_short = config.ALLOCATION_CHART_SERIES
    records = []
    for name, line in zip(names, ordered):
        gets = line["cases_by_release"][release_name]
        records.append({"Customer": name, "Series": series_gets, "Cases": gets, "Stack": 0})
        records.append({"Customer": name, "Series": series_short,
                        "Cases": max(line["eligible_cases"] - gets, 0), "Stack": 1})
    chart = alt.Chart(pd.DataFrame(records), height=max(120, 22 * len(ordered))).mark_bar().encode(
        y=alt.Y("Customer:N", sort=names, title=None),
        x=alt.X("Cases:Q", title="Cases"),
        color=alt.Color("Series:N", title=None, sort=config.ALLOCATION_CHART_SERIES,
                        scale=alt.Scale(domain=config.ALLOCATION_CHART_SERIES,
                                        range=config.ALLOCATION_CHART_COLOURS),
                        legend=alt.Legend(orient="top")),
        order=alt.Order("Stack:Q"),
        tooltip=["Customer", "Series", "Cases"])
    st.altair_chart(chart, width="stretch")


def _division_tab(detail):
    summary, lines = detail["summary"], detail["allocation_lines"]
    release_names = list(summary["releases"])
    release_name = st.segmented_control("Release", release_names, default=release_names[0],
                                        key=f"release_{detail['profile']['product_code']}")
    release_name = release_name or release_names[0]
    st.graphviz_chart(_division_graph(lines, release_name, summary["releases"][release_name],
                                      summary["cases_available"]))
    _division_chart(lines, release_name)


def _read_as(rules, key):
    if key == "history":
        return f"{rules['history_weeks']} weeks"
    if key == "cap":
        return f"{rules['cap']} times usual demand"
    if key == "protected":
        return ", ".join(rules["protected_channels"]) or "No customer"
    if rules["reserve_percent"] is None:
        return "No reserve"
    return f"Up to {rules['reserve_percent']} percent"


def _customers_tab(detail):
    rules = detail["policy_rules"]
    st.markdown(f"**Rules from {rules['document_name']}**")
    st.dataframe(pd.DataFrame({
        "Rule": [config.POLICY_RULE_LABELS[key] for key in config.POLICY_RULE_LABELS],
        "Read as": [_read_as(rules, key) for key in config.POLICY_RULE_LABELS],
        "Policy": [rules["rule_sentences"][key] for key in config.POLICY_RULE_LABELS]}),
        hide_index=True, width="stretch")
    lines = detail["allocation_lines"]
    release_names = list(detail["summary"]["releases"])
    table = {"Customer": [line["customer_name"] for line in lines],
             "Account": [line["account_code"] for line in lines],
             "Channel": [line["channel"] for line in lines],
             "Protected": ["Yes" if line["protected"] else "No" for line in lines],
             "Ordered": [line["cases_ordered"] for line in lines],
             "Usual demand": [line["usual_demand"] for line in lines],
             "May have": [line["eligible_cases"] for line in lines]}
    for release_name in release_names:
        table[release_name] = [line["cases_by_release"][release_name] for line in lines]
    table["Most urgent request"] = [detail["customer_requests"].get(line["account_code"], "")
                                    for line in lines]
    st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch", column_config={
        "Usual demand": st.column_config.NumberColumn(format="%.2f")})


def _orders_and_emails_tab(detail):
    orders = detail["order_lines"]
    bands = [config.INFLATED_ORDER_BAND_HIGH, config.INFLATED_ORDER_BAND_MEDIUM,
             config.INFLATED_ORDER_BAND_LOW]
    st.markdown(" ".join(_badge(f"{band} {sum(o['inflated_order_band'] == band for o in orders)}",
                                config.INFLATED_ORDER_BAND_BADGE_COLOURS[band])
                         for band in bands))
    st.dataframe(pd.DataFrame({
        "Order line": [o["order_line_id"] for o in orders],
        "Purchase order": [o["order_reference"] for o in orders],
        "Route": [o["order_route"] for o in orders],
        "Customer": [o["customer_name"] for o in orders],
        "Cases": [o["ordered_quantity"] for o in orders],
        "Inflated-order score": [o["inflated_order_score"] for o in orders],
        "Band": [o["inflated_order_band"] for o in orders]}),
        hide_index=True, width="stretch", column_config={
            "Inflated-order score": st.column_config.NumberColumn(format="%.4f")})
    accounts = {line["account_code"]: line["customer_name"] for line in detail["allocation_lines"]}
    if not detail["email_requests"]:
        st.caption("This product has no open customer email.")
    for email in detail["email_requests"]:
        with st.container(border=True):
            request_type = email[config.EMAIL_TARGET_COLUMN]
            st.markdown(f"{_badge(request_type, 'blue')} "
                        f"**{accounts.get(email['account_code'], email['account_code'])}**, "
                        f"{email['sender_role'].lower()}, {email['received_at']} · "
                        f"{email['email_subject']}")
            st.write(email[config.EMAIL_TEXT_COLUMN])


def _review_tab(product_code, review):
    if review is not None:
        st.info(f"{review['review_id']} already reviews {product_code}; a product on "
                f"allocation is reviewed once.")
        return
    if "review" not in st.session_state[USER_KEY]["areas"]:
        st.info("Your role does not open reviews.")
        return
    with st.form(f"start_review_{product_code}"):
        request_text = st.text_area("What do you want looked at?")
        started = st.form_submit_button("Start the review", icon=":material/play_arrow:",
                                        type="primary")
    if not started:
        return
    if len(request_text.strip()) < config.PLANNER_REQUEST_MINIMUM_CHARACTERS:
        st.warning(f"Write at least {config.PLANNER_REQUEST_MINIMUM_CHARACTERS} characters "
                   f"about what you want looked at.")
        return
    call_service("POST", "/reviews", payload={"product_code": product_code,
                                              "planner_request": request_text.strip()},
                 waiting_message=REVIEW_WAIT_MESSAGE)
    st.rerun()


def render_allocation_detail_page():
    """Show one product with every fact its review works from, and let a person open the review."""
    st.title(config.PAGE_DETAIL)
    queue = call_service("GET", "/queue", query={"product_category": config.QUEUE_CATEGORY_ALL})
    labels = {row["product_code"]: f"{row['product_code']} {row['product_name']}"
              for row in queue}
    codes = list(labels)
    wanted = st.session_state.get(SELECTED_PRODUCT_KEY)
    product_code = st.selectbox("Product", codes, format_func=labels.get,
                                index=codes.index(wanted) if wanted in codes else 0)
    st.session_state[SELECTED_PRODUCT_KEY] = product_code

    detail = call_service("GET", f"/queue/{product_code}")
    reviews = call_service("GET", "/reviews", query={"status_value": config.REVIEW_STATUS_OPTIONS})
    review = next((row for row in reviews if row["product_code"] == product_code), None)
    summary, profile = detail["summary"], detail["profile"]

    if review is not None and review["status"] == config.REVIEW_STATUS_PENDING:
        st.success(f"{review['review_id']} is waiting for a Supply Planning Manager, "
                   f"recommending {review['recommended_decision']}.",
                   icon=":material/hourglass_top:")
    badges = [_badge(profile[config.PRODUCT_CATEGORY_COLUMN], "gray")]
    badges += [_badge(f"{decision} open", config.DECISION_BADGE_COLOURS[decision])
               for decision in summary["decisions_open"]]
    badges.append(_badge(f"Queue position {detail['queue_position']} of {len(queue)}", "gray"))
    if review is not None:
        badges.append(_badge(config.REVIEW_STATUS_LABELS[review["status"]],
                             config.STATUS_BADGE_COLOURS[review["status"]]))
    st.markdown(" ".join(badges))
    st.caption(f"{profile[config.PACK_DESCRIPTION_COLUMN]}, "
               f"{profile[config.UNITS_PER_CASE_COLUMN]} units to a case and "
               f"{profile[config.CASES_PER_PALLET_COLUMN]} cases to a pallet. Short because: "
               f"{profile[config.SHORTAGE_REASON_COLUMN]}. Divided under "
               f"{detail['policy_rules']['document_name']}.")

    columns = st.columns(5)
    columns[0].metric(":material/package_2: Cases this week", f"{summary['cases_available']:,}",
                      border=True)
    columns[1].metric(":material/shopping_cart: Ordered", f"{summary['cases_ordered']:,}",
                      border=True)
    columns[2].metric(":material/checklist: Customers may have",
                      f"{summary['cases_eligible']:,}", border=True)
    columns[3].metric(":material/percent: Cover", _cover_text(summary["cover_percent"]),
                      border=True)
    columns[4].metric(":material/trending_down: Shortfall next week",
                      f"{detail['forecast_shortfall']:,}", border=True)

    division, customers, orders, start = st.tabs(
        ["Division", "Customers and rules", "Orders and emails", "Review"])
    with division:
        _division_tab(detail)
    with customers:
        _customers_tab(detail)
    with orders:
        _orders_and_emails_tab(detail)
    with start:
        _review_tab(product_code, review)