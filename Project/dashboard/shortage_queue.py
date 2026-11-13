"""The shortage queue page."""
import pandas as pd
import streamlit as st

from config import config
from dashboard.api_client import call_service

SELECTED_PRODUCT_KEY = "selected_product"


def render_shortage_queue_page():
    """List the products on allocation, largest forecast shortfall first, by category."""
    st.title(config.PAGE_QUEUE)
    category = st.pills("Category", config.QUEUE_CATEGORY_OPTIONS, selection_mode="single",
                        default=config.QUEUE_CATEGORY_ALL, key="queue_category")
    category = category or config.QUEUE_CATEGORY_ALL
    rows = call_service("GET", "/queue", query={"product_category": category})

    with_release = sum(config.DECISION_RELEASE in row["decisions_open"] for row in rows)
    escalate_alone = sum(row["decisions_open"] == [config.DECISION_ESCALATE] for row in rows)
    first, second, third, fourth = st.columns(4)
    first.metric(":material/inventory_2: Products on allocation", len(rows), border=True)
    second.metric(":material/local_shipping: With RELEASE open", with_release, border=True)
    third.metric(":material/priority_high: With ESCALATE alone open", escalate_alone,
                 border=True)
    fourth.metric(":material/package_2: Cases available this week",
                  f"{sum(row['cases_available_this_week'] for row in rows):,}", border=True)
    if not rows:
        st.info("No product on allocation is in this category.")
        return

    table = pd.DataFrame({
        "#": [row["queue_position"] for row in rows],
        "Product": [row["product_code"] for row in rows],
        "Name": [row["product_name"] for row in rows],
        "Category": [row["product_category"] for row in rows],
        "This week": [row["cases_available_this_week"] for row in rows],
        "Ordered": [row["cases_ordered"] for row in rows],
        "May have": [row["cases_eligible"] for row in rows],
        "Cover %": [row["cover_percent"] for row in rows],
        "Protected in full": ["Yes" if row["protected_served_in_full"] else "No"
                              for row in rows],
        "Decisions open": [", ".join(row["decisions_open"]) for row in rows],
        "Shortfall next week": [row["forecast_shortfall"] for row in rows],
    })
    event = st.dataframe(
        table, hide_index=True, width="stretch", on_select="rerun",
        selection_mode="single-row", key=f"queue_table_{category}",
        column_config={
            "#": st.column_config.NumberColumn(format="%d"),
            "This week": st.column_config.NumberColumn(format="localized"),
            "Ordered": st.column_config.NumberColumn(format="localized"),
            "May have": st.column_config.NumberColumn(format="localized"),
            "Cover %": st.column_config.NumberColumn(format="%.2f"),
            "Shortfall next week": st.column_config.NumberColumn(format="localized"),
        })
    chosen = event.selection.rows
    if chosen:
        st.session_state[SELECTED_PRODUCT_KEY] = rows[chosen[0]]["product_code"]
        st.caption(f"{rows[chosen[0]]['product_code']} is the product the Allocation detail "
                   f"page opens on.")