"""The order the nodes run in, the gate, the branch and the checkpoint."""
import os
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from agentic import nodes
from agentic.state import AllocationReviewState
from config import config

REVIEWING_STEPS = [
    ("load_shortage_context", nodes.load_shortage_context_node),
    ("plan_allocation_review", nodes.plan_allocation_review_node),
    ("analyse_demand", nodes.analyse_demand_node),
    ("review_customer_requests", nodes.review_customer_requests_node),
    ("compute_allocation", nodes.compute_allocation_node),
    ("compose_allocation_recommendation", nodes.compose_allocation_recommendation_node),
    ("request_allocation_decision", nodes.request_allocation_decision_node),
]


def build_workflow(checkpoint_path):
    """Assemble the ten nodes into a workflow that can be run, paused and resumed."""
    builder = StateGraph(AllocationReviewState)
    for node_name, node_function in REVIEWING_STEPS:
        builder.add_node(node_name, node_function)
    builder.add_node("draft_allocation_notice", nodes.draft_allocation_notice_node)
    builder.add_node("draft_sales_briefing", nodes.draft_sales_briefing_node)
    builder.add_node("record_allocation_outcome", nodes.record_allocation_outcome_node)

    builder.add_edge(START, REVIEWING_STEPS[0][0])
    for (source, _), (target, _) in zip(REVIEWING_STEPS, REVIEWING_STEPS[1:]):
        builder.add_edge(source, target)
    builder.add_conditional_edges(
        "request_allocation_decision", nodes.route_allocation_decision,
        {config.DECISION_RELEASE: "draft_allocation_notice",
         config.DECISION_RELEASE_WITH_RESERVE: "draft_allocation_notice",
         config.DECISION_ESCALATE: "draft_sales_briefing"})
    builder.add_edge("draft_allocation_notice", "draft_sales_briefing")
    builder.add_edge("draft_sales_briefing", "record_allocation_outcome")
    builder.add_edge("record_allocation_outcome", END)

    folder = os.path.dirname(checkpoint_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    connection = sqlite3.connect(checkpoint_path, check_same_thread=False)
    return builder.compile(checkpointer=SqliteSaver(connection))