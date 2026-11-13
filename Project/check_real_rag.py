"""Run on YOUR machine, with the real all-MiniLM-L6-v2 model, after run_knowledge_indexing.py.

Shows what the three retrieval readers return for every category and request type,
and the nearest distance a policy question finds, so the retrieval test can be judged.
Usage:  python3 check_real_rag.py
"""
import re

from config import config
from knowledge.vector_index import find_nearest_passage, open_vector_index, search_documents
from tools.retrieval_tools import (read_customer_advice, read_planner_checks,
                                   read_sales_briefing_guidance)


def heading(passage):
    found = re.match(r"\s*([0-9]{1,2}\. [A-Z][^.]{0,50})", passage)
    return found.group(1) if found else passage[:50]


collection = open_vector_index(config.VECTOR_STORE_DIRECTORY)
all_distinct = True
for reader, subjects in [(read_planner_checks, config.REQUEST_GUIDE_DOCUMENTS),
                         (read_customer_advice, config.CATEGORY_POLICY_DOCUMENTS),
                         (read_sales_briefing_guidance, config.CATEGORY_POLICY_DOCUMENTS)]:
    print(f"\n{reader.__name__}")
    seen = {}
    for subject in subjects:
        passage = reader(collection, subject)
        seen[subject] = passage
        print(f"  {subject:<24} -> {heading(passage)}")
    distinct = len(set(seen.values())) == len(subjects)
    all_distinct = all_distinct and distinct
    print(f"  every subject has its own passage: {distinct}")

question = "Which customers does the hand hygiene policy protect?"
found = search_documents(collection, [question])[0]
print(f"\nclosest passage to {question!r}")
print(f"  {found[0]['document_name']}  distance {found[0]['distance']:.4f}  "
      f"(limit {config.CHAT_MAX_RELEVANCE_DISTANCE}; within limit: "
      f"{found[0]['distance'] <= config.CHAT_MAX_RELEVANCE_DISTANCE})")
print(f"\nALL THREE READERS DISTINCT: {all_distinct}")