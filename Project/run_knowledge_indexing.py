"""Builds the knowledge base and reads every category's rules back from its own policy.

Run from the Project folder:  python3 run_knowledge_indexing.py
"""
import re

from config import config
from knowledge.chunk_splitting import build_chunk_records, split_documents_into_sections
from knowledge.pdf_extraction import extract_text_from_directory
from knowledge.policy_reading import read_allocation_policy
from knowledge.vector_index import build_vector_index, open_vector_index


def describe_rules(policy_rules):
    cap_written = re.search(config.CAP_SENTENCE_PATTERN,
                            policy_rules["rule_sentences"]["cap"]).group(1)
    channels = policy_rules["protected_channels"]
    protected = "nobody" if not channels else config.PROTECTED_CHANNEL_SEPARATOR.join(channels)
    reserve = ("no reserve" if policy_rules["reserve_percent"] is None
               else f"a reserve of up to {policy_rules['reserve_percent']} percent")
    return (f"usual demand over {policy_rules['history_weeks']} weeks, an order capped at "
            f"{cap_written} times it, {protected} protected, {reserve}")


def main():
    print("building the allocation knowledge index")
    documents = extract_text_from_directory(config.DOCS_DIRECTORY)
    sections = split_documents_into_sections(documents)
    chunks = build_chunk_records(sections)
    stored = build_vector_index(chunks, config.VECTOR_STORE_DIRECTORY)
    print(f"  documents read:  {len(documents)}")
    print(f"  sections found:  {len(sections)}")
    print(f"  chunks stored:   {stored}")
    print(f"  store location:  {config.VECTOR_STORE_DIRECTORY}")
    print()
    print("rules read from each product category's own allocation policy")
    collection = open_vector_index(config.VECTOR_STORE_DIRECTORY)
    for product_category in config.CATEGORY_POLICY_DOCUMENTS:
        policy_rules = read_allocation_policy(collection, product_category)
        print(f"  {product_category:<22} {policy_rules['document_name']}")
        print(f"      {describe_rules(policy_rules)}")


if __name__ == "__main__":
    main()