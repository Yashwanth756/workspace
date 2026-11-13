"""Settles what kind of question was asked and answers it from the documents the
chat may read, in one language model call, refusing when they do not ground an
answer. It knows nothing about any account's or product's allocation."""
import re

from agentic import agents
from config import config
from knowledge.vector_index import read_document_chunks, search_documents

REPLY_LABELS = ["QUESTION TYPE", "ACCOUNT", "PRODUCT CODE", "ANSWER"]
MINIMUM_OVERLAP_CHARACTERS = 10

KIND_MEANINGS = {
    config.CHAT_TYPE_POLICY_RULES: (
        "what a category's policy says: the weeks of shipments behind usual demand, the cap "
        "on an order, the reserve, the protected customers, the advice to customers and what "
        "the sales team is told"),
    config.CHAT_TYPE_ALLOCATION_RULES: (
        "how allocation works: account codes and quantities in each route's notation, an "
        "order sent twice, usual demand and the silent week, what a customer may have, the "
        "division and the largest remainder, the decisions and who takes them, the shortage "
        "queue"),
    config.CHAT_TYPE_REQUEST_HANDLING: (
        "what the request guides say about a customer's email: how a request shows up, what "
        "the planner checks, the evidence to ask for, what to tell the customer and when to "
        "escalate"),
    config.CHAT_TYPE_ALLOCATION_STATUS: (
        "how many cases one account or one product is getting, whether its allocation is "
        "under review, released or escalated, or anything else about one account's or one "
        "product's allocation"),
    config.CHAT_TYPE_OUT_OF_SCOPE: "anything the documents do not answer",
}


def _undecorated(text):
    """The text less the decoration at either end."""
    return re.sub(config.RESPONSE_DECORATION_EDGE_PATTERN, "", str(text).strip())


def read_account_code(account_text):
    """The account in canonical form where one route's notation reads it, whatever its
    case; otherwise None."""
    text = _undecorated(account_text)
    for notation in config.ACCOUNT_CODE_NOTATIONS.values():
        found = re.fullmatch(notation, text, flags=re.IGNORECASE)
        if found is not None:
            return config.CANONICAL_ACCOUNT_CODE_FORMAT.format(
                account_number=int(found.group(config.ACCOUNT_NUMBER_PART)))
    return None


def _read_whole(passages):
    """A document's passages joined back into its text, the overlap each one repeats
    from the one before written once."""
    text = ""
    for passage in passages:
        if not text:
            text = passage
            continue
        overlap = 0
        for size in range(min(len(text), len(passage), config.CHUNK_OVERLAP_CHARACTERS),
                          MINIMUM_OVERLAP_CHARACTERS - 1, -1):
            if text.endswith(passage[:size]):
                overlap = size
                break
        text += passage[overlap:] if overlap else "\n" + passage
    return text


def _found_passages(collection, question_text, search_text):
    """The passages the question and the search text find, each once, closest first."""
    by_id = {}
    for found in search_documents(collection, [question_text, search_text]):
        for passage in found:
            held = by_id.get(passage["chunk_id"])
            if held is None or passage["distance"] < held["distance"]:
                by_id[passage["chunk_id"]] = passage
    return sorted(by_id.values(), key=lambda passage: passage["distance"])


def _build_prompt(collection, question_text, conversation, passages):
    policies = []
    for category, document_name in config.CATEGORY_POLICY_DOCUMENTS.items():
        policies.append(f"POLICY: {category} ({document_name})\n"
                        + _read_whole(read_document_chunks(collection, document_name)))
    found = [f"[{passage['document_name']}] {passage['chunk_text']}" for passage in passages]
    kinds = "\n".join(f"- {kind}: {KIND_MEANINGS[kind]}" for kind in config.CHAT_TYPE_OPTIONS)
    labels = "\n".join(f"{label}: ..." for label in REPLY_LABELS)
    return "\n\n".join([
        f"You answer questions for the supply allocation desk of {config.ORGANISATION_NAME}. "
        "Answer only from the documents and passages given below, never from general knowledge.",
        "THE FIVE CATEGORY POLICIES, EACH GIVEN WHOLE\n\n" + "\n\n".join(policies),
        "PASSAGES FOUND FOR THE QUESTION, CLOSEST FIRST\n" + "\n\n".join(found),
        "THE CONVERSATION SO FAR\n" + conversation["history_text"],
        f"THE QUESTION\n{question_text}",
        "The kinds of question are:\n" + kinds,
        "A question that leaves out its subject continues the last exchange. A delivery "
        f"postcode has been withheld and shown as {config.CHAT_POSTCODE_WITHHELD_TEXT}; never "
        "write a postcode. A policy's rule, whether its weeks of history, its cap, its "
        "protected customers or its reserve, is quoted only from that category's own policy, "
        "and always with the policy named. You know nothing about any account's or any "
        "product's allocation: such a question is the kind "
        f"{config.CHAT_TYPE_ALLOCATION_STATUS}, and you write {config.CHAT_NO_VALUE_REPLY} as "
        "its ANSWER. A question the documents do not answer is "
        f"{config.CHAT_TYPE_OUT_OF_SCOPE}, with {config.CHAT_NO_VALUE_REPLY} as its ANSWER.",
        "Reply with exactly these labelled fields, each starting on its own line as the label, "
        "a colon and a space. QUESTION TYPE is one of the kinds above, on one line. ACCOUNT is "
        "the account as the question writes it, and PRODUCT CODE the product code it names, "
        f"each on one line, with {config.CHAT_NO_VALUE_REPLY} where the question names none.\n"
        + labels,
    ])


def _single_line(value, label):
    if "\n" in value:
        raise ValueError(f"{config.AGENT_NAME_ALLOCATION_ASSISTANT}: {label} runs over more "
                         f"than one line")
    return value


def _read_question_type(text):
    kind = re.sub(r"\s+", " ", _undecorated(_single_line(text, "QUESTION TYPE"))).upper()
    if kind not in config.CHAT_TYPE_OPTIONS:
        raise ValueError(f"{config.AGENT_NAME_ALLOCATION_ASSISTANT}: {text!r} is not one of "
                         f"the kinds {config.CHAT_TYPE_OPTIONS}")
    return kind


def _read_account(text):
    written = _undecorated(_single_line(text, "ACCOUNT"))
    if written.lower() == config.CHAT_NO_VALUE_REPLY:
        return None
    return read_account_code(written) or written


def _read_product(text):
    written = _undecorated(_single_line(text, "PRODUCT CODE"))
    if written.lower() == config.CHAT_NO_VALUE_REPLY:
        return None
    upper = written.upper()
    return upper if re.fullmatch(config.PRODUCT_CODE_PATTERN, upper) else None


def answer_from_documents(collection, question_text, conversation):
    """Answer a question from the documents the chat may read, in one model call."""
    passages = _found_passages(collection, question_text, conversation["search_text"])
    if not passages:
        raise ValueError("The knowledge base found no passage for the question")
    prompt = _build_prompt(collection, question_text, conversation, passages)
    reply = agents.call_language_model(prompt, config.AGENT_NAME_ALLOCATION_ASSISTANT)
    fields = agents.parse_labelled_response(reply["response_text"], REPLY_LABELS,
                                            config.AGENT_NAME_ALLOCATION_ASSISTANT)
    question_type = _read_question_type(fields["QUESTION TYPE"])
    account_code = _read_account(fields["ACCOUNT"])
    product_code = _read_product(fields["PRODUCT CODE"])
    answer = fields["ANSWER"]
    if (question_type in config.CHAT_ANSWERED_TYPES
            and _undecorated(answer).lower() == config.CHAT_NO_VALUE_REPLY):
        raise ValueError(f"{config.AGENT_NAME_ALLOCATION_ASSISTANT}: an answerable question "
                         f"was given {config.CHAT_NO_VALUE_REPLY} as its ANSWER")

    closest_distance = float(passages[0]["distance"])
    answered = (question_type in config.CHAT_ANSWERED_TYPES
                and closest_distance <= config.CHAT_MAX_RELEVANCE_DISTANCE)
    source_documents = []
    if answered:
        if question_type == config.CHAT_TYPE_POLICY_RULES:
            source_documents = list(config.CATEGORY_POLICY_DOCUMENTS.values())
        for passage in passages:
            if passage["document_name"] not in source_documents:
                source_documents.append(passage["document_name"])
    elif question_type != config.CHAT_TYPE_ALLOCATION_STATUS:
        question_type = config.CHAT_TYPE_OUT_OF_SCOPE
        answer = config.CHAT_REFUSAL_MESSAGE
    return {"question_type": question_type, "account_code": account_code,
            "product_code": product_code, "answer": answer,
            "source_documents": source_documents, "closest_distance": closest_distance,
            "was_refused": not answered and question_type != config.CHAT_TYPE_ALLOCATION_STATUS}