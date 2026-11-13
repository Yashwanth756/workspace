"""Recalls the last few exchanges of a person's own conversation."""
from config import config
from persistence.models import ChatTurn
from persistence.repository import fetch_chat_turns

FIRST_QUESTION_TEXT = "This is the first question."


def recall_conversation(session, username, question_text):
    """Recall the person's recent exchanges as history and as search text.

    Only this person's own most recent turns are read, never anybody else's. The
    history shows them oldest first; an allocation status answer is not replayed,
    since nothing from the allocation records is ever put in front of the model.
    The search text is the question asked just before, when there is one, then
    the new question.
    """
    recent = fetch_chat_turns(session, ChatTurn.username == username,
                              config.CHAT_HISTORY_TURN_LIMIT)
    oldest_first = list(reversed(recent))
    if not oldest_first:
        return {"history_text": FIRST_QUESTION_TEXT, "search_text": question_text}
    exchanges = []
    for turn in oldest_first:
        answer = (config.CHAT_STATUS_ANSWER_IN_HISTORY
                  if turn["question_type"] == config.CHAT_TYPE_ALLOCATION_STATUS
                  else turn["answer_text"])
        exchanges.append(f"Question: {turn['question_text']}\nAnswer: {answer}")
    return {"history_text": "\n\n".join(exchanges),
            "search_text": oldest_first[-1]["question_text"] + "\n" + question_text}