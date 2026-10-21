"""What a model is asked when it answers from recalled memories.

``recall()`` decides which memories reach a model. What the model then does
with them decides the answer, and a plain "answer from these snippets" prompt
loses questions the memories do answer: a total over several conversations,
a value that changed, anything that needs arithmetic on dates. In the judged
runs of 2026-10-09, ten of twenty-five wrong answers had all of their
evidence in the context. On the same 600-token contexts of 115 questions the
plain prompt answered 92 and this one 99 (2026-10-12; the prompt was written
against those questions).

``answer_prompt`` builds the chat messages for the answering model:

- the memories in the order they were said, each with its date in front,
- an instruction to list the ones that matter, settle what changed and what
  has to be counted or computed, and only then answer,
- the date the question is asked, when the caller has one ("a month ago"
  has no meaning without it).

``final_answer`` takes the answer out of the model's reply. Nothing here
calls a model or touches the engine.
"""

import math
import re
from typing import Dict, List, Optional, Sequence, Union

ANSWER_MARK = "Answer:"

ANSWER_SYSTEM = (
    "You answer a question about a user from notes taken during their earlier "
    "conversations with an assistant. Each note starts with the date it was said. "
    "Use only the notes.\n"
    "Work in three short steps.\n"
    "1. Relevant notes: the notes that bear on the question, oldest first, each as "
    "its date and a few words (ten at most).\n"
    "2. Reasoning: settle what they add up to.\n"
    "   - When a later note changes something an earlier note said, the later one is "
    "how things are now; the earlier one is how they were before.\n"
    "   - To count or total, name each distinct item or amount once, then add them "
    "up. Different notes can describe the same item.\n"
    "   - A relative time inside a note (\"yesterday\", \"last week\", \"two months "
    "ago\") counts from that note's date. For \"how long ago\" or \"how long "
    "between\", compute from the dates, and from the current date when it is given.\n"
    "   - If the question asks for advice or a recommendation, give it, fitted to "
    "what the notes say about the user (what they own, like, have done or plan), and "
    "say what it is based on.\n"
    f"3. Finish with one line that starts with \"{ANSWER_MARK}\" and states the answer "
    "directly, with the value or name asked for. If the notes do not contain the "
    f"answer, finish with \"{ANSWER_MARK} NO ANSWER\"."
)

# What recall() puts in front of a memory: "[2024-01-15] ", "[2024-01-15 10:00] ",
# "[turn 12] ", or for a gist the span it covers, "[2024-01-05 to 2024-03-02] "
# (ordered by where it starts).
_TAG = re.compile(
    r"^\[(?:(\d{4}-\d{2}-\d{2}(?: \d{2}:\d{2})?)(?: to \d{4}-\d{2}-\d{2})?|turn (\d+))\] ")

Memory = Union[str, Dict]


def _text(memory: Memory) -> str:
    if isinstance(memory, str):
        return memory
    return memory.get("context") or memory.get("text") or ""


def _order_key(memory: Memory, position: int):
    """Dated memories first, by date; then those that only have a turn; then
    the rest (gists) in the order given. Ties keep the order given."""
    match = _TAG.match(_text(memory))
    date, turn = match.groups() if match else (None, None)
    if not match and isinstance(memory, dict):
        # recall(time_tags=False): the fields are still there.
        from .memory import time_tag

        match = _TAG.match(time_tag(memory.get("timestamp"), memory.get("turn_index"),
                                    minutes=True))
        date, turn = match.groups() if match else (None, None)
    if isinstance(memory, dict) and turn is None:
        index = memory.get("turn_index")
        if isinstance(index, (int, float)) and not isinstance(index, bool) and math.isfinite(index):
            turn = index
    order = float(turn) if turn is not None else math.inf
    if date is not None:
        return (0, date, order, position)
    if turn is not None:
        return (1, "", order, position)
    return (2, "", order, position)


def in_order(memories: Sequence[Memory]) -> List[Memory]:
    """Recall results (or their ``"context"`` strings) in the order they were
    said: by date, then by turn within a date."""
    keyed = [(_order_key(memory, position), memory) for position, memory in enumerate(memories)]
    return [memory for _, memory in sorted(keyed, key=lambda pair: pair[0])]


def answer_prompt(question: str, memories: Sequence[Memory],
                  today: Optional[str] = None) -> List[Dict[str, str]]:
    """Chat messages that ask a model to answer ``question`` from ``memories``.

    ``memories`` is what ``recall()`` returned (their ``"context"`` is shown:
    the text with its date, marked when it has since changed), or plain
    strings. ``today`` is the date the question is asked. Take the answer out
    of the model's reply with ``final_answer``. Give the model room for about
    800 tokens: a reply is some 200 on average, but a question that draws on
    many notes needs more, and a reply cut off before its last line has no
    answer in it.
    """
    notes = [text for text in (_text(memory) for memory in in_order(memories)) if text]
    listing = "\n".join(f"{number}. {text}" for number, text in enumerate(notes, 1))
    when = f"Current date: {today}\n" if today else ""
    user = f"Notes:\n{listing or '(none)'}\n\n{when}Question: {question}"
    return [{"role": "system", "content": ANSWER_SYSTEM}, {"role": "user", "content": user}]


def final_answer(reply: Optional[str]) -> str:
    """The answer in a reply to ``answer_prompt``: what follows the last
    ``Answer:``. A reply without one is returned whole (a model that skipped
    the steps still answered)."""
    text = (reply or "").strip()
    cut = text.lower().rfind(ANSWER_MARK.lower())
    return text[cut + len(ANSWER_MARK):].strip() if cut >= 0 else text
