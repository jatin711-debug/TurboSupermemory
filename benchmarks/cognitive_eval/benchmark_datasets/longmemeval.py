"""LongMemEval dataset loader.

Three sources, one shape:

- the HuggingFace parquet copy (``MemoryAsModality/LongMemEval``), which is
  the benchmark's *oracle* variant: each question comes with its evidence
  sessions only (about two sessions and 7k tokens of history);
- the official files (``xiaowu0162/longmemeval-cleaned``): ``oracle`` (the
  same questions and sessions, plus the question date and a ``has_answer``
  flag on every evidence turn), ``s`` (about 40 sessions and 115k tokens per
  question) and ``m`` (about 500 sessions);
- a small JSON fixture.

Every session carries the date it took place. Sessions are kept in file
order, which is NOT always chronological (34 of the 133 temporal-reasoning
questions of the oracle file list them out of order); ``chronological``
gives the order a memory system would have received them in.
"""
import json
import re
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, Union

OFFICIAL_DIR = "longmemeval_official"
OFFICIAL_FILES = {
    "oracle": "longmemeval_oracle.json",
    "s": "longmemeval_s_cleaned.json",
    "m": "longmemeval_m_cleaned.json",
}
_DATE_HEADER = re.compile(r"\[Date:\s*([^\]]+)\]")
_DATE = re.compile(r"(\d{4})/(\d{1,2})/(\d{1,2})(?:\s*\(\w+\))?(?:\s+(\d{1,2}):(\d{2}))?")


@dataclass
class Message:
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[float] = None  # position in the file (kept for older runners)
    session: int = 0          # which session of the conversation, in file order
    date: str = ""            # that session's date as given, e.g. "2023/05/20 (Sat) 11:19"
    has_answer: bool = False  # official files: the turn holds evidence for the question


@dataclass
class Query:
    query_id: str
    query_text: str
    answer_text: str
    message_index: int  # Which message in the conversation this query refers to
    question_type: str = ""   # e.g. 'knowledge-update', 'temporal-reasoning', ...
    is_abstention: bool = False
    question_date: str = ""   # official files: when the question is asked
    answer_sessions: Tuple[int, ...] = ()  # official files: the evidence sessions (file order)


@dataclass
class Conversation:
    conv_id: str
    messages: List[Message]
    queries: List[Query]


def parse_date(text: str) -> Optional[datetime]:
    """A LongMemEval date ("2023/05/20 (Sat) 11:19") as a datetime, or None."""
    match = _DATE.search(text or "")
    if not match:
        return None
    year, month, day, hour, minute = match.groups()
    try:
        return datetime(int(year), int(month), int(day), int(hour or 0), int(minute or 0))
    except ValueError:
        return None


def chronological(messages: Sequence[Message]) -> List[Message]:
    """The messages with their sessions in the order they took place.

    Turns keep their order within a session. When any session has no
    readable date the file order is returned unchanged.
    """
    when = {}
    for m in messages:
        if m.session not in when:
            when[m.session] = parse_date(m.date)
    if not when or any(d is None for d in when.values()):
        return list(messages)
    first = {}
    for position, m in enumerate(messages):
        first.setdefault(m.session, position)
    order = sorted(when, key=lambda session: (when[session], first[session]))
    rank = {session: i for i, session in enumerate(order)}
    return sorted(messages, key=lambda m: rank[m.session])  # stable: turns stay in order


def _clean(text: str) -> str:
    """A turn as the parquet copy holds it: lines stripped, blank lines dropped.

    The official files keep the original whitespace. Cleaning them the same
    way makes the two sources give identical messages (checked on all 10,960
    turns of the oracle variant), so facts extracted from one are cache hits
    for the other.
    """
    return "\n".join(line.strip() for line in text.split("\n") if line.strip()).strip()


def load_from_json(path: Union[str, Path]) -> List[Conversation]:
    """Load LongMemEval data from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    conversations = []
    for item in data:
        messages = [
            Message(
                role=m.get("role", "user"),
                content=m["content"],
                timestamp=m.get("timestamp"),
            )
            for m in item["messages"]
        ]

        queries = [
            Query(
                query_id=q["query_id"],
                query_text=q["query_text"],
                answer_text=q["answer_text"],
                message_index=q["message_index"],
                question_type=q.get("question_type", ""),
                is_abstention=q.get("is_abstention", False),
            )
            for q in item.get("queries", [])
        ]

        conversations.append(
            Conversation(
                conv_id=item["conv_id"],
                messages=messages,
                queries=queries,
            )
        )

    return conversations


def load_from_parquet(path: Union[str, Path]) -> List[Conversation]:
    """Load LongMemEval data from HuggingFace parquet file.
    
    The parquet format has columns:
    - id: conversation ID
    - question_id: unique question ID
    - question: the query text
    - answer: the expected answer
    - question_type: e.g. 'temporal-reasoning', 'single-hop', 'multi-hop'
    - is_abstention: whether this is an abstention question
    - documents: array of conversation documents/messages
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required to load parquet files. "
            "Install with: pip install pandas pyarrow"
        )

    df = pd.read_parquet(path)
    conversations = []

    for _, row in df.iterrows():
        conv_id = str(row.get("id", f"conv_{len(conversations)}"))
        docs = row.get("documents", [])
        if not hasattr(docs, '__iter__') or isinstance(docs, str):
            docs = []
        messages = parse_documents(docs)

        # Create a single query for this row
        query = Query(
            query_id=str(row.get("question_id", f"q_{len(conversations)}")),
            query_text=str(row.get("question", "")),
            answer_text=str(row.get("answer", "")),
            message_index=len(messages) - 1 if messages else 0,
            question_type=str(row.get("question_type", "")),
            is_abstention=bool(row.get("is_abstention", False)),
        )

        conversations.append(
            Conversation(
                conv_id=conv_id,
                messages=messages,
                queries=[query],
            )
        )

    return conversations


def parse_documents(docs: Iterable[str]) -> List[Message]:
    """Messages of a parquet row: one document per session, each starting
    with a ``[Date: ...]`` line followed by ``User:`` / ``Assistant:`` turns."""
    messages: List[Message] = []
    for session, doc in enumerate(docs):
        if not isinstance(doc, str):
            continue
        date = ""
        current_role = None
        current_content: List[str] = []

        def close():
            if current_role and current_content:
                messages.append(Message(
                    role=current_role,
                    content='\n'.join(current_content).strip(),
                    timestamp=float(len(messages)),
                    session=session,
                    date=date,
                ))

        for line in doc.split('\n'):
            line = line.strip()
            if not line:
                continue
            if line.startswith("User:"):
                close()
                current_role = "user"
                current_content = [line[5:].strip()]
            elif line.startswith("Assistant:"):
                close()
                current_role = "assistant"
                current_content = [line[10:].strip()]
            elif line.startswith("[Date:"):
                header = _DATE_HEADER.match(line)
                if header and not date:
                    date = header.group(1).strip()
            elif current_role:
                # Continuation of current message
                current_content.append(line)
        close()
    return messages


def iter_json_array(path: Union[str, Path], chunk: int = 1 << 20) -> Iterator[dict]:
    """The items of a JSON file that is one big array, one at a time.

    The ``s`` file is 277 MB and the ``m`` file 2.7 GB; loading either whole
    costs several times that in memory.
    """
    decoder = json.JSONDecoder()
    with open(path, "r", encoding="utf-8") as f:
        buffer, started = "", False
        while True:
            data = f.read(chunk)
            buffer += data
            if not started:
                opening = buffer.find("[")
                if opening < 0:
                    if not data:
                        return
                    continue
                buffer, started = buffer[opening + 1:], True
            while True:
                buffer = buffer.lstrip(" \r\n\t,")
                if not buffer or buffer.startswith("]"):
                    break
                try:
                    item, end = decoder.raw_decode(buffer)
                except ValueError:
                    break  # the item is not complete yet
                yield item
                buffer = buffer[end:]
            if not data:
                return


def conversation_from_official(item: dict) -> Conversation:
    """One question of an official file, in the shape the parquet loader gives."""
    messages: List[Message] = []
    dates = item.get("haystack_dates") or []
    for session, turns in enumerate(item.get("haystack_sessions") or []):
        date = dates[session] if session < len(dates) else ""
        for turn in turns:
            content = _clean(turn.get("content") or "")
            if not content:
                continue
            messages.append(Message(
                role=turn.get("role", "user"),
                content=content,
                timestamp=float(len(messages)),
                session=session,
                date=date,
                has_answer=bool(turn.get("has_answer", False)),
            ))
    question_id = str(item.get("question_id", ""))
    evidence = set(item.get("answer_session_ids") or [])
    session_ids = item.get("haystack_session_ids") or []
    query = Query(
        query_id=question_id,
        query_text=str(item.get("question", "")),
        answer_text=str(item.get("answer", "")),
        message_index=len(messages) - 1 if messages else 0,
        question_type=str(item.get("question_type", "")),
        is_abstention=question_id.endswith("_abs"),
        question_date=str(item.get("question_date", "")),
        answer_sessions=tuple(i for i, sid in enumerate(session_ids) if sid in evidence),
    )
    return Conversation(conv_id=question_id, messages=messages, queries=[query])


def load_official(path: Union[str, Path],
                  question_ids: Optional[Iterable[str]] = None) -> List[Conversation]:
    """Questions of an official file, optionally only the given ids (in the
    order given)."""
    wanted = None if question_ids is None else list(question_ids)
    keep = None if wanted is None else set(wanted)
    found = {}
    for item in iter_json_array(path):
        question_id = str(item.get("question_id", ""))
        if keep is None or question_id in keep:
            found[question_id] = conversation_from_official(item)
            if keep is not None and len(found) == len(keep):
                break
    if wanted is None:
        return list(found.values())
    return [found[q] for q in wanted if q in found]


def attach_official_labels(conversations: Sequence[Conversation],
                           oracle_path: Union[str, Path]) -> int:
    """Copy what the official oracle file adds (evidence turns, evidence
    sessions, the question date) onto conversations loaded from the parquet
    copy. Returns how many conversations were labelled; one whose messages do
    not line up with the official file is left as it is."""
    official = {c.queries[0].query_id: c for c in load_official(oracle_path)}
    labelled = 0
    for conv in conversations:
        if not conv.queries:
            continue
        source = official.get(conv.queries[0].query_id)
        if source is None or len(source.messages) != len(conv.messages):
            continue
        if any(a.content != b.content or a.role != b.role
               for a, b in zip(conv.messages, source.messages)):
            continue
        for mine, theirs in zip(conv.messages, source.messages):
            mine.has_answer = theirs.has_answer
            mine.date = mine.date or theirs.date
        for query in conv.queries:
            query.question_date = source.queries[0].question_date
            query.answer_sessions = source.queries[0].answer_sessions
        labelled += 1
    return labelled


def load_longmemeval(data_dir: Optional[Union[str, Path]] = None,
                     variant: Optional[str] = None,
                     question_ids: Optional[Iterable[str]] = None) -> List[Conversation]:
    """Load LongMemEval dataset.

    With no ``variant``: the parquet copy (the oracle variant), with the
    official oracle file's evidence labels and question dates attached when
    that file has been downloaded; falls back to the JSON test data.
    ``variant`` ``"oracle"``, ``"s"`` or ``"m"`` reads the official file
    instead (``download.py --official s``), optionally only ``question_ids``.
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data"
    else:
        data_dir = Path(data_dir)

    if variant is not None:
        if variant not in OFFICIAL_FILES:
            raise ValueError(f"unknown LongMemEval variant {variant!r}: use one of "
                             f"{', '.join(OFFICIAL_FILES)}")
        path = data_dir / OFFICIAL_DIR / OFFICIAL_FILES[variant]
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run 'python benchmarks/cognitive_eval/benchmark_datasets/"
                f"download.py --official {variant}' first.")
        print(f"Loading LongMemEval ({variant}) from {path.name}...")
        return load_official(path, question_ids)

    # Try parquet files first (HuggingFace format)
    parquet_files = list((data_dir / "longmemeval").glob("*.parquet"))
    if parquet_files:
        print(f"Loading LongMemEval from {len(parquet_files)} parquet file(s)...")
        all_conversations = []
        for pf in sorted(parquet_files):
            all_conversations.extend(load_from_parquet(pf))
        oracle = data_dir / OFFICIAL_DIR / OFFICIAL_FILES["oracle"]
        if oracle.exists():
            attach_official_labels(all_conversations, oracle)
        if question_ids is not None:
            by_id = {c.queries[0].query_id: c for c in all_conversations if c.queries}
            return [by_id[q] for q in question_ids if q in by_id]
        return all_conversations

    # Fall back to JSON test data
    json_path = data_dir / "test_longmemeval.json"
    if json_path.exists():
        print(f"Loading LongMemEval test data from {json_path}")
        return load_from_json(json_path)

    raise FileNotFoundError(
        f"No LongMemEval data found in {data_dir}. "
        "Run 'python benchmarks/cognitive_eval/benchmark_datasets/download.py' first."
    )


if __name__ == "__main__":
    # Test loading
    convs = load_longmemeval()
    print(f"Loaded {len(convs)} conversations")
    if convs:
        print(f"  First conv: {len(convs[0].messages)} messages, {len(convs[0].queries)} queries")
