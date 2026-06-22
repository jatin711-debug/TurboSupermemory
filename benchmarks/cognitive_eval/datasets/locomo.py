"""LoCoMo dataset loader.

Supports both HuggingFace JSONL format and test JSON data.
The LoCoMo-MC10 format is a multiple-choice benchmark with haystack sessions.
"""
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Union


@dataclass
class Session:
    session_id: str
    timestamp: float
    facts: List[str]  # Session summaries/facts


@dataclass
class Query:
    query_id: str
    query_text: str
    answer_text: str
    relevant_session: str  # Which session contains the answer
    query_type: str  # "current" or "past" for temporal reasoning
    choices: List[str]  # Multiple choice options
    correct_choice_index: int


@dataclass
class LoCoMoDataset:
    sessions: List[Session]
    queries: List[Query]


def load_from_json(path: Union[str, Path]) -> LoCoMoDataset:
    """Load LoCoMo test data from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sessions = [
        Session(
            session_id=s["session_id"],
            timestamp=s["timestamp"],
            facts=s["facts"],
        )
        for s in data["sessions"]
    ]

    queries = [
        Query(
            query_id=q["query_id"],
            query_text=q["query_text"],
            answer_text=q["answer_text"],
            relevant_session=q["relevant_session"],
            query_type=q.get("query_type", "current"),
            choices=q.get("choices", []),
            correct_choice_index=q.get("correct_choice_index", 0),
        )
        for q in data["queries"]
    ]

    return LoCoMoDataset(sessions=sessions, queries=queries)


def load_from_locomo_jsonl(path: Union[str, Path]) -> LoCoMoDataset:
    """Load LoCoMo-MC10 data from the official JSONL format.
    
    Each line is a JSON object with:
    - question_id: unique question ID
    - question: the query text
    - question_type: e.g. 'recent', 'distant', 'temporal'
    - answer: correct answer text
    - correct_choice_index: index of correct answer in choices
    - num_choices: number of multiple choice options
    - choices: list of answer choices
    - num_sessions: number of haystack sessions
    - haystack_session_ids: list of session IDs
    - haystack_session_summaries: list of session summaries (facts)
    - haystack_session_datetimes: list of session timestamps
    - haystack_sessions: list of full session texts
    """
    conversations = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                conv = json.loads(line)
                conversations.append(conv)
            except json.JSONDecodeError:
                continue
    
    all_sessions = []
    all_queries = []
    
    for conv in conversations:
        # Extract haystack sessions
        session_ids = conv.get("haystack_session_ids", [])
        session_summaries = conv.get("haystack_session_summaries", [])
        session_datetimes = conv.get("haystack_session_datetimes", [])
        
        # Create sessions for this conversation
        conv_sessions = []
        for i, (sid, summary, dt) in enumerate(zip(session_ids, session_summaries, session_datetimes)):
            session = Session(
                session_id=str(sid),
                timestamp=float(dt) if isinstance(dt, (int, float)) else float(i),
                facts=[summary] if isinstance(summary, str) else summary if isinstance(summary, list) else [],
            )
            conv_sessions.append(session)
            all_sessions.append(session)
        
        # Create query
        question_type = conv.get("question_type", "current")
        # Map question types: 'recent' -> 'current', 'distant' -> 'past'
        if question_type == "recent":
            query_type = "current"
        elif question_type == "distant":
            query_type = "past"
        else:
            query_type = question_type
        
        # Determine relevant session from haystack
        correct_idx = conv.get("correct_choice_index", 0)
        choices = conv.get("choices", [])
        answer = conv.get("answer", choices[correct_idx] if choices and correct_idx < len(choices) else "")
        
        # Find which session contains the answer
        relevant_session = ""
        if conv_sessions:
            # Try to match answer to a session summary
            for sess in conv_sessions:
                if answer.lower() in " ".join(sess.facts).lower():
                    relevant_session = sess.session_id
                    break
            if not relevant_session:
                # Default to first session (most recent in haystack)
                relevant_session = conv_sessions[0].session_id
        
        all_queries.append(
            Query(
                query_id=conv.get("question_id", f"q_{len(all_queries)}"),
                query_text=conv.get("question", ""),
                answer_text=answer,
                relevant_session=relevant_session,
                query_type=query_type,
                choices=choices,
                correct_choice_index=correct_idx,
            )
        )
    
    return LoCoMoDataset(sessions=all_sessions, queries=all_queries)


def load_locomo(data_dir: Optional[Union[str, Path]] = None) -> LoCoMoDataset:
    """Load LoCoMo dataset.

    Tries to load from the official JSONL format first, then falls back to test JSON.
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data"
    else:
        data_dir = Path(data_dir)

    # Try official LoCoMo-MC10 JSONL format first
    locomo_json = data_dir / "locomo" / "locomo_mc10.json"
    if locomo_json.exists():
        print(f"Loading LoCoMo-MC10 from {locomo_json}")
        return load_from_locomo_jsonl(locomo_json)

    # Fall back to JSON test data
    json_path = data_dir / "test_locomo.json"
    if json_path.exists():
        print(f"Loading LoCoMo test data from {json_path}")
        return load_from_json(json_path)

    raise FileNotFoundError(
        f"No LoCoMo data found in {data_dir}. "
        "Run 'python datasets/download.py' first."
    )


if __name__ == "__main__":
    # Test loading
    dataset = load_locomo()
    print(f"Loaded {len(dataset.sessions)} sessions, {len(dataset.queries)} queries")
    if dataset.sessions:
        print(f"  First session: {len(dataset.sessions[0].facts)} facts")
    if dataset.queries:
        print(f"  First query: {dataset.queries[0].query_text[:100]}...")
        print(f"  Query type: {dataset.queries[0].query_type}")
        print(f"  Choices: {len(dataset.queries[0].choices)}")
