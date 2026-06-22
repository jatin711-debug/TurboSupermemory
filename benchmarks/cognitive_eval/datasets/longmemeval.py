"""LongMemEval dataset loader.

Supports both HuggingFace parquet files and JSON test data.
"""
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Union


@dataclass
class Message:
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[float] = None


@dataclass
class Query:
    query_id: str
    query_text: str
    answer_text: str
    message_index: int  # Which message in the conversation this query refers to


@dataclass
class Conversation:
    conv_id: str
    messages: List[Message]
    queries: List[Query]


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
        
        # Parse documents into messages
        messages = []
        docs = row.get("documents", [])
        if hasattr(docs, '__iter__') and not isinstance(docs, str):
            for i, doc in enumerate(docs):
                if isinstance(doc, str):
                    # Each document may contain multiple turns separated by "\nUser:" and "\nAssistant:"
                    # Split by lines and parse each turn
                    lines = doc.split('\n')
                    current_role = None
                    current_content = []
                    
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                            
                        if line.startswith("User:"):
                            # Save previous message if exists
                            if current_role and current_content:
                                messages.append(Message(
                                    role=current_role,
                                    content='\n'.join(current_content).strip(),
                                    timestamp=float(len(messages)),
                                ))
                            current_role = "user"
                            current_content = [line[5:].strip()]
                        elif line.startswith("Assistant:"):
                            # Save previous message if exists
                            if current_role and current_content:
                                messages.append(Message(
                                    role=current_role,
                                    content='\n'.join(current_content).strip(),
                                    timestamp=float(len(messages)),
                                ))
                            current_role = "assistant"
                            current_content = [line[10:].strip()]
                        elif line.startswith("[Date:"):
                            # Date header - skip or use as metadata
                            continue
                        else:
                            # Continuation of current message
                            if current_role:
                                current_content.append(line)
                    
                    # Don't forget the last message
                    if current_role and current_content:
                        messages.append(Message(
                            role=current_role,
                            content='\n'.join(current_content).strip(),
                            timestamp=float(len(messages)),
                        ))
        
        # Create a single query for this row
        query = Query(
            query_id=str(row.get("question_id", f"q_{len(conversations)}")),
            query_text=str(row.get("question", "")),
            answer_text=str(row.get("answer", "")),
            message_index=len(messages) - 1 if messages else 0,
        )

        conversations.append(
            Conversation(
                conv_id=conv_id,
                messages=messages,
                queries=[query],
            )
        )

    return conversations


def load_longmemeval(data_dir: Optional[Union[str, Path]] = None) -> List[Conversation]:
    """Load LongMemEval dataset.

    Tries to load from parquet files first, falls back to JSON test data.
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data"
    else:
        data_dir = Path(data_dir)

    # Try parquet files first (HuggingFace format)
    parquet_files = list((data_dir / "longmemeval").glob("*.parquet"))
    if parquet_files:
        print(f"Loading LongMemEval from {len(parquet_files)} parquet file(s)...")
        all_conversations = []
        for pf in sorted(parquet_files):
            all_conversations.extend(load_from_parquet(pf))
        return all_conversations

    # Fall back to JSON test data
    json_path = data_dir / "test_longmemeval.json"
    if json_path.exists():
        print(f"Loading LongMemEval test data from {json_path}")
        return load_from_json(json_path)

    raise FileNotFoundError(
        f"No LongMemEval data found in {data_dir}. "
        "Run 'python datasets/download.py' first."
    )


if __name__ == "__main__":
    # Test loading
    convs = load_longmemeval()
    print(f"Loaded {len(convs)} conversations")
    if convs:
        print(f"  First conv: {len(convs[0].messages)} messages, {len(convs[0].queries)} queries")
