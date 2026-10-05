"""Concept extraction for the memory graph.

Cheap, model-free tagging of a fact with the concepts that link it to other
memories (the engine also runs its own statistical extraction; these tags are
merged with it at insert time).
"""

import re
from typing import List

# Common single-word sentence starters that are capitalized for syntactic
# reasons rather than because they are proper nouns. Multi-word capitalized
# spans are always kept (they are almost always named entities).
SENTENCE_START_WORDS = frozenset({
    "the", "a", "an", "i", "it", "he", "she", "they", "we", "you",
    "this", "that", "these", "those", "there", "here", "what", "which",
    "when", "where", "why", "how", "if", "but", "and", "or", "so",
    "because", "although", "however", "therefore", "moreover", "furthermore",
    "actually", "basically", "honestly", "hopefully", "unfortunately",
    "fortunately", "interestingly", "surprisingly", "obviously", "clearly",
    "sure", "yes", "no", "maybe", "ok", "okay", "right", "wrong",
})

# Stop words used to filter content-word extraction.
STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare",
    "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
    "from", "as", "into", "through", "during", "before", "after", "above",
    "below", "between", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "each", "few",
    "more", "most", "other", "some", "such", "only", "own", "same", "than",
    "too", "very", "just", "and", "but", "if", "or", "because", "until",
    "while", "this", "that", "these", "those", "me", "my", "myself", "our",
    "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
    "him", "his", "himself", "her", "hers", "herself", "its", "itself",
    "them", "their", "theirs", "themselves", "what", "which", "who", "whom",
    "whose", "whoever", "whomever", "whatever", "whichever", "also",
    "about", "any", "both", "either", "neither", "nor", "not", "out",
    "over", "off", "down", "up", "now", "still", "even", "well", "back",
    "away", "around", "along", "since", "though", "unless", "whether",
})


def extract_concepts(text: str) -> List[str]:
    """Extract salient concepts from a fact for the memory graph.

    Multi-strategy: capitalized phrases (proper nouns), hyphenated compounds,
    then content words (4+ chars, not stop words). Returns at most 15
    deduplicated lowercase concepts, most salient first.
    """
    concepts: List[str] = []
    for m in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text):
        if len(m.split()) > 1 or m.lower() not in SENTENCE_START_WORDS:
            concepts.append(m.lower())
    concepts.extend(re.findall(r"\b[a-z]+(?:-[a-z]+)+\b", text.lower()))
    for w in re.findall(r"\b[a-zA-Z]{4,}\b", text.lower()):
        if w not in STOP_WORDS:
            concepts.append(w)

    seen = set()
    unique: List[str] = []
    for c in concepts:
        c = c.strip()
        if c and c not in seen and len(c) > 2:
            seen.add(c)
            unique.append(c)
    return unique[:15]
