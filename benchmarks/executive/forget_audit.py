"""Does ``Memory.forget`` leave the forgotten content anywhere?

``forget`` promises that the content is not served again: the memories,
what was stored from the same message, the intentions and episodes formed
from it, the gists that carry it (redacted), and the backends' cache
entries. Its tests check each of those through the API. This audit checks
the promise from the outside, on a store large enough to be tiered:

The forget call passes what recall finds of the secret by id and the
person's own words for it (``texts``), which reach what no memory holds any
more (a fact folded into a gist, or dropped by it, whose intention or episode
remains).

1. Recall, ``due`` and ``episodes`` probed with the secret's own
   words, before and after a restart (the write-ahead log must not bring it
   back).
2. Every file under ``db_path`` scanned for the secret's bytes, after
   ``close()`` (so everything is flushed) and again after a reopen.

Free: a fake embedder, a sentence-splitting extractor, a model-free gist
summarizer and a fake chat client for the intention tracker; no keys.

    python benchmarks/executive/forget_audit.py [--fillers 5000] [--keep DIR]
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from types import SimpleNamespace

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tsm import Memory  # noqa: E402
from tsm.gist import ExtractiveGistSummarizer  # noqa: E402
from tsm.intentions import LLMIntentionTracker  # noqa: E402
from tsm.tests.test_memory import FakeEmbedder, FakeExtractor  # noqa: E402

# Words that occur nowhere else, so a byte match is the secret and nothing else.
SECRET = "My gym locker code is 409173 and the spare key is under the quokkaberry pot."
TOKENS = ["quokkaberry", "409173"]
NOTE = "Locker 409173 opens with the quokkaberry key."
OTHER = "I like green tea in the morning."
FORGET_WORDS = ["gym locker code", "quokkaberry"]

TOPICS = ["garden", "football", "recipes", "trains", "chess", "cycling", "painting", "jazz",
          "astronomy", "pottery", "sailing", "birds", "coffee", "hiking", "baking", "films"]


def filler(i):
    topic = TOPICS[i % len(TOPICS)]
    return f"Note {i} about {topic}: the {topic} club meets on day {i % 28 + 1} at hall {i % 9}."


def fake_tracker_client():
    """Creates an intention for every new message that asks to be reminded,
    with the message's own words (as a model would)."""

    def create(**kwargs):
        user = kwargs["messages"][-1]["content"]
        new = user.split("NEW MESSAGES:\n", 1)[1].splitlines()
        ops = [{"op": "create", "message": k, "what": line.split("] ", 1)[1][:120],
                "due_from": "2024-03-08", "due_until": "2024-03-08", "situation": None}
               for k, line in enumerate(new, 1) if "remind" in line.lower()]
        reply = json.dumps({"ops": ops})
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def open_memory(path, config):
    kwargs = dict(db_path=path, embedder=FakeEmbedder(), extractor=FakeExtractor(),
                  intentions=True, experience=True)
    if config == "messages":
        kwargs.update(keep_messages=True, intention_tracker=LLMIntentionTracker(
            client=fake_tracker_client(), cache_dir=os.path.join(path, "tsm_cache")))
    else:  # facts folded into gists under a per-user budget
        kwargs.update(keep_messages=False, max_user_tokens=400,
                      gist_summarizer=ExtractiveGistSummarizer(max_tokens=200))
    return Memory(**kwargs)


def scan(path):
    """Files under ``path`` holding a secret token, in UTF-8 or UTF-16."""
    hits = {}
    needles = [(t, enc, t.encode(enc)) for t in TOKENS for enc in ("utf-8", "utf-16-le")]
    for root, _dirs, files in os.walk(path):
        for name in files:
            full = os.path.join(root, name)
            try:
                with open(full, "rb") as f:
                    data = f.read()
            except OSError:
                continue
            found = sorted({t for t, _enc, b in needles if b in data})
            if found:
                hits[os.path.relpath(full, path)] = found
    return hits


def probe(mem):
    """What the API still serves that carries a secret token."""
    out = []
    for query in (SECRET, "gym locker code", "quokkaberry", "spare key"):
        for r in mem.recall(query, user_id="u", top_k=20):
            if any(t in (r.get("context") or r["text"]) for t in TOKENS):
                out.append(("recall", r["role"], r["text"][:80]))
    for item in mem.intentions("u"):
        if any(t in item["text"] for t in TOKENS):
            out.append(("intention", item.get("status"), item["text"][:80]))
    for item in mem.episodes("open the gym locker", user_id="u", k=10):
        if any(t in json.dumps(item) for t in TOKENS):
            out.append(("episode", "", str(item.get("text", ""))[:80]))
    return sorted(set(out))


def run(config, fillers, keep):
    path = keep or tempfile.mkdtemp(prefix=f"tsm_forget_{config}_")
    if keep:
        os.makedirs(path, exist_ok=True)
    print(f"\n=== {config}: {fillers} other messages, store at {path}")
    mem = open_memory(path, config)
    batch = []
    for i in range(fillers):
        batch.append({"role": "user", "content": filler(i), "timestamp": "2024-03-01"})
        if len(batch) == 200:
            mem.add(batch, user_id="u")
            batch = []
        if i == fillers // 2:
            if batch:
                mem.add(batch, user_id="u")
                batch = []
            mem.add([{"role": "user", "timestamp": "2024-03-01",
                      "content": SECRET + " Remind me to move the quokkaberry pot on Friday."}],
                    user_id="u")
    if batch:
        mem.add(batch, user_id="u")
    mem.add([{"role": "user", "content": OTHER, "timestamp": "2024-03-02"}], user_id="u")
    secret_ids = [r["id"] for r in mem.recall(SECRET, user_id="u", top_k=20)
                  if "409173" in r["text"] and not r["role"].startswith("summary")]
    mem.feedback("open the gym locker", "success", secret_ids[:1], user_id="u", note=NOTE)
    mem.consolidate()
    if config == "gists":
        # The secret may have been folded into a gist already: forget what is left of it.
        secret_ids = [r["id"] for r in mem.recall(SECRET, user_id="u", top_k=20)
                      if "409173" in r["text"] and not r["role"].startswith("summary")]
    before = probe(mem)
    print(f"records: {len(mem.engine.scope_ids('u'))}; served before forget: {len(before)}")
    for row in before:
        print("   ", row)
    # What the person asks to forget, in their words: the app passes the
    # memories recall finds (by id) and the words themselves, for what no
    # memory holds any more (a fact folded into a gist or dropped by it).
    report = mem.forget(secret_ids[:1], user_id="u", texts=FORGET_WORDS)
    print("forget:", {k: (len(v) if isinstance(v, list) else v) for k, v in report.items()})
    after = probe(mem)
    print(f"served after forget: {after or 'nothing'}")
    kept = [r for r in mem.recall(OTHER, user_id="u", top_k=5) if "green tea" in r["text"]]
    print(f"an unrelated memory still served: {bool(kept)}")
    mem.close()
    on_disk = scan(path)
    print(f"files holding a secret token after close: {len(on_disk)}")
    for name, toks in sorted(on_disk.items()):
        print(f"    {name}: {', '.join(toks)}")
    mem = open_memory(path, config)
    reopened = probe(mem)
    print(f"served after reopen: {reopened or 'nothing'}")
    mem.consolidate()
    mem.close()
    on_disk2 = scan(path)
    print(f"files holding a secret token after reopen + consolidate + close: {len(on_disk2)}")
    for name, toks in sorted(on_disk2.items()):
        print(f"    {name}: {', '.join(toks)}")
    if not keep:
        shutil.rmtree(path, ignore_errors=True)
    return {"config": config, "served_before": len(before), "served_after": len(after),
            "served_after_reopen": len(reopened), "verified": report["verified"],
            "files_after_close": on_disk, "files_after_reopen": on_disk2}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fillers", type=int, default=5000,
                    help="other messages (over 4,096 records makes the store tiered)")
    ap.add_argument("--configs", default="messages,gists")
    ap.add_argument("--keep", help="build the stores under this directory and keep them")
    ap.add_argument("--json", help="write the summary here")
    args = ap.parse_args()
    out = []
    for config in args.configs.split(","):
        keep = os.path.join(args.keep, config) if args.keep else None
        out.append(run(config, args.fillers, keep))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
