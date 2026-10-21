"""Which embedder wrote a database's vectors.

The engine knows a dimension and nothing else: it cannot tell a database
written by one 768-dimensional model from another's. Searching one model's
vectors with another's query vectors returns plausible-looking, wrong
results, with no error. ``tsm.Memory`` therefore records the embedder's
fingerprint next to the engine's files the first time it writes to a
database and refuses to open that database with a different one.

The record is an SDK concern (the engine stays embedder-agnostic), kept in
``tsm_embedder.json`` inside the database directory and replaced atomically.
A database written before this record existed has none: it is adopted by
whichever embedder opens it next.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("tsm.identity")

IDENTITY_FILE = "tsm_embedder.json"


class EmbedderMismatchError(ValueError):
    """The database holds vectors from another embedder than the one given."""


def read_identity(db_path: str) -> Optional[dict]:
    """The recorded identity, or ``None`` when there is none (or it is unreadable)."""
    path = os.path.join(db_path, IDENTITY_FILE)
    try:
        with open(path, encoding="utf-8") as fh:
            record = json.load(fh)
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as e:
        logger.warning("%s is unreadable and ignored (%s)", IDENTITY_FILE, e)
        return None
    if not isinstance(record, dict) or not isinstance(record.get("fingerprint"), str):
        logger.warning("%s has no fingerprint and is ignored", IDENTITY_FILE)
        return None
    return record


def _write_identity(db_path: str, fingerprint: str, dimension: int) -> None:
    from . import __version__

    path = os.path.join(db_path, IDENTITY_FILE)
    tmp = path + ".tmp"
    record = {
        "fingerprint": fingerprint,
        "dimension": int(dimension),
        "written_by": f"tsm {__version__}",
        "written_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def bind_embedder(db_path: str, fingerprint: str, dimension: int, records: int,
                  allow_change: bool = False) -> None:
    """Record ``fingerprint`` as the database's embedder, or check it.

    ``records`` is how many memories the database holds. With none, nothing
    can be in the wrong space, so a different embedder simply takes over.
    Otherwise a different fingerprint raises ``EmbedderMismatchError``,
    unless ``allow_change`` says the caller knows the two are compatible (the
    record is then replaced).
    """
    recorded = read_identity(db_path)
    if recorded is not None and recorded["fingerprint"] != fingerprint:
        if records and not allow_change:
            raise EmbedderMismatchError(
                f"{db_path!r} holds {records} memories embedded by "
                f"{recorded['fingerprint']!r}, but this Memory was given an embedder "
                f"that identifies as {fingerprint!r}. Their vectors are not "
                f"comparable and search would return wrong results without an "
                f"error. Open it with the embedder that wrote it, or re-ingest "
                f"the memories into a new database. Pass "
                f"allow_embedder_change=True only if the two produce the same "
                f"vectors (for example, the same model reached another way).")
        logger.warning("embedder of %r changed from %r to %r (%s)", db_path,
                       recorded["fingerprint"], fingerprint,
                       "database is empty" if not records else "allow_embedder_change")
    elif recorded is not None:
        return
    elif records:
        logger.info("%r has %d memories and no embedder record (written before "
                    "tsm kept one): recording %r", db_path, records, fingerprint)
    _write_identity(db_path, fingerprint, dimension)
