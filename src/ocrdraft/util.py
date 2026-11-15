"""Small helpers shared by the local and GPU commands."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, file=sys.stderr, flush=True)


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def write_atomic(path: Path, obj) -> None:
    """Write JSON so a crash never leaves a half-written file that looks finished."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def jsonl_ids(path: Path) -> set[str]:
    """The `id` of every row in a JSONL file (empty if the file does not exist yet)."""
    if not path.exists():
        return set()
    return {json.loads(line)["id"] for line in path.read_text().splitlines() if line}
