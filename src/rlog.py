"""Minimal rlog reader using pycapnp + cereal/log.capnp schema directly.

Avoids needing the full openpilot package installed. Loads schema once at
import; iterates Event entries from a bz2-compressed rlog file.
"""

from __future__ import annotations

import bz2
from pathlib import Path
from typing import Iterator

import capnp

CAPNP_DIR = Path(__file__).resolve().parents[1] / "references" / "openpilot-v0.9.7" / "cereal_pkg"

# Load schema (must include imports' search paths)
capnp.remove_import_hook()
_log_schema = capnp.load(str(CAPNP_DIR / "log.capnp"), imports=[str(CAPNP_DIR)])
Event = _log_schema.Event


def iter_events(path: Path | str) -> Iterator:
    """Yield each Event from an rlog.bz2 (or uncompressed)."""
    path = Path(path)
    with open(path, "rb") as f:
        data = f.read()
    if path.suffix == ".bz2" or data.startswith(b"BZh"):
        data = bz2.decompress(data)
    yield from Event.read_multiple_bytes(data)


def first_event_of(path: Path | str, which: str):
    for ev in iter_events(path):
        try:
            if ev.which() == which:
                return ev
        except capnp.KjException:
            continue
    return None
