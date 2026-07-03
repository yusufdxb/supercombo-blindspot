"""Minimal rlog reader using pycapnp + cereal/log.capnp schema directly.

Avoids needing the full openpilot package installed. `capnp` and the schema
are loaded lazily on first use, not at import time, so importing this module
(e.g. transitively via src.probe_model) doesn't require pycapnp to be
installed. CI's unit-test job deliberately omits pycapnp (see
requirements-ci.txt); only code paths that actually decode an rlog need it.
"""

from __future__ import annotations

import bz2
from pathlib import Path
from typing import Iterator

_capnp = None
_Event = None


def _schema():
    """Load capnp + the Event schema once, on first real use."""
    global _capnp, _Event
    if _Event is None:
        import capnp

        capnp.remove_import_hook()
        capnp_dir = Path(__file__).resolve().parents[1] / "references" / "openpilot-v0.9.7" / "cereal_pkg"
        log_schema = capnp.load(str(capnp_dir / "log.capnp"), imports=[str(capnp_dir)])
        _capnp = capnp
        _Event = log_schema.Event
    return _capnp, _Event


def iter_events(path: Path | str) -> Iterator:
    """Yield each Event from an rlog.bz2 (or uncompressed)."""
    _capnp_mod, event_type = _schema()
    path = Path(path)
    with open(path, "rb") as f:
        data = f.read()
    if path.suffix == ".bz2" or data.startswith(b"BZh"):
        data = bz2.decompress(data)
    yield from event_type.read_multiple_bytes(data)


def first_event_of(path: Path | str, which: str):
    capnp_mod, _event_type = _schema()
    for ev in iter_events(path):
        try:
            if ev.which() == which:
                return ev
        except capnp_mod.KjException:
            continue
    return None
