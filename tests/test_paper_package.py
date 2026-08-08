"""Regression tests for the submission package proof boundary."""

import numpy as np
import pytest

from scripts import build_pdf_ieee, verify_paper


def test_metrics_npz_writer_is_byte_deterministic(tmp_path):
    from scripts.build_metrics import save_npz_deterministic

    arrays = {"b": np.arange(4, dtype=np.float32), "a": np.array(3, dtype=np.int32)}
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    save_npz_deterministic(first, arrays)
    save_npz_deterministic(second, arrays)
    assert first.read_bytes() == second.read_bytes()


def test_paper_claims_and_artifact_hashes() -> None:
    assert verify_paper.main() == 0


def test_every_manuscript_table_maps_to_a_distinct_float() -> None:
    """The manuscript's pipe tables must each claim exactly one hand-set float.

    Regression: the router previously sent any table without "Bucket" to the E6
    float, so a third table was silently rendered as a duplicate of the E6 one.
    """
    text = build_pdf_ieee.DRAFT.read_text(encoding="utf-8")
    dropped = build_pdf_ieee.drop_markdown_tables(text)
    for token in build_pdf_ieee.TABLE_FLOATS:
        assert dropped.count(token) == 1, f"{token} appeared != 1 time"


def test_unknown_table_raises_instead_of_falling_through() -> None:
    unknown = "intro\n\n| alpha | beta |\n|---|---|\n| 1 | 2 |\n\noutro\n"
    with pytest.raises(ValueError, match="matched 0 table markers"):
        build_pdf_ieee.drop_markdown_tables(unknown)


def test_duplicate_table_raises() -> None:
    row = "| readout | CARLA (raw) |\n|---|---|\n| accel_t0 | 0.0040 |\n"
    with pytest.raises(ValueError):
        build_pdf_ieee.drop_markdown_tables(f"a\n\n{row}\nb\n\n{row}\nc\n")
