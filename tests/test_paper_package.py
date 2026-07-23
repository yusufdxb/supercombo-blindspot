"""Regression tests for the submission package proof boundary."""

import numpy as np

from scripts import build_pdf, verify_paper


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


def test_author_and_anonymous_sources_validate() -> None:
    source = build_pdf.SOURCE.read_text(encoding="utf-8")
    body = build_pdf.insert_figures(build_pdf.strip_source_header(source))
    references = build_pdf.render_references(
        build_pdf.parse_bibliography(build_pdf.BIB.read_text(encoding="utf-8"))
    )
    build_pdf.validate_source(build_pdf.metadata(False) + body + references, False)
    build_pdf.validate_source(build_pdf.metadata(True) + body + references, True)
