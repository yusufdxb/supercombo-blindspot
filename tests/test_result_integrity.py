"""Integrity regression test: no committed report/*_results.md file may
contain the literal tokens 'nan', 'NaN', or 'None' on any line that also
contains a number or a verdict keyword.

Rationale: these tokens indicate a missing crossing or uninitialised
variable that was formatted as a string rather than caught explicitly.
Code-fence / example lines (```-delimited blocks) are skipped to avoid
false positives from prose that legitimately quotes the old broken output.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPORT_DIR = Path(__file__).resolve().parents[1] / "report"

# Tokens whose presence signals a broken result.
# None inside a Python slice literal (e.g. slice(5992, None)) is intentional
# technical prose and must not be flagged; strip those patterns before checking.
BAD_TOKENS = re.compile(r"\bnan\b|\bNaN\b|\bNone\b")
_SLICE_NONE = re.compile(r"slice\([^)]*\bNone\b[^)]*\)")

# A line "has a number" if it contains at least one digit after stripping
# markdown formatting.  This catches table rows, verdict sentences, etc.
# We also flag lines containing explicit verdict keywords even without digits.
VERDICT_KEYWORDS = re.compile(
    r"\bcliff\b|\bgradient\b|\bVERDICT\b|\bCOLLAPSED\b|\balive\b",
    re.IGNORECASE,
)
HAS_NUMBER = re.compile(r"\d")


def _numeric_or_verdict_line(line: str) -> bool:
    stripped = line.strip()
    return bool(HAS_NUMBER.search(stripped) or VERDICT_KEYWORDS.search(stripped))


def _result_files() -> list[Path]:
    return sorted(REPORT_DIR.glob("*_results.md"))


@pytest.mark.parametrize("md_file", _result_files(), ids=lambda p: p.name)
def test_no_nan_in_result_file(md_file: Path) -> None:
    """Fail if any numeric/verdict line in this report file contains nan/NaN/None."""
    text = md_file.read_text(encoding="utf-8")
    lines = text.splitlines()

    in_code_fence = False
    bad_lines: list[tuple[int, str]] = []

    for lineno, line in enumerate(lines, start=1):
        # Track code-fence boundaries; skip content inside fences.
        if line.strip().startswith("```"):
            in_code_fence = not in_code_fence
            continue
        if in_code_fence:
            continue

        # Strip intentional Python slice literals before checking for bad tokens
        sanitised = _SLICE_NONE.sub("slice(...)", line)
        if _numeric_or_verdict_line(sanitised) and BAD_TOKENS.search(sanitised):
            bad_lines.append((lineno, line.rstrip()))

    if bad_lines:
        detail = "\n".join(f"  line {n}: {l}" for n, l in bad_lines)
        pytest.fail(
            f"{md_file.name}: found nan/NaN/None on {len(bad_lines)} "
            f"numeric/verdict line(s):\n{detail}"
        )
