"""Literal per-share rows for model context, without inferred financial periods."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
import re

from .indexer import Chunk
from .quantities import finite_number
from .reasoner import _read_value

_UNSIGNED = r"(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
_VALUE = rf"(?:\(\s*{_UNSIGNED}\s*\)|[+-]?{_UNSIGNED})"
_CELL = re.compile(rf"\s*\$?\s*(?P<value>{_VALUE})(?![\w.,])")
_LABEL = re.compile(
    r"(?P<label>diluted\s+(?:earnings|income|loss)\s+per\s+(?:common\s+)?share|"
    r"earnings\s*(?:\(loss\))?\s+per\s+(?:common\s+)?share"
    r"(?:\s+of\s+common\s+stock)?\s*[-—–:]?\s*(?:diluted|assuming\s+dilution))"
    r"(?:\s*\((?:[a-z]|\d+)\))?",
    re.I,
)
_CAPTION = re.compile(
    r"earnings\s*(?:\(loss\))?\s+per\s+(?:common\s+)?share\s*:?\s*"
    rf"Basic(?:\s*\$?\s*{_VALUE}){{2,4}}\s*(?P<label>Diluted)\b",
    re.I,
)
_MONTH = r"[A-Za-z]+"
_MONTHS = {
    alias: i
    for i, m in enumerate(
        (
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ),
        1,
    )
    for alias in (m, m[:3])
}


def _columns(prefix: str, count: int) -> list[str | None]:
    """Bind only a complete, unambiguous two-column quarter heading."""
    unknown = [None] * count
    if count != 2:
        return unknown
    starts = list(re.finditer(r"(?:three months|quarter)\s+ended\s+", prefix, re.I))
    if not starts:
        return unknown
    header = prefix[starts[-1].end() :]
    if re.search(r"(?:six|nine|twelve)\s+months|year\s+ended", header, re.I):
        return unknown
    month_day = rf"(?P<m1>{_MONTH})\s+(?P<d1>\d{{1,2}}),?\s+"
    forms = (
        month_day
        + rf"(?P<y1>20\d{{2}})\s+(?P<m2>{_MONTH})\s+(?P<d2>\d{{1,2}}),?\s+(?P<y2>20\d{{2}})\b",
        month_day
        + rf"(?P<m2>{_MONTH})\s+(?P<d2>\d{{1,2}}),?\s+(?:\([^()]*\)\s*)?(?P<y1>20\d{{2}})\s+(?P<y2>20\d{{2}})\b",
        month_day + r"(?P<y1>20\d{2})\s+(?P<y2>20\d{2})\b",
    )
    for form in forms:
        match = re.match(form, header, re.I)
        if match:
            if re.match(r"\s+20\d{2}\b", header[match.end() :]):
                return unknown
            fields = match.groupdict()
            try:
                dates = [
                    date(
                        int(fields[f"y{i}"]),
                        _MONTHS[fields.get(f"m{i}", fields["m1"]).lower()],
                        int(fields.get(f"d{i}", fields["d1"])),
                    ).isoformat()
                    for i in (1, 2)
                ]
            except (KeyError, ValueError):
                return unknown
            return dates if dates[0] != dates[1] else unknown
    return unknown


def eps_table_context(chunk: Chunk) -> list[dict]:
    """Read literal EPS rows and left-to-right arithmetic; unknown dates stay null.

    Abbreviated EPS headings are deliberately excluded: a clipped 'diluted EPS'
    can be the tail of a net-income numerator label, with no way to tell locally.
    """
    text = chunk.text.replace("\u200b", " ")
    result = []
    for match in sorted(
        [*_LABEL.finditer(text), *_CAPTION.finditer(text)], key=lambda m: m.start()
    ):
        prefix = text[: match.start("label")]
        if re.search(
            r"(?:net\s+income|shares|numerator|denominator)[^.!?\n]{0,80}(?:for|calculate|calculation\s+of)\s*$",
            prefix[-100:],
            re.I,
        ):
            continue
        position = match.end()
        cells = []
        while len(cells) < 5:
            cell = _CELL.match(text, position)
            if not cell:
                break
            raw = cell["value"]
            value = _read_value(raw)
            if not finite_number(value):
                cells = []
                break
            cells.append(
                {
                    "text": chunk.text[cell.start("value") : cell.end("value")],
                    "value": value,
                    "span_start": chunk.span_start + cell.start("value"),
                    "span_end": chunk.span_start + cell.end("value"),
                }
            )
            position = cell.end()
        if len(cells) not in (2, 4) or not all("." in c["text"] for c in cells):
            continue
        if re.match(
            r"\s*(?:[$(+\-]?\s*(?:\d|\.)|NaN\b|Inf\b|[—–])", text[position:], re.I
        ):
            continue
        difference = float(
            Decimal(str(cells[0]["value"])) - Decimal(str(cells[1]["value"]))
        )
        if not finite_number(difference):
            continue
        periods = _columns(prefix, len(cells))
        for cell, period in zip(cells, periods):
            cell["period_end"] = period
        result.append(
            {
                "label": chunk.text[match.start("label") : match.end("label")],
                "label_span": [
                    chunk.span_start + match.start("label"),
                    chunk.span_start + match.end("label"),
                ],
                "source_values_left_to_right": cells,
                "first_minus_second": difference,
                "first_greater_than_second": cells[0]["value"] > cells[1]["value"],
            }
        )
    return result
