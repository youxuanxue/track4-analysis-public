"""Strict dated pipe tables and arithmetic over source observations only.

Column names are literal, missing cells stay missing, and dates must be unique
and increasing. Summaries describe the selected history, never future outcomes
or calibrated prediction intervals.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from .indexer import Chunk, dated_on_or_before
from .schema import target_name


def pipe_cells(text: str, max_chars: int) -> list[str]:
    if len(text) > max_chars or "|" not in text:
        return []
    try:
        cells = next(
            csv.reader([text], delimiter="|", skipinitialspace=True, strict=True)
        )
    except csv.Error:
        return []
    if text.strip().startswith("|") and text.strip().endswith("|"):
        cells = cells[1:-1]
    return [cell.strip() for cell in cells]


def dated_header(text: str, max_chars: int) -> list[str]:
    cells = pipe_cells(text, max_chars)
    if (
        len(cells) < 2
        or not all(cells)
        or len(set(c.lower() for c in cells)) != len(cells)
        or not re.fullmatch(r"(?:[a-z]+_)*date", cells[0], re.I)
    ):
        return []
    return cells


@dataclass(frozen=True)
class TableRow:
    cells: tuple[str, ...]
    start: int
    end: int


@dataclass(frozen=True)
class DatedTable:
    header: tuple[str, ...]
    start: int
    rows: tuple[TableRow, ...]


def dated_tables(text: str, cutoff: str, max_chars: int) -> list[DatedTable]:
    """Read contiguous tables; never bridge blank lines or malformed rows."""
    lines = list(re.finditer(r"[^\n]*(?:\n|$)", text))
    result = []
    for i, line in enumerate(lines):
        header = dated_header(line.group().rstrip("\r\n"), max_chars)
        if not header:
            continue
        rows = []
        last_date = ""
        invalid_order = False
        for row in lines[i + 1 :]:
            raw = row.group().rstrip("\r\n")
            cells = pipe_cells(raw, max_chars)
            if len(cells) != len(header) or not dated_on_or_before(cells[0], cutoff):
                break
            if cells[0] <= last_date:
                invalid_order = True
                break
            last_date = cells[0]
            rows.append(TableRow(tuple(cells), row.start(), row.start() + len(raw)))
        if rows and not invalid_order:
            result.append(
                DatedTable(
                    tuple(header),
                    line.start(),
                    tuple(rows),
                )
            )
    return result


def _number(raw: str) -> Decimal | None:
    if not re.fullmatch(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", raw):
        return None
    try:
        value = Decimal(raw.replace(",", ""))
        return value if value.is_finite() and abs(value) < Decimal("1e100") else None
    except InvalidOperation:
        return None


def percent_declaration(text: str, column: str) -> re.Match | None:
    """Recognize a standalone declaration, not an arbitrary nearby unit mention."""
    return re.search(
        r"(?:^|[.:]\s+)(?P<declaration>"
        + re.escape(column)
        + r" observations in percent)(?=[.,;]|$)",
        text,
        re.I,
    )


def table_summaries(chunk: Chunk, cutoff: str, columns: list[str]) -> list[dict]:
    """Keep literal column identity and source row offsets with each computation."""
    result = []
    for table in dated_tables(chunk.text, cutoff, len(chunk.text)):
        for column in columns:
            if column not in table.header:
                continue
            position = table.header.index(column)
            observations = [(row, _number(row.cells[position])) for row in table.rows]
            # Missing values are not zero and cannot be silently crossed for a delta.
            if any(value is None for _, value in observations):
                continue
            values = [value for _, value in observations if value is not None]
            first, last = table.rows[0], table.rows[-1]
            prefix = chunk.text[: table.start].rstrip("\r\n")
            previous = prefix.rfind("\n") + 1
            declaration = percent_declaration(prefix[previous:], column)
            unit = "percent" if column.endswith("_%") or declaration else None
            record = {
                "column": column,
                "date_column": table.header[0],
                "count": len(values),
                "first_date": first.cells[0],
                "last_date": last.cells[0],
                "first": float(values[0]),
                "last": float(values[-1]),
                "historical_min": float(min(values)),
                "historical_max": float(max(values)),
                "last_minus_first": float(values[-1] - values[0]),
                "source_span": [
                    chunk.span_start + table.start,
                    chunk.span_start + last.end,
                ],
                "last_row_span": [
                    chunk.span_start + last.start,
                    chunk.span_start + last.end,
                ],
                "unit": unit,
            }
            if len(values) > 1:
                record["last_minus_previous"] = float(values[-1] - values[-2])
                record["previous_date"] = table.rows[-2].cells[0]
            if unit == "percent":
                record["last_minus_first_bps"] = float((values[-1] - values[0]) * 100)
                if len(values) > 1:
                    record["last_minus_previous_bps"] = float(
                        (values[-1] - values[-2]) * 100
                    )
            result.append(record)
    return result


def summary_columns(task: dict, entity: dict, chunk: Chunk) -> list[str]:
    series = entity.get("series_fred") or entity.get("series_id")
    if series:
        return [str(series)]
    name = target_name(task)

    def normalize(value: str) -> str:
        value = re.sub(r"[_\W]+", " ", value.lower()).strip()
        return re.sub(r"\s+(?:ratio|rank|direction)$", "", value)

    return list(
        dict.fromkeys(
            column
            for table in dated_tables(chunk.text, task["cutoff_date"], len(chunk.text))
            for column in table.header[1:]
            if normalize(column) == normalize(name)
        )
    )
