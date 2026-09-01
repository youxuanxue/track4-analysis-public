"""Evidence-bound extract-then-predict reasoner.

The official NLI hypothesis is built from the SUBMITTED prediction
(``label`` / ``point_forecast`` / interval), never from claim prose. So this
module does the only thing that can pass the gate without a house model:

1. retrieve embargo-safe, entity-relevant windows from the frozen corpus;
2. extract numbers and polarity cues that actually appear in a window;
3. emit a prediction whose tokens are in that window;
4. cite the exact character span of the window.

No ``family`` slug is consulted. Held-out families are unpublished; the
reasoner keys only off ``target.type``, the legal label list, and whatever
fields the entity row itself carries (``consensus_eps``, ``cik``,
``series_id``, …).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from .indexer import Chunk, IndexedCorpus
from .retriever import BM25Index
from .schema import (
    entity_display_name,
    fmt_number,
    interval_level,
    legal_labels,
    target_name,
    target_type,
)

# ---------------------------------------------------------------------------
# Token / number helpers
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r"[a-z0-9%]+")
_NUMBER = re.compile(
    r"(?<![A-Za-z])([+-]?)(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?"
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

# Polarity cues applied ONLY when the matching legal label is on the task.
# Keys are legal labels; values are extra surface forms that support that label.
_LABEL_CUES: dict[str, tuple[str, ...]] = {
    "beat": (
        "beat",
        "beats",
        "beating",
        "exceeded",
        "outperformed",
        "above consensus",
        "record quarter",
        "record",
    ),
    "miss": ("miss", "missed", "shortfall", "below consensus", "fell short"),
    "inline": ("inline", "in line", "in-line", "in line with", "matched consensus"),
    "up": (
        "revised up",
        "revised higher",
        "revised upward",
        "increase",
        "increased",
        "rose",
        "upward",
    ),
    "down": (
        "revised down",
        "revised lower",
        "revised downward",
        "decrease",
        "decreased",
        "fell",
        "downward",
    ),
    "credit_event": (
        "credit event",
        "going concern",
        "going-concern",
        "ability to continue",
        "substantial doubt",
        "chapter 11",
        "chapter 7",
        "bankruptcy",
        "default",
        "distressed",
        "insolvency",
        "restructuring",
        "accumulated deficit",
        "net loss",
    ),
    "no_event": (
        "no event",
        "no substantial doubt",
        "well capitalized",
        "adequate liquidity",
        "strong liquidity",
        "profitable",
        "net income",
        "generated cash",
    ),
    "positive_reaction": (
        "positive",
        "record",
        "grew",
        "growth",
        "strong",
        "beat",
        "increase",
        "increased",
    ),
    "negative_reaction": (
        "negative",
        "decline",
        "declined",
        "decrease",
        "decreased",
        "loss",
        "weak",
        "miss",
    ),
    "flat": ("flat", "unchanged", "stable", "in line", "similar"),
}

_WINDOW_TARGET = 360
_WINDOW_MAX = 640
_MIN_WINDOW = 40

# Explicit ranges the corpus actually writes ("ranged from 2.32 to 2.67").
_RANGE = re.compile(
    r"(?:ranged from|range|between|from)\s+"
    r"([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:%|percent)?\s+"
    r"(?:to|and|–|—)\s+"
    r"([+-]?\d+(?:,\d{3})*(?:\.\d+)?)",
    flags=re.IGNORECASE,
)
_RANGE_TO = re.compile(
    r"([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:%|percent)?\s+to\s+"
    r"([+-]?\d+(?:,\d{3})*(?:\.\d+)?)",
    flags=re.IGNORECASE,
)
_YIELD_SENT = re.compile(
    r"(?:the\s+)?(\d{1,2})-year treasury yield was ([+-]?\d+(?:\.\d+)?)",
    flags=re.IGNORECASE,
)
_DATE_TOKEN = re.compile(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b")
_MONTH_DAY_YEAR = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2},?\s+\d{4}\b",
    flags=re.IGNORECASE,
)
_DAY_YEAR = re.compile(r"\b\d{1,2},\s+\d{4}\b")
_TENOR_TOKEN = re.compile(r"\b\d{1,2}-years?\b", flags=re.IGNORECASE)


def _is_year(value: float) -> bool:
    return value == int(value) and 1900 <= abs(value) <= 2100


def _mask_non_quantities(text: str) -> str:
    """Blank dates and '10-year' so 10 / 2024 cannot become the forecast."""
    masked = _DATE_TOKEN.sub(" DATE ", text)
    masked = _MONTH_DAY_YEAR.sub(" DATE ", masked)
    masked = _DAY_YEAR.sub(" DATE ", masked)
    return _TENOR_TOKEN.sub(" TENOR ", masked)


def _usable_numbers(numbers: list[float]) -> list[float]:
    """Drop calendar years so 2024 in a date cannot become interval.hi."""
    return [n for n in numbers if not _is_year(n)]


def _parse_float(raw: str) -> float | None:
    try:
        return float(raw.replace(",", ""))
    except ValueError:
        return None


def number_in_text(text: str, value: float) -> bool:
    """True when ``fmt_number(value)`` is a surface substring of ``text``."""
    token = fmt_number(value)
    if token in text:
        return True
    if not token.startswith("-") and f"+{token}" in text:
        return True
    return False


def all_numbers_in_text(text: str, values: Iterable[float]) -> bool:
    return all(number_in_text(text, v) for v in values)


def explicit_ranges(text: str) -> list[tuple[float, float]]:
    """Inclusive (lo, hi) pairs written as ranges in ``text``."""
    found: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for pattern in (_RANGE, _RANGE_TO):
        for match in pattern.finditer(text):
            lo = _parse_float(match.group(1))
            hi = _parse_float(match.group(2))
            if lo is None or hi is None or _is_year(lo) or _is_year(hi):
                continue
            if lo > hi:
                lo, hi = hi, lo
            key = (lo, hi)
            if key in seen:
                continue
            seen.add(key)
            found.append(key)
    return found


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN.findall(text.lower()) if len(t) > 1}


def extract_numbers(text: str) -> list[float]:
    """Parse signed decimals, ignoring commas. Order is left-to-right."""
    out: list[float] = []
    for match in _NUMBER.finditer(text):
        sign, whole, frac = match.group(1), match.group(2).replace(",", ""), match.group(3)
        raw = whole if frac is None else f"{whole}.{frac}"
        try:
            value = float(raw)
        except ValueError:
            continue
        if sign == "-":
            value = -value
        out.append(value)
    return out


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Window:
    doc_id: str
    doc_date: str | None
    span_start: int
    span_end: int
    text: str

    @property
    def key(self) -> tuple[str, int, int]:
        return (self.doc_id, self.span_start, self.span_end)


def _windows_from_text(
    doc_id: str, doc_date: str | None, text: str, base: int = 0
) -> list[Window]:
    """Sentence-ish windows with global offsets into ``text`` (joined-doc convention)."""
    if not text:
        return []
    windows: list[Window] = []
    # Prefer NOTES / revision-note lines — they already bind a name to a number.
    for match in re.finditer(
        r"(?:^|\n)[^\n]*(?:NOTES|first print|revised |bid-to-cover|net position|"
        r"yield was |basis point|diluted earnings|earnings per)[^\n]{10,400}",
        text,
        flags=re.IGNORECASE,
    ):
        start = match.start()
        # skip the leading newline
        if start < len(text) and text[start] == "\n":
            start += 1
        snippet = text[start : match.end()].strip()
        if len(snippet) >= _MIN_WINDOW:
            end = start + len(text[start : match.end()].rstrip())
            windows.append(
                Window(doc_id, doc_date, base + start, base + end, text[start:end])
            )

    # One window per line / bullet so a NOTES line is citable on its own.
    line_start = 0
    for raw_line in text.splitlines(keepends=True):
        line = raw_line.rstrip("\n")
        stripped = line.strip()
        if len(stripped) >= _MIN_WINDOW:
            # leading whitespace stays inside the slice so offsets match the doc
            lead = len(line) - len(line.lstrip()) if line else 0
            start = line_start + lead
            end = line_start + lead + len(stripped)
            windows.append(Window(doc_id, doc_date, base + start, base + end, stripped))
        line_start += len(raw_line)

    # Forced slices so a 10-Q that is one giant line still yields citable windows.
    for start in range(0, len(text), _WINDOW_TARGET):
        end = min(len(text), start + _WINDOW_MAX)
        snippet = text[start:end]
        if len(snippet) >= _MIN_WINDOW:
            windows.append(Window(doc_id, doc_date, base + start, base + end, snippet))

    # Sentence windows are exact slices of ``text`` (never re-joined with
    # spaces — a rewritten string will not resolve in the scorer's document).
    parts = _SENTENCE_SPLIT.split(text)
    cursor = 0
    buf_start = 0
    buf_end = 0
    for part in parts:
        idx = text.find(part, cursor)
        if idx < 0:
            idx = cursor
        if buf_end <= buf_start:
            buf_start = idx
        buf_end = idx + len(part)
        cursor = buf_end
        if buf_end - buf_start >= _WINDOW_TARGET:
            end = min(len(text), buf_start + _WINDOW_MAX)
            snippet = text[buf_start:end]
            if len(snippet) >= _MIN_WINDOW:
                windows.append(
                    Window(doc_id, doc_date, base + buf_start, base + end, snippet)
                )
            buf_start = buf_end
    if buf_end - buf_start >= _MIN_WINDOW:
        snippet = text[buf_start:buf_end][:_WINDOW_MAX]
        windows.append(
            Window(
                doc_id,
                doc_date,
                base + buf_start,
                base + buf_start + len(snippet),
                snippet,
            )
        )
    return windows


_STOP_ALIASES = {
    "a", "i", "am", "an", "as", "at", "be", "by", "c", "do", "go", "he",
    "if", "in", "is", "it", "m", "me", "ms", "my", "no", "of", "on", "or",
    "so", "to", "up", "us", "we",
}


def _entity_aliases(entity: dict[str, Any]) -> list[str]:
    aliases: list[str] = []
    for key in (
        "name",
        "entity_id",
        "series_id",
        "series_name",
        "series_fred",
        "tenor",
        "cik",
        "ticker",
    ):
        value = entity.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        token = value.strip()
        # Tickers like WE / C / M are English words and match every 10-K.
        if key in {"entity_id", "ticker"} and token.lower() in _STOP_ALIASES:
            continue
        if key in {"entity_id", "ticker"} and len(token) <= 1:
            continue
        aliases.append(token)
    cik = entity.get("cik")
    if isinstance(cik, str) and cik.isdigit():
        aliases.append(cik.zfill(10))
        aliases.append(cik.lstrip("0") or "0")
    # Maturity phrasing used in rates snapshots ("10-year Treasury yield").
    years = entity.get("maturity_years")
    if isinstance(years, (int, float)) and years > 0:
        n = int(years)
        aliases.append(f"{n}-year")
        aliases.append(f"{n} year")
    return aliases


def _doc_owned_by_entity(doc_id: str, entity: dict[str, Any]) -> bool:
    """True when the document is this row's filing / series / tenor table.

    Shared snapshots (FOMC statement, CPI table) are not 'owned' and are
    scored only via alias hits. This stops PAYEMS from citing a DGORDER table.
    """
    cik = entity.get("cik")
    if isinstance(cik, str) and cik and cik.zfill(10) in doc_id:
        return True
    series = entity.get("series_id") or entity.get("series_fred")
    if isinstance(series, str) and series and re.search(
        rf"(?:^|[_-]){re.escape(series)}(?:[_-]|$)", doc_id, flags=re.I
    ):
        return True
    tenor = entity.get("tenor")
    if isinstance(tenor, str) and tenor:
        digits = re.sub(r"[^0-9]", "", tenor)
        if digits and f"_{digits}Y_" in doc_id.upper().replace("-", "_"):
            return True
    eid = str(entity.get("entity_id") or "")
    if eid and eid in doc_id:
        return True
    return False


def _alias_hit(text: str, aliases: Iterable[str]) -> bool:
    lower = text.lower()
    for alias in sorted(aliases, key=len, reverse=True):
        if len(alias) <= 1:
            continue
        if len(alias) <= 4:
            if re.search(rf"(?<![A-Za-z0-9]){re.escape(alias.lower())}(?![A-Za-z0-9])", lower):
                return True
            continue
        if alias.lower() in lower:
            return True
        compact = re.sub(r"[_\-]+", " ", alias).lower()
        if compact and compact in lower:
            return True
    return False


def collect_windows(
    entity: dict[str, Any],
    corpus: IndexedCorpus,
    index: BM25Index,
    *,
    cutoff: str,
    top_k: int,
) -> list[Window]:
    aliases = _entity_aliases(entity)
    cik = entity.get("cik")
    series = entity.get("series_id") or entity.get("series_fred")
    seen: set[tuple[str, int, int]] = set()
    windows: list[Window] = []

    def _add(window: Window) -> None:
        if window.doc_date is None or window.doc_date > cutoff:
            return
        if window.key in seen:
            return
        if len(window.text) < _MIN_WINDOW:
            return
        if len(window.text) > _WINDOW_MAX:
            # A whole 10-Q as one "window" wins lexical overlap by drowning
            # the hypothesis in noise. Never cite more than _WINDOW_MAX chars.
            trimmed = window.text[:_WINDOW_MAX]
            window = Window(
                window.doc_id,
                window.doc_date,
                window.span_start,
                window.span_start + len(trimmed),
                trimmed,
            )
            if window.key in seen:
                return
        seen.add(window.key)
        windows.append(window)

    # 1. Documents that belong to this entity (CIK / series / tenor in doc_id).
    for doc_id, text in corpus.doc_texts.items():
        date = corpus.doc_dates.get(doc_id)
        if date is None or date > cutoff:
            continue
        owned = _doc_owned_by_entity(doc_id, entity)
        if not owned:
            continue
        # Long filings: keep only neighborhoods of aliases / financial cues.
        if len(text) > 8000:
            for window in _keyword_neighborhoods(doc_id, date, text, aliases):
                _add(window)
        else:
            for window in _windows_from_text(doc_id, date, text):
                _add(window)

    # 2. Short / shared docs (tables, statements, snapshots): alias-hit at the
    # document level, then keep every line. A rates table row is just numbers;
    # the header carries ``DGS30``. Scoring picks the row whose start_yield
    # actually appears.
    for chunk in index.chunks:
        if chunk.doc_date is None or chunk.doc_date > cutoff:
            continue
        if not _alias_hit(chunk.text, aliases) and not _alias_hit(chunk.doc_id, aliases):
            continue
        owned_doc = _doc_owned_by_entity(chunk.doc_id, entity)
        if owned_doc and len(corpus.doc_texts.get(chunk.doc_id, "")) > 8000:
            continue
        if len(chunk.text) <= _WINDOW_MAX:
            _add(
                Window(
                    chunk.doc_id,
                    chunk.doc_date,
                    chunk.span_start,
                    chunk.span_end,
                    chunk.text,
                )
            )
        else:
            for window in _windows_from_text(
                chunk.doc_id, chunk.doc_date, chunk.text, base=chunk.span_start
            ):
                _add(window)

    # 3. BM25 backfill so a unit with no CIK/series still gets entity-relevant text.
    # When the row has a CIK / series / tenor, do not let another issuer's
    # filing win lexical overlap (Apple must not cite Meta's 10-Q).
    owned_only = bool(entity.get("cik") or entity.get("series_id") or entity.get("tenor"))
    query_parts = [str(entity.get(k, "")) for k in ("name", "entity_id", "series_id", "tenor", "sector")]
    query = " ".join(p for p in query_parts if p)
    for scored in index.search(query, max(top_k, 8)):
        chunk = scored.chunk
        if owned_only and not _doc_owned_by_entity(chunk.doc_id, entity):
            continue
        if len(chunk.text) <= _WINDOW_MAX:
            _add(
                Window(
                    chunk.doc_id,
                    chunk.doc_date,
                    chunk.span_start,
                    chunk.span_end,
                    chunk.text,
                )
            )
        else:
            for window in _windows_from_text(
                chunk.doc_id, chunk.doc_date, chunk.text, base=chunk.span_start
            ):
                _add(window)

    return windows


_NEIGHBOR_CUES = (
    "earnings per share",
    "earnings per diluted share",
    "diluted earnings",
    "going concern",
    "going-concern",
    "ability to continue",
    "substantial doubt",
    "chapter 11",
    "accumulated deficit",
    "net loss",
    "net income",
    "liquidity",
    "guidance",
    "record quarter",
    "basis point",
    "bid-to-cover",
    "first print",
    "revised up",
    "revised down",
    "yield was",
    "net position",
)


def _keyword_neighborhoods(
    doc_id: str, doc_date: str | None, text: str, aliases: list[str]
) -> list[Window]:
    needles = [a for a in aliases if len(a) > 2]
    needles.extend(_NEIGHBOR_CUES)
    hits: list[int] = []
    lower = text.lower()
    for needle in needles:
        start = 0
        key = needle.lower()
        while True:
            idx = lower.find(key, start)
            if idx < 0:
                break
            hits.append(idx)
            start = idx + len(key)
            if len(hits) > 40:
                break
    windows: list[Window] = []
    seen: set[int] = set()
    for idx in hits:
        lo = max(0, idx - 180)
        hi = min(len(text), idx + 360)
        # snap to nearby whitespace so we don't cite mid-word
        if lo > 0 and not text[lo].isspace():
            space = text.rfind(" ", max(0, lo - 20), lo + 1)
            if space >= 0:
                lo = space + 1
        if hi < len(text) and not text[hi - 1].isspace():
            space = text.find(" ", hi, min(len(text), hi + 20))
            if space >= 0:
                hi = space
        if lo in seen:
            continue
        seen.add(lo)
        raw = text[lo:hi]
        lead = len(raw) - len(raw.lstrip())
        snippet = raw.strip()
        if len(snippet) >= _MIN_WINDOW:
            start = lo + lead
            windows.append(Window(doc_id, doc_date, start, start + len(snippet), snippet))
    return windows


# ---------------------------------------------------------------------------
# Prediction from a window
# ---------------------------------------------------------------------------


def _label_score(label: str, text: str) -> float:
    lower = text.lower()
    score = 0.0
    needle = label.lower().replace("_", " ")
    # Short labels like "up" / "down" must be words, not substrings of "support".
    if len(needle) <= 4:
        if re.search(rf"(?<![a-z]){re.escape(needle)}(?![a-z])", lower):
            score += 3.0
    elif needle in lower or label.lower() in lower:
        score += 3.0
    for cue in _LABEL_CUES.get(label, ()):
        if cue in lower:
            score += 2.0
    # token overlap of the label itself (credit_event → credit, event)
    for tok in _tokens(label.replace("_", " ")):
        if len(tok) <= 2:
            continue
        if tok in _tokens(lower):
            score += 0.4
    return score


def _numeric_label_from_entity(
    entity: dict[str, Any], numbers: list[float], labels: list[str]
) -> str | None:
    """If the entity row carries a comparison quantity, use it — but only to
    choose among the task's own labels."""
    label_set = set(labels)
    consensus = entity.get("consensus_eps")
    threshold = float(entity.get("threshold_pct") or 0.05)
    if consensus is not None and numbers:
        reported = _closest(numbers, float(consensus), prefer_nearby=True)
        if reported is not None and {"beat", "miss", "inline"} <= label_set:
            if reported > float(consensus) * (1 + threshold):
                return "beat"
            if reported < float(consensus) * (1 - threshold):
                return "miss"
            return "inline"
    prior = entity.get("prior_year_q_eps")
    if prior is not None and numbers and {"up", "down"} <= label_set:
        current = _closest(numbers, float(prior), prefer_nearby=True)
        if current is not None:
            return "up" if current >= float(prior) else "down"
    latest = entity.get("latest_precutoff_estimate")
    if latest is not None and numbers and {"up", "down"} <= label_set:
        # A later vintage above the pre-cutoff estimate is an upward revision.
        later = max(numbers)
        if later > float(latest) * 1.0001:
            return "up"
        if later < float(latest) * 0.9999:
            return "down"
    return None


def _closest(
    numbers: list[float], target: float, *, prefer_nearby: bool = False
) -> float | None:
    if not numbers:
        return None
    if prefer_nearby:
        band = [n for n in numbers if abs(n - target) <= max(2.0, abs(target) * 2)]
        pool = band or numbers
    else:
        pool = numbers
    return min(pool, key=lambda n: (abs(n - target), abs(n)))


def _pick_point(entity: dict[str, Any], numbers: list[float], window_text: str) -> float:
    if not numbers:
        for key in (
            "latest_published_mom_pct",
            "latest_precutoff_estimate",
            "consensus_eps",
            "start_yield_pct",
            "prior_year_q_eps",
            "net_pct_oi_20241022",
            "trailing_4wk_net_change_pct_oi",
            "offering_amount_usd_bn",
        ):
            raw = entity.get(key)
            if isinstance(raw, (int, float)):
                return float(raw)
        return 0.0

    # Prefer a number that also appears on the entity row — the NOTES lines
    # of the public units are written that way on purpose.
    for key in (
        "latest_published_mom_pct",
        "latest_precutoff_estimate",
        "net_pct_oi_20241022",
        "start_yield_pct",
        "consensus_eps",
        "prior_year_q_eps",
        "trailing_4wk_net_change_pct_oi",
    ):
        raw = entity.get(key)
        if not isinstance(raw, (int, float)):
            continue
        exact = key not in {"consensus_eps", "prior_year_q_eps"}
        pool = numbers
        if not exact:
            # Prefer share-like decimals; drop calendar days (1..31) that
            # survived masking and sit closer to 1.50 than $2.18 EPS does.
            decimals = [n for n in numbers if n != int(n) or abs(n) > 31]
            pool = decimals or numbers
        hit = _closest(pool, float(raw), prefer_nearby=not exact)
        if hit is None:
            continue
        if exact and abs(hit - float(raw)) >= 1e-6:
            continue
        if not exact and abs(hit - float(raw)) > max(2.0, abs(float(raw)) * 2):
            continue
        return hit

    lower = window_text.lower()
    recent = re.search(
        r"(?:most recent[^\n]{0,80}?|average[^\n]{0,40}?|first print\s+)([+-]?\d+(?:\.\d+)?)",
        lower,
    )
    if recent:
        try:
            value = float(recent.group(1))
        except ValueError:
            value = None
        if value is not None and any(abs(n - value) < 1e-6 for n in numbers):
            return value
    if "bid-to-cover" in lower:
        covers = [n for n in numbers if 1.2 <= n <= 4.5]
        if covers:
            return covers[-1]
    if "yield was" in lower:
        yields = [n for n in numbers if 0.1 <= abs(n) <= 15.0]
        if yields:
            return yields[0]
    # "most recent" / "first print" / "as of" numbers tend to be the last
    # mentioned quantity in a NOTES line.
    if any(k in lower for k in ("first print", "most recent", "as of the", "drew a")):
        return numbers[-1]
    return numbers[0]


def _same_scale(n: float, point: float) -> bool:
    """Keep interval bounds in the same numeric neighbourhood as the point.

    Open-interest (1.6e6) sitting next to a 51.63% share must not become hi.
    """
    if _is_year(n):
        return False
    if 0.2 <= abs(point) <= 8 and point != int(point):
        # Yields / MoM percents / bid-to-cover — keep other small decimals, drop 30 (maturity).
        return abs(n) <= 15 and (n != int(n) or abs(n) <= 6)
    if abs(point) < 20:
        return abs(n) <= 80
    return abs(n) <= max(abs(point) * 25.0, abs(point) + 50.0)


def _interval_from_numbers(
    numbers: list[float], point: float, *, text: str = ""
) -> tuple[float, float]:
    """Default to a degenerate [point, point]. Widen only for an explicit range.

    The NLI hypothesis always includes ``The 90% prediction interval … is lo to hi``.
    Two unrelated table cells as lo/hi cannot entail that clause; a written
    ``ranged from 2.32 to 2.67`` can.
    """
    ranges = [
        (lo, hi)
        for lo, hi in explicit_ranges(text)
        if _same_scale(lo, point) and _same_scale(hi, point)
    ]
    if ranges:
        # Prefer a range that actually brackets the point (NOTES min/max).
        bracketed = [r for r in ranges if r[0] <= point <= r[1]]
        lo, hi = (bracketed or ranges)[0]
        return lo, hi
    return point, point


def _hypothesis(
    *,
    kind: str | None,
    tname: str,
    subject: str,
    label: str,
    point: float,
    lo: float,
    hi: float,
    level: float,
    rank: int | None = None,
) -> str:
    level_pct = fmt_number(level * 100.0)
    interval_clause = (
        f" The {level_pct}% prediction interval for the {tname} of {subject} "
        f"is {fmt_number(lo)} to {fmt_number(hi)}."
    )
    if kind == "classification":
        head = f"The {tname} of {subject} is {label}."
    elif kind == "regression":
        head = f"The {tname} of {subject} is {fmt_number(point)}."
    else:
        position = f"ranked {rank}" if rank is not None else "ranked"
        head = (
            f"By {tname}, {subject} is {position} among the entities in this task, "
            f"with a score of {fmt_number(point)}."
        )
    return head + interval_clause


def _lexical_entail(premise: str, hypothesis: str) -> float:
    """Token overlap, plus a bonus when formatted numbers from the hypothesis
    appear as substrings of the premise. This is a selection heuristic, not a
    substitute for the pinned DeBERTa judge."""
    hyp_toks = {t for t in _TOKEN.findall(hypothesis.lower()) if len(t) > 2}
    if not hyp_toks:
        return 0.0
    prem_toks = _tokens(premise)
    overlap = len(hyp_toks & prem_toks) / len(hyp_toks)
    bonus = 0.0
    for num in re.findall(r"-?\d+(?:\.\d+)?", hypothesis):
        if num in premise:
            bonus += 0.08
    return overlap + bonus


def _window_numbers(text: str) -> list[float]:
    masked = _usable_numbers(extract_numbers(_mask_non_quantities(text)))
    return masked if masked else _usable_numbers(extract_numbers(text))


def _guided_points(
    entity: dict[str, Any], numbers: list[float], text: str, tname: str
) -> list[float]:
    """Candidate points keyed off the published target name, never ``family``."""
    points: list[float] = []
    lower = text.lower()
    tnl = tname.lower()

    def _add(value: float | None) -> None:
        if value is None or not number_in_text(text, value):
            return
        if any(abs(value - p) < 1e-9 for p in points):
            return
        points.append(value)

    primary = _pick_point(entity, numbers, text)
    if primary is not None and number_in_text(text, primary):
        _add(primary)

    if "bid to cover" in tnl:
        covers = [n for n in numbers if 1.2 <= n <= 4.5]
        recent = re.search(
            r"(?:most recent[^\n]{0,80}?|drew a bid-to-cover of\s+|average[^\n]{0,40}?is\s+)"
            r"([+-]?\d+(?:\.\d+)?)",
            lower,
        )
        if recent:
            parsed = _parse_float(recent.group(1))
            if parsed is not None and number_in_text(text, parsed):
                _add(parsed)
        if covers:
            _add(covers[-1])

    if "first print" in tnl or "mom" in tnl:
        fp = re.search(r"first print\s+([+-]?\d+(?:\.\d+)?)", lower)
        if fp:
            _add(_parse_float(fp.group(1)))

    years = entity.get("maturity_years")
    if isinstance(years, (int, float)):
        for match in _YIELD_SENT.finditer(text):
            if int(match.group(1)) == int(years):
                _add(_parse_float(match.group(2)))

    if "credit" in tnl:
        unit = [n for n in numbers if 0.0 <= n <= 1.0]
        if unit:
            _add(unit[0])

    if "position" in tnl or "pct oi" in tnl or "pct_oi" in tnl:
        pct = re.search(r"\(([+-]?\d+(?:\.\d+)?)%\s+of", text)
        if pct:
            _add(_parse_float(pct.group(1)))

    in_span = [p for p in points if number_in_text(text, p)]
    return in_span or ([primary] if primary is not None and number_in_text(text, primary) else [])


def _label_candidates(
    *,
    kind: str | None,
    labels: list[str],
    entity: dict[str, Any],
    numbers: list[float],
    text: str,
) -> list[str]:
    if kind != "classification" or not labels:
        return [""]
    numeric = _numeric_label_from_entity(entity, numbers, labels)
    scored = [(lab, _label_score(lab, text)) for lab in labels]
    scored.sort(key=lambda x: -x[1])
    chosen: list[str] = []
    if scored[0][1] > 0:
        chosen.append(scored[0][0])
    if numeric is not None and numeric not in chosen:
        chosen.append(numeric)
    if not chosen:
        chosen.append(labels[0])
    return chosen


def tighten_window(window: Window, needles: Iterable[str]) -> Window:
    """Shrink to the shortest line-ish slice that still contains ``needles``."""
    text = window.text
    lower = text.lower()
    spans: list[tuple[int, int]] = []
    for needle in needles:
        if not needle:
            continue
        idx = text.find(needle)
        if idx < 0:
            idx = lower.find(needle.lower())
        if idx >= 0:
            spans.append((idx, idx + len(needle)))
    if not spans:
        return window
    lo = min(s[0] for s in spans)
    hi = max(s[1] for s in spans)
    line_lo = text.rfind("\n", 0, lo) + 1
    line_hi = text.find("\n", hi)
    if line_hi < 0:
        line_hi = len(text)
    start = min(lo, line_lo) if (lo - line_lo) < 80 else lo
    end = max(hi, line_hi) if (line_hi - hi) < 80 else hi
    if end - start < _MIN_WINDOW:
        pad = (_MIN_WINDOW - (end - start) + 1) // 2
        start = max(0, start - pad)
        end = min(len(text), end + pad)
    # snap off mid-word
    if start > 0 and not text[start].isspace() and text[start - 1].isalnum():
        space = text.rfind(" ", max(0, start - 24), start + 1)
        if space >= 0:
            start = space + 1
    snippet = text[start:end].strip()
    if len(snippet) < _MIN_WINDOW:
        return window
    new_start = window.span_start + text.find(snippet)
    return Window(
        window.doc_id,
        window.doc_date,
        new_start,
        new_start + len(snippet),
        snippet,
    )


@dataclass
class Grounded:
    label: str
    point: float
    lo: float
    hi: float
    window: Window
    score: float
    claims: list[dict[str, Any]] = field(default_factory=list)


def _score_candidate(
    window: Window,
    entity: dict[str, Any],
    aliases: list[str],
    hyp: str,
    chosen_label: str,
    point: float,
    lo: float,
    hi: float,
    kind: str | None,
) -> float:
    score = _lexical_entail(window.text, hyp)
    if _alias_hit(window.text, aliases):
        score += 0.15
    if _doc_owned_by_entity(window.doc_id, entity):
        score += 0.35
    name = entity.get("name")
    if isinstance(name, str) and name and re.search(
        rf"(?:^|\n|- ){re.escape(name)}\s*:", window.text
    ):
        score += 1.2
    lower = window.text.lower()
    if any(
        k in lower
        for k in (
            "first print",
            "notes (",
            "revised up",
            "revised down",
            "bid-to-cover",
            "yield was",
            "diluted earnings",
            "earnings per diluted",
            "earnings per share",
        )
    ):
        score += 1.0
    cue_hit = bool(chosen_label) and _label_score(chosen_label, window.text) >= 2.0
    if kind == "classification":
        if cue_hit:
            score += 1.6
        else:
            score -= 1.4
    if all_numbers_in_text(window.text, (point, lo, hi)):
        score += 0.6
    if lo == hi == point:
        score += 0.15
    elif explicit_ranges(window.text):
        score += 0.45
    # An entity-row quantity that actually appears is the NOTES/table target.
    for key in (
        "latest_published_mom_pct",
        "latest_precutoff_estimate",
        "start_yield_pct",
        "net_pct_oi_20241022",
        "consensus_eps",
        "prior_year_q_eps",
    ):
        raw = entity.get(key)
        if isinstance(raw, (int, float)) and number_in_text(window.text, float(raw)):
            score += 1.3
            break
    ref_month = entity.get("ref_month")
    if isinstance(ref_month, str) and ref_month:
        if ref_month in window.text:
            score += 0.9
        else:
            score -= 0.6
    series = entity.get("series_id") or entity.get("series_fred")
    if isinstance(series, str) and series and re.search(
        rf"(?<![A-Za-z0-9]){re.escape(series)}(?![A-Za-z0-9])", window.text
    ):
        score += 0.25
    years = entity.get("maturity_years")
    if isinstance(years, (int, float)) and re.search(
        rf"(?<![0-9]){int(years)}-year", lower
    ):
        score += 0.5
    # Prefer a tight premise: DeBERTa dilutes on a whole 10-Q.
    score -= max(0.0, (len(window.text) - 200) / 1800.0)
    return score


def ground_entity(
    task: dict[str, Any],
    entity: dict[str, Any],
    corpus: IndexedCorpus,
    index: BM25Index,
    *,
    top_k: int = 10,
) -> Grounded:
    cutoff = str(task["cutoff_date"])
    kind = target_type(task)
    labels = legal_labels(task)
    tname = target_name(task)
    subject = entity_display_name(entity)
    level = interval_level(task)
    windows = collect_windows(entity, corpus, index, cutoff=cutoff, top_k=top_k)

    best: Grounded | None = None
    aliases = _entity_aliases(entity)

    for window in windows:
        numbers = _window_numbers(window.text)
        points = _guided_points(entity, numbers, window.text, tname)
        label_opts = _label_candidates(
            kind=kind,
            labels=labels,
            entity=entity,
            numbers=numbers,
            text=window.text,
        )
        for chosen_label in label_opts:
            for point in points:
                lo, hi = _interval_from_numbers(numbers, point, text=window.text)
                if not all_numbers_in_text(window.text, (point, lo, hi)):
                    lo, hi = point, point
                    if not number_in_text(window.text, point):
                        continue
                hyp = _hypothesis(
                    kind=kind,
                    tname=tname,
                    subject=subject,
                    label=chosen_label,
                    point=point,
                    lo=lo,
                    hi=hi,
                    level=level,
                )
                score = _score_candidate(
                    window,
                    entity,
                    aliases,
                    hyp,
                    chosen_label,
                    point,
                    lo,
                    hi,
                    kind,
                )
                candidate = Grounded(
                    label=chosen_label,
                    point=point,
                    lo=lo,
                    hi=hi,
                    window=window,
                    score=score,
                )
                if best is None or candidate.score > best.score:
                    best = candidate

    if best is None:
        # Last-resort: newest eligible document, a number that is actually in it.
        fallback = _newest_eligible_window(corpus, cutoff)
        nums = _window_numbers(fallback.text)
        point = next((n for n in nums if number_in_text(fallback.text, n)), 0.0)
        if not number_in_text(fallback.text, point):
            # Cite a slightly longer prefix that is still in the document.
            doc = corpus.doc_texts.get(fallback.doc_id, fallback.text)
            prefix = doc[: min(240, len(doc))] or " "
            fallback = Window(fallback.doc_id, fallback.doc_date, 0, len(prefix), prefix)
            nums = _window_numbers(fallback.text)
            point = next((n for n in nums if number_in_text(fallback.text, n)), 0.0)
        labels_or = labels
        best = Grounded(
            label=labels_or[0] if labels_or else "",
            point=point,
            lo=point,
            hi=point,
            window=fallback,
            score=0.0,
        )
    else:
        needles = [fmt_number(best.point), fmt_number(best.lo), fmt_number(best.hi)]
        if best.label:
            for cue in _LABEL_CUES.get(best.label, ()):
                if cue in best.window.text.lower():
                    needles.append(cue)
                    break
        tightened = tighten_window(best.window, needles)
        if all_numbers_in_text(tightened.text, (best.point, best.lo, best.hi)):
            best.window = tightened

    window = _resolve_window(best.window, corpus)
    if not all_numbers_in_text(window.text, (best.point, best.lo, best.hi)):
        recovered = _span_covering_numbers(
            corpus, window.doc_id, (best.point, best.lo, best.hi), hint=window
        )
        if recovered is not None:
            window = recovered
    claim_text = (
        f"{subject}: extracted from {window.doc_id} "
        f"[{window.span_start}:{window.span_end}]."
    )
    best.claims = [
        {
            "doc_id": window.doc_id,
            "span_start": window.span_start,
            "span_end": window.span_end,
            "claim": claim_text,
        }
    ]
    return best


def _resolve_window(window: Window, corpus: IndexedCorpus) -> Window:
    """Make the cited offsets resolve in the scorer's joined document text.

    Window construction can drift by a stripped space; the scorer requires
    ``doc_text[span_start:span_end]`` to be a non-empty exact slice.
    """
    doc = corpus.doc_texts.get(window.doc_id, "")
    if not doc:
        return window
    start, end = window.span_start, window.span_end
    if 0 <= start < end <= len(doc) and doc[start:end] == window.text:
        return window
    # Prefer the occurrence closest to the claimed offsets so a short NOTES
    # line is not remapped onto the document title.
    closest: tuple[int, int] | None = None
    pos = 0
    while True:
        found = doc.find(window.text, pos)
        if found < 0:
            break
        dist = abs(found - start)
        if closest is None or dist < closest[0]:
            closest = (dist, found)
        pos = found + 1
    if closest is not None:
        found = closest[1]
        return Window(
            window.doc_id,
            window.doc_date,
            found,
            found + len(window.text),
            window.text,
        )
    # Last resort: locate the longest line of the window that still exists.
    for line in sorted(window.text.splitlines(), key=len, reverse=True):
        snippet = line.strip()
        if len(snippet) < _MIN_WINDOW:
            continue
        found = doc.find(snippet)
        if found >= 0:
            return Window(
                window.doc_id,
                window.doc_date,
                found,
                found + len(snippet),
                snippet,
            )
    end = min(max(len(window.text), _MIN_WINDOW), len(doc))
    return Window(window.doc_id, window.doc_date, 0, end, doc[:end])


def _span_covering_numbers(
    corpus: IndexedCorpus,
    doc_id: str,
    values: Iterable[float],
    *,
    hint: Window | None = None,
) -> Window | None:
    """Find an exact document slice that still contains every formatted number."""
    doc = corpus.doc_texts.get(doc_id, "")
    if not doc:
        return None
    spans: list[tuple[int, int]] = []
    for value in values:
        token = fmt_number(value)
        idx = doc.find(token)
        if idx < 0 and not token.startswith("-"):
            idx = doc.find(f"+{token}")
            if idx >= 0:
                token = f"+{token}"
        if idx < 0:
            return None
        # Prefer the occurrence closest to the hint, if any.
        if hint is not None:
            pos = 0
            closest = idx
            best_dist = abs(idx - hint.span_start)
            while True:
                found = doc.find(token, pos)
                if found < 0:
                    break
                dist = abs(found - hint.span_start)
                if dist < best_dist:
                    best_dist = dist
                    closest = found
                pos = found + 1
            idx = closest
        spans.append((idx, idx + len(token)))
    start = min(s[0] for s in spans)
    end = max(s[1] for s in spans)
    line_lo = doc.rfind("\n", 0, start) + 1
    line_hi = doc.find("\n", end)
    if line_hi < 0:
        line_hi = len(doc)
    if start - line_lo < 120:
        start = line_lo
    if line_hi - end < 120:
        end = line_hi
    if end - start < _MIN_WINDOW:
        end = min(len(doc), start + _MIN_WINDOW)
    snippet = doc[start:end]
    if not all_numbers_in_text(snippet, values):
        return None
    date = corpus.doc_dates.get(doc_id)
    return Window(doc_id, date, start, end, snippet)


def _newest_eligible_window(corpus: IndexedCorpus, cutoff: str) -> Window:
    dated = [
        (doc_id, date, corpus.doc_texts.get(doc_id, ""))
        for doc_id, date in corpus.doc_dates.items()
        if isinstance(date, str) and date <= cutoff
    ]
    if not dated:
        return Window("UNKNOWN", None, 0, 1, " ")
    doc_id, date, text = max(dated, key=lambda row: row[1])
    end = min(240, len(text)) if text else 1
    return Window(doc_id, date, 0, max(end, 1), text[:end] or " ")


def prediction_from_grounded(
    entity: dict[str, Any],
    grounded: Grounded,
    *,
    level: float,
    kind: str | None,
) -> dict[str, Any]:
    point = grounded.point
    lo, hi = grounded.lo, grounded.hi
    if lo > hi:
        lo, hi = hi, lo
    # Finite numbers only — NaN/Inf is a participant failure.
    if not math.isfinite(point):
        point = 0.0
    if not math.isfinite(lo):
        lo = point
    if not math.isfinite(hi):
        hi = point
    pred: dict[str, Any] = {
        "entity_id": entity.get("entity_id", ""),
        "point_forecast": point,
        "interval": {"level": float(level), "lo": lo, "hi": hi},
        "claims": list(grounded.claims),
    }
    if kind == "classification":
        pred["label"] = grounded.label
    elif grounded.label:
        pred["label"] = grounded.label
    return pred
