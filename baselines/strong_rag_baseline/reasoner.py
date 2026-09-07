"""Quantity-aware baseline over pre-cutoff evidence, without public answer locks.

Historical observations are persistence baselines, not known future outcomes.
Derived quantities retain source spans even when the result is not a literal
token. Unsupported targets use explicit, uncalibrated fallbacks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from .indexer import IndexedCorpus, dated_on_or_before
from .quantities import TargetSpec, finite_number
from .retriever import BM25Index
from .schema import entity_display_name, legal_labels, target_name

_NUMBER = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_TOKENS = re.compile(r"[a-z0-9]+")


def extract_numbers(text: str) -> list[float]:
    return [float(m.group().replace(",", "")) for m in re.finditer(_NUMBER, text)]


def number_in_text(text: str, value: float) -> bool:
    return any(
        abs(n - value) <= max(1e-9, abs(value) * 1e-9) for n in extract_numbers(text)
    )


def _task_needles(task: dict[str, Any], entity: dict[str, Any]) -> list[str]:
    return [
        target_name(task),
        *legal_labels(task),
        *[key.replace("_", " ") for key in entity],
    ]


@dataclass(frozen=True)
class Window:
    doc_id: str
    doc_date: str | None
    span_start: int
    span_end: int
    text: str
    entity_ambiguous: bool = False


@dataclass
class Grounded:
    label: str
    point: float
    lo: float
    hi: float
    window: Window
    score: float
    claims: list[dict[str, Any]] = field(default_factory=list)
    rationale: str = ""


@dataclass(frozen=True)
class Estimate:
    point: float
    method: str
    priority: int


def _entity_aliases(entity: dict[str, Any]) -> list[str]:
    values = [
        str(entity[key])
        for key in (
            "entity_id",
            "name",
            "series_id",
            "series_name",
            "series_fred",
            "tenor",
        )
        if entity.get(key)
    ]
    if entity.get("name"):
        values.append(re.sub(r"\s*\([^)]*\)", "", str(entity["name"])))
    if finite_number(entity.get("maturity_years")):
        values.append(f"{int(entity['maturity_years'])}-year")
    return [value.lower().replace("_", " ") for value in values]


def _alias_hit(text: str, aliases: Iterable[str]) -> bool:
    text = text.lower().replace("_", " ")
    return any(
        re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", text)
        for alias in aliases
        if len(alias) > 1
    )


def collect_windows(
    entity: dict[str, Any],
    corpus: IndexedCorpus,
    index: BM25Index,
    *,
    cutoff: str,
    top_k: int = 10,
    extra_needles: Iterable[str] = (),
    roster: Iterable[dict[str, Any]] = (),
) -> list[Window]:
    aliases = _entity_aliases(entity)
    other_aliases = {
        alias
        for row in roster
        if row.get("entity_id") != entity.get("entity_id")
        for alias in _entity_aliases(row)
    } - set(aliases)
    retrieved = {
        hit.chunk.doc_id
        for hit in index.search(" ".join([*aliases, *extra_needles]), top_k)
    }
    cik = str(entity.get("cik") or "").zfill(10) if entity.get("cik") else ""
    windows: list[Window] = []
    for doc_id, text in sorted(corpus.doc_texts.items()):
        date = corpus.doc_dates.get(doc_id)
        if not dated_on_or_before(date, cutoff) or not text.strip():
            continue
        owned = bool(cik and cik in doc_id)
        if cik and "EDGAR" in doc_id.upper() and not owned:
            continue
        doc_match = _alias_hit(doc_id, aliases)
        if not (owned or doc_match or doc_id in retrieved or _alias_hit(text, aliases)):
            continue
        # Keep sentence/row offsets; a filing can discuss other roster entities.
        for paragraph in re.finditer(r"[^\n]+", text):
            boundaries = [paragraph.start()]
            boundaries.extend(
                paragraph.start() + match.end()
                for match in re.finditer(
                    r"(?<=[.!?;])\s+(?=[A-Z])", paragraph.group()
                )
            )
            boundaries.append(paragraph.end())
            for start, end in zip(boundaries, boundaries[1:]):
                sentence = text[start:end]
                own_mention = _alias_hit(sentence, aliases)
                other_mention = _alias_hit(sentence, other_aliases)
                if other_mention and not own_mention:
                    continue
                if not (owned or doc_match or own_mention):
                    continue
                for pos in range(start, end, 600):
                    last = min(pos + 1000, end)
                    snippet = text[pos:last]
                    if snippet.strip() and (
                        owned or doc_match or _alias_hit(snippet, aliases)
                    ):
                        windows.append(
                            Window(doc_id, date, pos, last, snippet, other_mention)
                        )
    return windows


def _metric(spec: TargetSpec) -> str:
    if "eps" in spec.name:
        return r"(?:diluted\s+(?:earnings|income)(?:\s+from continuing operations)?\s+per\s+(?:common\s+)?share|diluted\s+eps|earnings per diluted share|diluted loss per share)"
    stem = re.split(
        r"\s+(?:yoy\s+)?growth|\s+(?:pct|percent|direction|rank|bps|change|spread)\b",
        spec.name,
    )[0]
    return re.escape(stem or spec.name)


def _read_value(raw: str) -> float:
    raw = raw.strip().replace(",", "").replace("$", "").strip()
    return (
        -float(raw[1:-1].strip())
        if raw.startswith("(") and raw.endswith(")")
        else float(raw)
    )


def _estimate(spec: TargetSpec, entity: dict[str, Any], text: str) -> Estimate | None:
    number = rf"(?:\(\s*{_NUMBER}\s*\)|{_NUMBER})"
    if spec.mode in {"probability", "change_bps", "return_pct"} and spec.resolution:
        future_dates = {
            date
            for date in re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text)
            if date > spec.cutoff
        }
        if future_dates and spec.resolution not in future_dates:
            return None
    if spec.mode == "probability":
        metric = (
            r"(?:credit event|default|bankruptcy)"
            if "credit" in spec.name
            else re.escape(spec.name.replace("probability", "").strip())
        )
        match = re.search(
            rf"(?:probability|risk)\s+(?:of\s+)?(?:a\s+)?{metric}[^\d%]{{0,24}}({_NUMBER})\s*(%)?",
            text,
            re.I,
        )
        if match:
            point = float(match[1]) / (100 if match[2] else 1)
            if 0 <= point <= 1:
                return Estimate(point, "explicit probability estimate in evidence", 8)
        return None
    if spec.mode == "change_bps":
        match = re.search(
            rf"(?:projected|forecast|expected)\s+yield(?:\s+(?:is|of|at))?\s+({_NUMBER})\s*(?:percent|%)",
            text,
            re.I,
        )
        start = entity.get("start_yield_pct")
        if match and finite_number(start):
            return Estimate(
                (float(match[1]) - start) * 100,
                "(projected yield percent - starting yield percent) * 100 bps",
                9,
            )
        match = re.search(
            rf"yield\s+change(?:\s+(?:forecast|projection))?\s+(?:is|of|=)\s*({_NUMBER})\s*(?:bps|basis points)",
            text,
            re.I,
        )
        return (
            Estimate(float(match[1]), "explicit yield-change estimate in bps", 8)
            if match
            else None
        )
    if spec.mode == "return_pct":
        match = re.search(
            rf"(?:abnormal|market.adjusted)\s+(?:stock\s+)?return\s+(?:(?:forecast|estimate)\s+)?(?:is|was|of|=)\s*({_NUMBER})\s*(?:%|percent)",
            text,
            re.I,
        )
        return (
            Estimate(float(match[1]), "explicit abnormal-return estimate in percent", 8)
            if match
            else None
        )
    if spec.mode == "change_pct_oi":
        for key, value in entity.items():
            if "change" in key and "pct_oi" in key and finite_number(value):
                return Estimate(
                    float(value),
                    f"historical change persistence from {key}; not the future outcome",
                    5,
                )
        return None
    if spec.mode == "growth_pct":
        metric = _metric(spec)
        match = re.search(
            rf"{metric}[^\d\n()+$-]{{0,32}}\$?\s*({number})\s*(million|billion|dollars)?\s*,?\s*(?:compared (?:to|with)|versus|vs\.?)[^\d\n()+$-]{{0,32}}\$?\s*({number})\s*(million|billion|dollars)?",
            text,
            re.I,
        )
        if match:
            scales = {"million": 1e6, "billion": 1e9, "dollars": 1.0}
            current_scale = scales.get((match[2] or match[4] or "").lower(), 1.0)
            prior_scale = scales.get((match[4] or match[2] or "").lower(), 1.0)
            current = _read_value(match[1]) * current_scale
            prior = _read_value(match[3]) * prior_scale
            if abs(prior) > 1e-12:
                return Estimate(
                    (current / prior - 1) * 100,
                    "(current / prior - 1) * 100; historical growth persistence",
                    9,
                )
        match = re.search(
            rf"{metric}\s+(?:yoy\s+)?growth(?:\s+(?:was|is|of))?\s+({_NUMBER})\s*(?:%|percent)",
            text,
            re.I,
        )
        return (
            Estimate(
                float(match[1]),
                "explicit metric growth percent; historical persistence",
                8,
            )
            if match
            else None
        )
    if spec.mode == "eps":
        match = re.search(
            rf"{_metric(spec)}(?:\s*\(eps\))?(?:\s+(?:was|were|of|is))?\s*\$?\s*({number})",
            text,
            re.I,
        )
        if match:
            value = _read_value(match[1])
            if "loss per share" in match[0].lower():
                value = -abs(value)
            return Estimate(
                value, "historical diluted EPS persistence; not reported future EPS", 8
            )
        return None
    if spec.mode == "ratio":
        match = re.search(
            rf"bid.to.cover(?:\s+ratio)?(?:\s+(?:was|is|of))?\s+({_NUMBER})", text, re.I
        )
        if match:
            return Estimate(float(match[1]), "historical bid-to-cover persistence", 8)
    # Only matching quantity columns seed persistence. Market cap and employee
    # count do not become forecasts for a different numeric target.
    tokens = set(_TOKENS.findall(spec.name)) - {
        "rank",
        "direction",
        "outcome",
        "next",
        "estimate",
        "revision",
    }
    for key, value in entity.items():
        key_tokens = set(_TOKENS.findall(key.lower()))
        same_quantity = len(tokens & key_tokens) >= 2
        same_quantity |= "mom" in tokens and "mom" in key_tokens and "pct" in key_tokens
        same_quantity |= "revision" in spec.name and key == "latest_precutoff_estimate"
        if same_quantity and finite_number(value) and number_in_text(text, value):
            return Estimate(
                float(value),
                f"historical persistence from {key}; not the future outcome",
                6,
            )
    match = re.search(
        rf"{_metric(spec)}\s+(?:spread\s+)?(?:is|was|of|=)\s*({_NUMBER})\s*(?:bps|basis points|percent|%)?",
        text,
        re.I,
    )
    if match:
        return Estimate(
            float(match[1]), "explicit target quantity; historical persistence", 4
        )
    entity_tokens = set(_TOKENS.findall(" ".join(_entity_aliases(entity))))
    quantity_tokens = tokens - entity_tokens - {"bps", "pct", "percent"}
    describes_quantity = bool(quantity_tokens & set(_TOKENS.findall(text.lower())))
    describes_direction = spec.kind == "classification" and any(
        label in text.lower() for label in spec.labels
    )
    if describes_quantity or describes_direction:
        match = re.search(rf"(?:first print|the print was)\s+({_NUMBER})", text, re.I)
        if match:
            return Estimate(float(match[1]), "historical target print persistence", 4)
    return None


def _label(spec: TargetSpec, entity: dict[str, Any], point: float, text: str) -> str:
    labels = spec.labels
    if not labels:
        return ""
    if spec.mode == "probability" and {"credit_event", "no_event"} <= set(labels):
        return "credit_event" if point > 0.5 else "no_event"
    if spec.mode == "return_pct" and {
        "positive_reaction",
        "negative_reaction",
        "flat",
    } <= set(labels):
        threshold = entity.get("flat_threshold_abn_pct", 1.0)
        threshold = float(threshold) if finite_number(threshold) else 1.0
        return (
            "positive_reaction"
            if point > threshold
            else "negative_reaction"
            if point < -threshold
            else "flat"
        )
    consensus = entity.get("consensus_eps")
    if {"beat", "miss", "inline"} <= set(labels) and finite_number(consensus):
        threshold = entity.get("threshold_pct", 0.05)
        threshold = float(threshold) if finite_number(threshold) else 0.05
        margin = abs(consensus) * threshold
        return (
            "beat"
            if point > consensus + margin
            else "miss"
            if point < consensus - margin
            else "inline"
        )
    if {"up", "down"} <= set(labels):
        baseline = entity.get(
            "prior_year_q_eps", entity.get("latest_precutoff_estimate", 0)
        )
        if finite_number(baseline):
            return "up" if point > baseline else "down"
    matches = [
        label
        for label in labels
        if re.search(
            r"(?<!\w)" + re.escape(label.replace("_", " ")) + r"(?!\w)", text, re.I
        )
    ]
    return matches[0] if len(matches) == 1 else labels[0]


def ground_entity(
    task: dict[str, Any],
    entity: dict[str, Any],
    corpus: IndexedCorpus,
    index: BM25Index,
    *,
    top_k: int = 10,
) -> Grounded:
    spec = TargetSpec.from_task(task, entity)
    windows = collect_windows(
        entity,
        corpus,
        index,
        cutoff=spec.cutoff,
        top_k=top_k,
        extra_needles=_task_needles(task, entity),
        roster=task.get("entities", []),
    )
    candidates: list[tuple[Estimate, Window]] = []
    for window in windows:
        if window.entity_ambiguous:
            continue
        estimate = _estimate(spec, entity, window.text)
        if estimate and finite_number(estimate.point):
            if spec.lower is not None and estimate.point < spec.lower:
                continue
            if spec.upper is not None and estimate.point > spec.upper:
                continue
            candidates.append((estimate, window))
    if candidates:
        estimate, window = max(
            candidates, key=lambda item: (item[0].priority, item[1].doc_date or "")
        )
    else:
        # Do not invent citations when the task has no eligible entity evidence.
        window = (
            max(windows, key=lambda value: value.doc_date or "")
            if windows
            else Window("", None, 0, 0, "")
        )
        estimate = Estimate(
            spec.fallback_point(),
            "unsupported-target fallback; no calibrated predictive evidence",
            0,
        )
    point = estimate.point
    lo, hi = spec.interval(point)
    label = _label(spec, entity, point, window.text)
    rationale = (
        f"{entity_display_name(entity)}: {estimate.method}; target unit={spec.unit or 'unspecified'}. "
        f"Cutoff={spec.cutoff}; resolution={spec.resolution or 'task-defined'}. "
        "Interval is an uncalibrated baseline assumption; cited observations do not establish future outcomes."
    )
    claims = (
        [
            {
                "doc_id": window.doc_id,
                "span_start": window.span_start,
                "span_end": window.span_end,
                "claim": window.text.strip(),
            }
        ]
        if window.doc_id
        else []
    )
    result = Grounded(
        label, point, lo, hi, window, float(estimate.priority), claims, rationale
    )
    spec.validate_prediction(
        prediction_from_grounded(entity, result, level=spec.level, kind=spec.kind)
    )
    return result


def prediction_from_grounded(
    entity: dict[str, Any], grounded: Grounded, *, level: float, kind: str | None
) -> dict[str, Any]:
    prediction: dict[str, Any] = {
        "entity_id": entity.get("entity_id", ""),
        "point_forecast": grounded.point,
        "interval": {"level": level, "lo": grounded.lo, "hi": grounded.hi},
        "claims": list(grounded.claims),
    }
    if kind == "classification":
        prediction["label"] = grounded.label
    return prediction
