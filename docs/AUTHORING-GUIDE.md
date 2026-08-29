# Track 4 — How a Unit Is Built (Participant Guide)

## Executive summary (read this first)

A Track 4 **unit** is a self-contained directory: a task file, a task card, a checksum manifest,
and a frozen corpus of documents. This page walks through that layout field by field and says, for
each one, **who reads it** — your agent, the embargo gate, the faithfulness judge, the scorer, or
nobody. Knowing which fields are load-bearing is what stops you building a retrieval pipeline
around a provenance label the scorer never looks at, or missing the one field that decides whether
a citation resolves at all. Everything here is read off the shipped units and the shipped scorer,
so you can check any claim on this page against
`units/t4-EXAMPLE-eps-beat/` and `qfbench2_track_analysis/scoring.py`.

This is a guide to *reading* a unit, not to writing one. Task authoring is an organiser process and
its material — resolved outcomes, adversarial variant construction, generation and review logs — is
not published, by design: it is the answer key. If you are looking for what to predict, start with
[`CATEGORIES.md`](CATEGORIES.md); if a term is unfamiliar, start with [`CONCEPTS.md`](CONCEPTS.md).

---

## The directory

```
units/t4-EXAMPLE-eps-beat/
  card.toml       # the unit's parameters: gates, thresholds, environment, embargo
  task.json       # what to predict, for which entities, from what evidence
  manifest.json   # per-file checksum manifest for the whole unit
  corpus/         # the frozen evidence, one JSON document per file
```

Your container does not see this path. The harness mounts the **unit directory itself** at
`/input`, so your agent reads `/input/task.json` and `/input/corpus/` — see
[`../SUBMISSION_CLI.md`](../SUBMISSION_CLI.md) for the full invocation.

---

## `task.json` — the participant-facing task

The exemplar is `units/t4-EXAMPLE-eps-beat/task.json`. The fields that matter:

| Field | Read by | What it is |
|---|---|---|
| `task_id` | the scorer | Copy it verbatim into `answer.json`. A mismatch is a whole-submission failure. |
| `schema_version` | the scorer | The answer-schema generation this unit speaks. |
| `family` | nobody | A descriptive slug. No enum is enforced; do not switch on it. |
| `target` | **you and the scorer** | The nested block `{name, type, labels}`. `type` is one of `classification`, `regression`, `ranking`. |
| `prompt` | **you** | The plain-English statement of what to predict, including the units the number must be in. |
| `cutoff_date` | **the embargo gate** | Cited evidence must be dated on or before this. |
| `resolution_date` | context | When the outcome became known. The gap to `cutoff_date` is the horizon; it varies widely across units. |
| `interval_level` | the scorer | The level your `interval.level` must equal. |
| `corpus_manifest` | tooling | Relative path to the unit's manifest. |
| `faithfulness_rubric` | **you** | Prose restatement of the admission rule for this unit. |
| `entities` | **you and the scorer** | The table. One object per row; `entity_id` is the key the scorer aligns on. |
| `notes` | nobody | Free text. |

**The target block is nested.** It is `target.type`, not a flat `target_type` key, in `task.json`.
The flat spelling belongs to two other places: the `answer.json` you emit, and `[scoring.params]`
in `card.toml`. The shipped baseline reads `task.get("target", {}).get("type")`, and no reader in
this repository consumes a flat `target_type` out of `task.json`.

**Entity columns are pre-cutoff facts.** Feature columns differ per family — `consensus_eps`,
`prior_year_q_eps`, `start_yield_pct`, `latest_precutoff_estimate`, `open_interest_20241022`. None
of them is the answer; several are the answer's value one period earlier, which is the naive
baseline a regression unit measures your skill against.

---

## `card.toml` — the unit's parameters

`card.toml` is authoritative for everything the scorer is configured with. The blocks you should
read before designing an agent:

- `[scoring.params]` — `faithfulness_threshold`, `interval_level`, `target_type`,
  `composite_weights`, `tau_citation`, and the pinned NLI ensemble. These are the numbers the
  gate and the composite actually use.
- `[environment]` — the compute grant and network mode the harness enforces.
- `[agent] timeout_sec` — the per-unit wall clock. Exceeding it is a `g2` timeout failure.
- `[embargo]` — which field on the task carries the cutoff (`cutoff_date`), which field on each
  corpus document carries its date (`doc_date`), and that the check is strict.
- `[contamination] canary_guid` — a per-unit contamination marker. Never emit it.

Prose in this repository is not allowed to hand-restate these values: the guards in
`baselines/tests/test_docs_match_artifacts.py` fail CI when a document quotes a threshold or an
`[environment]` grant the exemplar card does not carry. Read the card.

---

## `manifest.json` — the checksum manifest

The manifest at the **unit root** is a per-file checksum manifest: a `files[]` array whose entries
each carry a `path`, a real `sha256`, a `bytes` count, and provenance fields (`role`, `source`,
`license`, `split`, `redistributable`, `pii_stripped`). `verify_manifest` — run by the
`validate-units` CI job and by the harness before mounting — checks it **exactly, in both
directions**: every declared entry must exist and hash to the recorded digest, and every regular
file inside the subtrees the manifest covers must be declared. Which subtrees those are varies:
ten of the eleven published units declare `card.toml` and `task.json` alongside their corpus files,
while the exemplar declares its corpus only.

It is **not** a corpus index. Per-document metadata (`doc_date`, `form_type`, ticker) lives in each
corpus document, where the scorer reads it, and in `task.json`. A `documents[]`-style index in a
file named `manifest.json` fails verification.

Ten of the eleven published units also carry a **second** manifest inside `corpus/`, covering the
corpus files alone; the exemplar does not. Resolve citations through the document files themselves
rather than assuming one manifest layout.

---

## `corpus/` — the frozen evidence

One JSON document per file, at `corpus/<doc_id>.json`. Two fields are load-bearing:

| Field | Read by | Why it matters |
|---|---|---|
| `text` | **the faithfulness judge** | The document's plain text. This is the premise every entailment check is run against. |
| `doc_date` | **the embargo gate** | ISO-8601. A citation into a document dated after `cutoff_date` fails the unit. |

`doc_id` must equal the filename stem: the scorer resolves a citation by looking up
`corpus/<doc_id>.json`. Nothing parses the id's internal structure — the shipped convention is
`EDGAR_{cik}_{form}_{YYYYMMDD}` for filings and `{SOURCE}_{SERIES}_{YYYYMMDD}` for data snapshots,
but uniqueness and the filename match are the only real constraints.

The remaining fields — `source`, `form_type`, `ticker`, `cik`, `title`, `note`, `pii_stripped` —
are provenance records. No scoring code reads them. They are useful to you for filtering and
retrieval; do not build a contract on them.

**Where the text lives matters.** `qfbench2_common.scoring.faithfulness._doc_text` accepts a raw
string, a dict with a flat `text` string, or a dict with a `spans` list of `{text: ...}` objects,
which it joins with spaces. Content anywhere else resolves to the empty string, every citation into
that document yields an empty premise, and faithfulness scores zero. Treat the flat `text` field as
the normal case.

---

## Character offsets: how a citation points at a passage

A citation is `{doc_id, span_start, span_end, claim}`. The offsets are zero-indexed character
positions into the resolved document text — the same string `_doc_text` returns — so
`text[span_start:span_end]` is the premise the judge is shown. Compute them against the document
you actually loaded, and verify by slicing before you emit.

Your `claim` string is parsed and reported, but it is **not** what the judge is asked about. The
hypothesis is built from the values you submitted — `label` / `point_forecast` / `rank` /
`interval` — plus the trusted task schema (`qfbench2_track_analysis/hypothesis.py`). Describing a
passage accurately cannot make a wrong prediction faithful, and writing more claims cannot help:
the denominator is the roster.

---

## What "the outcome" is, and where it is not

Each unit resolves against a per-entity outcome roster held in the private repository. It is not
published, and no field of it appears in this repository. Two properties of it are worth knowing
because they shape what a well-formed submission looks like:

- The roster covers the entity roster **exactly** — the same `entity_id` set, no more and no less.
  Your `entity_predictions[]` must do the same.
- Where a family has a numeric target, **every** row carries it or none does. There is no unit in
  which some rows are scored numerically and others are not. A unit with no numeric target is a
  pure-label unit, and its composite drops the calibration leg entirely.

---

## Checking a unit yourself

```bash
# Card schema, track, split and canary checks — standard library only.
python .github/validate_units.py analysis --stdlib-only

# The same, plus manifest checksums and the public-safety firewall (needs the shared toolkit).
python .github/validate_units.py analysis

# Faithfulness preview for one answer against one unit.
python faithfulness/judge.py --answer /tmp/answer.json --unit units/t4-EXAMPLE-eps-beat
```

`--unit` is the unit *directory*, not `corpus/`: the roster and target schema come from
`task.json`, the thresholds from `card.toml`, and a `doc_id` resolves only to a document the
manifest declares.
