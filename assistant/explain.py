"""Plain-English reason per flagged row: the LLM words it, a validator checks it, a template backs it up."""
import json
import re

from .llm import LLM
from .notes import batches
from .verdict import Verdict

REASONS_SCHEMA = {
    "type": "object",
    "properties": {"reasons": {"type": "array", "items": {
        "type": "object", "properties": {"id": {"type": "string"}, "text": {"type": "string"}},
        "required": ["id", "text"], "additionalProperties": False}}},
    "required": ["reasons"], "additionalProperties": False,
}

REASON_SYSTEM = """You write the 'reason' cell of freight-cost alerts: one or two plain-English sentences each.
You get one or more alerts, each starting with its id like [R1]. Return JSON {"reasons": [{"id": ..., "text": ...}]}
with exactly one entry per id. Each reason may use ONLY the facts of its own alert.
The verdict label is added for you; write only the explanation that follows it.
Rules:
- Use ONLY the facts provided. Do not invent causes, numbers, dates or note ids.
- If justified, say in your own words what the matched note reports and why it covers this week.
- If unexplained, say no note explains the rise, mention the closest note id and why it does not apply,
  and recommend a human review.
- Refer to a note's timing by its date. Never write "this week" or "the current week" about a note.
- Use your own words; do not copy the note text verbatim."""

NOTE_ID = re.compile(r"\bN\d{3}\b")
PERCENT = re.compile(r"\d+(?:\.\d+)?%")  # unsigned, so "29.5%" matches the fact "+29.5%"
# Known limit: only ISO dates are checked; a reworded date ("Jan 20") is trusted to the prompt.
ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
ROUTE = re.compile(r"\b[A-Z][a-z]+-[A-Z][a-z]+\b")
GROUNDING_CHECKS = ((NOTE_ID, "note id"), (PERCENT, "percentage"), (ISO_DATE, "date"), (ROUTE, "route"))
# Phrases that would claim the opposite of the code's verdict.
CONTRADICTS = {True: re.compile(r"unexplained|no note (explains|justifies)", re.I),
               False: re.compile(r"\b(is|was|are|were) justified\b|\bjustified by\b|\bexplained by\b", re.I)}
# gpt-oss writes "2025‑01‑20" with U+2011 non-breaking hyphens, which would slip past the date/route checks
# above. Every LLM text is mapped to plain ASCII hyphens before it is validated or shown.
# Hyphen-like characters (and en dash, used in ranges) become "-"; em dashes are punctuation, so " - ".
UNICODE_DASHES = str.maketrans({**{c: "-" for c in "‐‑‒–−"}, "—": " - ", "―": " - "})
LEADING_LABEL = re.compile(r"^(verdict:\s*)?(justified|unexplained)\s*[:.\-]?\s*", re.I)
MAX_REASON_CHARS = 500


def facts(row, verdict: Verdict) -> str:
    lines = [f"Verdict: {'Justified' if verdict.justified else 'Unexplained'}",
             f"Route: {row.route}, week of {row.week_of.date()}",
             f"Cost change: {row.vs_own_pct:+.1f}% vs this route's past average, "
             f"{row.vs_peer_pct:+.1f}% vs similar routes this week"]
    if verdict.note:
        n = verdict.note
        label = "Matched note" if verdict.justified else "Closest note"
        lines.append(f"{label}: {n.note_id} dated {n.date} ({n.applies_to}): {n.text}")
        if verdict.rejection:
            lines.append(f"Why it does not apply: {verdict.rejection}")
    else:
        lines.append("No note mentions this route or all routes.")
    return "\n".join(lines)


def pattern_line(row) -> str:
    """Code-written context for a sustained rise ('' for a first or one-off flagged week), appended after
    the LLM's text like the verdict label is prepended, so it is always present and exact."""
    if row.episode_week_no < 2:
        return ""
    return (f" Sustained rise: flagged week {row.episode_week_no} on this route since {row.episode_start.date()}, "
            f"now {row.vs_pre_episode_pct:+.1f}% above its level before that.")


def verdict_label(verdict: Verdict) -> str:
    """Code, not the LLM, writes the verdict and the citation, so neither can drift."""
    if verdict.justified:
        return f"Justified by note {verdict.note.note_id} ({verdict.note.date})"
    return "Unexplained"


def template_reason(row, verdict: Verdict) -> str:
    n = verdict.note
    if verdict.justified:
        return f"{verdict_label(verdict)}: {n.text} The note covers the week of {row.week_of.date()}.{pattern_line(row)}"
    closest = (f" The closest note ({n.note_id}, {n.date}) does not apply because {verdict.rejection}."
               if n else "")
    return f"{verdict_label(verdict)}: no note explains this rise.{closest} Flagged for human review.{pattern_line(row)}"


def clean(text: str) -> str:
    """Plain hyphens, single spaces: what the validator checks is exactly what gets written."""
    return " ".join(text.translate(UNICODE_DASHES).split())


def ungrounded(text: str, allowed: str) -> list[str]:
    """Note ids, percentages, ISO dates or route names in `text` that never appear in `allowed`."""
    problems = []
    for pattern, kind in GROUNDING_CHECKS:
        invented = set(pattern.findall(text)) - set(pattern.findall(allowed))
        if invented:
            problems.append(f"{kind} not in facts: {sorted(invented)}")
    return problems


def validate(reason: str, allowed_facts: str, verdict: Verdict) -> list[str]:
    """Problems with an LLM reason; empty list means it is grounded in the facts."""
    problems = ungrounded(reason, allowed_facts)
    if len(reason.split()) < 4:
        problems.append("empty or too short")
    if CONTRADICTS[verdict.justified].search(reason):
        problems.append("wording contradicts the verdict")
    if len(reason) > MAX_REASON_CHARS:
        problems.append("too long")
    return problems


def parse_reasons(reply: str) -> dict[str, str]:
    try:
        return {str(r["id"]): str(r["text"]) for r in json.loads(reply)["reasons"]}
    except (ValueError, TypeError, KeyError):  # unusable reply: every row falls back to its template
        return {}


def write_reasons(items: list[tuple], llm: LLM) -> list[tuple[str, str, list[str]]]:
    """(row, verdict) pairs -> (reason, source, problems) each, in batches of llm.batch_size.
    Every reason is validated against its OWN row's facts, so a batch can't leak one row's note into another.
    source is 'llm' or 'template' (validator fallback)."""
    results = []
    for start, chunk in zip(range(0, len(items), llm.batch_size), batches(items, llm.batch_size)):
        ids = [f"R{start + i + 1}" for i in range(len(chunk))]
        row_facts = [facts(row, verdict) for row, verdict in chunk]
        reply = llm.chat(REASON_SYSTEM, "\n\n".join(f"[{i}]\n{f}" for i, f in zip(ids, row_facts)),
                         schema=REASONS_SCHEMA)
        texts = parse_reasons(reply)
        for rid, allowed, (row, verdict) in zip(ids, row_facts, chunk):
            body = LEADING_LABEL.sub("", clean(texts.get(rid, "")))
            problems = validate(body, allowed, verdict)
            results.append((template_reason(row, verdict), "template", problems) if problems
                           else (f"{verdict_label(verdict)}: {body}{pattern_line(row)}", "llm", []))
    return results
