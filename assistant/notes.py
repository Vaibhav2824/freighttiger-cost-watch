"""Context notes: the LLM classifies each note, code grounds its dates, FAISS indexes the text for retrieval."""
import calendar
import json
import re
from dataclasses import dataclass
from datetime import date, timedelta

import faiss
import pandas as pd

from .llm import LLM
from .metrics import HISTORY_WEEKS

ALL_ROUTES = "all routes"
MAX_RANGE_DAYS = 180  # sanity bound on a stated date range
ONGOING_BASIS = f"no end date stated: credited for {HISTORY_WEEKS} weeks (policy), until the baseline absorbs it"
COST_WORDS = re.compile(r"cost|rate|price|charge|toll|freight|fare|tariff|surcharge", re.I)
PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*%")  # "5-7%" -> 7: the number right before % is the upper bound
# Forward-looking wording: a forecast is not evidence that costs went up. ("may" is left out: it's a month.)
SPECULATIVE = re.compile(r"\b(expect(s|ed)?|forecast|projected|likely to|might|could|proposed|"
                         r"under consideration|next (week|month|quarter|year))\b", re.I)
WEEK_COUNT = re.compile(r"\b(\d+|two|three|four|five|six|seven|eight)[\s-]weeks?\b|\b(fortnight)\b", re.I)
NUMBER_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "fortnight": 2}

EXTRACT_SYSTEM = """You read short logistics notes and classify each one's effect on freight costs.
You get one or more notes, each with a note_id. Judge every note independently, on its own text only.
Return JSON {"notes": [...]} with exactly one entry per note_id.

cost_effect:
- "increase": the note says freight costs, rates or surcharges WENT UP (already happened) for the routes it names.
- "no_change": the note says costs were not (significantly) affected, were absorbed, had no rate change,
  stayed stable/normal, the affected routes are not part of this dataset, OR it only predicts, expects,
  proposes or plans a future change (a forecast is not an increase).
- "decrease": costs fell, a surcharge was withdrawn, or conditions improved / returned to normal.

duration ("not_applicable" whenever cost_effect is not "increase"):
- "one_week": the effect itself lasts a single week (e.g. "festival week", "for the week").
  A lasting change that merely HAPPENED this week (rates raised, revised, introduced) is "ongoing".
- "n_weeks": a stated number of weeks (e.g. "lasted two weeks").
- "date_range": the note states BOTH a start and an end date. Put them in range_start/range_end as YYYY-MM-DD,
  using the note's year. range_end is the LAST affected day (the day before things were back to normal).
- "ongoing": a lasting change with no stated end (e.g. "starting this week", "from October 1").
Use "" for range_start/range_end when not a date_range. Never invent an end date."""

ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "note_id": {"type": "string"},
        "cost_effect": {"type": "string", "enum": ["increase", "no_change", "decrease"]},
        "duration": {"type": "string", "enum": ["one_week", "n_weeks", "date_range", "ongoing", "not_applicable"]},
        "range_start": {"type": "string"},
        "range_end": {"type": "string"},
    },
    "required": ["note_id", "cost_effect", "duration", "range_start", "range_end"],
    "additionalProperties": False,
}
EXTRACT_SCHEMA = {"type": "object", "properties": {"notes": {"type": "array", "items": ITEM_SCHEMA}},
                  "required": ["notes"], "additionalProperties": False}


def route_key(route: str) -> str:
    """Direction-insensitive route key: 'Bangalore - Chennai' and 'chennai-bangalore' -> 'bangalore|chennai'.
    A corridor note (flood, toll, festival) affects trucks in both directions."""
    return "|".join(sorted(c.strip().casefold() for c in route.split("-")))


def note_routes(applies_to: str) -> frozenset[str]:
    """'All Routes', 'Mumbai-Pune', or a list like 'Mumbai-Pune; Delhi-Jaipur and Delhi-Chennai'."""
    parts = [p.strip() for p in re.split(r"[,;/&]|\band\b", applies_to) if p.strip()]
    return frozenset(ALL_ROUTES if p.casefold() in (ALL_ROUTES, "all") else route_key(p) for p in parts)


@dataclass(frozen=True)
class Note:
    note_id: str
    date: date
    applies_to: str               # as written in the notes file, for display
    text: str
    routes: frozenset[str]        # normalised route keys, or {ALL_ROUTES}
    raises_cost: bool             # only grounded "increase" notes can ever justify a rise
    effect_start: date | None     # inclusive window in which the note can explain a rise
    effect_end: date | None
    max_increase_pct: float | None  # parsed from cost sentences by regex, not by the LLM
    extraction: dict              # raw LLM fields + how the window was grounded, for the audit trail

    @property
    def all_routes(self) -> bool:
        return ALL_ROUTES in self.routes

    def covers(self, route: str) -> bool:
        return self.all_routes or route_key(route) in self.routes


def load_notes(path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = {"note_id", "date", "applies_to", "note"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    return df


def stated_increase_pct(text: str) -> float | None:
    """Largest % in a sentence that talks about cost ("costs up 5-7%" -> 7; "70% of fleets" ignored)."""
    pcts = [float(p) for s in re.split(r"[.;]", text) if COST_WORDS.search(s) for p in PERCENT.findall(s)]
    return max(pcts) if pcts else None


def stated_weeks(text: str) -> int | None:
    m = WEEK_COUNT.search(text)
    if not m:
        return None
    word = (m.group(1) or m.group(2)).lower()
    return int(word) if word.isdigit() else NUMBER_WORDS[word]


def mentioned(d: date, text: str) -> bool:
    """Is this calendar date written in the text (e.g. 'Mar 8', '8th March', '2025-03-08')?"""
    months = f"{calendar.month_name[d.month]}|{calendar.month_abbr[d.month]}"
    day = rf"0?{d.day}(?:st|nd|rd|th)?"
    return bool(d.isoformat() in text
                or re.search(rf"\b(?:{months})\.?\s+{day}\b", text, re.I)
                or re.search(rf"\b(?:{months})\.?\s+\d{{1,2}}\s*[-–]\s*{day}\b", text, re.I)  # "Sep 9-10"
                or re.search(rf"\b{day}\s+(?:of\s+)?(?:{months})\b", text, re.I))


def stated_range(note_date: date, fields: dict, text: str) -> tuple[date, date] | None:
    """The LLM's range_start/range_end, only if both are written in the note text and plausible."""
    try:
        start = date.fromisoformat(fields["range_start"]) if fields["range_start"] else note_date
        end = date.fromisoformat(fields["range_end"])
    except ValueError:
        return None
    grounded = mentioned(end, text) and (start == note_date or mentioned(start, text))
    sane = start <= end and (end - start).days <= MAX_RANGE_DAYS and abs((start - note_date).days) <= 31
    return (start, end) if grounded and sane else None


def effect_window(note_date: date, fields: dict, text: str) -> tuple[date, date, str] | None:
    """Inclusive window in which the note can explain a rise, and how it was derived.
    Precedence: dates written in the note > a week count written in the note > the LLM's duration label.
    The label is the least stable part of a small model's reading (it flipped at temperature 0.8 in the
    stress check), so it only decides when the text itself states nothing."""
    if span := stated_range(note_date, fields, text):
        return *span, "date range stated in the note"
    if weeks := stated_weeks(text):
        return note_date, note_date + timedelta(weeks=weeks, days=-1), f"{weeks} weeks stated in the note"
    duration = fields["duration"]
    if duration in ("n_weeks", "not_applicable"):  # no count written / no duration at all: unusable
        return None
    if duration == "one_week":
        return note_date, note_date + timedelta(days=6), "one week from the note date"
    # Ongoing: after HISTORY_WEEKS the trailing baseline already contains the new cost level,
    # so a lasting change can only explain a rise vs own history for that long.
    return note_date, note_date + timedelta(weeks=HISTORY_WEEKS, days=-1), ONGOING_BASIS


def check_fields(item) -> dict:
    """One note's fields, or {'error': ...} if unusable (the note then cannot justify anything)."""
    try:
        for key, prop in ITEM_SCHEMA["properties"].items():
            if "enum" in prop and item.get(key) not in prop["enum"]:
                raise ValueError(f"{key}={item.get(key)!r}")
        return {k: item.get(k) or "" for k in ITEM_SCHEMA["properties"] if k != "note_id"}
    except (ValueError, TypeError, AttributeError) as exc:
        return {"error": f"unreadable LLM reading ({exc}): {str(item)[:120]!r}"}


def parse_extraction(reply: str) -> dict[str, dict]:
    """note_id -> fields. A reply that isn't the expected JSON yields {} (every note then errors)."""
    try:
        items = json.loads(reply)["notes"]
        return {str(i.get("note_id")): check_fields(i) for i in items if isinstance(i, dict)}
    except (ValueError, TypeError, KeyError):  # JSONDecodeError is a ValueError
        return {}


def batches(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def extract_notes(rows: list, llm: LLM) -> list[Note]:
    """Read notes in batches of llm.batch_size (one call for all 10 notes)."""
    notes = []
    for chunk in batches(rows, llm.batch_size):
        user = "\n\n".join(f"note_id: {r.note_id}\nNote date: {r.date}\nApplies to: {r.applies_to}\nNote: {r.note}"
                           for r in chunk)
        readings = parse_extraction(llm.chat(EXTRACT_SYSTEM, user, schema=EXTRACT_SCHEMA))
        notes += [build_note(r, readings.get(r.note_id, {"error": "missing from the LLM reply"})) for r in chunk]
    return notes


def build_note(row, fields: dict) -> Note:
    note_date = date.fromisoformat(row.date)
    window = None
    if fields.get("cost_effect") == "increase":
        if SPECULATIVE.search(row.note):
            fields["guard"] = "forward-looking wording: treated as not raising cost"
        else:
            window = effect_window(note_date, fields, row.note)
    if window:
        fields["window_basis"] = window[2]
    return Note(note_id=row.note_id, date=note_date, applies_to=row.applies_to, text=row.note,
                routes=note_routes(row.applies_to),
                raises_cost=window is not None,
                effect_start=window[0] if window else None,
                effect_end=window[1] if window else None,
                max_increase_pct=stated_increase_pct(row.note),
                extraction=fields)


class NoteIndex:
    """Vector index over note text with a metadata pre-filter on route (filtered vector search)."""

    def __init__(self, notes: list[Note], llm: LLM):
        self.notes, self.llm = notes, llm
        vectors = llm.embed([f"{n.applies_to}: {n.text}" for n in notes])
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def search(self, query: str, route: str, k: int | None = None) -> list[tuple[Note, float]]:
        """Route-relevant notes, most similar first. k=None returns all of them, so a justifying note can
        never be lost to ranking; similarity only orders candidates and picks the closest near-miss.
        Known limit: gating every route-relevant note suits tens of notes; pass k once there are thousands."""
        scores, ids = self.index.search(self.llm.embed([query]), len(self.notes))
        hits = [(self.notes[i], round(float(s), 4)) for s, i in zip(scores[0], ids[0])
                if self.notes[i].covers(route)]
        hits.sort(key=lambda h: (-h[1], h[0].note_id))  # deterministic tie-break
        return hits[:k] if k else hits
