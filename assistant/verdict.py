"""Deterministic verdict: a note justifies a flagged week only if it passes every gate below."""
from dataclasses import dataclass
from datetime import timedelta

import pandas as pd

from .metrics import HISTORY_WEEKS
from .notes import ONGOING_BASIS, Note

MAGNITUDE_TOLERANCE_PCT = 3.0  # rise may exceed a note's stated % by this much (weekly noise) and still count


@dataclass(frozen=True)
class Verdict:
    justified: bool
    note: Note | None                     # the justifying note, or the closest rejected one
    rejection: str | None                 # why `note` does not justify (None when justified)
    considered: tuple[tuple[str, float, str | None], ...]  # (note_id, similarity, rejection) audit trail


def outside_window(note: Note, week_start) -> str:
    """Why the time gate failed. For a note with no stated end, the 8-week cut-off is OUR policy, so its
    end date is deliberately not shown: the LLM must not be able to present it as something the note said."""
    if note.extraction.get("window_basis") == ONGOING_BASIS:
        return (f"it reports a lasting change from {note.effect_start} with no end date, and such a change is "
                f"only credited for {HISTORY_WEEKS} weeks (after that the route's own baseline already includes "
                f"it), so it cannot explain a rise in the week of {week_start}")
    return f"its stated period ({note.effect_start} to {note.effect_end}) does not cover the week of {week_start}"


def check_note(note: Note, row) -> tuple[int, str | None]:
    """(gates passed, rejection). rejection is None when `note` explains this route-week's rise.
    Gates run in order, so a higher count means a nearer miss."""
    week_start = pd.Timestamp(row.week_of).date()
    week_end = week_start + timedelta(days=6)
    gates = (
        (lambda: note.covers(row.route),
         lambda: f"it is about {note.applies_to}, not {row.route}"),
        (lambda: note.raises_cost,
         lambda: "it does not describe a cost increase on these routes"),
        (lambda: note.effect_start <= week_end and note.effect_end >= week_start,
         lambda: outside_window(note, week_start)),
        (lambda: not (note.all_routes and row.anomaly_peer),
         lambda: "it affects all routes equally, so it cannot explain why this route costs more than similar routes"),
        (lambda: note.max_increase_pct is None
                 or row.vs_own_pct <= note.max_increase_pct + MAGNITUDE_TOLERANCE_PCT,
         lambda: f"it accounts for up to about {note.max_increase_pct:g}%, "
                 f"but cost rose {row.vs_own_pct:+.1f}% vs this route's past average"),
    )
    for passed, (ok, why) in enumerate(gates):
        if not ok():
            return passed, why()
    return len(gates), None


def decide(row, candidates: list[tuple[Note, float]]) -> Verdict:
    """Most similar candidate that passes every gate wins; otherwise unexplained,
    citing the nearest miss (most gates passed, then most similar) as the closest note."""
    checks = [(note, score, *check_note(note, row)) for note, score in candidates]
    considered = tuple((n.note_id, score, why) for n, score, _, why in checks)
    for note, _, _, why in checks:
        if why is None:
            return Verdict(True, note, None, considered)
    if not checks:
        return Verdict(False, None, None, considered)
    note, _, _, why = max(checks, key=lambda c: (c[2], c[1]))  # stable: ties keep retrieval order
    return Verdict(False, note, why, considered)
