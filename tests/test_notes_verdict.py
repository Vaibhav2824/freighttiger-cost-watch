"""Note reading guards and verdict gates. No LLM: a fake returns canned replies."""
import json
from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from assistant.checks import GOLD_NOTES, weeks_covered
from assistant.notes import (ONGOING_BASIS, Note, effect_window, extract_notes, mentioned, note_routes, parse_extraction,
                             route_key, stated_increase_pct)
from assistant.pipeline import NOTES
from assistant.verdict import decide


class FakeLLM:
    batch_size = 25

    def __init__(self, reply: str):
        self.reply = reply

    def chat(self, system, user, schema=None):
        return self.reply


def increase(duration, start="", end=""):
    return json.dumps({"cost_effect": "increase", "duration": duration, "range_start": start, "range_end": end})


def batch_reply(*items):
    """What the LLM returns for a batch: {"notes": [{note_id, ...fields}]}."""
    return json.dumps({"notes": [{"note_id": "X", **json.loads(i)} for i in items]})


def read_one(text, reply, applies_to="Mumbai-Pune"):
    row = SimpleNamespace(note_id="X", date="2025-04-07", applies_to=applies_to, note=text)
    return extract_notes([row], FakeLLM(reply))[0]


@pytest.fixture(scope="module")
def notes():
    """Gold note metadata, so gate tests don't depend on the LLM."""
    raw = pd.read_csv(NOTES, dtype=str).set_index("note_id")
    out = {}
    for g in json.loads(GOLD_NOTES.read_text()):
        r = raw.loc[g["note_id"]]
        out[g["note_id"]] = Note(g["note_id"], date.fromisoformat(r.date), r.applies_to, r.note,
                                 note_routes(r.applies_to), g["raises_cost"],
                                 date.fromisoformat(g["effect_start"]) if g["effect_start"] else None,
                                 date.fromisoformat(g["effect_end"]) if g["effect_end"] else None,
                                 g["max_increase_pct"], {})
    return out


def row(route, week, own=15.0, peer=5.0, by_peer=False):
    return SimpleNamespace(route=route, week_of=pd.Timestamp(week), vs_own_pct=own, vs_peer_pct=peer,
                           anomaly_peer=by_peer)


# --- reading a note ------------------------------------------------------------------------------

D = date(2025, 4, 7)


@pytest.mark.parametrize("fields, text, expected", [
    (increase("one_week"), "", (D, date(2025, 4, 13))),
    (increase("n_weeks"), "lasted two weeks", (D, date(2025, 4, 20))),
    (increase("n_weeks"), "lasted a while", None),                                   # no count -> unusable
    (increase("not_applicable"), "rates went up", None),                             # raise without a duration
    (increase("ongoing"), "", (D, date(2025, 6, 1))),                                # 8-week policy
    (increase("date_range", "2025-04-07", "2025-04-12"), "from Apr 7 to Apr 12", (D, date(2025, 4, 12))),
    (increase("date_range", "2025-04-07", "2025-04-30"), "raised from April 7", (D, date(2025, 6, 1))),  # invented end
    (increase("date_range", "2025-04-07", "2025-04-30"), "strike lasted 3 weeks", (D, date(2025, 4, 27))),
    (increase("date_range", "2025-04-07", "not a date"), "", (D, date(2025, 6, 1))),
    # The label is the unstable part: text-stated dates / week counts win over it (stress-check finding).
    (increase("one_week", "2025-04-07", "2025-04-12"), "from Apr 7 to Apr 12", (D, date(2025, 4, 12))),
    (increase("one_week"), "a strike that lasted two weeks", (D, date(2025, 4, 20))),
    (increase("ongoing"), "a surcharge for three weeks", (D, date(2025, 4, 27))),
])
def test_effect_window_only_trusts_dates_written_in_the_note(fields, text, expected):
    window = effect_window(D, json.loads(fields), text)
    assert (window[:2] if window else None) == expected


@pytest.mark.parametrize("d, text, ok", [
    (date(2025, 3, 8), "from Feb 24 to Mar 8", True),
    (date(2025, 3, 8), "until 8th March", True),
    (date(2025, 9, 10), "a strike on Sep 9-10", True),
    (date(2025, 3, 8), "on 2025-03-08", True),
    (date(2025, 3, 8), "from Feb 24 to Mar 18", False),
    (date(2025, 3, 8), "in March", False),
])
def test_mentioned(d, text, ok):
    assert mentioned(d, text) is ok


def test_percent_only_counts_cost_sentences():
    assert stated_increase_pct("pushing up transportation costs across all routes by roughly 5-7%.") == 7
    assert stated_increase_pct("About 70% of fleets reported GPS outages; freight rates were unchanged.") is None


@pytest.mark.parametrize("reply", ["not json", "[]", '{"notes": "x"}', '{"notes": [{"note_id": "X", "cost_effect": "up"}]}',
                                   '{"notes": [{"note_id": "OTHER", "cost_effect": "increase", "duration": "ongoing"}]}'])
def test_unreadable_or_missing_llm_reading_cannot_justify(reply):
    n = read_one("Rates rose.", reply)
    assert not n.raises_cost and "error" in n.extraction
    assert "error" in parse_extraction(reply).get("X", {"error": "absent"})


def test_batch_reading_maps_each_note_by_id():
    rows = [SimpleNamespace(note_id=i, date="2025-04-07", applies_to="Mumbai-Pune", note="Rates rose.") for i in "AB"]
    reply = json.dumps({"notes": [{"note_id": "B", **json.loads(increase("one_week"))},
                                  {"note_id": "A", "cost_effect": "no_change", "duration": "one_week",
                                   "range_start": "", "range_end": ""}]})
    a, b = extract_notes(rows, FakeLLM(reply))
    assert (a.note_id, a.raises_cost, b.note_id, b.raises_cost) == ("A", False, "B", True)


def test_forecast_is_never_a_raise_even_if_the_llm_says_so():
    n = read_one("Analysts expect costs to rise next quarter.", batch_reply(increase("ongoing")), "Delhi-Jaipur")
    assert not n.raises_cost and "guard" in n.extraction


def test_route_normalisation():
    assert route_key("Bangalore - Chennai") == route_key("chennai-bangalore")
    assert note_routes("All routes") == note_routes("All Routes") == frozenset({"all routes"})
    assert note_routes("Mumbai-Pune; Delhi-Jaipur and Delhi-Chennai") == {
        route_key("Mumbai-Pune"), route_key("Delhi-Jaipur"), route_key("Delhi-Chennai")}


def test_weeks_covered():
    assert weeks_covered("2025-09-08", "2025-09-10") == weeks_covered("2025-09-09", "2025-09-10")
    assert len(weeks_covered("2025-02-24", "2025-03-08")) == 2
    assert weeks_covered("", "") == frozenset()


# --- verdict gates: every trap note must be rejected, every real one accepted only in its window -----

@pytest.mark.parametrize("route, week, note_id, justified", [
    ("Ahmedabad-Mumbai", "2025-01-20", "N002", True),
    ("Ahmedabad-Mumbai", "2025-01-27", "N002", False),   # festival week is over
    ("Chennai-Bangalore", "2025-03-03", "N001", True),   # week overlaps flood (ends Mar 8)
    ("Chennai-Bangalore", "2025-03-10", "N001", False),  # after the flood window
    ("Chennai-Bangalore", "2025-03-17", "N009", False),  # "returned to normal" is not a cause
    ("Mumbai-Pune", "2025-02-24", "N001", False),        # right time, wrong route
    ("Mumbai-Delhi", "2024-07-29", "N005", False),       # "costs were not significantly affected"
    ("Delhi-Jaipur", "2024-05-20", "N007", False),       # road improvement
    ("Kolkata-Bhubaneswar", "2025-06-09", "N008", False),
    ("Mumbai-Pune", "2025-09-22", "N006", False),
    ("Mumbai-Pune", "2025-10-27", "N010", False),        # "without a rate change"
    ("Delhi-Chennai", "2024-03-11", "N004", False),      # toll on routes outside the dataset
])
def test_gates(notes, route, week, note_id, justified):
    assert decide(row(route, week), [(notes[note_id], 1.0)]).justified is justified


def test_corridor_note_covers_the_reverse_direction(notes):
    flood = notes["N001"]
    reverse = Note(**{**flood.__dict__, "applies_to": "Bangalore-Chennai", "routes": note_routes("Bangalore-Chennai")})
    assert decide(row("Chennai-Bangalore", "2025-02-24"), [(reverse, 1.0)]).justified


def test_all_routes_note_justifies_own_rise_within_its_size_and_window(notes):
    diesel = notes["N003"]
    assert decide(row("Delhi-Chennai", "2025-05-05", own=6.2), [(diesel, 1.0)]).justified
    assert not decide(row("Mumbai-Delhi", "2025-05-12", own=11.6), [(diesel, 1.0)]).justified  # > 7% + 3
    assert not decide(row("Mumbai-Pune", "2025-06-23", own=6.2, peer=20.6, by_peer=True),
                      [(diesel, 1.0)]).justified  # peers got the same diesel hike
    assert not decide(row("Delhi-Chennai", "2025-09-01", own=6.0), [(diesel, 1.0)]).justified  # baseline absorbed it


def test_policy_end_date_is_never_shown_as_if_the_note_said_it(notes):
    diesel = notes["N003"]
    ongoing = Note(**{**diesel.__dict__, "extraction": {"window_basis": ONGOING_BASIS}})
    why = decide(row("Mumbai-Pune", "2025-12-22", own=5.7), [(ongoing, 1.0)]).rejection
    assert "no end date" in why and "2025-06-29" not in why
    stated = decide(row("Chennai-Bangalore", "2025-03-10"), [(notes["N001"], 1.0)]).rejection
    assert "stated period (2025-02-24 to 2025-03-08)" in stated


def test_unexplained_cites_nearest_miss(notes):
    v = decide(row("Mumbai-Delhi", "2025-05-12", own=11.6), [(notes["N005"], 0.9), (notes["N003"], 0.5)])
    assert not v.justified and v.note.note_id == "N003" and "7%" in v.rejection


def test_justifying_note_found_even_when_ranked_last(notes):
    candidates = [(notes[n], s) for n, s in (("N009", 0.9), ("N003", 0.8), ("N006", 0.7), ("N004", 0.6), ("N001", 0.1))]
    v = decide(row("Chennai-Bangalore", "2025-02-24"), candidates)
    assert v.justified and v.note.note_id == "N001"
