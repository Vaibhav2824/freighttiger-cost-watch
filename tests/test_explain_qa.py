"""Grounding validator for reasons and Q&A answers. No LLM: fakes return canned replies."""
import json
from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from assistant.explain import clean, facts, validate, write_reasons
from assistant.notes import Note, note_routes
from assistant.qa import ask, parse_scope
from assistant.verdict import Verdict

FESTIVAL = Note("N002", date(2025, 1, 20), "Ahmedabad-Mumbai", "A regional festival week saw a surcharge.",
                note_routes("Ahmedabad-Mumbai"), True, date(2025, 1, 20), date(2025, 1, 26), None, {})


class FakeLLM:
    batch_size = 25

    def __init__(self, reply=""):
        self.reply, self.calls = reply, 0

    def chat(self, system, user, schema=None):
        self.calls += 1
        return self.reply


def flagged_row(**kw):
    base = dict(route="Ahmedabad-Mumbai", week_of=pd.Timestamp("2025-01-20"), vs_own_pct=29.5, vs_peer_pct=22.5,
                episode_week_no=1, episode_start=pd.Timestamp("2025-01-20"), vs_pre_episode_pct=29.5)
    return SimpleNamespace(**{**base, **kw})


def test_validator_rejects_invented_or_contradicting_text():
    facts_text = "Route: Ahmedabad-Mumbai. Matched note: N002 dated 2025-01-20. Cost change: +29.5%"
    ok = Verdict(True, FESTIVAL, None, ())
    assert validate("The festival surcharge in N002 explains the 29.5% rise.", facts_text, ok) == []
    assert validate("Note N003 also shows diesel costs.", facts_text, ok)          # invented note id
    assert validate("Costs rose 40% due to the festival.", facts_text, ok)         # invented number
    assert validate("Same as on Mumbai-Pune.", facts_text, ok)                      # invented route
    assert validate("The rise is unexplained.", facts_text, ok)                    # contradicts verdict
    assert validate("The rise is justified by the festival.", facts_text, Verdict(False, FESTIVAL, "window", ()))


def reasons_reply(*texts):
    return json.dumps({"reasons": [{"id": f"R{i}", "text": t} for i, t in enumerate(texts, start=1)]})


def test_batched_reasons_are_validated_row_by_row():
    good = "The regional festival week surcharge explains the rise on this corridor."
    items = [(flagged_row(), Verdict(True, FESTIVAL, None, ()))] * 2
    (r1, s1, p1), (r2, s2, p2) = write_reasons(items, FakeLLM(reasons_reply(good, "Costs rose 99% because of N007.")))
    assert (s1, p1) == ("llm", []) and r1 == f"Justified by note N002 (2025-01-20): {good}"
    assert s2 == "template" and p2 and r2.startswith("Justified by note N002 (2025-01-20)")


def test_unicode_hyphens_cannot_smuggle_an_invented_date_or_route():
    sneaky = "The festival surcharge from 2025‑01‑13 on Pune‑Goa explains it."  # U+2011 hyphens
    (reason, source, problems), = write_reasons([(flagged_row(), Verdict(True, FESTIVAL, None, ()))],
                                                FakeLLM(reasons_reply(sneaky)))
    assert source == "template" and any("date" in p for p in problems) and any("route" in p for p in problems)
    good = "The regional festival week surcharge on Ahmedabad‑Mumbai explains the rise."
    (reason, source, _), = write_reasons([(flagged_row(), Verdict(True, FESTIVAL, None, ()))],
                                         FakeLLM(reasons_reply(good)))
    assert source == "llm" and "Ahmedabad-Mumbai" in reason and "‑" not in reason
    assert clean("each week—from 2025‑09‑15") == "each week - from 2025-09-15"  # em dash stays a pause


def test_unusable_reasons_reply_falls_back_to_templates():
    (reason, source, problems), = write_reasons([(flagged_row(), Verdict(True, FESTIVAL, None, ()))], FakeLLM("oops"))
    assert source == "template" and "empty or too short" in problems


def test_sustained_rise_context_is_code_written_and_appended():
    later = flagged_row(episode_week_no=8, episode_start=pd.Timestamp("2025-09-15"), vs_pre_episode_pct=20.6)
    v = Verdict(False, None, None, ())
    (first, _, _), (eighth, _, _) = write_reasons([(flagged_row(), v), (later, v)], FakeLLM(
        reasons_reply(*["No note explains the rise on this route; a human should review it."] * 2)))
    assert "Sustained rise" not in first
    assert eighth.endswith("Sustained rise: flagged week 8 on this route since 2025-09-15, now +20.6% above its level before that.")
    assert "Sustained" not in facts(later, v)  # not shown to the LLM, so it can't be mangled or parroted


def test_scope_parsing():
    s = parse_scope("Why did Chennai-Bangalore get pricier in March 2025?", ["Chennai-Bangalore", "Mumbai-Pune"])
    assert s.cities == {"Chennai", "Bangalore"} and s.months == {3} and s.years == {2025}
    assert s.matches("Chennai-Bangalore", pd.Timestamp("2025-03-10"))
    assert not s.matches("Mumbai-Pune", pd.Timestamp("2025-03-10"))
    assert s.matches("All Routes", pd.Timestamp("2025-03-03"))          # all-routes notes stay in scope
    assert not s.matches("Chennai-Bangalore", pd.Timestamp("2024-03-11"))
    since = parse_scope("What is going on with Mumbai-Pune since September 2025?", ["Mumbai-Pune"])
    assert since.matches("Mumbai-Pune", pd.Timestamp("2025-12-22"))      # "since" is open-ended, not "in"
    assert not since.matches("Mumbai-Pune", pd.Timestamp("2025-08-25"))


@pytest.fixture
def results(tmp_path):
    pd.DataFrame([{"route": "Chennai-Bangalore", "week_of": "2025-03-03", "cost_per_tonne_km": "3.48",
                   "vs_own_history": "+22.8% vs this route's past average",
                   "vs_similar_routes": "+32.0% vs similar-length routes this week", "flagged": "No (justified)",
                   "matched_note_id": "N001", "reason": "Justified by note N001 (2025-02-24): flooding."}]
                 ).to_csv(tmp_path / "output.csv", index=False)
    pd.DataFrame([{"route": "Chennai-Bangalore", "week_of": "2025-03-03", "cost_per_tonne_km": 3.48,
                   "vs_own_pct": 22.8, "vs_peer_pct": 32.0}]).to_csv(tmp_path / "weekly_metrics.csv", index=False)
    (tmp_path / "notes_extracted.json").write_text(json.dumps(
        [{"note_id": "N001", "date": "2025-02-24", "applies_to": "Chennai-Bangalore", "text": "Flooding."}]))
    return tmp_path


def test_unknown_route_answered_without_llm(results):
    llm = FakeLLM("should not be called")
    assert "not in the shipment data" in ask("Why did Pune-Goa get pricier?", results, llm)
    assert llm.calls == 0


def test_list_questions_are_answered_by_code(results):
    llm = FakeLLM("should not be called")
    answer = ask("Which routes had justified rises in March 2025?", results, llm)
    assert "1 justified route-week(s) across 1 route(s)" in answer and "justified by N001" in answer
    assert "No unexplained route-weeks" in ask("How many unexplained weeks did Chennai-Bangalore have?", results, llm)
    assert llm.calls == 0


def test_ungrounded_answer_falls_back_to_the_findings(results):
    answer = ask("Why did Chennai-Bangalore get pricier in March?", results, FakeLLM("Costs rose 55% due to N004."))
    assert "couldn't give a fully grounded summary" in answer and "justified by N001" in answer


def test_grounded_answer_passes_through(results):
    reply = "The week of 2025-03-03 was justified by N001 (flooding); cost was +22.8% vs its past average."
    assert ask("Why did Chennai-Bangalore get pricier in March?", results, FakeLLM(reply)).startswith(reply)
