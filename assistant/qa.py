"""Stretch goal: answer plain-English questions using only the assistant's own results."""
import calendar
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .explain import ROUTE, clean, ungrounded
from .llm import LLM
from .notes import ALL_ROUTES, note_routes, route_key

QA_SYSTEM = """You answer questions about a freight-cost monitoring report.
Use ONLY the findings, weekly figures and notes provided. Each finding already states its verdict;
repeat verdicts and note ids exactly, never change them. A week marked "justified" IS explained by its note.
Copy numbers exactly as written; do not compute new ones. If the provided data does not answer the
question, say so plainly. Never guess a cause that is not written in a finding. Answer in 2-4 sentences."""

MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
MONTHS |= {m.lower(): i for i, m in enumerate(calendar.month_abbr) if m}
MAX_WEEKLY_ROWS = 16  # most recent weeks in scope
MAX_FALLBACK_LINES = 8
# Listing/counting questions are answered by code. "Why ..." questions go to the LLM.
LIST_QUESTION = re.compile(r"^\s*(which|list|how many|show)\b", re.I)


@dataclass(frozen=True)
class Scope:
    cities: frozenset
    months: frozenset
    years: frozenset
    since: pd.Timestamp | None = None  # "since/from/after September 2025" -> everything from that month on

    def matches(self, routes: str, day: pd.Timestamp) -> bool:
        """`routes` is a route ('Mumbai-Pune') or a note's applies_to ('All Routes', 'A-B; C-D')."""
        if self.since is not None:
            if day < self.since:
                return False
        elif (self.months and day.month not in self.months) or (self.years and day.year not in self.years):
            return False
        keys = note_routes(routes)
        if not self.cities or ALL_ROUTES in keys:
            return True
        # one city -> every route touching it; two -> the route between them
        wanted = {c.casefold() for c in self.cities}
        return any(wanted <= set(k.split("|")) if len(wanted) > 1 else wanted & set(k.split("|")) for k in keys)


def parse_scope(question: str, routes) -> Scope:
    q = question.lower()
    since = re.search(r"\b(?:since|from|after)\s+([a-z]+)\s+(20\d\d)\b", q)
    return Scope(frozenset(c for r in routes for c in r.split("-") if c.lower() in q),
                 frozenset(MONTHS[w] for w in re.findall(r"[a-z]+", q) if w in MONTHS),
                 frozenset(int(y) for y in re.findall(r"\b(20\d\d)\b", q)),
                 pd.Timestamp(int(since[2]), MONTHS[since[1]], 1) if since and since[1] in MONTHS else None)


def finding_line(r) -> str:
    verdict = f"justified by {r.matched_note_id}" if r.matched_note_id else "unexplained, flagged"
    return (f"- {r.route}, week of {r.week_of}: {r.cost_per_tonne_km}/tonne-km ({r.vs_own_history}; "
            f"{r.vs_similar_routes}). Verdict: {verdict}. {r.reason}")


def build_context(scope: Scope, findings: pd.DataFrame, weekly: pd.DataFrame,
                  notes: pd.DataFrame) -> tuple[str, list[str], list[str]]:
    """Deterministic retrieval over the results. Returns (context, finding lines, source labels)."""
    def keep(df, route_col, date_col):
        return df[[scope.matches(r, pd.Timestamp(d)) for r, d in zip(df[route_col], df[date_col])]]

    f = keep(findings, "route", "week_of")
    w = keep(weekly, "route", "week_of").tail(MAX_WEEKLY_ROWS) if scope.cities else weekly.iloc[0:0]
    n = keep(notes, "applies_to", "date")
    found = [finding_line(r) for r in f.itertuples()]
    lines = ["Flagged findings:", *(found or ["- none"])]
    if not w.empty:
        lines += ["Weekly figures (all weeks, flagged or not):"]
        lines += [f"- {r.route}, week of {r.week_of}: {r.cost_per_tonne_km:.2f}/tonne-km, "
                  f"{r.vs_own_pct:+.1f}% vs own history, {r.vs_peer_pct:+.1f}% vs similar routes"
                  for r in w.itertuples()]
    lines += ["Notes:"] + [f"- {r.note_id} ({r.date}, {r.applies_to}): {r.text}" for r in n.itertuples()]
    sources = [f"{r.route} {r.week_of}" for r in f.itertuples()] + list(n.note_id)
    return "\n".join(lines), found, sources if (len(f) or len(w)) else []


def list_answer(question: str, findings: pd.DataFrame, scope: Scope) -> str:
    """'Which routes / how many weeks ...' questions: counted and listed by code, so nothing can be invented."""
    q = question.lower()
    rows = findings[[scope.matches(r, pd.Timestamp(d)) for r, d in zip(findings.route, findings.week_of)]]
    if "unexplained" in q:  # checked first: "unexplained" contains "explained"
        rows, kind = rows[rows.matched_note_id == ""], "unexplained"
    elif "justified" in q or "explained" in q:
        rows, kind = rows[rows.matched_note_id != ""], "justified"
    else:
        kind = "flagged"
    if rows.empty:
        return f"No {kind} route-weeks match that question."
    lines = [f"{len(rows)} {kind} route-week(s) across {rows.route.nunique()} route(s):"]
    for route, g in rows.groupby("route", sort=False):
        notes = sorted(set(g.matched_note_id) - {""})
        lines.append(f"- {route}: {len(g)} week(s), {g.week_of.min()} to {g.week_of.max()}"
                     + (f", justified by {', '.join(notes)}" if notes else ""))
    return "\n".join(lines)


def ask(question: str, out_dir: Path, llm: LLM) -> str:
    findings = pd.read_csv(out_dir / "output.csv", dtype=str, keep_default_na=False)
    weekly = pd.read_csv(out_dir / "weekly_metrics.csv")
    notes = pd.DataFrame(json.loads((out_dir / "notes_extracted.json").read_text(encoding="utf-8")))
    scope = parse_scope(question, weekly.route.unique())
    named = {route_key(m) for m in ROUTE.findall(question)}
    if named and not named & {route_key(r) for r in weekly.route.unique()}:
        return "That route is not in the shipment data, so I have no findings for it."  # no LLM call
    if LIST_QUESTION.search(question):
        return f"{list_answer(question, findings, scope)}\n\nSources: output.csv (counted by code, no LLM call)"
    context, found, sources = build_context(scope, findings, weekly, notes)
    if not sources:
        return "I have no findings or weekly figures matching that question."  # nothing to ground on: no LLM call
    answer = clean(llm.chat(QA_SYSTEM, f"{context}\n\nQuestion: {question}"))
    problems = ungrounded(answer, context) if answer else ["empty answer"]
    if problems:  # same grounding check as the reasons: fall back to the matching findings themselves
        listing = "\n".join(found[:MAX_FALLBACK_LINES]) or "(no flagged findings in scope)"
        answer = (f"I couldn't give a fully grounded summary ({'; '.join(problems)}). "
                  f"The matching findings are:\n{listing}")
    return f"{answer}\n\nSources: {', '.join(sources)}"
