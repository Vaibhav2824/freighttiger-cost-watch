"""End-to-end run: metrics -> flags -> retrieval -> gates -> reasons -> output files."""
import json
import logging
from pathlib import Path

import pandas as pd

from .explain import write_reasons
from .llm import LLM
from .metrics import load_shipments, weekly_metrics
from .notes import NoteIndex, extract_notes, load_notes
from .verdict import decide

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SHIPMENTS = DATA / "shipment_records.csv"
NOTES = DATA / "context_notes.csv"
SAMPLE_FORMAT = DATA / "sample_output_format_v2.csv"
OUT_DIR = ROOT / "outputs"

log = logging.getLogger(__name__)


def output_columns() -> list[str]:
    """The sample file's header is the contract."""
    return list(pd.read_csv(SAMPLE_FORMAT, nrows=0).columns)


def retrieval_query(row) -> str:
    return f"Reason for a freight cost increase on the {row.route} route in the week of {row.week_of:%d %B %Y}"


def output_row(row, verdict, reason: str) -> dict:
    return {"route": row.route,
            "week_of": row.week_of.date().isoformat(),
            "cost_per_tonne_km": f"{row.cost_per_tonne_km:.2f}",  # always 2 decimals, as in the sample
            "vs_own_history": f"{row.vs_own_pct:+.1f}% vs this route's past average",
            "vs_similar_routes": f"{row.vs_peer_pct:+.1f}% vs similar-length routes this week",
            "flagged": "No (justified)" if verdict.justified else "Yes",
            "matched_note_id": verdict.note.note_id if verdict.justified else "",
            "reason": reason}


def route_summary(weekly: pd.DataFrame, output: pd.DataFrame) -> str:
    """One line per episode, so 12 Mumbai-Pune alerts read as one story."""
    flagged = weekly[weekly.is_anomaly].assign(week=weekly.week_of.dt.strftime("%Y-%m-%d"))
    merged = flagged.merge(output, left_on=["route", "week"], right_on=["route", "week_of"], suffixes=("", "_out"))
    lines = ["# Flagged episodes by route", "",
             "| route | episode | flagged weeks | unexplained | peak vs own history | peak vs similar routes | "
             "latest vs pre-episode level | notes cited |", "|---|---|---|---|---|---|---|---|"]
    for (route, start), ep in merged.groupby(["route", "episode_start"]):
        cited = ", ".join(sorted(set(ep.matched_note_id) - {""})) or "-"
        lines.append(f"| {route} | {start:%Y-%m-%d} to {ep.week.max()} | {len(ep)} | "
                     f"{(ep.flagged == 'Yes').sum()} | {ep.vs_own_pct.max():+.1f}% | {ep.vs_peer_pct.max():+.1f}% | "
                     f"{ep.vs_pre_episode_pct.iloc[-1]:+.1f}% | {cited} |")
    return "\n".join(lines) + "\n"


def notes_audit(notes) -> list[dict]:
    return [{"note_id": n.note_id, "date": str(n.date), "applies_to": n.applies_to, "text": n.text,
             "raises_cost": n.raises_cost, "effect_start": str(n.effect_start or ""),
             "effect_end": str(n.effect_end or ""), "max_increase_pct": n.max_increase_pct,
             "llm_fields": n.extraction} for n in notes]


def run(shipments_path: Path = SHIPMENTS, notes_path: Path = NOTES, out_dir: Path = OUT_DIR,
        options: dict | None = None, cache_dir: Path | None = None) -> pd.DataFrame:
    llm = LLM(options, cache_dir)
    weekly = weekly_metrics(load_shipments(shipments_path))
    anomalies = weekly[weekly.is_anomaly]
    log.info("%d route-weeks computed, %d flagged", len(weekly), len(anomalies))

    notes = extract_notes(list(load_notes(notes_path).itertuples()), llm)
    index = NoteIndex(notes, llm)
    log.info("%d notes read by %s and indexed", len(notes), llm.chat_model)

    flagged = list(anomalies.itertuples())
    verdicts = [decide(row, index.search(retrieval_query(row), row.route)) for row in flagged]
    reasons = write_reasons(list(zip(flagged, verdicts)), llm)

    rows, audit = [], []
    for i, (row, verdict, (reason, source, problems)) in enumerate(zip(flagged, verdicts, reasons), start=1):
        rows.append(output_row(row, verdict, reason))
        audit.append({"route": row.route, "week_of": row.week_of.date().isoformat(),
                      "flagged_by": [k for k in ("own", "peer") if getattr(row, f"anomaly_{k}")],
                      "candidates": [{"note_id": n, "similarity": s, "rejected_because": r}
                                     for n, s, r in verdict.considered],
                      "reason_source": source, "validator_problems": problems})
        log.info("[%d/%d] %s %s -> %s", i, len(flagged), row.route, row.week_of.date(),
                 rows[-1]["flagged"] + (f" ({rows[-1]['matched_note_id']})" if verdict.justified else ""))

    output = pd.DataFrame(rows, columns=output_columns())
    out_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_dir / "output.csv", index=False)
    weekly.assign(week_of=weekly.week_of.dt.date, episode_start=weekly.episode_start.dt.date) \
          .round(4).to_csv(out_dir / "weekly_metrics.csv", index=False)
    (out_dir / "notes_extracted.json").write_text(json.dumps(notes_audit(notes), indent=2), encoding="utf-8")
    (out_dir / "decisions.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (out_dir / "summary.md").write_text(route_summary(weekly, output), encoding="utf-8")
    (out_dir / "token_log.json").write_text(json.dumps(
        {**llm.usage_report(), "flagged_rows": len(output),
         "reasons_from_llm": sum(a["reason_source"] == "llm" for a in audit),
         "reasons_from_template": sum(a["reason_source"] == "template" for a in audit)}, indent=2))
    return output
