"""Trust checks: reproducibility, temperature stress, and scoring against the hand-labelled gold sets."""
import hashlib
import json
import tempfile
from pathlib import Path

import pandas as pd

from .llm import GENERATION_OPTIONS, LLM
from .notes import extract_notes, load_notes
from .pipeline import NOTES, OUT_DIR, ROOT, SHIPMENTS, run

GOLD_NOTES = ROOT / "eval" / "gold_notes.json"
GOLD_VERDICTS = ROOT / "eval" / "gold_verdicts.csv"
ADVERSARIAL_NOTES = ROOT / "eval" / "adversarial_notes.csv"
STRICT_COLUMNS = ["route", "week_of", "cost_per_tonne_km", "vs_own_history", "vs_similar_routes",
                  "flagged", "matched_note_id"]  # must be identical run to run; `reason` wording may vary
STRESS_TEMPERATURE = 0.8
STRESS_SEEDS = (1, 2, 3, 4, 5)
NOTE_FIELDS = ["raises_cost", "effect_start", "effect_end"]
HOLDOUT_TARGET = 0.9  # unsafe misreads must be zero; conservative misses are tolerated up to this


def fingerprint(df: pd.DataFrame) -> str:
    """SHA-256 (first 12 hex) of the strict columns: equal fingerprints = identical flags, numbers, verdicts, notes."""
    return hashlib.sha256(df[STRICT_COLUMNS].astype(str).to_csv(index=False).encode()).hexdigest()[:12]


def cold_run(shipments: Path, notes: Path, options: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """A full run from scratch in a throwaway folder: notes re-read, index rebuilt, every reason regenerated."""
    with tempfile.TemporaryDirectory() as tmp:
        df = run(shipments, notes, Path(tmp), options=options)
        return df.astype(str), json.loads((Path(tmp) / "token_log.json").read_text())


def _report(title: str, intro: str, reference: pd.DataFrame, results: list[tuple[str, pd.DataFrame, dict]]) -> str:
    ref = fingerprint(reference)
    lines = [f"# {title}", "", intro, "", f"Reference fingerprint: `{ref}`", "",
             "| run | fingerprint | matches reference | reason wording identical | LLM calls | input tokens | output tokens |",
             "|---|---|---|---|---|---|---|"]
    for label, df, log in results:
        lines.append(f"| {label} | `{fingerprint(df)}` | {'yes' if fingerprint(df) == ref else 'NO'} | "
                     f"{'yes' if df.reason.equals(reference.reason) else 'no (wording only)'} | "
                     f"{log['chat_calls']} | {log['input_tokens']} | {log['output_tokens']} |")
    ok = all(fingerprint(df) == ref for _, df, _ in results)
    lines += ["", f"**Result: {'PASS' if ok else 'FAIL'}**: flags, numbers, verdicts and cited notes "
              f"{'are identical in every run' if ok else 'DIFFER between runs'}.", "",
              f"The fingerprint is a SHA-256 of these columns: {', '.join(STRICT_COLUMNS)}. "
              "The free-text `reason` is excluded; the brief allows its wording to vary."]
    return "\n".join(lines) + "\n"


def reproducibility(runs: int = 3, out_dir: Path = OUT_DIR, shipments: Path = SHIPMENTS, notes: Path = NOTES) -> str:
    """Brief: 'three untouched runs on the same input: identical flags and numbers'."""
    results = [(f"run {i}", *cold_run(shipments, notes)) for i in range(1, runs + 1)]
    report = _report("Reproducibility check",
                     f"{runs} untouched runs at the shipped settings {GENERATION_OPTIONS}, each starting cold. "
                     "Reference = run 1.", results[0][1], results)
    (out_dir / "repro_report.md").write_text(report, encoding="utf-8")
    return report


def stress(out_dir: Path = OUT_DIR, shipments: Path = SHIPMENTS, notes: Path = NOTES) -> str:
    """Brief: 'if anything is random, show it does not move the verdict'. Re-run with sampling switched ON
    (temperature 0.8, several seeds) and compare with the shipped temperature-0 output."""
    reference = pd.read_csv(out_dir / "output.csv", dtype=str, keep_default_na=False)
    results = [(f"T={STRESS_TEMPERATURE}, seed {seed}",
                *cold_run(shipments, notes, {**GENERATION_OPTIONS, "temperature": STRESS_TEMPERATURE, "seed": seed}))
               for seed in STRESS_SEEDS]
    report = _report("Temperature stress check",
                     f"{len(STRESS_SEEDS)} cold runs with sampling switched ON (temperature {STRESS_TEMPERATURE}, "
                     f"seeds {', '.join(map(str, STRESS_SEEDS))}). Reference = the shipped temperature-0 "
                     "`outputs/output.csv`.", reference, results)
    (out_dir / "stress_report.md").write_text(report, encoding="utf-8")
    return report


def weeks_covered(start: str, end: str) -> frozenset:
    """The Monday-to-Sunday weeks a window touches: exactly what the time gate can see."""
    if not start:
        return frozenset()
    days = pd.date_range(start, end)
    return frozenset((days - pd.to_timedelta(days.weekday, unit="D")).date)


def _same_reading(got: dict, want: dict) -> bool:
    return (str(got["raises_cost"]) == str(want["raises_cost"])
            and weeks_covered(got["effect_start"], got["effect_end"])
            == weeks_covered(want["effect_start"], want["effect_end"]))


def _note_errors(got: dict, gold: dict) -> list[str]:
    return [f"{nid}: got {[got[nid][f] for f in NOTE_FIELDS]}, gold {[g[f] for f in NOTE_FIELDS]}"
            for nid, g in gold.items() if not _same_reading(got[nid], g)]


def adversarial(llm: LLM) -> tuple[pd.DataFrame, list[str]]:
    """Read unseen tricky notes (forecasts, withdrawn surcharges, stated durations...) and score vs gold."""
    gold = list(load_notes(ADVERSARIAL_NOTES).itertuples())
    rows, errors = [], []
    for g, n in zip(gold, extract_notes(gold, llm)):
        got = {"raises_cost": str(n.raises_cost), "effect_start": str(n.effect_start or ""),
               "effect_end": str(n.effect_end or "")}
        want = {"raises_cost": g.raises_cost, "effect_start": g.effect_start, "effect_end": g.effect_end}
        ok = _same_reading(got, want)
        # Unsafe = could produce a false "justified": a raise that isn't there, or weeks the note doesn't cover.
        # A shorter window or a missed raise is conservative: it can only yield "unexplained".
        unsafe = bool(weeks_covered(got["effect_start"], got["effect_end"])
                      - weeks_covered(want["effect_start"], want["effect_end"]))
        rows.append({"set": g.set, "ok": ok, "unsafe": unsafe})
        if not ok:
            errors.append(f"{g.note_id} ({g.set}, {'UNSAFE' if unsafe else 'conservative'} miss): "
                          f"got {got}, gold {want}. {g.why}")
    return pd.DataFrame(rows), errors


def evaluate(out_dir: Path = OUT_DIR) -> str:
    """Score the latest run against hand labels, then read the adversarial notes (20 LLM calls)."""
    gold_notes = {n["note_id"]: n for n in json.loads(GOLD_NOTES.read_text())}
    got_notes = {n["note_id"]: n for n in json.loads((out_dir / "notes_extracted.json").read_text(encoding="utf-8"))}
    note_errors = _note_errors(got_notes, gold_notes)

    gold = pd.read_csv(GOLD_VERDICTS, dtype=str, keep_default_na=False)
    out = pd.read_csv(out_dir / "output.csv", dtype=str, keep_default_na=False)
    merged = gold.merge(out, on=["route", "week_of"], how="outer", indicator=True, suffixes=("_gold", ""))
    missing, extra = merged[merged._merge == "left_only"], merged[merged._merge == "right_only"]
    both = merged[merged._merge == "both"]
    verdict_errors = both[(both.flagged != both.flagged_gold) | (both.matched_note_id != both.matched_note_id_gold)]
    fabricated = both[(both.flagged == "No (justified)") & (both.matched_note_id != both.matched_note_id_gold)]

    decisions = {(d["route"], d["week_of"]): d for d in json.loads((out_dir / "decisions.json").read_text())}
    ranks = [[c["note_id"] for c in decisions[(r.route, r.week_of)]["candidates"]].index(r.matched_note_id) + 1
             for r in gold[gold.matched_note_id != ""].itertuples()]
    from_llm = sum(d["reason_source"] == "llm" for d in decisions.values())
    adv, adv_errors = adversarial(LLM())
    held = adv[adv.set == "holdout"]

    checks = [
        ("Note reading, dataset notes (raises cost + weeks covered)",
         f"{len(gold_notes) - len(note_errors)}/{len(gold_notes)}", not note_errors),
        ("Note reading, adversarial probe set (informed the fixes; accuracy only)",
         f"{adv[adv.set == 'probe'].ok.sum()}/{(adv.set == 'probe').sum()}", True),
        (f"Note reading, adversarial held-out set (target >= {HOLDOUT_TARGET:.0%})",
         f"{held.ok.sum()}/{len(held)}", held.ok.mean() >= HOLDOUT_TARGET),
        ("UNSAFE misreads on adversarial notes (false raise or window too wide)", str(adv.unsafe.sum()),
         not adv.unsafe.any()),
        ("Flagged route-weeks match gold set", f"missing {len(missing)}, extra {len(extra)}", missing.empty and extra.empty),
        ("Verdict + matched_note_id match gold", f"{len(both) - len(verdict_errors)}/{len(both)}", verdict_errors.empty),
        ("Fabricated justifications in output", str(len(fabricated)), fabricated.empty),
        ("Retrieval rank of the gold note (justified rows)", ", ".join(f"#{r}" for r in ranks), all(r == 1 for r in ranks)),
        ("Reasons that passed the grounding validator (rest used the template)", f"{from_llm}/{len(decisions)}", True),
    ]
    lines = ["# Evaluation against the hand-labelled sets", "", "| check | result | pass |", "|---|---|---|"]
    lines += [f"| {name} | {res} | {'yes' if ok else 'NO'} |" for name, res, ok in checks]
    details = note_errors + adv_errors + [
        f"verdict mismatch: {r.route} {r.week_of}: got {r.flagged}/{r.matched_note_id or '-'}, "
        f"gold {r.flagged_gold}/{r.matched_note_id_gold or '-'}" for r in verdict_errors.itertuples()]
    if details:
        lines += ["", "## Mismatches", *[f"- {d}" for d in details]]
    lines += ["", f"**Overall: {'PASS' if all(ok for *_, ok in checks) else 'FAIL'}**"]
    report = "\n".join(lines) + "\n"
    (out_dir / "eval_report.md").write_text(report, encoding="utf-8")
    return report
