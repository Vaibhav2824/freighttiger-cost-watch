"""CLI: python -m assistant {run,ask,repro,stress,eval}"""
import argparse
import json
import logging
import sys
from pathlib import Path

from .checks import evaluate, reproducibility, stress
from .llm import LLM
from .pipeline import NOTES, OUT_DIR, ROOT, SHIPMENTS, run
from .qa import ask


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m assistant", description="Freight cost watch assistant")
    parser.add_argument("--shipments", type=Path, default=SHIPMENTS, help="shipment records CSV")
    parser.add_argument("--notes", type=Path, default=NOTES, help="context notes CSV")
    parser.add_argument("--out", type=Path, default=OUT_DIR, help="output directory")
    parser.add_argument("--cache", action="store_true",
                        help="replay identical LLM requests from .llm_cache/ (byte-identical reruns; off for repro/stress)")
    parser.add_argument("-q", "--quiet", action="store_true", help="no progress lines")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run", help="full run -> OUT/output.csv (+ audit files, token log)")
    q = sub.add_parser("ask", help="ask a plain-English question about the latest run")
    q.add_argument("question")
    sub.add_parser("repro", help="3 cold runs, diffed -> OUT/repro_report.md")
    sub.add_parser("stress", help="5 runs at temperature 0.8 vs the shipped output -> OUT/stress_report.md")
    sub.add_parser("eval", help="score the latest run + adversarial notes vs eval/ gold -> OUT/eval_report.md")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="  %(message)s")
    for path in (args.shipments, args.notes):
        if args.command in ("run", "repro", "stress") and not path.exists():
            sys.exit(f"Input not found: {path}")
    if args.command in ("ask", "stress", "eval") and not (args.out / "output.csv").exists():
        sys.exit(f"No results in {args.out} yet - run `python -m assistant run` first.")

    try:
        dispatch(args)
    except PermissionError as exc:  # typically output.csv open in Excel, which locks it on Windows
        sys.exit(f"Cannot write {exc.filename}: close it in Excel (or any other program) and re-run.")
    except RuntimeError as exc:  # LLM server down, quota used up, missing key: say so plainly
        sys.exit(str(exc))


def dispatch(args: argparse.Namespace) -> None:
    if args.command == "run":
        output = run(args.shipments, args.notes, args.out, cache_dir=ROOT / ".llm_cache" if args.cache else None)
        log = json.loads((args.out / "token_log.json").read_text())
        print(f"{len(output)} flagged route-weeks: {(output.flagged == 'Yes').sum()} unexplained, "
              f"{(output.flagged != 'Yes').sum()} justified -> {args.out / 'output.csv'}")
        print(f"LLM ({log['chat_model']}): {log['chat_calls']} calls, {log['input_tokens']} in / {log['output_tokens']} out tokens; "
              f"embeddings: {log['embed_calls']} calls, {log['embed_input_tokens']} tokens")
    elif args.command == "ask":
        llm = LLM()
        print(ask(args.question, args.out, llm))
        print(f"\n[{llm.usage.chat_calls} LLM call(s), {llm.usage.input_tokens} in / {llm.usage.output_tokens} out tokens]")
    elif args.command == "repro":
        print(reproducibility(out_dir=args.out, shipments=args.shipments, notes=args.notes))
    elif args.command == "stress":
        print(stress(args.out, args.shipments, args.notes))
    elif args.command == "eval":
        print(evaluate(args.out))


if __name__ == "__main__":
    main()
