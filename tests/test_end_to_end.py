"""Whole pipeline, checks, CLI and the HTTP client, offline: fakes stand in for Groq and Ollama."""
import io
import json
import sys
import urllib.error

import numpy as np
import pandas as pd
import pytest

import assistant.__main__ as cli
import assistant.checks as checks
import assistant.llm as llm_module
import assistant.pipeline as pipeline
from assistant.checks import GOLD_VERDICTS

READINGS = {  # what a correct reader returns for the dataset notes; everything else reads as no_change
    "Heavy flooding": {"cost_effect": "increase", "duration": "date_range",
                       "range_start": "2025-02-24", "range_end": "2025-03-08"},
    "regional festival": {"cost_effect": "increase", "duration": "one_week", "range_start": "", "range_end": ""},
    "Diesel prices rose": {"cost_effect": "increase", "duration": "ongoing", "range_start": "", "range_end": ""},
}
NO_CHANGE = {"cost_effect": "no_change", "duration": "not_applicable", "range_start": "", "range_end": ""}


class FakeLLM(llm_module.LLM):
    """Same batching contract as the real client; answers like a correct reader would."""

    def __init__(self, options=None, cache_dir=None):
        super().__init__(options, cache_dir, key="test-key")

    def chat(self, system, user, schema=None):
        self.usage.chat_calls += 1
        if "notes" in schema["properties"]:
            blocks = [b for b in user.split("\n\n") if b.startswith("note_id:")]
            return json.dumps({"notes": [
                {"note_id": b.splitlines()[0].split(": ")[1],
                 **next((r for key, r in READINGS.items() if key in b), NO_CHANGE)} for b in blocks]})
        ids = [line.strip("[]") for line in user.splitlines() if line.startswith("[R")]
        return json.dumps({"reasons": [{"id": i, "text": "The facts above show what happened to this route in that week."}
                                       for i in ids]})

    def embed(self, texts):
        vecs = np.array([[t.count(w) + 1e-3 for w in ("Chennai", "Mumbai", "Delhi", "cost", "diesel", "flood")]
                         for t in texts], dtype="float32")
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    def model_versions(self):
        return {}


@pytest.fixture
def fake(monkeypatch):
    for module in (pipeline, checks, cli):
        monkeypatch.setattr(module, "LLM", FakeLLM)


def test_full_run_matches_gold_and_writes_every_artifact(fake, tmp_path):
    out = pipeline.run(out_dir=tmp_path)
    gold = pd.read_csv(GOLD_VERDICTS, dtype=str, keep_default_na=False)
    assert out[["route", "week_of", "flagged", "matched_note_id"]].values.tolist() == \
        gold[["route", "week_of", "flagged", "matched_note_id"]].values.tolist()
    for name in ("output.csv", "weekly_metrics.csv", "decisions.json", "notes_extracted.json",
                 "summary.md", "token_log.json"):
        assert (tmp_path / name).exists(), name
    assert "| Mumbai-Pune | 2025-09-15 to 2025-12-22 | 11 | 11 |" in (tmp_path / "summary.md").read_text()
    assert pd.read_csv(tmp_path / "output.csv", dtype=str).cost_per_tonne_km.str.match(r"^\d+\.\d\d$").all()


def test_checks_pass_on_a_correct_reader(fake, tmp_path):
    pipeline.run(out_dir=tmp_path)
    assert "**Result: PASS**" in checks.reproducibility(runs=2, out_dir=tmp_path)
    assert "**Result: PASS**" in checks.stress(out_dir=tmp_path)
    shipped = pd.read_csv(tmp_path / "output.csv", dtype=str, keep_default_na=False)
    assert f"Reference fingerprint: `{checks.fingerprint(shipped)}`" in (tmp_path / "stress_report.md").read_text()
    assert not any(p.is_dir() for p in tmp_path.iterdir())  # cold runs leave no folders behind, only the reports
    changed = shipped.assign(flagged=shipped.flagged.replace({"Yes": "No (justified)"}))
    assert checks.fingerprint(changed) != checks.fingerprint(shipped)  # any verdict change shows up
    report = checks.evaluate(tmp_path)
    assert "| Verdict + matched_note_id match gold | 20/20 | yes |" in report
    assert "| UNSAFE misreads on adversarial notes (false raise or window too wide) | 0 | yes |" in report


def test_cli_run_and_input_errors(fake, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["assistant", "--out", str(tmp_path), "-q", "run"])
    cli.main()
    assert "20 flagged route-weeks: 17 unexplained, 3 justified" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["assistant", "--notes", str(tmp_path / "missing.csv"), "run"])
    with pytest.raises(SystemExit, match="Input not found"):
        cli.main()
    monkeypatch.setattr(sys, "argv", ["assistant", "--out", str(tmp_path / "empty"), "ask", "why?"])
    with pytest.raises(SystemExit, match="No results"):
        cli.main()


# --- the real HTTP client, with urlopen faked --------------------------------------------------------

class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def fake_server(monkeypatch, groq_replies=()):
    """urlopen stand-in: Ollama answers normally; Groq pops queued replies (or raises them)."""
    queue, sent = list(groq_replies), []

    def urlopen(req, timeout):
        url = req if isinstance(req, str) else req.full_url
        if "groq.com" in url:
            sent.append(json.loads(req.data))
            reply = queue.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return FakeResponse(json.dumps(reply).encode())
        body = ({"embeddings": [[3.0, 4.0]], "prompt_eval_count": 2} if url.endswith("/api/embed")
                else {"models": [{"name": "all-minilm:latest", "digest": "1b226e2802dbffff"}]})
        return FakeResponse(json.dumps(body).encode())

    monkeypatch.setattr(llm_module.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(llm_module.time, "sleep", lambda s: None)
    return sent


def http_error(code, body="", retry_after=None):
    return urllib.error.HTTPError("https://api.groq.com", code, "err", {"Retry-After": retry_after} if retry_after else {},
                                  io.BytesIO(body.encode()))


OK = {"choices": [{"message": {"content": " ok "}}], "system_fingerprint": "fp_abc",
      "usage": {"prompt_tokens": 100, "completion_tokens": 20}}


def test_client_sends_settings_counts_usage_and_logs_versions(monkeypatch):
    sent = fake_server(monkeypatch, [http_error(429, "tokens per minute", "1"), OK])
    client = llm_module.LLM(key="gsk-test")
    assert client.chat("s", "u", schema={"type": "object"}) == "ok"
    assert np.allclose(client.embed(["x"]), [[0.6, 0.8]])
    assert len(sent) == 2  # retried the rate limit
    assert (sent[0]["model"], sent[0]["temperature"], sent[0]["seed"], sent[0]["reasoning_effort"]) == \
        ("openai/gpt-oss-120b", 0, 42, "low")
    assert sent[0]["response_format"]["json_schema"]["strict"] is True
    report = client.usage_report()
    assert (report["chat_calls"], report["input_tokens"], report["output_tokens"], report["embed_input_tokens"]) == \
        (1, 100, 20, 2)
    assert report["model_versions"] == {"openai/gpt-oss-120b": "Groq system_fingerprint ['fp_abc']",
                                        "all-minilm": "1b226e2802db"}
    assert report["cost_usd_actual"] == 0.0 and report["cost_usd_at_reference_rate"] > 0


def test_schema_miss_is_retried(monkeypatch):
    miss = http_error(400, '{"error":{"message":"Generated JSON does not match the expected schema","code":"json_validate_failed"}}')
    sent = fake_server(monkeypatch, [miss, OK])
    assert llm_module.LLM(key="gsk-test").chat("s", "u", schema={"type": "object"}) == "ok" and len(sent) == 2


@pytest.mark.parametrize("failure, message", [
    (http_error(429, "Rate limit reached on requests per day (RPD)"), "daily quota used up"),
    (http_error(401, "Invalid API Key"), "failed \\(401\\)"),
    ({"error": {"code": 400, "message": "bad request"}}, "failed \\(400\\)"),
])
def test_client_fails_fast_on_quota_and_client_errors(monkeypatch, failure, message):
    fake_server(monkeypatch, [failure])
    with pytest.raises(RuntimeError, match=message):
        llm_module.LLM(key="gsk-test").chat("s", "u")


def test_cache_replays_identical_requests_without_calling(monkeypatch, tmp_path):
    sent = fake_server(monkeypatch, [OK])
    client = llm_module.LLM(cache_dir=tmp_path, key="gsk-test")
    assert client.chat("s", "u") == client.chat("s", "u") == "ok"
    assert len(sent) == 1 and client.usage.cache_hits == 1 and client.usage.chat_calls == 1


def test_key_loading(monkeypatch, tmp_path):
    monkeypatch.setattr(llm_module, "KEY_FILE", tmp_path / "groq api key.txt")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="No Groq key"):
        llm_module.api_key()
    (tmp_path / "groq api key.txt").write_text(" gsk-from-file \n")
    assert llm_module.api_key() == "gsk-from-file"
    monkeypatch.setenv("GROQ_API_KEY", "gsk-from-env")
    assert llm_module.api_key() == "gsk-from-env"  # env var wins over the file


def test_local_embeddings_down_is_explained(monkeypatch):
    def urlopen(req, timeout):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(llm_module.urllib.request, "urlopen", urlopen)
    with pytest.raises(RuntimeError, match="Is Ollama running"):
        llm_module.LLM(key="gsk-test").embed(["x"])
