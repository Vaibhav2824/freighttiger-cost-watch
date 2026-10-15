"""The one LLM client: chat on Groq (gpt-oss-120B), embeddings on local Ollama (all-MiniLM). Stdlib HTTP only.

Counts every call and token, retries rate limits, and can replay identical requests from an opt-in cache.
Free tier for this model: 1,000 requests/day, 8,000 tokens/min. A full run is 2 batched chat calls.
"""
import hashlib
import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

GROQ_URL = "https://api.groq.com/openai/v1"
CHAT_MODEL = "openai/gpt-oss-120b"  # open weights; strict JSON-schema decoding on Groq
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "all-minilm"          # all-MiniLM-L6-v2 sentence embeddings, 384-d, local
# temperature 0 = greedy decoding; the seed pins what's left. The stress check re-runs at 0.8 to show
# that flags, numbers and verdicts don't move. "low" reasoning: a short think; decisions are made in code.
GENERATION_OPTIONS = {"temperature": 0, "seed": 42}
REASONING_EFFORT = "low"
BATCH_SIZE = 25  # one call reads all notes, one call writes all reasons
# Free tier costs $0. Reference: Groq's published rate for this model (console.groq.com, checked 2026-10-15).
REFERENCE_RATE_USD_PER_M = (0.15, 0.60)
KEY_FILE = Path(__file__).resolve().parent.parent / "groq api key.txt"  # git-ignored
TIMEOUT_S = 300
RETRY_STATUSES = {429, 500, 502, 503, 504}
MAX_ATTEMPTS = 5

log = logging.getLogger(__name__)


@dataclass
class Usage:
    chat_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    embed_calls: int = 0
    embed_input_tokens: int = 0
    cache_hits: int = 0


def api_key() -> str:
    """GROQ_API_KEY first, else the git-ignored key file. Never logged."""
    key = os.environ.get("GROQ_API_KEY") or (KEY_FILE.read_text(encoding="utf-8").strip()
                                             if KEY_FILE.exists() else "")
    if not key:
        raise RuntimeError(f"No Groq key: set GROQ_API_KEY or create '{KEY_FILE.name}' in the project folder.")
    return key


def _post_json(url: str, payload: dict, headers: dict | None = None) -> dict:
    req = urllib.request.Request(url, json.dumps(payload).encode(),
                                 {"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        return json.load(resp)


class LLM:
    batch_size = BATCH_SIZE
    chat_model = CHAT_MODEL

    def __init__(self, options: dict | None = None, cache_dir: Path | None = None, key: str | None = None):
        self.options = options or GENERATION_OPTIONS
        self.cache_dir = cache_dir  # opt-in (--cache): replay identical requests byte-for-byte
        self._headers = {"Authorization": f"Bearer {key or api_key()}", "User-Agent": "freight-cost-watch"}
        self.usage = Usage()
        self.fingerprints: set[str] = set()

    def chat(self, system: str, user: str, schema: dict | None = None) -> str:
        """One Groq call, or a replay from the opt-in cache when the exact same request was made before."""
        path = None
        if self.cache_dir:
            request = json.dumps([CHAT_MODEL, self.options, system, user, schema], sort_keys=True)
            path = self.cache_dir / f"{hashlib.sha256(request.encode()).hexdigest()}.txt"
            if path.exists():
                self.usage.cache_hits += 1
                return path.read_text(encoding="utf-8")
        payload = {"model": CHAT_MODEL, **self.options, "reasoning_effort": REASONING_EFFORT,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if schema:
            payload["response_format"] = {"type": "json_schema",
                                          "json_schema": {"name": "reply", "strict": True, "schema": schema}}
        out = self._post_with_retry(payload)
        usage = out.get("usage") or {}
        self.usage.chat_calls += 1
        self.usage.input_tokens += usage.get("prompt_tokens", 0)
        self.usage.output_tokens += usage.get("completion_tokens", 0)
        self.fingerprints.add(out.get("system_fingerprint") or "unknown")
        reply = (out["choices"][0]["message"].get("content") or "").strip()
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(reply, encoding="utf-8")
        return reply

    def _post_with_retry(self, payload: dict) -> dict:
        """POST with backoff on rate limits / transient errors; a used-up daily quota fails fast."""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            retry_after = None
            try:
                out = _post_json(f"{GROQ_URL}/chat/completions", payload, self._headers)
                if "error" not in out:
                    return out
                status, message = out["error"].get("code"), str(out["error"].get("message", ""))
            except urllib.error.HTTPError as exc:
                status, message = exc.code, exc.read().decode(errors="replace")[:300]
                retry_after = exc.headers.get("Retry-After")
            except urllib.error.URLError as exc:
                status, message = None, str(exc)
            if "per day" in message:
                raise RuntimeError(f"Groq daily quota used up: {message}. Wait for the reset.")
            # Strict mode returns 400 when a sampled generation fails the schema; a fresh try usually passes.
            schema_miss = "does not match the expected schema" in message or "json_validate_failed" in message
            if not (status is None or status in RETRY_STATUSES or schema_miss) or attempt == MAX_ATTEMPTS:
                raise RuntimeError(f"Groq request failed ({status}): {message}")
            wait = float(retry_after) if retry_after else min(60.0, 5.0 * 2 ** (attempt - 1))
            log.warning("Groq %s, retrying in %.0fs (attempt %d/%d)", status, wait, attempt, MAX_ATTEMPTS)
            time.sleep(wait)
        raise AssertionError("unreachable")

    def embed(self, texts: list[str]) -> np.ndarray:
        """L2-normalised local embeddings, so inner product == cosine similarity."""
        try:
            out = _post_json(f"{OLLAMA_URL}/api/embed", {"model": EMBED_MODEL, "input": texts})
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Local embeddings failed ({exc}). Is Ollama running with "
                               f"`ollama pull {EMBED_MODEL}` done?") from exc
        self.usage.embed_calls += 1
        self.usage.embed_input_tokens += out.get("prompt_eval_count", 0)
        vectors = np.asarray(out["embeddings"], dtype="float32")
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def model_versions(self) -> dict:
        """Groq's system_fingerprint(s) and the local embedding build: what produced this run."""
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=TIMEOUT_S) as resp:
            tags = {m["name"]: m["digest"][:12] for m in json.load(resp)["models"]}
        return {CHAT_MODEL: f"Groq system_fingerprint {sorted(self.fingerprints) or '-'}",
                EMBED_MODEL: tags.get(EMBED_MODEL) or tags.get(f"{EMBED_MODEL}:latest")}

    def usage_report(self) -> dict:
        rate_in, rate_out = REFERENCE_RATE_USD_PER_M
        reference = (self.usage.input_tokens * rate_in + self.usage.output_tokens * rate_out) / 1e6
        return {"chat_model": CHAT_MODEL, "embed_model": EMBED_MODEL, "model_versions": self.model_versions(),
                "generation_options": {**self.options, "reasoning_effort": REASONING_EFFORT},
                **asdict(self.usage), "cost_usd_actual": 0.0,
                "reference_rate_usd_per_m_tokens": REFERENCE_RATE_USD_PER_M,
                "cost_usd_at_reference_rate": round(reference, 6)}
