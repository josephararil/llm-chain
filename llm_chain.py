"""Provider-agnostic LLM calls with a cross-model fallback chain.

This module knows NOTHING about any project. It never imports `config` or `common`,
never assumes a caller's domain, and never hardcodes a model name outside the DEFAULT_*
block below. That is what makes it liftable into any pipeline unchanged, and it is the
whole point of the module -- if a change would require importing project config or
baking in a caller-specific assumption, that change belongs in the caller instead.

Written after a provider capacity dip produced a blank email: gemini-flash-latest
returned 503 on 13 of 14 reasoning calls inside three minutes while a different model on
the same API key served 3 of 3. The old retry policy allowed 14 seconds of total patience
against an event lasting minutes, and had no way to reach the healthy model. This module
fixes both: a two-level loop that retries the same model, then advances to the next one.

Everything is configured through LLM_* environment variables, read at CALL time (not
import time) so CI and tests can change them without reimporting. See resolved_config().
"""

import os, json, time, dataclasses, typing
import requests

# ---- Defaults. The ONLY place any model name appears in the codebase. ----
DEFAULT_PROVIDER             = "gemini"
DEFAULT_MODEL_CHAIN          = "gemini-flash-latest,gemini-3.1-flash-lite"
DEFAULT_SEARCH_MODEL_CHAIN   = "gemini-3.1-flash-lite,gemini-flash-latest"
DEFAULT_ATTEMPTS_PER_MODEL   = 4
DEFAULT_BACKOFF_SECONDS      = "5,15,45"
DEFAULT_TIMEOUT_SECONDS      = 180
DEFAULT_RETRY_STATUSES       = "429,500,502,503,504"
DEFAULT_ADVANCE_STATUSES     = "400,404,429,500,502,503,504"
DEFAULT_TOTAL_BUDGET_SECONDS = 1200
DEFAULT_RETRY_AFTER_CAP      = 120
DEFAULT_ANTHROPIC_MODEL      = "claude-haiku-4-5-20251001"

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"

# Used when the caller passes no search_prompt: wraps its task text in a generic
# lead-generation directive. Deliberately free of any domain vocabulary.
_DEFAULT_SEARCH_WRAPPER = (
    "Search the web for current, concrete facts relevant to the task below: real dates,\n"
    "times, locations, prices, availability, and named sources. Return a thorough list of\n"
    "findings. Do not write final analysis or JSON.\n"
    "\n"
    "TASK:\n")

# Used when the caller passes no search_preamble: frames the grounded findings for the
# reasoning model. Callers with their own house style pass their own.
_DEFAULT_SEARCH_PREAMBLE = (
    "### LIVE SEARCH RESULTS (a web search was run for you moments ago)\n\n"
    "{leads}\n\n"
    "### END OF SEARCH RESULTS\n\n")


@dataclasses.dataclass
class LLMResult:
    text:      str  = ""     # "" when every model failed
    ok:        bool = False  # True iff text is non-empty
    model:     str  = ""     # model that actually answered; "" if none did
    provider:  str  = ""     # "gemini" | "anthropic"
    fell_back: bool = False  # True iff the FIRST model in the chain did not answer
    grounded:  bool = False  # True iff search grounding was injected into the prompt
    truncated: bool = False  # True iff finishReason != "STOP"
    attempts:  int  = 0      # total HTTP requests made, across all models
    error:     str  = ""     # last failure summary; "" when ok


# ------------------------------ Environment ------------------------------
# Every reader below runs at CALL time. common.py reads env at import; that makes the
# knobs untestable and unsettable from a workflow env: block, so it is not copied here.
# An empty or whitespace-only variable resolves to the DEFAULT, never to an empty list:
# an empty chain would make every call fail with no diagnosis.

def _env_raw(name):
    v = os.environ.get(name)
    return v if v is not None and v.strip() else None


def _split(raw):
    return [p.strip() for p in raw.split(",") if p.strip()]


def _env_list(name, default):
    parts = _split(_env_raw(name) or "")
    return parts or _split(default)


def _env_int(name, default):
    raw = _env_raw(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"  [llm] WARNING: {name}={raw!r} is not an integer; using default {default}")
        return default


def _env_int_list(name, default):
    raw = _env_raw(name)
    if raw is None:
        return [int(p) for p in _split(default)]
    try:
        vals = [int(p) for p in _split(raw)]
    except ValueError:
        print(f"  [llm] WARNING: {name}={raw!r} is not a comma-separated integer list; "
              f"using default {default!r}")
        vals = []
    return vals or [int(p) for p in _split(default)]


def resolved_provider(provider=None):
    """The provider that will actually handle a call: the explicit arg if given,
    else the LLM_PROVIDER environment variable. Lets callers tailor prompts per provider."""
    return (provider or _env_raw("LLM_PROVIDER") or DEFAULT_PROVIDER).strip().lower()


def _config():
    """Every effective knob, resolved now. Called once per call_llm."""
    return {
        "model_chain":          _env_list("LLM_MODEL_CHAIN", DEFAULT_MODEL_CHAIN),
        "search_model_chain":   _env_list("LLM_SEARCH_MODEL_CHAIN", DEFAULT_SEARCH_MODEL_CHAIN),
        "attempts_per_model":   _env_int("LLM_ATTEMPTS_PER_MODEL", DEFAULT_ATTEMPTS_PER_MODEL),
        "backoff_seconds":      _env_int_list("LLM_BACKOFF_SECONDS", DEFAULT_BACKOFF_SECONDS),
        "timeout_seconds":      _env_int("LLM_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS),
        "retry_statuses":       set(_env_int_list("LLM_RETRY_STATUSES", DEFAULT_RETRY_STATUSES)),
        "advance_statuses":     set(_env_int_list("LLM_ADVANCE_STATUSES", DEFAULT_ADVANCE_STATUSES)),
        "total_budget_seconds": _env_int("LLM_TOTAL_BUDGET_SECONDS", DEFAULT_TOTAL_BUDGET_SECONDS),
        "retry_after_cap":      _env_int("LLM_RETRY_AFTER_CAP", DEFAULT_RETRY_AFTER_CAP),
        "anthropic_model":      _env_raw("LLM_ANTHROPIC_MODEL") or DEFAULT_ANTHROPIC_MODEL,
    }


def resolved_config():
    """Every effective knob with its value and whether it came from env or default.
    Printed once per run: it is the only record of what the chain was actually configured
    with, and a repo variable that silently failed to reach the job looks identical to a
    correctly-defaulted one without it."""
    cfg = _config()
    pairs = [
        ("LLM_PROVIDER",             resolved_provider()),
        ("LLM_MODEL_CHAIN",          ",".join(cfg["model_chain"])),
        ("LLM_SEARCH_MODEL_CHAIN",   ",".join(cfg["search_model_chain"])),
        ("LLM_ATTEMPTS_PER_MODEL",   cfg["attempts_per_model"]),
        ("LLM_BACKOFF_SECONDS",      ",".join(str(s) for s in cfg["backoff_seconds"])),
        ("LLM_TIMEOUT_SECONDS",      cfg["timeout_seconds"]),
        ("LLM_RETRY_STATUSES",       ",".join(str(s) for s in sorted(cfg["retry_statuses"]))),
        ("LLM_ADVANCE_STATUSES",     ",".join(str(s) for s in sorted(cfg["advance_statuses"]))),
        ("LLM_TOTAL_BUDGET_SECONDS", cfg["total_budget_seconds"]),
        ("LLM_RETRY_AFTER_CAP",      cfg["retry_after_cap"]),
        ("LLM_ANTHROPIC_MODEL",      cfg["anthropic_model"]),
    ]
    return {name: {"value": value, "source": "env" if _env_raw(name) else "default"}
            for name, value in pairs}


# ------------------------------ Budget valve ------------------------------
# A hard wall-clock ceiling across every LLM call in the process. This is the only thing
# that turns the job's timeout-minutes into a proof rather than an estimate: without it,
# seven logical calls each exhausting a three-model chain is ~24 minutes of pure sleeping.
# The clock starts at the FIRST call_llm rather than at import, so a slow harvest does not
# eat the LLM budget.

_BUDGET_START = None


def reset_budget():
    """Restart the process-wide budget clock. For tests and long-lived callers."""
    global _BUDGET_START
    _BUDGET_START = None


def _budget_left(budget):
    global _BUDGET_START
    if _BUDGET_START is None:
        _BUDGET_START = time.time()
    return budget - (time.time() - _BUDGET_START)


# ------------------------------ The chain ------------------------------

@dataclasses.dataclass
class _Outcome:
    text:      str  = ""
    ok:        bool = False
    model:     str  = ""
    fell_back: bool = False
    truncated: bool = False
    attempts:  int  = 0
    error:     str  = ""


def _log(stage, msg):
    print(f"  [llm:{stage}] {msg}" if stage else f"  [llm] {msg}")


def _delay(n, backoff, retry_after, cap):
    """Sleep before attempt n+1. A server-supplied Retry-After wins over our schedule but
    is capped -- providers occasionally return values in the hundreds of seconds, which
    would blow the whole budget on one model."""
    if retry_after is not None:
        try:
            return min(int(float(retry_after)), cap)
        except (TypeError, ValueError):
            pass
    return backoff[min(n - 1, len(backoff) - 1)]


def _run_chain(models, build, extract, stage, provider, cfg):
    """Try each model in `models` in order. Within a model, retry on a RETRY status up to
    attempts_per_model times; then advance to the next model. See the status table:

      2xx + text        -> success
      RETRY             -> sleep, retry the SAME model, then advance
      ADVANCE not RETRY -> advance immediately, no retry, no sleep (400, 404 by default)
      neither list      -> FAIL FAST, try no further model (401, 403 by default)
      RequestException  -> exactly as a RETRY status

    The fail-fast row is the non-obvious one: a bad API key returns 401 on every model, so
    advancing would burn the entire budget and bury the real error behind a generic
    "all models failed", sending the operator to wait on the provider instead of the key.
    """
    per_model = cfg["attempts_per_model"]
    attempts  = 0
    error     = ""

    for idx, model in enumerate(models):
        reason = ""
        for n in range(1, per_model + 1):
            if _budget_left(cfg["total_budget_seconds"]) <= 0:
                error = error or "total LLM budget exhausted before the request"
                _log(stage, f"budget exhausted after {attempts} attempt(s); abandoning chain")
                return _Outcome(attempts=attempts, error=error)

            url, headers, body = build(model)
            attempts += 1
            retry_after = None
            try:
                r = requests.post(url, headers=headers, json=body,
                                  timeout=cfg["timeout_seconds"])
            except requests.exceptions.RequestException as exc:
                action  = "retry"
                outcome = type(exc).__name__
                error   = f"{type(exc).__name__}: {exc}"
            else:
                status = r.status_code
                if 200 <= status < 300:
                    text, truncated = extract(r)
                    if text:
                        _log(stage, f"{provider}/{model} attempt {n}/{per_model} -> "
                                    f"ok, {len(text)} chars"
                                    + (" (TRUNCATED)" if truncated else ""))
                        return _Outcome(text=text, ok=True, model=model, fell_back=idx > 0,
                                        truncated=truncated, attempts=attempts)
                    # 2xx with no text is not covered by the status table. Retrying the same
                    # model cannot help (an empty body is usually a token-budget or safety
                    # outcome, not congestion), so advance and give the next model a turn.
                    action  = "advance"
                    outcome = f"HTTP {status} with an empty body"
                    error   = f"HTTP {status} returned no text (truncated={truncated})"
                elif status in cfg["retry_statuses"]:
                    action, outcome = "retry", f"HTTP {status}"
                    error = f"HTTP {status}: {r.text[:500]}"
                    retry_after = r.headers.get("Retry-After")
                elif status in cfg["advance_statuses"]:
                    action, outcome = "advance", f"HTTP {status}"
                    error = f"HTTP {status}: {r.text[:500]}"
                else:
                    action, outcome = "failfast", f"HTTP {status}"
                    error = f"HTTP {status}: {r.text[:500]}"

            _log(stage, f"{provider}/{model} attempt {n}/{per_model} -> {outcome}")

            if action == "failfast":
                _log(stage, f"ALL MODELS FAILED after {attempts} attempt(s): {error}")
                return _Outcome(attempts=attempts, error=error)
            reason = outcome
            if action == "advance" or n == per_model:
                break
            if _budget_left(cfg["total_budget_seconds"]) <= 0:
                _log(stage, f"budget exhausted after {attempts} attempt(s); abandoning chain")
                return _Outcome(attempts=attempts, error=error)
            time.sleep(_delay(n, cfg["backoff_seconds"], retry_after, cfg["retry_after_cap"]))

        nxt = models[idx + 1] if idx + 1 < len(models) else None
        _log(stage, f"{model} exhausted ({reason}); "
                    + (f"advancing to {nxt}" if nxt else "no models left"))

    _log(stage, f"ALL MODELS FAILED after {attempts} attempt(s): {error}")
    return _Outcome(attempts=attempts, error=error or "no models configured")


# ------------------------------ Gemini ------------------------------

def _gemini_extract(r):
    """(text, truncated) from a generateContent response. finishReason != "STOP"
    (e.g. "MAX_TOKENS") means the output was cut off -- on thinking models, hidden
    reasoning tokens can exhaust maxOutputTokens before the visible answer completes,
    which would otherwise look like an empty result rather than a budget problem."""
    cand = (r.json().get("candidates") or [{}])[0]
    parts = [p["text"] for p in cand.get("content", {}).get("parts", []) if "text" in p]
    finish = cand.get("finishReason")
    return "".join(parts).strip(), bool(finish) and finish != "STOP"


def _gemini(prompt, stage, max_tokens, want_search, search_prompt, search_preamble,
            response_schema, cfg):
    """Search and reasoning are split across two chains. Grounding runs on the search
    chain with the google_search tool; the reasoning chain then runs tools-free with the
    grounding injected as context. Attaching google_search to a flagship reasoning model
    times out on Google's grounding gateway, and the split also keeps responseSchema off
    the search call (they conflict)."""
    key = os.environ.get("GEMINI_API_KEY", "")
    headers = {"x-goog-api-key": key, "Content-Type": "application/json"}
    text = prompt
    grounded = False
    search_attempts = 0

    if want_search:
        search_text = search_prompt if search_prompt is not None \
            else _DEFAULT_SEARCH_WRAPPER + prompt

        def build_search(model):
            # NO-SILENT-MODEL-DEFAULT: `model` goes into the URL verbatim. There is no
            # name-mapping dict and no .get(model, fallback) anywhere in this module.
            return (f"{GEMINI_BASE}/models/{model}:generateContent", headers, {
                "contents": [{"role": "user", "parts": [{"text": search_text}]}],
                "generationConfig": {"maxOutputTokens": max_tokens},
                "tools": [{"google_search": {}}],
            })

        out = _run_chain(cfg["search_model_chain"], build_search, _gemini_extract,
                         f"{stage}:search" if stage else "search", "gemini", cfg)
        search_attempts = out.attempts
        if out.ok:
            preamble = search_preamble if search_preamble is not None \
                else _DEFAULT_SEARCH_PREAMBLE
            # .replace, never .format -- grounding routinely contains braces.
            text = preamble.replace("{leads}", out.text) + prompt
            grounded = True
        else:
            # Search failure degrades to knowledge-only reasoning; it is NOT a call failure.
            _log(stage, f"search grounding unavailable ({out.error}); reasoning knowledge-only")

    def build_reason(model):
        body = {
            "contents": [{"role": "user", "parts": [{"text": text}]}],
            "generationConfig": {"maxOutputTokens": max_tokens},
        }
        if response_schema is not None:
            body["generationConfig"]["responseMimeType"] = "application/json"
            body["generationConfig"]["responseSchema"] = response_schema
        return f"{GEMINI_BASE}/models/{model}:generateContent", headers, body

    out = _run_chain(cfg["model_chain"], build_reason, _gemini_extract, stage, "gemini", cfg)
    return LLMResult(text=out.text, ok=out.ok, model=out.model, provider="gemini",
                     fell_back=out.fell_back, grounded=grounded, truncated=out.truncated,
                     attempts=search_attempts + out.attempts, error=out.error)


# ------------------------------ Anthropic ------------------------------
# Anthropic is a manual lever (LLM_PROVIDER=anthropic), never an automatic fallback link:
# it ignores response_schema, so an automatic cross-provider hop would silently drop
# schema-enforced JSON mid-run. It still gets the same retry/backoff/budget behaviour.

def _anthropic_extract(r):
    body = r.json()
    text = "".join(b.get("text", "") for b in body.get("content", [])
                   if b.get("type") == "text").strip()
    return text, body.get("stop_reason") == "max_tokens"


def _anthropic(prompt, stage, max_tokens, want_search, web_search_max_uses, cfg):
    key = os.environ.get("ANTHROPIC_API_KEY", "")

    def build(model):
        body = {"model": model, "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}]}
        if want_search:
            body["tools"] = [{"type": "web_search_20250305", "name": "web_search",
                              "max_uses": web_search_max_uses}]
        return "https://api.anthropic.com/v1/messages", {
            "x-api-key": key, "anthropic-version": "2023-06-01",
            "content-type": "application/json"}, body

    out = _run_chain([cfg["anthropic_model"]], build, _anthropic_extract,
                     stage, "anthropic", cfg)
    return LLMResult(text=out.text, ok=out.ok, model=out.model, provider="anthropic",
                     fell_back=out.fell_back, grounded=out.ok and want_search,
                     truncated=out.truncated, attempts=out.attempts, error=out.error)


# ------------------------------ Public entry point ------------------------------

def call_llm(prompt,
             *,
             stage="",
             max_tokens=4000,
             want_search=False,
             search_prompt=None,
             search_preamble=None,
             response_schema=None,
             provider=None,
             web_search_max_uses=6):
    """Single entry point for all LLM calls. Returns an LLMResult; NEVER raises on a
    provider failure -- total failure is LLMResult(ok=False, text="", error=...) and
    callers branch on .ok. A malformed response_schema producing 400 is reported the same
    way rather than as an exception.

    prompt is a plain string (one user message), not a messages list.
    search_prompt   -- verbatim prompt for the search step; None wraps `prompt` generically.
    search_preamble -- must contain "{leads}"; None uses this module's default.
    """
    cfg = _config()
    p = resolved_provider(provider)
    if p == "anthropic":
        return _anthropic(prompt, stage, max_tokens, want_search, web_search_max_uses, cfg)
    return _gemini(prompt, stage, max_tokens, want_search, search_prompt, search_preamble,
                   response_schema, cfg)


def available_models(provider: typing.Optional[str] = None) -> typing.List[str]:
    """Every model id the configured API key can actually list, sorted. [] on any failure.
    This exists so the chain is configured from the real list rather than from plausible-
    looking model names -- see the LLM_MODEL_CHAIN setup notes in CLAUDE.md."""
    if resolved_provider(provider) != "gemini":
        return []
    try:
        r = requests.get(f"{GEMINI_BASE}/models",
                         headers={"x-goog-api-key": os.environ.get("GEMINI_API_KEY", "")},
                         timeout=30)
        if not r.ok:
            print(f"  [llm] available_models: HTTP {r.status_code}: {r.text[:200]}")
            return []
        return sorted(m.get("name", "").split("/")[-1]
                      for m in r.json().get("models", []) if m.get("name"))
    except (requests.exceptions.RequestException, ValueError) as exc:
        print(f"  [llm] available_models: {type(exc).__name__}: {exc}")
        return []


# ------------------------------ JSON parsing ------------------------------

def parse_json_block(text):
    """Strip markdown fences and parse the outermost JSON value the model returned,
    choosing object vs array by whichever bracket appears first."""
    t = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    starts = [(t.find(c), c) for c in ("[", "{") if t.find(c) != -1]
    if not starts:
        return None
    _, open_c = min(starts)
    close_c = "]" if open_c == "[" else "}"
    i, j = t.find(open_c), t.rfind(close_c)
    if i != -1 and j != -1 and j > i:
        try:
            return json.loads(t[i:j + 1])
        except json.JSONDecodeError:
            return None
    return None


# ------------------------------ Smoke test ------------------------------

if __name__ == "__main__":
    print("=== resolved config ===")
    for name, info in resolved_config().items():
        print(f"  {name:<26} {info['value']!r}  ({info['source']})")

    print("\n=== models this key can list ===")
    models = available_models()
    print(f"  {len(models)} model(s)")
    for m in models:
        print(f"  {m}")

    print("\n=== live ping ===")
    print(f"  {call_llm('Reply with the single word: ok', stage='smoke', max_tokens=100)}")
