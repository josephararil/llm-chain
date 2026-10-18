# llm-chain

One shared LLM layer for every pipeline in this account. `call_llm()` returns an
`LLMResult` and **never raises on a provider failure** — total failure is
`LLMResult(ok=False, text="", error=...)` and callers branch on `.ok`.

What it buys you over a hand-rolled `requests.post`:

- a **two-level fallback loop** — retry the same model on a transient status, then advance
  to the next model in the chain;
- a **wall-clock budget valve** (`LLM_TOTAL_BUDGET_SECONDS`) checked before every request
  *and* before every sleep, which is what turns a job's `timeout-minutes` into a proof
  rather than an estimate;
- **fail-fast on client errors** (401/403) so a dead API key is not disguised as a
  provider outage;
- **no silent model default** — a model name reaches the API verbatim, so a typo 404s
  loudly instead of quietly resolving to some other model;
- a **search/reasoning split** for Gemini: grounding runs on the search chain carrying the
  `google_search` tool, then the reasoning chain runs tools-free with the grounding
  injected as context.

It exists because a provider capacity dip produced a blank email in a downstream pipeline:
`gemini-flash-latest` returned 503 on 13 of 14 reasoning calls inside three minutes while a
different model **on the same API key** served 3 of 3. The retry policy it replaced allowed
14 seconds of total patience against an event lasting minutes, and had no way to reach the
healthy model.

## Design constraint — read before editing

`llm_chain.py` **imports only `os`, `json`, `time`, `dataclasses`, `typing` and
`requests`.** It never imports a caller's `config` or `common`, never assumes a caller's
domain, and never hardcodes a model name outside the `DEFAULT_*` block. Consumers are
unrelated pipelines, so anything project-specific added here is imposed on all of them.

If a change would require importing project config or baking in a caller-specific
assumption, **that change belongs in the caller**.

Corollary that bites in practice: because it reads only the process environment, it does
**not** load a `.env` file. See [Troubleshooting](#troubleshooting).

## Install

Consumers pin a **moving major tag**, not `main` and not an immutable version:

```
llm-chain @ git+https://github.com/josephararil/llm-chain@v1
```

Put that line in `requirements.txt`. The repo is public, so no token or auth is needed.

```bash
pip install -r requirements.txt
```

`@v1` is deliberately mutable — see [Releasing](#releasing-and-the-v1-tag). pip does not
cache VCS installs from a mutable ref, so consumers re-resolve `v1` on each install and
pick up a moved tag without any per-repo edit.

## Usage

```python
import llm_chain as L

res = L.call_llm(
    prompt,
    stage="find",              # log prefix only
    max_tokens=20000,
    want_search=True,          # Gemini: run the search chain first, inject the grounding
    search_prompt=None,        # verbatim prompt for the search step; None wraps `prompt`
    search_preamble=None,      # must contain "{leads}"; None uses the built-in default
    response_schema=SCHEMA,    # Gemini responseSchema; ignored by the Anthropic path
    provider=None,             # None -> LLM_PROVIDER
)
if not res.ok:
    ...                       # degrade; res.error carries the last failure summary
data = L.parse_json_block(res.text)
```

`prompt` is a plain string (one user message), not a messages list.

`LLMResult` fields: `text`, `ok`, `model` (which model actually answered), `provider`,
`fell_back` (the first model in the chain did **not** answer), `grounded` (search was
injected), `truncated` (`finishReason != "STOP"`), `attempts` (total HTTP requests across
all models), `error`, `sources` (a list of `{"uri", "title"}` dicts from Gemini's search
grounding, in chunk order, duplicates kept; `[]` when the call was not grounded, when
grounding returned no chunks, and always on the Anthropic path). `uri` is a
`vertexaisearch.cloud.google.com` redirect, not the page itself; `title` is normally the
bare source domain (observed on 21 of 21 chunks across `gemini-flash-latest` and
`gemini-3.1-flash-lite`, 2026-10-18), so read `title` when you want the domain.

Two of these are worth acting on rather than logging. `truncated` means `max_tokens` cut
the answer off mid-JSON — which looks downstream like the model simply found nothing.
`fell_back` means the chain worked and is normal operation, not a failure: it deserves a
log line, not an alarm.

Other public helpers: `parse_json_block()`, `resolved_provider()`, `available_models()`,
`resolved_config()`, `reset_budget()`.

## Smoke test

```bash
python -m llm_chain
```

Prints the resolved config (every knob, with `env` or `default` as its source), every model
the configured key can actually list, and one live ping. Run it before setting a chain —
configure `LLM_MODEL_CHAIN` from that real list, not from plausible-looking model names.

> Before extraction this module was invoked as `python llm_chain.py` from inside a
> consumer repo. It is now installed rather than checked out, so the invocation is
> `python -m llm_chain`.

## Configuration

Every knob is an `LLM_*` environment variable read at **call** time, not import time — so
CI and tests can change it without reimporting. An empty or whitespace-only value resolves
to the default, never to an empty list.

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | `gemini` or `anthropic` |
| `LLM_MODEL_CHAIN` | `gemini-flash-latest,gemini-3.1-flash-lite` | Reasoning models, tried left to right |
| `LLM_SEARCH_MODEL_CHAIN` | `gemini-3.1-flash-lite,gemini-flash-latest` | Search-step models |
| `LLM_ATTEMPTS_PER_MODEL` | `4` | HTTP attempts per model before advancing |
| `LLM_BACKOFF_SECONDS` | `5,15,45` | Sleep between attempts; the last value repeats |
| `LLM_TIMEOUT_SECONDS` | `180` | Per-request HTTP timeout |
| `LLM_RETRY_STATUSES` | `429,500,502,503,504` | Retry the **same** model |
| `LLM_ADVANCE_STATUSES` | `400,404,429,500,502,503,504` | Advance to the **next** model |
| `LLM_TOTAL_BUDGET_SECONDS` | `1200` | Wall-clock ceiling across all calls in the process |
| `LLM_RETRY_AFTER_CAP` | `120` | Upper bound on a server `Retry-After` header |
| `LLM_ANTHROPIC_MODEL` | `claude-haiku-4-5-20251001` | Model used when provider is `anthropic` |

Keys: `GEMINI_API_KEY`, or `ANTHROPIC_API_KEY` when `LLM_PROVIDER=anthropic`.

### Status handling

| Response | Behaviour |
|---|---|
| 2xx with text | success |
| in `RETRY_STATUSES` | sleep, retry the **same** model, then advance |
| in `ADVANCE` but not `RETRY` (400, 404) | advance immediately — no retry, no sleep |
| in neither list (401, 403) | **fail fast**, try no further model |
| `requests.RequestException` | exactly as a RETRY status |
| 2xx with an empty body | advance without retry |

The two non-obvious rows:

- **401/403 fail fast** because a bad key returns 401 on *every* model. Advancing would
  burn the whole budget and bury the real cause behind a generic "all models failed",
  sending you to wait on the provider instead of fixing the key.
- **400 advances but is never retried**, because a malformed `response_schema` returns 400
  deterministically — retrying it wastes `attempts × models` requests on a fixed error.

## The fallback chain is Gemini-only

**By design, and documented here so nobody later assumes otherwise.**

`_anthropic()` calls `_run_chain([cfg["anthropic_model"]], ...)` with a **single-element
list**. So the Anthropic path gets the retry-with-backoff loop, the `Retry-After` handling,
the fail-fast rule and the budget valve — but **no cross-model fallback**, because there is
no second model in its chain. When that one model is down, the call fails.

There is no `ANTHROPIC_MODEL_CHAIN`. Anthropic is also never an automatic fallback link
*from* Gemini: it ignores `response_schema`, so an automatic cross-provider hop would
silently drop schema-enforced JSON mid-run. Switching provider is a manual lever
(`LLM_PROVIDER=anthropic`), never automatic.

This is a deliberate, accepted asymmetry: production runs Gemini 100%, and the Anthropic
path exists as a hedge. **Do not "fix" it opportunistically** — widening it means a new env
var, new defaults and a change to every consumer's failure profile.

## Releasing, and the `v1` tag

`v1` is a **moving** tag. It is the auto-propagation mechanism *and* the only safety gate
in front of it, so promote in this order and do not skip step 3:

1. Merge to `main` (branch → PR, no direct commits).
2. **CI green** on `main`.
3. **Verify against one real pipeline** — install the new commit in a single consumer and
   let it complete a real run. `weekly-concierge` is the designated proving consumer.
4. Promote:

```bash
git tag -f v1 && git push -f origin v1
```

> **Moving `v1` changes behaviour in every consumer on its next scheduled run.** There is
> no per-repo review step, no lockfile and no staged rollout between step 4 and a cron job
> emailing a real person. Untested code on `v1` is untested code in production everywhere
> at once. That is the whole trade being made in exchange for "edit once, all consumers
> pick it up".

`main` may be ahead of `v1`; that is normal and is exactly the gap step 3 lives in. A
breaking change to `call_llm`'s signature or to `LLMResult` gets a **`v2` tag**, and
consumers migrate one at a time — never a moved `v1`.

## Tests

```bash
python test_llm_chain.py     # 23 tests, fully offline
```

Fully offline: `requests.post` and `time.sleep` are both monkeypatched, and sleeps are
*recorded* rather than slept, so the backoff schedule is asserted directly instead of
waited for. CI therefore needs no secrets and runs on every push and PR.

**What the suite does not cover, and it matters:** it proves the chain's control flow
against *canned* responses. It says nothing about whether Google actually returns 503 in
the shape assumed here, and nothing about whether the live fallback path works. Only a
forced-fallback run against the real API exercises that — put a deliberately bogus model
first in `LLM_MODEL_CHAIN` and confirm the run still succeeds on the second:

```bash
LLM_MODEL_CHAIN=zzz-not-a-model,gemini-flash-latest python -m llm_chain
```

`TestParity` compares `parse_json_block` against a **frozen verbatim copy** of the
`common.parse_json_block` that consumers historically used (weekly-concierge `common.py`
blob `17310bf6`). The copy is inlined in the test on purpose: importing a consumer's
`common.py` would violate the stdlib+requests constraint, and that file has already drifted
into three different versions across three repos.

## Troubleshooting

**`HTTP 403 ... unregistered callers`** — this is a **missing** key far more often than a
dead one. This module reads only the process environment and, by design, does **not** load
a `.env` file (`python-dotenv` is not a dependency). If your consumer repo keeps its key in
`.env`, the key is invisible here unless something else has already loaded it. Confirm with
`python -m llm_chain`, whose resolved-config block prints the source of every knob, and
export the key into the environment:

```bash
export GEMINI_API_KEY=...
```

**"all models failed" after ~15 seconds with one request** — that is the fail-fast path, so
the status was 401/403. Fix the key; the provider is fine.

**A stage returns nothing and looks like a quiet result** — check `res.truncated`. Hidden
reasoning tokens on thinking models can exhaust `maxOutputTokens` before the visible answer
completes, which truncates the JSON mid-structure and parses to `None`.

**A repo variable seems to have no effect** — a variable that silently failed to reach the
job is indistinguishable from a correctly-defaulted one. `resolved_config()` prints `env`
vs `default` per knob; log it once per run.
