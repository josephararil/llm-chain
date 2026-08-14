"""Offline tests for llm_chain.py. No network, no repo writes.

llm_chain.requests.post and llm_chain.time.sleep are both monkeypatched: sleeps are
recorded into a list rather than slept, so the backoff schedule is asserted directly
instead of waited for.

What these tests do NOT cover, and it matters: they prove the chain's control flow
against CANNED responses. They say nothing about whether Google actually returns 503 in
the shape assumed here, and nothing about whether the live fallback path works -- only a
forced-fallback run against the real API exercises that.
"""

import json, os, sys, time, unittest
from types import SimpleNamespace

import llm_chain

MODEL_A, MODEL_B = "model-alpha", "model-beta"


# ------------------------------ fake HTTP ------------------------------

class FakeResponse:
    def __init__(self, status, body=None, text="", headers=None):
        self.status_code = status
        self._body   = body if body is not None else {}
        self.text    = text
        self.headers = headers or {}

    def json(self):
        return self._body


def gemini_ok(text="answer", finish="STOP"):
    return FakeResponse(200, {"candidates": [
        {"content": {"parts": [{"text": text}]}, "finishReason": finish}]})


def err(status, text="upstream error", headers=None):
    return FakeResponse(status, {}, text, headers)


def is_search(body):
    """Search calls are the ones carrying the google_search tool."""
    return "tools" in (body or {})


class Post:
    """Records every request and delegates the response to `responder(url, body, i)`."""

    def __init__(self, responder):
        self.responder = responder
        self.calls = []

    def __call__(self, url, headers=None, json=None, timeout=None):
        self.calls.append(SimpleNamespace(url=url, body=json, headers=headers))
        return self.responder(url, json, len(self.calls) - 1)

    def urls_for(self, model):
        return [c for c in self.calls if model in c.url]


def seq(responses):
    """Responder returning each response in turn; the last one repeats forever."""
    def responder(url, body, i):
        return responses[min(i, len(responses) - 1)]
    return responder


def per_model(mapping, fallback=None):
    def responder(url, body, i):
        for name, resp in mapping.items():
            if name in url:
                return resp
        return fallback if fallback is not None else err(500)
    return responder


class ChainTest(unittest.TestCase):
    """Clears every LLM_* variable so a developer's real environment cannot change a
    result, and restarts the process-wide budget clock so test order cannot either."""

    def setUp(self):
        self._saved_env = {k: v for k, v in os.environ.items() if k.startswith("LLM_")}
        for k in self._saved_env:
            del os.environ[k]
        self._real_post  = llm_chain.requests.post
        self._real_sleep = llm_chain.time.sleep
        self.sleeps = []
        llm_chain.time.sleep = self.sleeps.append
        llm_chain.reset_budget()

    def tearDown(self):
        llm_chain.requests.post = self._real_post
        llm_chain.time.sleep    = self._real_sleep
        for k in [k for k in os.environ if k.startswith("LLM_")]:
            del os.environ[k]
        os.environ.update(self._saved_env)

    def install(self, responder):
        post = Post(responder)
        llm_chain.requests.post = post
        return post

    def chain(self, *models, **env):
        os.environ["LLM_MODEL_CHAIN"] = ",".join(models)
        os.environ["LLM_SEARCH_MODEL_CHAIN"] = ",".join(models)
        for k, v in env.items():
            os.environ[k] = str(v)


# ------------------------------ chain advance ------------------------------

class TestAdvance(ChainTest):

    def test_chain_advances_on_503(self):
        self.chain(MODEL_A, MODEL_B)
        post = self.install(per_model({MODEL_A: err(503), MODEL_B: gemini_ok("hi")}))
        res = llm_chain.call_llm("p")
        self.assertTrue(res.ok)
        self.assertEqual(res.model, MODEL_B)
        self.assertTrue(res.fell_back)
        self.assertEqual(res.attempts, 5)
        self.assertEqual(len(post.urls_for(MODEL_A)), 4)

    def test_no_advance_and_single_request_on_401(self):
        # FAIL-FAST-ON-CLIENT-ERROR: a bad key returns 401 on every model, so advancing
        # would burn the budget and bury the real cause behind "all models failed".
        self.chain(MODEL_A, MODEL_B)
        post = self.install(seq([err(401, "invalid api key")]))
        res = llm_chain.call_llm("p")
        self.assertFalse(res.ok)
        self.assertEqual(len(post.calls), 1)
        self.assertEqual(post.urls_for(MODEL_B), [])
        self.assertEqual(self.sleeps, [])

    def test_advance_immediately_on_404_without_retry(self):
        self.chain(MODEL_A, MODEL_B)
        post = self.install(seq([err(404, "model not found")]))
        res = llm_chain.call_llm("p")
        self.assertFalse(res.ok)
        self.assertEqual(res.attempts, 2)
        self.assertEqual(len(post.urls_for(MODEL_A)), 1)
        self.assertEqual(len(post.urls_for(MODEL_B)), 1)
        self.assertEqual(self.sleeps, [])

    def test_400_advances_but_never_retries(self):
        # A malformed response_schema returns 400 on every model; retrying it would waste
        # attempts x models requests on a deterministic error.
        self.chain(MODEL_A, MODEL_B)
        self.install(seq([err(400, "Invalid JSON payload: responseSchema")]))
        res = llm_chain.call_llm("p")
        self.assertFalse(res.ok)
        self.assertEqual(res.attempts, 2)
        self.assertEqual(self.sleeps, [])
        self.assertIn("responseSchema", res.error)

    def test_empty_2xx_body_advances_without_retry(self):
        # Not in the plan's status table: a 200 carrying no text. Retrying the same model
        # cannot help (an empty body is a token-budget or safety outcome, not congestion),
        # so it advances. Without this test, a change making an empty 200 count as success
        # passes every other test while returning ok=True with text="".
        self.chain(MODEL_A, MODEL_B)
        empty = FakeResponse(200, {"candidates": [
            {"content": {"parts": []}, "finishReason": "STOP"}]})
        self.install(per_model({MODEL_A: empty, MODEL_B: gemini_ok("real")}))
        res = llm_chain.call_llm("p")
        self.assertTrue(res.ok)
        self.assertEqual(res.text, "real")
        self.assertEqual(res.model, MODEL_B)
        self.assertEqual(res.attempts, 2)
        self.assertEqual(self.sleeps, [])


# ------------------------------ backoff and budget ------------------------------

class TestTiming(ChainTest):

    def test_backoff_schedule_is_5_15_45(self):
        self.chain(MODEL_A, LLM_ATTEMPTS_PER_MODEL=4)
        self.install(seq([err(503)]))
        llm_chain.call_llm("p")
        self.assertEqual(self.sleeps, [5, 15, 45])

    def test_backoff_last_value_repeats(self):
        self.chain(MODEL_A, LLM_ATTEMPTS_PER_MODEL=6, LLM_BACKOFF_SECONDS="5,15,45")
        self.install(seq([err(503)]))
        llm_chain.call_llm("p")
        self.assertEqual(self.sleeps, [5, 15, 45, 45, 45])

    def test_retry_after_header_honoured_and_capped(self):
        self.chain(MODEL_A, LLM_ATTEMPTS_PER_MODEL=2, LLM_RETRY_AFTER_CAP=120)
        self.install(seq([err(429, "slow down", {"Retry-After": "999"})]))
        llm_chain.call_llm("p")
        self.assertEqual(self.sleeps, [120])

    def test_budget_valve_stops_immediately(self):
        # BUDGET-VALVE: the only thing that makes the job's timeout-minutes a proof
        # rather than an estimate.
        self.chain(MODEL_A, MODEL_B, LLM_TOTAL_BUDGET_SECONDS=0)
        post = self.install(seq([err(503)]))
        res = llm_chain.call_llm("p")
        self.assertFalse(res.ok)
        self.assertEqual(self.sleeps, [])
        # Deliberately stricter than "at most one request": the valve is checked before
        # every request AND before every sleep, and an already-exhausted budget must make
        # ZERO requests. Asserting <= 1 passes even with the pre-request check deleted,
        # because the pre-sleep check still stops the run after one request -- verified by
        # mutation, so this assertion is the one that actually guards the invariant.
        self.assertEqual(len(post.calls), 0)

    def test_budget_valve_stops_before_sleeping_mid_chain(self):
        # The other half of BUDGET-VALVE, and the half that actually saves wall-clock in
        # production: the budget usually runs out DURING a chain, not before it. Exhausting
        # it from inside the responder reaches the pre-sleep check, which a budget of 0
        # never does because the pre-request check fires first.
        self.chain(MODEL_A, MODEL_B, LLM_TOTAL_BUDGET_SECONDS=60)

        def responder(url, body, i):
            llm_chain._BUDGET_START = time.time() - 999   # burn the budget, without waiting
            return err(503)

        post = self.install(responder)
        res = llm_chain.call_llm("p")
        self.assertFalse(res.ok)
        self.assertEqual(self.sleeps, [])
        self.assertEqual(len(post.calls), 1)


# ------------------------------ configuration ------------------------------

class TestConfig(ChainTest):

    def test_unknown_model_name_reaches_url_verbatim(self):
        # NO-SILENT-MODEL-DEFAULT: common.py resolves an unknown model to flash via
        # .get(model, "gemini-flash-latest"), so a typo'd chain runs green on the wrong
        # model and the log agrees with itself. That trap must not be reproduced.
        self.chain("zzz-not-a-model")
        post = self.install(seq([gemini_ok()]))
        llm_chain.call_llm("p")
        self.assertIn("zzz-not-a-model", post.calls[0].url)
        self.assertNotIn("gemini-flash-latest", post.calls[0].url)

    def test_empty_env_var_falls_back_to_default(self):
        for blank in ("", "   "):
            os.environ["LLM_MODEL_CHAIN"] = blank
            self.assertEqual(llm_chain._config()["model_chain"],
                             llm_chain.DEFAULT_MODEL_CHAIN.split(","),
                             f"blank {blank!r} must fall back to the default chain, not []")


# ------------------------------ search / reasoning split ------------------------------

class TestSearch(ChainTest):

    def test_search_failure_degrades_to_ungrounded_success(self):
        self.chain(MODEL_A, LLM_ATTEMPTS_PER_MODEL=1)
        self.install(lambda url, body, i: err(503) if is_search(body) else gemini_ok("x"))
        res = llm_chain.call_llm("p", want_search=True)
        self.assertTrue(res.ok)
        self.assertFalse(res.grounded)

    def test_search_leads_containing_braces_survive(self):
        # .replace, never .format. Note the preamble carries a literal brace of its own
        # besides {leads}: braces in the leads VALUE survive .format too (it does not
        # rescan substituted text), so a leads-only case does not discriminate between the
        # two -- verified by mutation. A brace anywhere in the template is what makes
        # .format raise, and callers supply their own templates.
        leads = "{not a placeholder}"
        preamble = 'LEADS:\n{leads}\nShape your answer like {"a": 1}\n---\n'
        self.chain(MODEL_A, LLM_ATTEMPTS_PER_MODEL=1)
        post = self.install(lambda url, body, i:
                            gemini_ok(leads) if is_search(body) else gemini_ok("x"))
        res = llm_chain.call_llm("p", want_search=True, search_preamble=preamble)
        self.assertTrue(res.ok)
        self.assertTrue(res.grounded)
        reasoning = [c for c in post.calls if not is_search(c.body)][0]
        self.assertIn(leads, reasoning.body["contents"][0]["parts"][0]["text"])

    def test_schema_only_on_reasoning_call(self):
        schema = {"type": "OBJECT", "properties": {"a": {"type": "STRING"}}}
        self.chain(MODEL_A, LLM_ATTEMPTS_PER_MODEL=1)
        post = self.install(lambda url, body, i: gemini_ok("leads" if is_search(body) else "x"))
        llm_chain.call_llm("p", want_search=True, response_schema=schema)
        search    = [c for c in post.calls if is_search(c.body)][0]
        reasoning = [c for c in post.calls if not is_search(c.body)][0]
        self.assertNotIn("responseSchema", search.body["generationConfig"])
        self.assertEqual(reasoning.body["generationConfig"]["responseSchema"], schema)

    def test_truncated_when_finish_reason_not_stop(self):
        self.chain(MODEL_A)
        self.install(seq([gemini_ok("partial", finish="MAX_TOKENS")]))
        res = llm_chain.call_llm("p")
        self.assertTrue(res.ok)
        self.assertTrue(res.truncated)


# ------------------------------ ported behaviour ------------------------------

def common_parse_json_block(text):
    """A FROZEN VERBATIM COPY of common.parse_json_block as it stood in the three consumer
    repos at extraction time (weekly-concierge common.py blob 17310bf6). It is duplicated
    here on purpose: this module must not import a consumer's common.py -- being importable
    with nothing but stdlib + requests is the whole point of the package. Comparing against
    a frozen copy is also strictly stronger than importing one consumer's live file, which
    can drift (and has: deal-hunter a7d1c256, shopping-assistant 07d55152 are three
    different files). Do not "simplify" this to call llm_chain.parse_json_block -- the test
    would then compare a function to itself and assert nothing."""
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


class TestParity(ChainTest):

    def test_parse_json_block_matches_common(self):
        cases = [
            '```json\n{"a": 1}\n```',      # fenced object
            '```json\n[{"a": 1}]\n```',    # fenced array
            '{"a": 1}',                    # bare object
            'Here you go:\n{"a": 1}',      # leading prose
            '{"a": ',                      # malformed
            '',                            # empty
        ]
        for case in cases:
            self.assertEqual(llm_chain.parse_json_block(case),
                             common_parse_json_block(case),
                             f"divergence on {case!r}")


if __name__ == "__main__":
    unittest.main(verbosity=1)
