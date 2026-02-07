# Assay Plugin + VCR Interference Analysis

## Root Cause: Evaluator runs inside VCR's cassette context

The `vcr` fixture in `pytest-recording` is a **yield fixture** (`pytest_recording/plugin.py:141-168`):

```python
@pytest.fixture(autouse=True)
def vcr(...) -> Iterator[Optional["Cassette"]]:
    if vcr_markers:
        with use_cassette(...) as cassette:
            yield cassette      # <- cassette stays open until teardown
    else:
        yield None
```

The cassette is opened during **setup** and closed during **teardown**. But the evaluator runs in between, during `pytest_runtest_makereport(when="call")`. Here's the execution order:

```
1. pytest_runtest_setup       -> VCR cassette OPENS (yield fixture setup)
2. pytest_runtest_call        -> test runs (assay wraps Agent.run, VCR intercepts HTTP)
3. pytest_runtest_makereport  -> assay runs evaluator here <- VCR cassette STILL OPEN
4. pytest_runtest_teardown    -> VCR cassette CLOSES (yield fixture teardown)
```

In step 3, the code at `plugin.py:355`:
```python
readout = asyncio.run(evaluator(item))
```

This causes **two compounding problems**:

### Problem 1: `asyncio.run()` creates a new event loop while VCR's transport is alive

The test ran in pytest-asyncio's event loop (loop A). `asyncio.run()` creates loop B. VCR patched `httpx` transports during the test. The evaluator's Agent creates a new `httpx.AsyncClient` in loop B, but VCR's patched transport may hold internal state (connection pools, pending operations) tied to loop A. Attempting to use this transport in loop B can **deadlock** -- the transport waits on loop A resources that will never resolve in loop B.

### Problem 2: Evaluator HTTP calls are captured by VCR

Since the cassette is still open, the evaluator's HTTP calls to Ollama (for Bradley-Terry tournament games) are intercepted by VCR:

- **During recording**: VCR records dozens of evaluator requests alongside the test's requests, polluting the cassette
- **During playback**: VCR can't match the evaluator's requests (the `adaptive_uncertainty_strategy` uses random pairs each run), so VCR either raises `CannotSendRequest` or blocks waiting for a match

This explains the **intermittent** nature -- the tournament's randomness means different pairs are evaluated each time, making cassette matching unpredictable.

### Why it works without `@pytest.mark.vcr()`

Without VCR, there's no cassette context, no HTTP patching, and `asyncio.run()` just works with a clean new event loop and real Ollama connections.

---

## Secondary concern: `asyncio.run()` in a sync hook

Using `asyncio.run()` inside `pytest_runtest_makereport` (`plugin.py:355`) is fragile even without VCR. If any other plugin or fixture maintains async state across the call/teardown boundary, the new event loop can conflict. This is a structural issue worth addressing regardless.

---

## Recommended fix

Move the evaluator execution out of `pytest_runtest_makereport` and into `pytest_runtest_teardown` **after** fixtures have torn down, or better yet, into the test's own async context (so it shares the event loop with the test and runs before VCR's fixture tears down -- but with its own non-VCR httpx client).

The cleanest approach: run the evaluator **after** the VCR cassette has closed. One way is to defer evaluation to a `pytest_sessionfinish` hook or a custom late-phase hook. Another is to check for VCR and skip the in-hook evaluation, serializing the data for a post-test evaluation step instead.
