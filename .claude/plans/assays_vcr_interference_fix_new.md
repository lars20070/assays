# Fix VCR Interference with Evaluator Execution

## Context

The evaluator (Bradley-Terry / Pairwise) runs inside `pytest_runtest_makereport(when="call")`, which executes **before** VCR's yield fixture tears down. This causes two problems:

1. **Event loop conflict**: `asyncio.run()` creates loop B while VCR's patched httpx transports hold state from loop A, causing deadlocks
2. **Cassette pollution**: VCR intercepts evaluator HTTP calls to Ollama -- recording unwanted requests or failing on playback due to the tournament's random pair selection

## Solution

Move evaluator execution from `pytest_runtest_makereport` to a **hookwrapper** on `pytest_runtest_teardown`. Code after `yield` in the hookwrapper runs **after** all fixture finalizers (VCR cassette closed), but `item` and `item.stash` remain accessible.

**Note on post-teardown data access**: `item.funcargs` and `item.stash` are plain dicts on the `Item` object. Fixture finalization tears down resources but does not clear these dicts. `AssayContext` (a Pydantic model with `dataset`, `path`, `assay_mode`) holds no external resources that a finalizer would invalidate. The same applies to `item.stash[AGENT_RESPONSES_KEY]` and `item.stash[BASELINE_DATASET_KEY]`. This is safe, but worth documenting with a code comment for future contributors.

### Hook execution order (after fix)

```
1. pytest_runtest_setup       -> VCR opens, assay loads dataset
2. pytest_runtest_call        -> test runs, Agent.run() captured
3. pytest_runtest_makereport  -> logs test outcome (no evaluation)
4. pytest_runtest_teardown    -> VCR closes (default impl)
                              -> [hookwrapper resumes after yield]
                              -> serialize baseline OR run evaluator
```

## Changes

### 1. [plugin.py](src/assays/plugin.py) -- `pytest_runtest_teardown`

Replace the current `trylast` regular hook with a `hookwrapper`:

```python
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item: Item, nextitem: Item | None) -> Generator[None, None, None]:
    yield  # Default teardown runs: all fixtures finalized, VCR cassette closed

    if not _is_assay(item):
        return

    assay: AssayContext | None = item.funcargs.get("context")  # type: ignore[attr-defined]
    if assay is None:
        return

    if assay.assay_mode == "new_baseline":
        _serialize_baseline(item, assay)
    elif assay.assay_mode == "evaluate":
        _run_evaluation(item, assay)
```

Add `Generator` to the imports (use `collections.abc.Generator` to match the codebase style).

Extract two helper functions from the current inline code:

- `_serialize_baseline(item, assay)` -- current teardown body (merge responses into cases, write to disk)
- `_run_evaluation(item, assay)` -- current makereport evaluation body (get evaluator from marker, `asyncio.run(evaluator(item))`, serialize readout). **Must preserve the existing try/except from makereport** that catches evaluation errors and logs via `logger.exception()`.

### 2. [plugin.py](src/assays/plugin.py) -- `pytest_runtest_makereport`

Strip evaluation logic. Keep only test outcome logging:

```python
@pytest.hookimpl(tryfirst=True)
def pytest_runtest_makereport(item: Item, call: CallInfo) -> None:
    if not _is_assay(item):
        return
    if call.when != "call":
        return

    logger.info(f"Test: {item.nodeid}")
    test_outcome = "failed" if call.excinfo else "passed"
    logger.info(f"Test Outcome: {test_outcome}")
    logger.info(f"Test Duration: {call.duration:.5f} seconds")
```

### 3. [test_plugin.py](tests/test_plugin.py) -- Update tests

Three groups of tests are affected:

**a) `pytest_runtest_makereport` tests (lines ~755-983)**
These currently assert that evaluation runs inside makereport. After the change, makereport only logs test outcomes. The evaluation-related tests must be **moved** to the teardown test group and rewritten to invoke `pytest_runtest_teardown` (or the extracted `_run_evaluation` helper) instead. The remaining makereport tests should verify only the logging behavior (test outcome, duration).

**b) `pytest_runtest_teardown` tests (lines ~585-747)**
These test baseline serialization in `new_baseline` mode. They currently test a `trylast` regular hook. Since the hook becomes a `hookwrapper`, the test setup may need adjustment -- e.g., if tests call the hook directly, they now need to drive the generator (call `next()` then `send(None)` or use `pytest`'s hook caller). Alternatively, if they test the extracted `_serialize_baseline` helper directly, no generator mechanics are needed.

**c) Integration tests (`test_integration_pairwiseevaluator`, `test_integration_bradleyterryevaluator`, lines ~1619-1664)**
These use `@pytest.mark.vcr()` and test end-to-end behavior. They should continue to work unchanged, but verify they still pass after the migration -- they are the primary validation that VCR interference is resolved.

## Why hookwrapper, not trylast

- `hookwrapper` wraps **all** implementations (including defaults and other plugins)
- Semantically explicit: "run my code after teardown completes"
- `trylast` on a regular hook works today but is fragile if another plugin adds a later teardown

## Why not defer to `pytest_sessionfinish`

- Per-item evaluation is simpler and provides immediate feedback
- Session-level batching would require accumulating items and break the existing readout-per-test model
- No benefit since the hookwrapper approach solves both problems cleanly

## Verification

```bash
# 1. Run full test suite
uv run pytest -n auto

# 2. Run VCR+assay integration tests specifically
uv run pytest tests/test_plugin.py::test_integration_pairwiseevaluator -v
uv run pytest tests/test_plugin.py::test_integration_bradleyterryevaluator -v

# 3. Type check
uv run pyright .

# 4. Lint
uv run ruff check --fix . && uv run ruff format .
```
