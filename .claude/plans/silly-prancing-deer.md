# Restructure test_plugin.py

## Context

`tests/test_plugin.py` is 1097 lines mixing pure unit tests with Ollama integration tests. The module-level `model = OpenAIChatModel(...)` runs on every import. Context variable tests are orphaned between sections. `test_module_imports` is fragile (uses `importlib.reload`). There's a critical coverage gap: `_instrumented_agent_run` (the Agent.run monkeypatch body) is never actually invoked by any test.

## Changes

### 1. Split into two files

**`tests/test_plugin.py`** — unit tests (fast, mock-based, no Ollama)
**`tests/test_plugin_integration.py`** — Ollama integration tests (slow, `@pytest.mark.ollama`)

Move to `test_plugin_integration.py`:
- `generate_evaluation_cases()`
- `model`, `BASIC_PROMPT`, `CREATIVE_PROMPT`
- `_run_query_generation()`
- `test_integration_pairwiseevaluator`
- `test_integration_bradleyterryevaluator`

### 2. Remove fragile/duplicative tests

| Test | Reason |
|------|--------|
| `test_module_imports` | Fragile (`importlib.reload`), tests Python import system not plugin behavior |
| `test_assay_modes_constant` | Trivially covered by `test_pytest_addoption` and all tests that set `assay_mode` |
| `test_current_item_var_default` | Tests `contextvars` module, not plugin; covered by `test_pytest_runtest_call_sets_context_var` |
| `test_current_item_var_set_and_get` | Same as above |
| `test_response_capture_simulation` | Duplicates `test_pytest_runtest_call_initializes_stash` + `test_pytest_runtest_call_sets_context_var`; manual append doesn't reflect real mechanism |

### 3. Reorder remaining tests to match plugin.py code flow

```
# Helper Functions (_path, _is_assay)
# pytest_addoption
# pytest_configure
# pytest_runtest_setup
# pytest_runtest_call
# pytest_runtest_teardown (_serialize_baseline path)
# pytest_runtest_teardown (_run_evaluation path)
# Workflow tests (setup + teardown combined)
```

### 4. Add missing test: `_instrumented_agent_run` body

New async test that actually calls `Agent.run()` while the monkeypatch is active, verifying:
- Response is appended to `item.stash[AGENT_RESPONSES_KEY]`
- Original `Agent.run` is called (no infinite recursion)
- Return value is passed through

### 5. Simplify `test_pytest_runtest_call_initializes_stash`

Replace fragile index-based assertion (`list(stash.values())[0]`) with direct key assertion (`AGENT_RESPONSES_KEY in stash`).

### 6. Clean up imports in test_plugin.py

After splitting, remove unused imports: `OpenAIChatModel`, `OpenAIProvider`, `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `config`.

## Files modified

- `tests/test_plugin.py` — restructure, remove tests, add monkeypatch test
- `tests/test_plugin_integration.py` — new file with Ollama integration tests

## Verification

```bash
uv run ruff format .
uv run ruff check --fix .
uv run pyright .
uv run pytest tests/test_plugin.py tests/test_plugin_integration.py -v
```
