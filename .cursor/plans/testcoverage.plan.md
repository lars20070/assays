---
name: ""
overview: ""
todos: []
isProject: false
---

# Test Coverage Reporting Issue

## Problem

Despite having thorough tests for `config.py`, `logger.py`, and other modules, the reported coverage is unexpectedly low (overall line rate: 57.5%). Files like `config.py` and `logger.py` show **0% coverage** even though tests clearly exercise them.

## Root Cause: Pytest Entry-Point Plugin Loading Before Coverage Starts

The project registers `assays.plugin` as a pytest plugin via an entry point:

```toml
[project.entry-points.pytest11]
assays = "assays.plugin"
```

When pytest starts (including in each xdist worker with `-n auto`), it loads entry-point plugins **before** `pytest-cov` begins measuring coverage. The loading of `assays.plugin` triggers this import chain:

1. `plugin.py` line 16: `from .logger import logger` → imports `logger.py`
2. `logger.py` line 8: `from .config import config` → imports `config.py`
3. `plugin.py` line 17: `from .models import ...` → imports `models.py`

All the **module-level code** in `config.py`, `logger.py`, `models.py`, and `__init__.py` executes during this early plugin loading, **before coverage tracking begins**.

Later, when tests run:

```python
from assays.config import Config, config  # does NOT re-execute config.py
```

Python finds `assays.config` already cached in `sys.modules` and just returns the existing objects — no lines of `config.py` execute again. Since `config.py` and `logger.py` are **100% module-level code** (no function bodies to call later), they end up with **0 hits across the board**.

## Evidence

This also explains the partial-coverage pattern across all files:


| File                 | Coverage | Explanation                                                       |
| -------------------- | -------- | ----------------------------------------------------------------- |
| `plugin.py`          | 94.2%    | Function bodies execute during tests (when coverage IS active)    |
| `pairwise.py`        | 76.4%    | Only function bodies called by tests get hits                     |
| `bradleyterry.py`    | 38.6%    | Same — module-level imports/class defs show 0                     |
| `models.py`          | 10.5%    | Only a few method-body lines get hits                             |
| `config.py`          | 0%       | **Entirely** module-level code — nothing re-executes during tests |
| `logger.py`          | 0%       | **Entirely** module-level code — nothing re-executes during tests |
| `__init__.py` (both) | 0%       | Entirely module-level                                             |


Every line with `hits="0"` in the coverage XML is module-level code (imports, class definitions, top-level calls), while lines with `hits="1"` are inside function/method bodies called during tests.

The `disable_warnings = ["module-not-measured"]` in `pyproject.toml` also hints this issue was previously encountered.

## Fix Options

### Option 1: Lazy imports in `plugin.py` (most robust)

Defer importing `config`, `logger`, `models` from module-level to inside the functions that use them. This way they get imported during test execution when coverage is active.

**Pros:** Fixes coverage regardless of how tests are run.
**Cons:** Slightly more verbose code in plugin.py.

### Option 2: Separate coverage run from xdist

Use `-n0` (or omit `-n auto`) when running with `--cov`, so there's a single process where pytest-cov can start coverage before plugin loading.

**Pros:** Simple config change.
**Cons:** Coverage runs become slower (no parallelism).

### Option 3: Force early coverage via `COVERAGE_PROCESS_START`

Configure the `sitecustomize.py` / `.pth` mechanism so coverage starts at interpreter startup, before any plugin loads.

**Pros:** No code changes needed.
**Cons:** More complex CI/local setup; fragile across environments.

## Recommendation

Option 1 (lazy imports) is the most robust and portable solution. It fixes coverage regardless of test runner configuration and has no CI/environment dependencies.