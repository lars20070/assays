#!/usr/bin/env python3
"""
Unit tests for the pytest assay plugin.

These tests cover:
- Plugin hook functions (pytest_addoption, pytest_configure, etc.)
- Helper functions (_path, _is_assay)
- Agent.run() interception mechanism
- Evaluation strategies
"""

from __future__ import annotations as _annotations

import contextlib
import importlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest
from pydantic_ai import Agent
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings
from pydantic_evals import Case, Dataset
from pytest import Function, Item

import assays.plugin
from assays.config import config  # noqa: F401
from assays.logger import logger
from assays.models import AssayContext, Readout
from assays.plugin import (
    ASSAY_MODES,
    BASELINE_DATASET_KEY,
    BradleyTerryEvaluator,
    PairwiseEvaluator,
    _current_item_var,
    _is_assay,
    _path,
    pytest_addoption,
    pytest_configure,
    pytest_runtest_call,
    pytest_runtest_setup,
    pytest_runtest_teardown,
)

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def _drive_hookwrapper(item: Item, nextitem: Item | None = None) -> None:
    """Helper to drive pytest_runtest_teardown hookwrapper generator."""
    gen = pytest_runtest_teardown(item, nextitem)
    next(gen)  # Run pre-yield code (evaluation/serialization), then pause at yield
    with contextlib.suppress(StopIteration):
        next(gen)  # Resume after yield (fixture finalization)


# =============================================================================
# Module Import Tests
# =============================================================================


def test_module_imports() -> None:
    """
    Test that the plugin module imports correctly and exports expected symbols.

    This test forces a module reload to ensure coverage tracks import-time code.
    """
    # Reload the module to capture import-time coverage
    module = importlib.reload(assays.plugin)

    # Verify module-level exports
    assert module.ASSAY_MODES == ("evaluate", "new_baseline")
    assert callable(module.pytest_addoption)
    assert callable(module.pytest_configure)
    assert callable(module.pytest_runtest_setup)
    assert callable(module.pytest_runtest_call)
    assert callable(module.pytest_runtest_teardown)
    assert hasattr(module, "BradleyTerryEvaluator")
    assert hasattr(module, "PairwiseEvaluator")
    assert hasattr(module, "Readout")
    assert callable(module._path)
    assert callable(module._is_assay)


def test_assay_modes_constant() -> None:
    """Test that ASSAY_MODES contains the expected values."""
    assert ASSAY_MODES == ("evaluate", "new_baseline")
    assert len(ASSAY_MODES) == 2


# =============================================================================
# Helper Function Tests
# =============================================================================


def test_path_computation(mocker: MockerFixture) -> None:
    """Test _path computes the correct assay file path."""
    mock_item = mocker.MagicMock(spec=Item)
    mock_item.path = Path("/project/tests/test_example.py")
    mock_item.name = "test_my_function"

    result = _path(mock_item)

    expected = Path("/project/tests/assays/test_example/test_my_function.json")
    assert result == expected

    # Verify path is absolute
    assert result.is_absolute()

    # Verify path ends with .json
    assert result.suffix == ".json"

    # Verify assays directory is in the path
    assert "assays" in result.parts


def test_path_computation_with_parametrized_test(mocker: MockerFixture) -> None:
    """Test _path strips parameter suffix from parametrized test names."""
    mock_item = mocker.MagicMock(spec=Item)
    mock_item.path = Path("/project/tests/test_example.py")
    mock_item.name = "test_my_function[param1-param2]"

    result = _path(mock_item)

    # Should strip everything after the bracket
    expected = Path("/project/tests/assays/test_example/test_my_function.json")
    assert result == expected


def test_path_computation_nested_directory(mocker: MockerFixture) -> None:
    """Test _path with deeply nested test directories."""
    mock_item = mocker.MagicMock(spec=Item)
    mock_item.path = Path("/project/tests/integration/api/test_endpoints.py")
    mock_item.name = "test_get_user"

    result = _path(mock_item)

    expected = Path("/project/tests/integration/api/assays/test_endpoints/test_get_user.json")
    assert result == expected


def test_is_assay_with_marker(mocker: MockerFixture) -> None:
    """Test _is_assay returns True for Function items with assay marker."""
    mock_item = mocker.MagicMock(spec=Function)
    mock_item.get_closest_marker.return_value = mocker.MagicMock()

    assert _is_assay(mock_item) is True
    mock_item.get_closest_marker.assert_called_once_with("assay")


def test_is_assay_without_marker(mocker: MockerFixture) -> None:
    """Test _is_assay returns False for items without assay marker."""
    mock_item = mocker.MagicMock(spec=Function)
    mock_item.get_closest_marker.return_value = None

    assert _is_assay(mock_item) is False


def test_is_assay_non_function_item(mocker: MockerFixture) -> None:
    """Test _is_assay returns False for non-Function items (e.g., class)."""
    mock_item = mocker.MagicMock(spec=Item)

    # Ensure it's not a Function instance
    assert not isinstance(mock_item, Function)
    assert _is_assay(mock_item) is False


# =============================================================================
# pytest_addoption Tests
# =============================================================================


def test_pytest_addoption(mocker: MockerFixture) -> None:
    """Test pytest_addoption registers the --assay-mode option correctly."""
    mock_parser = mocker.MagicMock()

    pytest_addoption(mock_parser)

    mock_parser.addoption.assert_called_once_with(
        "--assay-mode",
        action="store",
        default="evaluate",
        choices=ASSAY_MODES,
        help='Assay mode. Defaults to "evaluate".',
    )


# =============================================================================
# pytest_configure Tests
# =============================================================================


def test_pytest_configure(mocker: MockerFixture) -> None:
    """Test pytest_configure registers the assay marker."""
    mock_config = mocker.MagicMock()
    mocker.patch("assays.plugin.logger")

    pytest_configure(mock_config)

    mock_config.addinivalue_line.assert_called_once()
    call_args = mock_config.addinivalue_line.call_args
    assert call_args[0][0] == "markers"
    assert "assay" in call_args[0][1]


# =============================================================================
# pytest_runtest_setup Tests
# =============================================================================


def test_pytest_runtest_setup_non_assay_item(mocker: MockerFixture) -> None:
    """Test pytest_runtest_setup skips non-assay items."""
    mock_item = mocker.MagicMock(spec=Item)
    mock_item.funcargs = {}
    mocker.patch("assays.plugin._is_assay", return_value=False)

    # Should not raise and should not modify funcargs
    pytest_runtest_setup(mock_item)

    # funcargs should not have been accessed/modified
    assert "context" not in mock_item.funcargs


def test_pytest_runtest_setup_with_existing_dataset(mocker: MockerFixture, tmp_path: Path) -> None:
    """Test pytest_runtest_setup loads existing dataset from file."""
    # Create a temporary dataset file with realistic topic data
    dataset_path = tmp_path / "assays" / "test_module" / "test_func.json"
    dataset_path.parent.mkdir(parents=True)
    dataset = Dataset[dict[str, str], type[None], Any](
        cases=[
            Case(name="case_001", inputs={"topic": "pangolin trafficking", "query": "existing query"}),
            Case(name="case_002", inputs={"topic": "molecular gastronomy", "query": "another query"}),
        ]
    )
    dataset.to_file(dataset_path, schema_path=None)

    # Mock the item
    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {}
    mock_item.stash = {}
    mock_marker = mocker.MagicMock()
    mock_marker.kwargs = {}
    mock_item.get_closest_marker.return_value = mock_marker
    mock_item.config.getoption.return_value = "evaluate"

    mocker.patch("assays.plugin._is_assay", return_value=True)
    mocker.patch("assays.plugin._path", return_value=dataset_path)
    mocker.patch("assays.plugin.logger")

    pytest_runtest_setup(mock_item)

    # Verify baseline dataset was stashed for evaluators
    assert assays.plugin.BASELINE_DATASET_KEY in mock_item.stash
    baseline = mock_item.stash[assays.plugin.BASELINE_DATASET_KEY]
    assert len(baseline.cases) == 2
    assert baseline.cases[0].inputs["query"] == "existing query"

    # Verify assay context was injected
    assert "context" in mock_item.funcargs
    assay_ctx = mock_item.funcargs["context"]

    # Check by attribute presence instead of isinstance (module reload can cause different class instances)
    assert hasattr(assay_ctx, "dataset")
    assert hasattr(assay_ctx, "path")
    assert hasattr(assay_ctx, "assay_mode")

    # Verify dataset content was loaded correctly
    assert len(assay_ctx.dataset.cases) == 2
    assert assay_ctx.path == dataset_path
    assert assay_ctx.assay_mode == "evaluate"

    # Verify case data integrity
    assert assay_ctx.dataset.cases[0].name == "case_001"
    assert assay_ctx.dataset.cases[0].inputs["topic"] == "pangolin trafficking"
    assert assay_ctx.dataset.cases[0].inputs["query"] == "existing query"

    assert assay_ctx.dataset.cases[1].name == "case_002"
    assert assay_ctx.dataset.cases[1].inputs["topic"] == "molecular gastronomy"
    assert assay_ctx.dataset.cases[1].inputs["query"] == "another query"

    # Verify cases are iterable (as used in test_curiosity.py)
    topics = [case.inputs["topic"] for case in assay_ctx.dataset.cases]
    assert topics == ["pangolin trafficking", "molecular gastronomy"]


def test_pytest_runtest_setup_with_generator(mocker: MockerFixture, tmp_path: Path) -> None:
    """Test pytest_runtest_setup calls generator when no dataset file exists."""
    dataset_path = tmp_path / "assays" / "test_module" / "test_func.json"

    # Mock generator that returns a dataset (like generate_evaluation_cases in test_curiosity.py)
    topics = ["pangolin trafficking", "molecular gastronomy", "dark kitchen economics"]
    generated_cases: list[Case[dict[str, str], type[None], Any]] = [
        Case(name=f"case_{idx:03d}", inputs={"topic": topic}) for idx, topic in enumerate(topics)
    ]
    generated_dataset = Dataset[dict[str, str], type[None], Any](cases=generated_cases)
    mock_generator = mocker.MagicMock(return_value=generated_dataset)

    # Mock the item
    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {}
    mock_item.stash = {}
    mock_marker = mocker.MagicMock()
    mock_marker.kwargs = {"generator": mock_generator}
    mock_item.get_closest_marker.return_value = mock_marker
    mock_item.config.getoption.return_value = "evaluate"

    mocker.patch("assays.plugin._is_assay", return_value=True)
    mocker.patch("assays.plugin._path", return_value=dataset_path)
    mocker.patch("assays.plugin.logger")

    pytest_runtest_setup(mock_item)

    # Verify baseline dataset was stashed
    assert assays.plugin.BASELINE_DATASET_KEY in mock_item.stash
    assert len(mock_item.stash[assays.plugin.BASELINE_DATASET_KEY].cases) == 3

    # Verify generator was called exactly once
    mock_generator.assert_called_once()

    # Verify dataset file was created
    assert dataset_path.exists()
    assert dataset_path.suffix == ".json"

    # Verify file contains valid JSON that can be reloaded
    reloaded_dataset = Dataset[dict[str, str], type[None], Any].from_file(dataset_path)
    assert len(reloaded_dataset.cases) == 3

    # Verify reloaded data matches original
    for idx, topic in enumerate(topics):
        assert reloaded_dataset.cases[idx].name == f"case_{idx:03d}"
        assert reloaded_dataset.cases[idx].inputs["topic"] == topic

    # Verify assay context was injected with correct data
    assay_ctx = mock_item.funcargs["context"]
    assert len(assay_ctx.dataset.cases) == 3
    assert assay_ctx.path == dataset_path
    assert assay_ctx.assay_mode == "evaluate"

    # Verify case content
    assert assay_ctx.dataset.cases[0].inputs["topic"] == "pangolin trafficking"
    assert assay_ctx.dataset.cases[1].inputs["topic"] == "molecular gastronomy"
    assert assay_ctx.dataset.cases[2].inputs["topic"] == "dark kitchen economics"


def test_pytest_runtest_setup_generator_invalid_return(mocker: MockerFixture, tmp_path: Path) -> None:
    """Test pytest_runtest_setup raises TypeError for invalid generator return."""
    dataset_path = tmp_path / "assays" / "test_module" / "test_func.json"

    # Mock generator that returns invalid type
    mock_generator = mocker.MagicMock(return_value="not a dataset")

    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {}
    mock_marker = mocker.MagicMock()
    mock_marker.kwargs = {"generator": mock_generator}
    mock_item.get_closest_marker.return_value = mock_marker

    mocker.patch("assays.plugin._is_assay", return_value=True)
    mocker.patch("assays.plugin._path", return_value=dataset_path)
    mocker.patch("assays.plugin.logger")

    with pytest.raises(TypeError, match="must return a Dataset instance"):
        pytest_runtest_setup(mock_item)


def test_pytest_runtest_setup_empty_dataset(mocker: MockerFixture, tmp_path: Path) -> None:
    """Test pytest_runtest_setup uses empty dataset when no file or generator."""
    dataset_path = tmp_path / "assays" / "test_module" / "test_func.json"

    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {}
    mock_item.stash = {}
    mock_marker = mocker.MagicMock()
    mock_marker.kwargs = {}  # No generator
    mock_item.get_closest_marker.return_value = mock_marker
    mock_item.config.getoption.return_value = "evaluate"

    mocker.patch("assays.plugin._is_assay", return_value=True)
    mocker.patch("assays.plugin._path", return_value=dataset_path)
    mocker.patch("assays.plugin.logger")

    pytest_runtest_setup(mock_item)

    assert assays.plugin.BASELINE_DATASET_KEY in mock_item.stash
    assert len(mock_item.stash[assays.plugin.BASELINE_DATASET_KEY].cases) == 0

    assay_ctx = mock_item.funcargs["context"]
    assert len(assay_ctx.dataset.cases) == 0


def test_pytest_runtest_setup_baseline_stash_is_copy(mocker: MockerFixture, tmp_path: Path) -> None:
    """Test that mutating assay.dataset does not change the stashed baseline."""
    dataset_path = tmp_path / "assays" / "test_module" / "test_func.json"
    dataset_path.parent.mkdir(parents=True)
    dataset = Dataset[dict[str, str], type[None], Any](
        cases=[
            Case(name="case_001", inputs={"topic": "topic A", "query": "baseline query A"}),
        ]
    )
    dataset.to_file(dataset_path, schema_path=None)

    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {}
    mock_item.stash = {}
    mock_marker = mocker.MagicMock()
    mock_marker.kwargs = {}
    mock_item.get_closest_marker.return_value = mock_marker
    mock_item.config.getoption.return_value = "evaluate"

    mocker.patch("assays.plugin._is_assay", return_value=True)
    mocker.patch("assays.plugin._path", return_value=dataset_path)
    mocker.patch("assays.plugin.logger")

    pytest_runtest_setup(mock_item)

    assay_ctx = mock_item.funcargs["context"]
    baseline = mock_item.stash[assays.plugin.BASELINE_DATASET_KEY]

    # Simulate test mutation (like test_curiosity.py)
    assay_ctx.dataset.cases.clear()
    assay_ctx.dataset.cases.append(Case(name="case_001", inputs={"topic": "topic A", "query": "novel query A"}))

    # Stashed baseline must be unchanged
    assert len(baseline.cases) == 1
    assert baseline.cases[0].inputs["query"] == "baseline query A"
    assert assay_ctx.dataset.cases[0].inputs["query"] == "novel query A"


# =============================================================================
# pytest_runtest_call Tests
# =============================================================================


def test_pytest_runtest_call_non_function_item(mocker: MockerFixture) -> None:
    """Test pytest_runtest_call yields immediately for non-Function items."""
    mock_item = mocker.MagicMock(spec=Item)

    gen = pytest_runtest_call(mock_item)
    next(gen)  # Should yield immediately

    with pytest.raises(StopIteration):
        next(gen)


def test_pytest_runtest_call_without_assay_marker(mocker: MockerFixture) -> None:
    """Test pytest_runtest_call yields immediately without assay marker."""
    mock_item = mocker.MagicMock(spec=Function)
    mock_item.get_closest_marker.return_value = None

    gen = pytest_runtest_call(mock_item)
    next(gen)

    with pytest.raises(StopIteration):
        next(gen)


def test_pytest_runtest_call_initializes_stash(mocker: MockerFixture) -> None:
    """Test pytest_runtest_call initializes the response stash."""
    mock_item = mocker.MagicMock(spec=Function)
    mock_item.stash = {}
    mock_marker = mocker.MagicMock()
    mock_item.get_closest_marker.return_value = mock_marker

    mocker.patch("assays.plugin.logger")

    gen = pytest_runtest_call(mock_item)
    next(gen)  # Run until yield

    # Verify stash was initialized - check that at least one key exists with empty list
    assert len(mock_item.stash) == 1
    stash_values = list(mock_item.stash.values())
    assert stash_values[0] == []

    # Clean up
    with contextlib.suppress(StopIteration):
        next(gen)


def test_pytest_runtest_call_sets_context_var(mocker: MockerFixture) -> None:
    """Test pytest_runtest_call sets the current item context variable."""
    mock_item = mocker.MagicMock(spec=Function)
    mock_item.stash = {}
    mock_marker = mocker.MagicMock()
    mock_item.get_closest_marker.return_value = mock_marker

    mocker.patch("assays.plugin.logger")

    gen = pytest_runtest_call(mock_item)
    next(gen)

    # During the yield, context var should be set
    assert assays.plugin._current_item_var.get() == mock_item

    # Clean up and verify context var is reset
    with contextlib.suppress(StopIteration):
        next(gen)

    assert assays.plugin._current_item_var.get() is None


# =============================================================================
# pytest_runtest_teardown Tests
# =============================================================================


def test_pytest_runtest_teardown_non_assay_item(mocker: MockerFixture) -> None:
    """Test pytest_runtest_teardown skips non-assay items."""
    mock_item = mocker.MagicMock(spec=Item)
    mocker.patch("assays.plugin._is_assay", return_value=False)

    # Drive the hookwrapper generator
    _drive_hookwrapper(mock_item, None)


def test_pytest_runtest_teardown_evaluate_mode(mocker: MockerFixture, tmp_path: Path) -> None:
    """Test pytest_runtest_teardown does not serialize in evaluate mode."""
    dataset_path = tmp_path / "assays" / "test.json"
    dataset = Dataset[dict[str, str], type[None], Any](cases=[])

    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {"context": AssayContext(dataset=dataset, path=dataset_path, assay_mode="evaluate")}
    mock_marker = mocker.MagicMock()
    mock_marker.kwargs = {"evaluator": AsyncMock(return_value=Readout(passed=True))}
    mock_item.get_closest_marker.return_value = mock_marker

    mocker.patch("assays.plugin._is_assay", return_value=True)
    mocker.patch("assays.plugin.logger")
    mocker.patch("assays.plugin.asyncio.run")  # Mock to prevent actual evaluation

    # Drive the hookwrapper generator
    _drive_hookwrapper(mock_item, None)

    # File should not be created in evaluate mode (evaluation doesn't serialize dataset)
    assert not dataset_path.exists()


def test_pytest_runtest_teardown_new_baseline_mode(mocker: MockerFixture, tmp_path: Path) -> None:
    """Test pytest_runtest_teardown merges captured responses and serializes dataset in new_baseline mode."""
    dataset_path = tmp_path / "assays" / "test.json"

    # Create dataset with cases that have empty expected_output (as generated)
    cases: list[Case[dict[str, str], str, Any]] = [
        Case(name="case_000", inputs={"topic": "topic A"}, expected_output=""),
        Case(name="case_001", inputs={"topic": "topic B"}, expected_output=""),
    ]
    dataset = Dataset[dict[str, str], str, Any](cases=cases)

    # Create mock AgentRunResult responses (captured by pytest_runtest_call)
    mock_response_a = mocker.MagicMock(spec=AgentRunResult)
    mock_response_a.output = "generated query A"
    mock_response_b = mocker.MagicMock(spec=AgentRunResult)
    mock_response_b.output = "generated query B"

    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {"context": AssayContext(dataset=dataset, path=dataset_path, assay_mode="new_baseline")}
    mock_item.stash = {
        assays.plugin.AGENT_RESPONSES_KEY: [mock_response_a, mock_response_b],
    }

    mocker.patch("assays.plugin._is_assay", return_value=True)
    mocker.patch("assays.plugin.logger")

    # Drive the hookwrapper generator
    _drive_hookwrapper(mock_item, None)

    # File should be created in new_baseline mode
    assert dataset_path.exists()
    assert dataset_path.suffix == ".json"

    # Verify serialized content can be reloaded
    reloaded = Dataset[dict[str, str], str, Any].from_file(dataset_path)
    assert len(reloaded.cases) == 2

    # Verify expected_output was populated from captured responses
    assert reloaded.cases[0].name == "case_000"
    assert reloaded.cases[0].inputs["topic"] == "topic A"
    assert reloaded.cases[0].expected_output == "generated query A"

    assert reloaded.cases[1].name == "case_001"
    assert reloaded.cases[1].inputs["topic"] == "topic B"
    assert reloaded.cases[1].expected_output == "generated query B"


def test_pytest_runtest_teardown_new_baseline_response_count_mismatch(mocker: MockerFixture, tmp_path: Path) -> None:
    """Test pytest_runtest_teardown skips serialization when response count mismatches case count."""
    dataset_path = tmp_path / "assays" / "test.json"

    cases: list[Case[dict[str, str], str, Any]] = [
        Case(name="case_000", inputs={"topic": "topic A"}, expected_output=""),
        Case(name="case_001", inputs={"topic": "topic B"}, expected_output=""),
    ]
    dataset = Dataset[dict[str, str], str, Any](cases=cases)

    # Only one response for two cases
    mock_response = mocker.MagicMock(spec=AgentRunResult)
    mock_response.output = "query A"

    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {"context": AssayContext(dataset=dataset, path=dataset_path, assay_mode="new_baseline")}
    mock_item.stash = {
        assays.plugin.AGENT_RESPONSES_KEY: [mock_response],
    }

    mocker.patch("assays.plugin._is_assay", return_value=True)
    mock_logger = mocker.patch("assays.plugin.logger")

    # Drive the hookwrapper generator
    _drive_hookwrapper(mock_item, None)

    # File should NOT be created due to mismatch
    assert not dataset_path.exists()
    mock_logger.error.assert_called_once()
    assert "Cannot merge responses" in str(mock_logger.error.call_args)


def test_pytest_runtest_teardown_new_baseline_none_output(mocker: MockerFixture, tmp_path: Path) -> None:
    """Test pytest_runtest_teardown uses empty string for None response output."""
    dataset_path = tmp_path / "assays" / "test.json"

    cases: list[Case[dict[str, str], str, Any]] = [
        Case(name="case_000", inputs={"topic": "topic A"}, expected_output=""),
    ]
    dataset = Dataset[dict[str, str], str, Any](cases=cases)

    mock_response = mocker.MagicMock(spec=AgentRunResult)
    mock_response.output = None

    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {"context": AssayContext(dataset=dataset, path=dataset_path, assay_mode="new_baseline")}
    mock_item.stash = {
        assays.plugin.AGENT_RESPONSES_KEY: [mock_response],
    }

    mocker.patch("assays.plugin._is_assay", return_value=True)
    mocker.patch("assays.plugin.logger")

    # Drive the hookwrapper generator
    _drive_hookwrapper(mock_item, None)

    assert dataset_path.exists()
    reloaded = Dataset[dict[str, str], str, Any].from_file(dataset_path)
    assert reloaded.cases[0].expected_output == ""


def test_pytest_runtest_teardown_new_baseline_no_responses(mocker: MockerFixture, tmp_path: Path) -> None:
    """Test pytest_runtest_teardown skips serialization when no responses captured but cases exist."""
    dataset_path = tmp_path / "assays" / "test.json"

    cases: list[Case[dict[str, str], str, Any]] = [
        Case(name="case_000", inputs={"topic": "topic A"}, expected_output=""),
    ]
    dataset = Dataset[dict[str, str], str, Any](cases=cases)

    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {"context": AssayContext(dataset=dataset, path=dataset_path, assay_mode="new_baseline")}
    mock_item.stash = {}  # No AGENT_RESPONSES_KEY

    mocker.patch("assays.plugin._is_assay", return_value=True)
    mock_logger = mocker.patch("assays.plugin.logger")

    # Drive the hookwrapper generator
    _drive_hookwrapper(mock_item, None)

    # File should NOT be created (0 responses vs 1 case)
    assert not dataset_path.exists()
    mock_logger.error.assert_called_once()


def test_pytest_runtest_teardown_no_assay_context(mocker: MockerFixture) -> None:
    """Test pytest_runtest_teardown handles missing assay context gracefully."""
    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {}  # No assay context

    mocker.patch("assays.plugin._is_assay", return_value=True)

    # Drive the hookwrapper generator - should not raise
    _drive_hookwrapper(mock_item, None)


def test_pytest_runtest_teardown_runs_evaluation(mocker: MockerFixture, tmp_path: Path) -> None:
    """Test pytest_runtest_teardown runs evaluation and serializes readout in evaluate mode."""
    dataset = Dataset[dict[str, str], type[None], Any](cases=[])
    assay_path = tmp_path / "assays" / "test_module" / "test_func.json"
    assay_path.parent.mkdir(parents=True, exist_ok=True)

    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {"context": AssayContext(dataset=dataset, path=assay_path, assay_mode="evaluate")}
    mock_marker = mocker.MagicMock()

    # Custom evaluator mock - returns Readout with details
    mock_evaluator = AsyncMock(return_value=Readout(passed=True, details={"test_key": "test_value"}))
    mock_marker.kwargs = {"evaluator": mock_evaluator}
    mock_item.get_closest_marker.return_value = mock_marker

    mocker.patch("assays.plugin._is_assay", return_value=True)
    mock_logger = mocker.patch("assays.plugin.logger")

    # Drive the hookwrapper generator
    _drive_hookwrapper(mock_item, None)

    # Evaluator should have been called
    mock_evaluator.assert_called_once_with(mock_item)
    # Should log the result
    mock_logger.info.assert_any_call("Evaluation result: passed=True")

    # Verify readout file was created
    readout_path = assay_path.with_suffix(".readout.json")
    assert readout_path.exists()

    # Verify readout content
    with readout_path.open() as f:
        readout_data = json.load(f)
    assert readout_data["passed"] is True
    assert readout_data["details"] == {"test_key": "test_value"}


def test_pytest_runtest_teardown_uses_default_evaluator(mocker: MockerFixture, tmp_path: Path) -> None:
    """Test pytest_runtest_teardown uses BradleyTerryEvaluator() as default."""
    dataset = Dataset[dict[str, str], type[None], Any](cases=[])
    assay_path = tmp_path / "assays" / "test_module" / "test_func.json"
    assay_path.parent.mkdir(parents=True, exist_ok=True)

    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {"context": AssayContext(dataset=dataset, path=assay_path, assay_mode="evaluate")}
    mock_item.stash = {
        assays.plugin.AGENT_RESPONSES_KEY: [],
        BASELINE_DATASET_KEY: dataset,
    }
    mock_marker = mocker.MagicMock()
    mock_marker.kwargs = {}  # No evaluator specified - should use default
    mock_item.get_closest_marker.return_value = mock_marker

    mocker.patch("assays.plugin._is_assay", return_value=True)
    mock_logger = mocker.patch("assays.plugin.logger")

    # Mock BradleyTerryEvaluator so the default evaluator is a mock we control
    mock_bt_evaluator = AsyncMock(return_value=Readout(passed=True, details={"default": True}))
    mocker.patch("assays.plugin.BradleyTerryEvaluator", return_value=mock_bt_evaluator)

    # Drive the hookwrapper generator
    _drive_hookwrapper(mock_item, None)

    # The default BradleyTerryEvaluator instance should have been called
    mock_bt_evaluator.assert_called_once_with(mock_item)
    # Should log evaluation completed (no error)
    assert any("Evaluation result" in str(call) for call in mock_logger.info.call_args_list)


def test_pytest_runtest_teardown_invalid_evaluator(mocker: MockerFixture) -> None:
    """Test pytest_runtest_teardown handles non-callable evaluator."""
    dataset = Dataset[dict[str, str], type[None], Any](cases=[])

    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {"context": AssayContext(dataset=dataset, path=Path("/tmp/test.json"), assay_mode="evaluate")}
    mock_marker = mocker.MagicMock()
    mock_marker.kwargs = {"evaluator": "not_callable"}  # Invalid
    mock_item.get_closest_marker.return_value = mock_marker

    mocker.patch("assays.plugin._is_assay", return_value=True)
    mock_logger = mocker.patch("assays.plugin.logger")

    # Drive the hookwrapper generator
    _drive_hookwrapper(mock_item, None)

    # Should log error for invalid evaluator
    mock_logger.error.assert_called_once()
    assert "Invalid evaluator type" in str(mock_logger.error.call_args)


def test_pytest_runtest_teardown_evaluation_exception(mocker: MockerFixture) -> None:
    """Test pytest_runtest_teardown handles evaluation exceptions."""
    dataset = Dataset[dict[str, str], type[None], Any](cases=[])

    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {"context": AssayContext(dataset=dataset, path=Path("/tmp/test.json"), assay_mode="evaluate")}
    mock_marker = mocker.MagicMock()

    # Evaluator that raises an exception
    async def failing_evaluator(item: Item) -> Readout:
        raise RuntimeError("Evaluation failed")

    mock_marker.kwargs = {"evaluator": failing_evaluator}
    mock_item.get_closest_marker.return_value = mock_marker

    mocker.patch("assays.plugin._is_assay", return_value=True)
    mock_logger = mocker.patch("assays.plugin.logger")

    # Drive the hookwrapper generator - should not raise, exception is handled internally
    _drive_hookwrapper(mock_item, None)

    # Should log exception
    mock_logger.exception.assert_called_once()


# =============================================================================
# Integration-style Tests (Realistic Workflows)
# =============================================================================


def test_full_assay_workflow_with_topic_generation(mocker: MockerFixture, tmp_path: Path) -> None:
    """
    Test a realistic assay workflow similar to test_curiosity.py.

    This test verifies the complete flow:
    1. Generator creates initial cases with topics
    2. Setup injects AssayContext
    3. Agent.run() responses are captured in AGENT_RESPONSES_KEY
    4. Teardown merges responses into expected_output and serializes
    """
    dataset_path = tmp_path / "assays" / "test_curiosity" / "test_search_queries.json"

    # Define topics like in test_curiosity.py
    topics = ["pangolin trafficking networks", "molecular gastronomy", "dark kitchen economics"]

    # Generator function (similar to generate_evaluation_cases)
    def generate_cases() -> Dataset[dict[str, str], str, Any]:
        cases: list[Case[dict[str, str], str, Any]] = []
        for idx, topic in enumerate(topics):
            case = Case(name=f"case_{idx:03d}", inputs={"topic": topic}, expected_output="")
            cases.append(case)
        return Dataset[dict[str, str], str, Any](cases=cases)

    # Setup: Create mock item with generator
    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {}
    mock_item.stash = {}
    mock_marker = mocker.MagicMock()
    mock_marker.kwargs = {"generator": generate_cases}
    mock_item.get_closest_marker.return_value = mock_marker
    mock_item.config.getoption.return_value = "new_baseline"

    mocker.patch("assays.plugin._is_assay", return_value=True)
    mocker.patch("assays.plugin._path", return_value=dataset_path)
    mocker.patch("assays.plugin.logger")

    # Run setup
    pytest_runtest_setup(mock_item)

    # Verify setup injected correct context
    assert "context" in mock_item.funcargs
    assay_ctx = mock_item.funcargs["context"]
    assert len(assay_ctx.dataset.cases) == 3
    assert assay_ctx.assay_mode == "new_baseline"

    # Simulate captured Agent.run() responses (populated by pytest_runtest_call)
    mock_responses = []
    for topic in topics:
        mock_response = mocker.MagicMock(spec=AgentRunResult)
        mock_response.output = f"search for: {topic}"
        mock_responses.append(mock_response)
    mock_item.stash[assays.plugin.AGENT_RESPONSES_KEY] = mock_responses

    # Run teardown (should merge responses and serialize in new_baseline mode)
    # Drive the hookwrapper generator
    _drive_hookwrapper(mock_item, None)

    # Verify file was created with updated data
    assert dataset_path.exists()

    # Reload and verify data integrity
    reloaded = Dataset[dict[str, str], str, Any].from_file(dataset_path)
    assert len(reloaded.cases) == 3

    for idx, topic in enumerate(topics):
        assert reloaded.cases[idx].name == f"case_{idx:03d}"
        assert reloaded.cases[idx].inputs["topic"] == topic
        assert reloaded.cases[idx].expected_output == f"search for: {topic}"


def test_response_capture_simulation(mocker: MockerFixture) -> None:
    """
    Test that the response capture mechanism correctly stores Agent outputs.

    This simulates what happens during pytest_runtest_call when Agent.run() is called.
    """
    # Setup mock item with assay marker
    mock_item = mocker.MagicMock(spec=Function)
    mock_item.stash = {}
    mock_marker = mocker.MagicMock()
    mock_item.get_closest_marker.return_value = mock_marker

    mocker.patch("assays.plugin.logger")

    # Run the hook to initialize stash
    gen = pytest_runtest_call(mock_item)
    next(gen)  # Run until yield

    # Verify stash was initialized with empty list
    assert assays.plugin.AGENT_RESPONSES_KEY in mock_item.stash
    responses = mock_item.stash[assays.plugin.AGENT_RESPONSES_KEY]
    assert responses == []

    # Verify context var was set
    assert assays.plugin._current_item_var.get() == mock_item

    # Simulate adding responses (normally done by wrapped Agent.run)
    mock_response = mocker.MagicMock(spec=AgentRunResult)
    mock_response.output = "test output"
    responses.append(mock_response)

    # Verify response was captured
    assert len(mock_item.stash[assays.plugin.AGENT_RESPONSES_KEY]) == 1
    assert mock_item.stash[assays.plugin.AGENT_RESPONSES_KEY][0].output == "test output"

    # Clean up
    with contextlib.suppress(StopIteration):
        next(gen)

    # Verify context var was reset
    assert _current_item_var.get() is None


# =============================================================================
# Context Variable Tests
# =============================================================================


def test_current_item_var_default() -> None:
    """Test that _current_item_var has None as default."""
    # Reset to ensure clean state
    token = _current_item_var.set(None)
    try:
        assert _current_item_var.get() is None
    finally:
        _current_item_var.reset(token)


def test_current_item_var_set_and_get(mocker: MockerFixture) -> None:
    """Test setting and getting the current item context variable."""
    mock_item = mocker.MagicMock(spec=Item)

    token = _current_item_var.set(mock_item)
    try:
        assert _current_item_var.get() == mock_item
    finally:
        _current_item_var.reset(token)

    # After reset, should be None again
    assert _current_item_var.get() is None


# =============================================================================
# Full integration test for the 'assay' pytest plugin
#
# The tests generate search queries for various research topics.
# The @pytest.mark.assay decorator triggers an *assay* which produces a *readout*.
# The assay evalutes the curiosity and originality of the generated search queries.
#
# Two evaluator strategies are tested:
# 1. PairwiseEvaluator: Compares baseline vs novel responses directly
# 2. BradleyTerryEvaluator: Ranks all responses using the Bradley-Terry model
#
# The assays compare two different sets of generated search queries:
# A. Some baseline results generated with BASIC_PROMPT in the assay-mode 'new_baseline'.
#    These results have been pre-recorded and stored in the `assays/` subfolder.`
# B. More creative results generated with CREATIVE_PROMPT in the (default) assay-mode 'evaluate'.
# =============================================================================


def generate_evaluation_cases() -> Dataset[dict[str, str], str, Any]:
    """Generate a list of Cases containing topics as input."""
    logger.info("Creating new assay dataset.")

    topics = [
        "pangolin trafficking networks",
        "molecular gastronomy",
        "dark kitchen economics",
        "kintsugi philosophy",
        "nano-medicine delivery systems",
        "Streisand effect dynamics",
        "Anne Brorhilker",
        "bioconcrete self-healing",
        "bacteriophage therapy revival",
        "Habsburg jaw genetics",
    ]

    cases: list[Case[dict[str, str], str, Any]] = []
    for idx, topic in enumerate(topics):
        logger.debug(f"Case {idx + 1} / {len(topics)} with topic: {topic}")
        case = Case(
            name=f"case_{idx:03d}",
            inputs={"topic": topic},
            expected_output="",
        )
        cases.append(case)

    return Dataset[dict[str, str], str, Any](cases=cases)


# Model for both (1) the search query generation and (2) the evaluation of the generated queries.
model = OpenAIChatModel(
    model_name="qwen3:14b",
    provider=OpenAIProvider(base_url="http://localhost:11434/v1"),  # Local Ollama server
)


BASIC_PROMPT = "Please generate a useful search query for the following research topic: <TOPIC>{topic}</TOPIC>"

CREATIVE_PROMPT = """\
Please generate a very creative search query for the research topic: <TOPIC>{topic}</TOPIC>
The query should show genuine originality and interest in the topic. AVOID any generic or formulaic phrases.

Examples of formulaic queries for the topic 'molecular gastronomy' (BAD):
- 'Definition molecular gastronomy'
- 'Molecular gastronomy techniques and applications'

Examples of curious, creative queries for the topic 'molecular gastronomy' (GOOD):
- 'Use of liquid nitrogen instead of traditional freezing for food texture'
- 'Failed molecular gastronomy experiments that led to new dishes'

Now generate one creative search query for: <TOPIC>{topic}</TOPIC>"""


async def _run_query_generation(context: AssayContext) -> None:
    """Run query generation for all cases in the dataset."""
    query_agent = Agent(
        model=model,
        output_type=str,
        system_prompt="Please generate a concise web search query for the given research topic. You must respond only in English.",
        retries=5,
        instrument=True,
    )

    for case in context.dataset.cases:
        logger.info(f"Case {case.name} with topic: {case.inputs['topic']}")

        # prompt = BASIC_PROMPT.format(topic=case.inputs["topic"])  # for assay-mode 'new_baseline'
        prompt = CREATIVE_PROMPT.format(topic=case.inputs["topic"])  # for assay-mode 'evaluate'

        async with query_agent:
            result = await query_agent.run(
                user_prompt=prompt,
                model_settings=ModelSettings(temperature=0.0, timeout=300),
            )

        logger.debug(f"Generated search query: {result.output}")


# =============================================================================
# Integration tests for the 'assay' pytest plugin with different evaluators
# =============================================================================


@pytest.mark.skip()
@pytest.mark.assay(
    generator=generate_evaluation_cases,
    evaluator=PairwiseEvaluator(
        model=model,
        criterion="Which of the two search queries shows more genuine curiosity and creativity, and is less formulaic?",
    ),
)
@pytest.mark.asyncio
async def test_integration_pairwiseevaluator(context: AssayContext) -> None:
    """
    Integration test for the 'assay' pytest plugin with PairwiseEvaluator evaluator.

    An agent generates search queries for various research topics.
    PairwiseEvaluator then evaluates the creativity of the generated queries.

    Args:
        context: The assay context containing the evaluation dataset context.dataset and other information.
    """
    logger.info("Integration test for assay pytest plugin with PairwiseEvaluator.")
    await _run_query_generation(context)


@pytest.mark.skip()
@pytest.mark.assay(
    generator=generate_evaluation_cases,
    evaluator=BradleyTerryEvaluator(
        model=model,
        criterion="Which of the two search queries shows more genuine curiosity and creativity, and is less formulaic?",
        max_standard_deviation=2.1,
    ),
)
@pytest.mark.asyncio
async def test_integration_bradleyterryevaluator(context: AssayContext) -> None:
    """
    Integration test for the 'assay' pytest plugin with BradleyTerryEvaluator evaluator.

    An agent generates search queries for various research topics.
    BradleyTerryEvaluator then evaluates the creativity of the generated queries.

    Args:
        context: The assay context containing the evaluation dataset context.dataset and other information.
    """
    logger.info("Integration test for assay pytest plugin with BradleyTerryEvaluator.")
    await _run_query_generation(context)
