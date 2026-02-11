#!/usr/bin/env python3
"""Unit tests for the PairwiseEvaluator."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_evals import Case, Dataset
from pytest import Function

import assays.plugin
from assays.evaluators.pairwise import PairwiseEvaluator
from assays.plugin import AGENT_RESPONSES_KEY, BASELINE_DATASET_KEY, AssayContext

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def test_pairwise_evaluator_init_defaults() -> None:
    """Test PairwiseEvaluator initializes with default values."""
    evaluator = PairwiseEvaluator()

    # Default model: OpenAIChatModel with qwen3:8b on Ollama
    assert evaluator.model is not None
    assert isinstance(evaluator.model, OpenAIChatModel)
    assert evaluator.model.model_name == "qwen3:8b"
    # model_settings (TypedDict)
    assert evaluator.model_settings.get("temperature") == 0.0
    assert evaluator.model_settings.get("timeout") == 300
    # system_prompt
    assert evaluator.system_prompt is not None
    assert "response" in evaluator.system_prompt or "A" in evaluator.system_prompt
    assert "A" in evaluator.system_prompt and "B" in evaluator.system_prompt
    # agent
    assert evaluator.agent is not None
    # criterion
    assert "curiosity" in evaluator.criterion or "better" in evaluator.criterion.lower()


def test_pairwise_evaluator_init_custom() -> None:
    """Test PairwiseEvaluator initializes with custom values."""
    evaluator = PairwiseEvaluator(criterion="Custom criterion")

    assert evaluator.criterion == "Custom criterion"


def test_pairwise_evaluator_init_with_custom_model(mocker: MockerFixture) -> None:
    """Test PairwiseEvaluator uses custom model when provided."""
    custom_model = OpenAIChatModel(
        model_name="custom-model",
        provider=OpenAIProvider(base_url="http://localhost:11434/v1"),
    )
    evaluator = PairwiseEvaluator(model=custom_model)

    assert evaluator.model is custom_model
    assert isinstance(evaluator.model, OpenAIChatModel)
    assert evaluator.model.model_name == "custom-model"
    assert evaluator.agent.model is custom_model


def test_pairwise_evaluator_init_with_model_string(mocker: MockerFixture) -> None:
    """Test PairwiseEvaluator accepts model as string (e.g. for OpenAI default)."""
    mocker.patch("assays.evaluators.pairwise.Agent")  # Mock Agent to avoid actual initialization and API keys
    evaluator = PairwiseEvaluator(model="openai:gpt-4o-mini")

    assert evaluator.model == "openai:gpt-4o-mini"


@pytest.mark.asyncio
async def test_pairwise_evaluator_call_no_pairs(mocker: MockerFixture) -> None:
    """Test PairwiseEvaluator.__call__ handles empty baseline and novel lists."""
    evaluator = PairwiseEvaluator()
    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {
        "context": AssayContext(
            dataset=Dataset[dict[str, str], type[None], Any](cases=[]),
            path=Path("/tmp/test.json"),
            assay_mode="evaluate",
        )
    }
    mock_item.stash = {
        AGENT_RESPONSES_KEY: [],
        BASELINE_DATASET_KEY: Dataset[dict[str, str], type[None], Any](cases=[]),
    }

    mocker.patch("assays.evaluators.pairwise.logger")

    result = await evaluator(mock_item)

    assert type(result).__name__ == "Readout"
    assert result.passed is False
    assert result.details is not None
    assert result.details.get("test_cases_count") == 0
    assert result.details.get("wins_baseline") == []
    assert result.details.get("wins_novel") == []


@pytest.mark.asyncio
async def test_pairwise_evaluator_call_with_pairs(mocker: MockerFixture) -> None:
    """Test PairwiseEvaluator.__call__ runs pairwise comparison with mocked agent."""
    evaluator = PairwiseEvaluator(criterion="Test criterion")

    cases: list[Case[dict[str, str], type[None], Any]] = [
        Case(name="case_001", inputs={"query": "baseline query 1"}),
        Case(name="case_002", inputs={"query": "baseline query 2"}),
    ]
    dataset = Dataset[dict[str, str], type[None], Any](cases=cases)

    mock_response1 = mocker.MagicMock(spec=AgentRunResult)
    mock_response1.output = "novel output 1"
    mock_response2 = mocker.MagicMock(spec=AgentRunResult)
    mock_response2.output = "novel output 2"

    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {"context": AssayContext(dataset=dataset, path=Path("/tmp/test.json"), assay_mode="evaluate")}
    mock_item.stash = {
        AGENT_RESPONSES_KEY: [mock_response1, mock_response2],
        assays.plugin.BASELINE_DATASET_KEY: dataset,
    }

    mocker.patch("assays.evaluators.pairwise.logger")

    # Mock agent.run to return "B" (novel wins) for both comparisons
    mock_run_result = mocker.MagicMock()
    mock_run_result.output = "B"
    mocker.patch.object(evaluator.agent, "run", new_callable=AsyncMock, return_value=mock_run_result)

    result = await evaluator(mock_item)

    # Novel wins both -> passed=True
    assert type(result).__name__ == "Readout"
    assert result.passed is True
    assert result.details is not None
    assert result.details.get("test_cases_count") == 2
    assert result.details.get("wins_novel") == [True, True]
    assert result.details.get("wins_baseline") == [False, False]


@pytest.mark.asyncio
async def test_pairwise_evaluator_call_baseline_wins(mocker: MockerFixture) -> None:
    """Test PairwiseEvaluator.__call__ when baseline wins more comparisons."""
    evaluator = PairwiseEvaluator(criterion="Test criterion")

    cases: list[Case[dict[str, str], type[None], Any]] = [
        Case(name="case_001", inputs={"query": "baseline query"}),
    ]
    dataset = Dataset[dict[str, str], type[None], Any](cases=cases)

    mock_response = mocker.MagicMock(spec=AgentRunResult)
    mock_response.output = "novel output"

    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {"context": AssayContext(dataset=dataset, path=Path("/tmp/test.json"), assay_mode="evaluate")}
    mock_item.stash = {
        AGENT_RESPONSES_KEY: [mock_response],
        assays.plugin.BASELINE_DATASET_KEY: dataset,
    }

    mocker.patch("assays.evaluators.pairwise.logger")

    # Mock agent.run to return "A" (baseline wins)
    mock_run_result = mocker.MagicMock()
    mock_run_result.output = "A"
    mocker.patch.object(evaluator.agent, "run", new_callable=AsyncMock, return_value=mock_run_result)

    result = await evaluator(mock_item)

    assert type(result).__name__ == "Readout"
    assert result.passed is False
    assert result.details is not None
    assert result.details.get("wins_novel") == [False]
    assert result.details.get("wins_baseline") == [True]


@pytest.mark.asyncio
async def test_pairwise_evaluator_call_mismatch_raises(mocker: MockerFixture) -> None:
    """Test PairwiseEvaluator.__call__ raises when baseline and novel counts differ."""
    evaluator = PairwiseEvaluator()
    cases: list[Case[dict[str, str], type[None], Any]] = [
        Case(name="case_001", inputs={"query": "baseline query"}),
    ]
    dataset = Dataset[dict[str, str], type[None], Any](cases=cases)

    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {"context": AssayContext(dataset=dataset, path=Path("/tmp/test.json"), assay_mode="evaluate")}
    mock_item.stash = {
        AGENT_RESPONSES_KEY: [],  # No novel responses
        assays.plugin.BASELINE_DATASET_KEY: dataset,
    }

    mocker.patch("assays.evaluators.pairwise.logger")

    with pytest.raises(AssertionError, match="Mismatch in response counts"):
        await evaluator(mock_item)


@pytest.mark.asyncio
async def test_pairwise_evaluator_protocol_conformance() -> None:
    """Test PairwiseEvaluator conforms to Evaluator Protocol."""
    evaluator = PairwiseEvaluator()
    # Should be callable with Item and return Coroutine[Any, Any, Readout]
    assert callable(evaluator)
