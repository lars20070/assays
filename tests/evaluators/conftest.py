#!/usr/bin/env python3
"""Shared fixtures for evaluator tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import httpx
import pytest
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from assays.evaluators.bradleyterry import EVALUATION_INSTRUCTIONS, EvalGame, EvalPlayer, EvalTournament

if TYPE_CHECKING:
    from collections.abc import Callable
    from unittest.mock import AsyncMock, MagicMock

    from pytest_mock import MockerFixture

OLLAMA_BASE_URL = "http://localhost:11434"

OLLAMA_MODEL = "qwen2.5:14b"


def _ollama_is_running() -> bool:
    """
    Check whether the local Ollama server is reachable.
    """
    try:
        response = httpx.get(OLLAMA_BASE_URL, timeout=5)
        return response.status_code == 200
    except httpx.ConnectError:
        return False


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "ollama: tests requiring a local Ollama instance")


@pytest.fixture(autouse=True)
def skip_ollama_tests(request: pytest.FixtureRequest) -> None:
    """
    Skip tests marked with 'ollama' when running in CI or when Ollama is not reachable.
    """
    if request.node.get_closest_marker("ollama") and not _ollama_is_running():
        pytest.skip(f"Ollama server is not running at {OLLAMA_BASE_URL}. Please start it with `ollama serve` before running this test.")


@pytest.fixture
def model_settings() -> ModelSettings:
    """Provide deterministic model settings for VCR-compatible tests."""
    return ModelSettings(temperature=0.0, timeout=300)


@pytest.fixture
def ice_cream_players() -> list[EvalPlayer]:
    """Provide a list of EvalPlayer instances with ice cream flavours."""
    return [
        EvalPlayer(idx=0, item="vanilla"),
        EvalPlayer(idx=1, item="chocolate"),
        EvalPlayer(idx=2, item="strawberry"),
        EvalPlayer(idx=3, item="peach"),
        EvalPlayer(idx=4, item="toasted rice & miso caramel ice cream"),
    ]


@pytest.fixture
def ice_cream_game() -> EvalGame:
    """Provide an EvalGame instance for ice cream flavour comparison."""
    return EvalGame(criterion="Which of the two ice cream flavours A or B is more creative?")


@pytest.fixture
def ice_cream_tournament(ice_cream_players: list[EvalPlayer], ice_cream_game: EvalGame) -> EvalTournament:
    """Provide an EvalTournament with ice cream players and game."""
    return EvalTournament(players=ice_cream_players, game=ice_cream_game)


@pytest.fixture
def mock_pydantic_agent(mocker: MockerFixture) -> MagicMock:
    """Provide a MagicMock spec'd to pydantic_ai.Agent for tournament tests."""
    return mocker.MagicMock(spec=Agent)


@pytest.fixture
def evaluation_model() -> OpenAIChatModel:
    """Provide the OpenAI-compatible model pointing at local Ollama."""
    return OpenAIChatModel(
        model_name=OLLAMA_MODEL,
        provider=OpenAIProvider(base_url=f"{OLLAMA_BASE_URL}/v1"),
    )


@pytest.fixture
def evaluation_agent(evaluation_model: OpenAIChatModel) -> Agent[None, Any]:
    """Provide the evaluation agent used by integration tests."""
    return Agent(
        model=evaluation_model,
        output_type=Literal["A", "B"],
        system_prompt=EVALUATION_INSTRUCTIONS,
        retries=5,
        instrument=True,
    )


@pytest.fixture
def make_mock_agent(mocker: MockerFixture) -> Callable[..., AsyncMock]:
    """Factory fixture that returns a mock Agent configured to output a given value."""

    def _make(output: str = "A") -> AsyncMock:
        mock_result = mocker.MagicMock()
        mock_result.output = output
        mock_agent = mocker.AsyncMock(spec=Agent)
        mock_agent.run = mocker.AsyncMock(return_value=mock_result)
        mock_agent.__aenter__ = mocker.AsyncMock(return_value=mock_agent)
        mock_agent.__aexit__ = mocker.AsyncMock(return_value=False)
        return mock_agent

    return _make
