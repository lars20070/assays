#!/usr/bin/env python3
"""Unit tests for the BradleyTerryEvaluator."""

from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings
from pydantic_evals import Case, Dataset
from pytest import Function

import assays.plugin
from assays.evaluators.bradleyterry import (
    EVALUATION_AGENT,
    BradleyTerryEvaluator,
    EvalGame,
    EvalPlayer,
    EvalTournament,
    adaptive_uncertainty_strategy,
    random_sampling_strategy,
    round_robin_strategy,
)
from assays.logger import logger
from assays.plugin import BASELINE_DATASET_KEY, AssayContext

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

MODEL_SETTINGS = ModelSettings(
    temperature=0.0,  # Model needs to be deterministic for VCR recording to work.
    timeout=300,
)


def test_evalplayer() -> None:
    """
    Test the EvalPlayer class.
    """
    logger.info("Testing EvalPlayer() class")

    player = EvalPlayer(
        idx=42,
        item="toasted rice & miso caramel ice cream",
    )
    assert player.idx == 42
    assert player.item == "toasted rice & miso caramel ice cream"


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_evalgame(ice_cream_players: list[EvalPlayer]) -> None:
    """
    Test the EvalGame class.
    """
    logger.info("Testing EvalGame() class")

    game = EvalGame(criterion="Which of the two ice cream flavours A or B is more creative?")
    assert game.criterion == "Which of the two ice cream flavours A or B is more creative?"

    result = await game.run(
        players=(ice_cream_players[0], ice_cream_players[4]),
        agent=EVALUATION_AGENT,
        model_settings=MODEL_SETTINGS,
    )
    logger.debug(f"Game result: {result}")

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert all(isinstance(r, int) for r in result)
    assert result[0] == 4  # Toasted rice & miso caramel ice cream flavour is more creative.


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_evaltournament(ice_cream_players: list[EvalPlayer], ice_cream_game: EvalGame) -> None:
    """
    Test the EvalTournament class.
    """
    random.seed(42)  # Make test deterministic for VCR recording

    logger.info("Testing EvalTournament() class")

    tournament = EvalTournament(players=ice_cream_players, game=ice_cream_game)

    assert len(tournament.players) == len(ice_cream_players)
    assert tournament.game.criterion == ice_cream_game.criterion

    # Test player retrieval
    player = tournament.get_player_by_idx(1)
    assert player is not None
    assert player.item == ice_cream_players[1].item

    # Test the default strategy
    players_with_scores = await tournament.run(
        agent=EVALUATION_AGENT,
        model_settings=MODEL_SETTINGS,
    )
    assert isinstance(players_with_scores, list)
    for player in players_with_scores:
        assert isinstance(player, EvalPlayer)
        assert hasattr(player, "score")
        assert isinstance(player.score, float)
        assert player.score is not None
        logger.debug(f"Player {player.idx} score: {player.score}")

    # Test the random sampling strategy
    players_with_scores = await tournament.run(
        agent=EVALUATION_AGENT,
        model_settings=MODEL_SETTINGS,
        strategy=random_sampling_strategy,
        fraction_of_games=0.3,
    )
    assert isinstance(players_with_scores, list)
    for player in players_with_scores:
        assert isinstance(player, EvalPlayer)
        assert hasattr(player, "score")
        assert isinstance(player.score, float)
        assert player.score is not None
        logger.debug(f"Player {player.idx} score: {player.score}")


@pytest.mark.vcr()
@pytest.mark.asyncio
@pytest.mark.parametrize("fraction_of_games", [None, 0.3, 42.0])  # Last value is non-sensical and will be ignored in the strategy.
async def test_random_sampling_strategy(ice_cream_players: list[EvalPlayer], ice_cream_game: EvalGame, fraction_of_games: float | None) -> None:
    """
    Test the random sampling tournament strategy.
    """
    random.seed(42)  # Make test deterministic for VCR recording

    logger.info("Testing random_sampling_strategy()")

    players_with_scores = await random_sampling_strategy(
        players=ice_cream_players,
        game=ice_cream_game,
        agent=EVALUATION_AGENT,
        model_settings=MODEL_SETTINGS,
        fraction_of_games=fraction_of_games,
    )
    assert isinstance(players_with_scores, list)
    for player in players_with_scores:
        assert isinstance(player, EvalPlayer)
        assert hasattr(player, "score")
        assert isinstance(player.score, float)
        assert player.score is not None
        logger.debug(f"Player {player.idx} score: {player.score}")


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_round_robin_strategy(ice_cream_players: list[EvalPlayer], ice_cream_game: EvalGame) -> None:
    """
    Test the round robin tournament strategy.
    """
    random.seed(42)  # Make test deterministic for VCR recording

    logger.info("Testing round_robin_strategy()")

    players_with_scores = await round_robin_strategy(
        players=ice_cream_players,
        game=ice_cream_game,
        agent=EVALUATION_AGENT,
        model_settings=MODEL_SETTINGS,
        number_of_rounds=1,
    )
    assert isinstance(players_with_scores, list)
    for player in players_with_scores:
        assert isinstance(player, EvalPlayer)
        assert hasattr(player, "score")
        assert isinstance(player.score, float)
        assert player.score is not None
        logger.debug(f"Player {player.idx} score: {player.score}")


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_adaptive_uncertainty_strategy(ice_cream_players: list[EvalPlayer], ice_cream_game: EvalGame) -> None:
    """
    Test the adaptive uncertainty tournament strategy.
    """
    random.seed(42)  # Make test deterministic for VCR recording

    logger.info("Testing adaptive_uncertainty_strategy()")

    players_with_scores = await adaptive_uncertainty_strategy(
        players=ice_cream_players,
        game=ice_cream_game,
        agent=EVALUATION_AGENT,
        model_settings=MODEL_SETTINGS,
        max_standard_deviation=1.0,
        alpha=0.01,
    )
    assert isinstance(players_with_scores, list)
    for player in players_with_scores:
        assert isinstance(player, EvalPlayer)
        assert hasattr(player, "score")
        assert isinstance(player.score, float)
        assert player.score is not None
        logger.debug(f"Player {player.idx} score: {player.score}")


def test_bradley_terry_evaluator_init_defaults() -> None:
    """Test BradleyTerryEvaluator initializes with default values."""
    evaluator = BradleyTerryEvaluator()

    # Default model: OpenAIChatModel with qwen3:8b on Ollama
    assert evaluator.model is not None
    assert isinstance(evaluator.model, OpenAIChatModel)
    assert evaluator.model.model_name == "qwen3:8b"
    # model_settings (TypedDict)
    assert evaluator.model_settings.get("temperature") == 0.0
    assert evaluator.model_settings.get("timeout") == 300
    # system_prompt
    assert evaluator.system_prompt is not None
    assert "response" in evaluator.system_prompt
    assert "A" in evaluator.system_prompt and "B" in evaluator.system_prompt
    # agent
    assert evaluator.agent is not None
    # criterion and max_standard_deviation
    assert evaluator.criterion == "Which of the two search queries shows more genuine curiosity and creativity, and is less formulaic?"
    assert evaluator.max_standard_deviation == 2.0


def test_bradley_terry_evaluator_init_custom() -> None:
    """Test BradleyTerryEvaluator initializes with custom values."""
    evaluator = BradleyTerryEvaluator(criterion="Custom criterion", max_standard_deviation=1.5)

    assert evaluator.criterion == "Custom criterion"
    assert evaluator.max_standard_deviation == 1.5


def test_bradley_terry_evaluator_init_with_custom_model(mocker: MockerFixture) -> None:
    """Test BradleyTerryEvaluator uses custom model when provided."""
    custom_model = OpenAIChatModel(
        model_name="custom-model",
        provider=OpenAIProvider(base_url="http://localhost:11434/v1"),
    )
    evaluator = BradleyTerryEvaluator(model=custom_model)

    assert evaluator.model is custom_model
    assert isinstance(evaluator.model, OpenAIChatModel)
    assert evaluator.model.model_name == "custom-model"
    assert evaluator.agent.model is custom_model


def test_bradley_terry_evaluator_init_with_model_string(mocker: MockerFixture) -> None:
    """Test BradleyTerryEvaluator accepts model as string (e.g. for OpenAI default)."""
    mocker.patch("assays.evaluators.bradleyterry.Agent")  # Mock Agent to avoid actual initialization and API keys
    evaluator = BradleyTerryEvaluator(model="openai:gpt-4o-mini")

    assert evaluator.model == "openai:gpt-4o-mini"


@pytest.mark.asyncio
async def test_bradley_terry_evaluator_call_no_players(mocker: MockerFixture) -> None:
    """Test BradleyTerryEvaluator.__call__ handles empty player list."""
    evaluator = BradleyTerryEvaluator()
    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {"context": None}
    mock_item.stash = {
        assays.plugin.AGENT_RESPONSES_KEY: [],
        BASELINE_DATASET_KEY: Dataset[dict[str, str], type[None], Any](cases=[]),
    }

    mocker.patch("assays.evaluators.bradleyterry.logger")

    result = await evaluator(mock_item)

    # Check result by attribute presence (module reload can cause isinstance to fail)
    assert type(result).__name__ == "Readout"
    assert result.passed is True
    assert result.details is not None
    assert result.details.get("message") == "No players to evaluate"


@pytest.mark.asyncio
async def test_bradley_terry_evaluator_call_with_players(mocker: MockerFixture) -> None:
    """Test BradleyTerryEvaluator.__call__ runs tournament with players."""
    # Mock ModelSettings before creating the evaluator (it's now called in __init__)
    mock_model_settings = mocker.patch("assays.evaluators.bradleyterry.ModelSettings")
    mock_model_settings.return_value = mocker.MagicMock()

    evaluator = BradleyTerryEvaluator(criterion="Test criterion", max_standard_deviation=1.5)

    # Verify ModelSettings was called with hard-coded values during __init__
    mock_model_settings.assert_called_once_with(temperature=0.0, timeout=300)

    cases: list[Case[dict[str, str], type[None], Any]] = [
        Case(name="case_001", inputs={"query": "baseline query"}),
    ]
    dataset = Dataset[dict[str, str], type[None], Any](cases=cases)

    mock_response = mocker.MagicMock(spec=AgentRunResult)
    mock_response.output = "novel output"

    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {"context": AssayContext(dataset=dataset, path=Path("/tmp/test.json"), assay_mode="evaluate")}
    mock_item.stash = {
        assays.plugin.AGENT_RESPONSES_KEY: [mock_response],
        BASELINE_DATASET_KEY: dataset,
    }

    mocker.patch("assays.evaluators.bradleyterry.logger")

    # Mock the tournament - use SimpleNamespace for players to allow formatting
    mock_tournament_class = mocker.patch("assays.evaluators.bradleyterry.EvalTournament")
    mock_tournament = mocker.MagicMock()
    mock_player = SimpleNamespace(idx=0, score=0.75, item="baseline query")
    mock_tournament.run = mocker.AsyncMock(return_value=[mock_player])
    mock_tournament.get_player_by_idx = MagicMock(return_value=mock_player)
    mock_tournament_class.return_value = mock_tournament

    result = await evaluator(mock_item)

    # Verify tournament was created with correct criterion
    mock_tournament_class.assert_called_once()
    call_kwargs = mock_tournament_class.call_args.kwargs
    assert call_kwargs["game"].criterion == "Test criterion"

    # Verify tournament.run was called with model_settings and max_standard_deviation
    mock_tournament.run.assert_called_once()
    run_kwargs = mock_tournament.run.call_args.kwargs
    assert "model_settings" in run_kwargs
    assert run_kwargs["max_standard_deviation"] == 1.5

    # Verify result (use type name check due to module reload)
    assert type(result).__name__ == "Readout"
    assert result.passed is False


@pytest.mark.asyncio
async def test_bradley_terry_evaluator_protocol_conformance() -> None:
    """Test BradleyTerryEvaluator conforms to Evaluator Protocol."""
    evaluator = BradleyTerryEvaluator()
    # Should be callable with Item and return Coroutine[Any, Any, Readout]
    assert callable(evaluator)
