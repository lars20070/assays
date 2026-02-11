#!/usr/bin/env python3
"""Unit tests for the BradleyTerryEvaluator."""

from __future__ import annotations

import inspect
import random
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal
from unittest.mock import MagicMock

import pytest
from pydantic_ai import Agent
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings
from pydantic_evals import Case, Dataset
from pytest import Function

import assays.plugin
from assays.evaluators.bradleyterry import (
    EVALUATION_INSTRUCTIONS,
    BradleyTerryEvaluator,
    EvalGame,
    EvalPlayer,
    EvalTournament,
    adaptive_uncertainty_strategy,
    random_sampling_strategy,
    round_robin_strategy,
)
from assays.logger import logger
from assays.plugin import BASELINE_DATASET_KEY, AssayContext, Readout

if TYPE_CHECKING:
    from collections.abc import Callable
    from unittest.mock import AsyncMock

    from pytest_mock import MockerFixture

MODEL_SETTINGS = ModelSettings(
    temperature=0.0,  # Model needs to be deterministic for VCR recording to work.
    timeout=300,
)

MODEL = OpenAIChatModel(
    model_name="qwen2.5:14b",
    provider=OpenAIProvider(base_url="http://localhost:11434/v1"),  # Local Ollama server
)


EVALUATION_AGENT = Agent(
    model=MODEL,
    output_type=Literal["A", "B"],
    system_prompt=EVALUATION_INSTRUCTIONS,
    retries=5,
    instrument=True,
)


# ---------------------------------------------------------------------------
# Unit tests for EvalPlayer, EvalGame, EvalTournament and BradleyTerryEvaluator
# ---------------------------------------------------------------------------


class TestEvalPlayer:
    """Unit tests for EvalPlayer."""

    def test_create_with_required_fields(self) -> None:
        player = EvalPlayer(idx=0, item="vanilla")
        assert player.idx == 0
        assert player.item == "vanilla"
        assert player.score is None

    def test_create_with_score(self) -> None:
        player = EvalPlayer(idx=1, item="chocolate", score=1.5)
        assert player.score == 1.5

    def test_score_is_mutable(self) -> None:
        player = EvalPlayer(idx=0, item="vanilla")
        player.score = 3.14
        assert player.score == 3.14

    def test_negative_idx(self) -> None:
        """Negative idx is allowed by the model; it is just an identifier."""
        player = EvalPlayer(idx=-1, item="pistachio")
        assert player.idx == -1

    def test_empty_item(self) -> None:
        player = EvalPlayer(idx=0, item="")
        assert player.item == ""


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


class TestEvalGame:
    """Unit tests for EvalGame (agent interaction mocked)."""

    def test_criterion_stored(self) -> None:
        game = EvalGame(criterion="Which answer is better?")
        assert game.criterion == "Which answer is better?"

    @pytest.mark.asyncio
    async def test_run_returns_winner_a(self, make_mock_agent: Callable[..., AsyncMock]) -> None:
        """When the agent picks A, the first player wins."""
        game = EvalGame(criterion="Which is better?")
        p1 = EvalPlayer(idx=0, item="alpha")
        p2 = EvalPlayer(idx=1, item="beta")
        mock_agent = make_mock_agent("A")

        result = await game.run(players=(p1, p2), agent=mock_agent, model_settings=MODEL_SETTINGS)

        assert result == (0, 1)
        mock_agent.run.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_run_returns_winner_b(self, make_mock_agent: Callable[..., AsyncMock]) -> None:
        """When the agent picks B, the second player wins."""
        game = EvalGame(criterion="Which is better?")
        p1 = EvalPlayer(idx=0, item="alpha")
        p2 = EvalPlayer(idx=1, item="beta")
        mock_agent = make_mock_agent("B")

        result = await game.run(players=(p1, p2), agent=mock_agent, model_settings=MODEL_SETTINGS)

        assert result == (1, 0)

    @pytest.mark.asyncio
    async def test_run_prompt_contains_player_items(self, make_mock_agent: Callable[..., AsyncMock]) -> None:
        """The prompt passed to the agent must include both player items and the criterion."""
        game = EvalGame(criterion="creativity")
        p1 = EvalPlayer(idx=0, item="vanilla")
        p2 = EvalPlayer(idx=1, item="miso caramel")
        mock_agent = make_mock_agent("A")

        await game.run(players=(p1, p2), agent=mock_agent, model_settings=MODEL_SETTINGS)

        call_kwargs = mock_agent.run.call_args.kwargs
        prompt = call_kwargs["user_prompt"]
        assert "vanilla" in prompt
        assert "miso caramel" in prompt
        assert "creativity" in prompt

    @pytest.mark.asyncio
    async def test_run_preserves_player_idx_order(self, make_mock_agent: Callable[..., AsyncMock]) -> None:
        """Result tuple uses the actual player idx values, not positional indices."""
        game = EvalGame(criterion="test")
        p1 = EvalPlayer(idx=7, item="x")
        p2 = EvalPlayer(idx=3, item="y")
        mock_agent = make_mock_agent("A")

        result = await game.run(players=(p1, p2), agent=mock_agent, model_settings=MODEL_SETTINGS)

        assert result == (7, 3)


class TestEvalTournament:
    """Unit tests for EvalTournament (strategy mocked)."""

    def test_construction(self, ice_cream_players: list[EvalPlayer], ice_cream_game: EvalGame) -> None:
        tournament = EvalTournament(players=ice_cream_players, game=ice_cream_game)
        assert len(tournament.players) == 5
        assert tournament.game.criterion == ice_cream_game.criterion

    @pytest.mark.parametrize(
        ("idx", "expected_item"),
        [(0, "vanilla"), (3, "peach"), (4, "toasted rice & miso caramel ice cream")],
    )
    def test_get_player_by_idx(self, ice_cream_players: list[EvalPlayer], ice_cream_game: EvalGame, idx: int, expected_item: str) -> None:
        tournament = EvalTournament(players=ice_cream_players, game=ice_cream_game)
        player = tournament.get_player_by_idx(idx)
        assert player.idx == idx
        assert player.item == expected_item

    def test_get_player_by_idx_not_found(self, ice_cream_players: list[EvalPlayer], ice_cream_game: EvalGame) -> None:
        tournament = EvalTournament(players=ice_cream_players, game=ice_cream_game)
        with pytest.raises(ValueError, match="Player with unique identifier 99 not found"):
            tournament.get_player_by_idx(99)

    @pytest.mark.asyncio
    async def test_run_uses_default_strategy(self, mocker: MockerFixture, ice_cream_players: list[EvalPlayer], ice_cream_game: EvalGame) -> None:
        """When no strategy is passed, adaptive_uncertainty_strategy is used."""
        scored = [EvalPlayer(idx=p.idx, item=p.item, score=float(i)) for i, p in enumerate(ice_cream_players)]
        mock_strategy = mocker.AsyncMock(return_value=scored)
        mocker.patch("assays.evaluators.bradleyterry.adaptive_uncertainty_strategy", mock_strategy)

        tournament = EvalTournament(players=ice_cream_players, game=ice_cream_game)
        mock_agent = mocker.MagicMock(spec=Agent)

        result = await tournament.run(agent=mock_agent, model_settings=MODEL_SETTINGS)

        mock_strategy.assert_awaited_once()
        assert len(result) == 5
        assert all(p.score is not None for p in result)

    @pytest.mark.asyncio
    async def test_run_uses_custom_strategy(self, mocker: MockerFixture, ice_cream_players: list[EvalPlayer], ice_cream_game: EvalGame) -> None:
        """A custom strategy function receives the correct arguments."""
        scored = [EvalPlayer(idx=p.idx, item=p.item, score=float(i)) for i, p in enumerate(ice_cream_players)]

        async def fake_strategy(*_args: object, **_kwargs: object) -> list[EvalPlayer]:
            return scored

        spy = mocker.AsyncMock(side_effect=fake_strategy)
        spy.__name__ = "fake_strategy"

        tournament = EvalTournament(players=ice_cream_players, game=ice_cream_game)
        mock_agent = mocker.MagicMock(spec=Agent)

        result = await tournament.run(agent=mock_agent, model_settings=MODEL_SETTINGS, strategy=spy)

        spy.assert_awaited_once()
        call_args = spy.call_args
        assert call_args[0][0] == ice_cream_players  # players
        assert call_args[0][1] is ice_cream_game  # game
        assert call_args[0][2] is mock_agent  # agent
        assert result == scored

    @pytest.mark.asyncio
    async def test_run_forwards_strategy_kwargs(self, mocker: MockerFixture, ice_cream_players: list[EvalPlayer], ice_cream_game: EvalGame) -> None:
        """Extra kwargs are forwarded to the strategy function."""
        scored = [EvalPlayer(idx=p.idx, item=p.item, score=0.0) for p in ice_cream_players]

        received_kwargs: dict[str, object] = {}

        async def capture_strategy(*_args: object, **kwargs: object) -> list[EvalPlayer]:
            received_kwargs.update(kwargs)
            return scored

        capture_strategy.__name__ = "capture_strategy"

        tournament = EvalTournament(players=ice_cream_players, game=ice_cream_game)
        mock_agent = mocker.MagicMock(spec=Agent)

        await tournament.run(
            agent=mock_agent,
            model_settings=MODEL_SETTINGS,
            strategy=capture_strategy,
            max_standard_deviation=1.5,
            alpha=0.2,
        )

        assert received_kwargs == {"max_standard_deviation": 1.5, "alpha": 0.2}

    @pytest.mark.asyncio
    async def test_run_updates_players_attribute(self, mocker: MockerFixture, ice_cream_players: list[EvalPlayer], ice_cream_game: EvalGame) -> None:
        """After run(), tournament.players is updated with the scored players."""
        scored = [EvalPlayer(idx=p.idx, item=p.item, score=float(i) * 0.5) for i, p in enumerate(ice_cream_players)]
        mock_strategy = mocker.AsyncMock(return_value=scored)
        mock_strategy.__name__ = "mock_strategy"

        tournament = EvalTournament(players=ice_cream_players, game=ice_cream_game)
        mock_agent = mocker.MagicMock(spec=Agent)

        await tournament.run(agent=mock_agent, model_settings=MODEL_SETTINGS, strategy=mock_strategy)

        assert tournament.players is scored


class TestBradleyTerryEvaluator:
    """Unit tests for BradleyTerryEvaluator."""

    def test_init_defaults(self) -> None:
        """Test BradleyTerryEvaluator initializes with default values."""
        evaluator = BradleyTerryEvaluator()

        # Default model: OpenAIChatModel with qwen2.5:14b on Ollama
        assert evaluator.model is not None
        assert isinstance(evaluator.model, OpenAIChatModel)
        assert evaluator.model.model_name == "qwen2.5:14b"
        # model_settings (TypedDict)
        assert evaluator.model_settings.get("temperature") == 0.0
        assert evaluator.model_settings.get("timeout") == 300
        # system_prompt
        assert evaluator.system_prompt is not None
        assert evaluator.system_prompt == EVALUATION_INSTRUCTIONS
        # agent
        assert evaluator.agent is not None
        # criterion and max_standard_deviation
        assert evaluator.criterion == "Which of the two agent responses is better?"
        assert evaluator.max_standard_deviation == 2.0

    def test_init_custom(self) -> None:
        """Test BradleyTerryEvaluator initializes with custom values."""
        evaluator = BradleyTerryEvaluator(criterion="Custom criterion", max_standard_deviation=1.5)

        assert evaluator.criterion == "Custom criterion"
        assert evaluator.max_standard_deviation == 1.5

    def test_init_with_custom_model(self) -> None:
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

    def test_init_with_model_string(self, mocker: MockerFixture) -> None:
        """Test BradleyTerryEvaluator accepts model as string (e.g. for OpenAI default)."""
        mocker.patch("assays.evaluators.bradleyterry.Agent")  # Mock Agent to avoid actual initialization and API keys
        evaluator = BradleyTerryEvaluator(model="openai:gpt-4o-mini")

        assert evaluator.model == "openai:gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_call_no_players(self, mocker: MockerFixture) -> None:
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

        assert isinstance(result, Readout)
        assert result.passed is True
        assert result.details is not None
        assert result.details.get("message") == "No players to evaluate"

    @pytest.mark.asyncio
    async def test_call_with_players(self, mocker: MockerFixture) -> None:
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

        assert isinstance(result, Readout)
        # passed is False because the mock uses a single player (score=0.75) for both
        # novel and baseline; novel must strictly exceed baseline to pass.
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_protocol_conformance(self) -> None:
        """Test BradleyTerryEvaluator conforms to Evaluator Protocol."""
        evaluator = BradleyTerryEvaluator()
        assert callable(evaluator)
        assert inspect.iscoroutinefunction(evaluator.__call__)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.skip()
@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_evalgame_integration(ice_cream_players: list[EvalPlayer]) -> None:
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


@pytest.mark.skip()
@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_evaltournament_integration(ice_cream_players: list[EvalPlayer], ice_cream_game: EvalGame) -> None:
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


@pytest.mark.skip()
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


@pytest.mark.skip()
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


@pytest.mark.skip()
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
