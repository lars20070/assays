#!/usr/bin/env python3
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from assays.evals import (
    EvalGame,
    EvalPlayer,
    EvalTournament,
    GameResult,
    adaptive_uncertainty_strategy,
)
from assays.logger import logger

if TYPE_CHECKING:
    from pytest import Item

    from assays.plugin import Readout


class BradleyTerryEvaluator:
    """Evaluates test outputs using Bradley-Terry tournament scoring.

    Configuration is set at instantiation; __call__ runs the evaluation.
    """

    def __init__(
        self,
        model: str | OpenAIChatModel | None = None,
        criterion: str = "Which of the two search queries shows more genuine curiosity and creativity, and is less formulaic?",
        max_standard_deviation: float = 2.0,
    ) -> None:
        """Configure the evaluator.

        Args:
            model: The language model or model string to use for evaluation. Defaults to qwen3:8b on Ollama.
            criterion: The evaluation criterion for pairwise comparison.
            max_standard_deviation: Convergence threshold for adaptive strategy.
        """
        if model is None:
            self.model = OpenAIChatModel(
                model_name="qwen3:8b",
                provider=OpenAIProvider(base_url="http://localhost:11434/v1"),  # Local Ollama server
            )
        else:
            self.model = model
        self.model_settings = ModelSettings(
            temperature=0.0,
            timeout=300,
        )
        self.system_prompt = """
            You are presented with a question and two possible answers A and B. Evaluate carefully whether answer A or answer B is the better reply.
            You have got only these two options. Your evaluations contribute to Bradley-Terry scores across multiple items. Consistency and
            objectivity are critical for reliable rankings. Each comparison should be independent but internally consistent.

            <EXAMPLES>
            Example 1:
            <QUESTION> Which of the two ice cream flavours below is more creative? </QUESTION>
            <A> Vanilla </A>
            <B> Pickled Citrus Ribbon </B>
            Expected output:
            {
                "response": "B",
            }

            Example 2:
            <QUESTION> Which search query shows more genuine curiosity? </QUESTION>
            <A> effect of ocean acidification feedback loops on Arctic methane release </A>
            <B> climate change effects </B>
            Expected output:
            {
                "response": "A",
            }

            Example 3:
            <QUESTION> Which reply is more insulting? </QUESTION>
            <A> Your argument lacks logical coherence and fails to address the core issue at hand. </A>
            <B> That's an interesting perspective, though I see it differently. </B>
            Expected output:
            {
                "response": "A",
            }
            </EXAMPLES>

            <REQUIREMENTS>
            1. Consider the question carefully. What aspects are important for the answer?
            2. Think about answer A. Is it a good answer to the question? Why (not)?
            3. Think about answer B. Is it a good answer to the question? Why (not)?
            4. Make a decision based on your analysis.
            </REQUIREMENTS>

            <OUTPUT_FORMAT>
            You must respond with valid JSON containing exactly one field called "response" with value "A" or "B":

            {
                "response": "A",
            }
            or
            {
                "response": "B",
            }

            Do NOT include explanations, reasoning, or any other fields.
            </OUTPUT_FORMAT>
            """
        self.agent = Agent(
            model=self.model,
            output_type=GameResult,
            system_prompt=self.system_prompt,
            retries=5,
            instrument=True,
        )
        self.criterion = criterion
        self.max_standard_deviation = max_standard_deviation

    async def __call__(self, item: Item) -> Readout:
        """Run Bradley-Terry tournament on baseline and novel responses.

        Args:
            item: The pytest test item with assay context and captured responses.

        Returns:
            Readout with passed status and details.
        """
        from assays.plugin import AGENT_RESPONSES_KEY, BASELINE_DATASET_KEY, Readout  # noqa: PLC0415

        logger.info("Running Bradley-Terry evaluation on captured agent responses")

        # Prepare the list of all players, baseline and novel
        players: list[EvalPlayer] = []

        # 1. Baseline players from previously serialized assay dataset
        baseline_dataset = item.stash.get(BASELINE_DATASET_KEY, None)
        baseline_case_count = 0
        if baseline_dataset is not None:
            for idx, case in enumerate(baseline_dataset.cases):
                players.append(EvalPlayer(idx=idx, item=str(case.expected_output)))
            baseline_case_count = len(baseline_dataset.cases)

        # 2. Novel players from current test run
        responses = item.stash.get(AGENT_RESPONSES_KEY, [])
        for idx, response in enumerate(responses):
            if response.output is None:
                logger.warning(f"Response #{idx} has None output.")
                continue
            players.append(EvalPlayer(idx=idx + baseline_case_count, item=response.output))

        # Log all players before tournament
        for player in players:
            logger.debug(f"Player #{player.idx} item: {repr(player.item)[:100]}")

        if not players:
            logger.debug("No players to evaluate in tournament.")
            return Readout(passed=True, details={"message": "No players to evaluate"})

        # Run the Bradley-Terry tournament to score both baseline and novel queries
        game = EvalGame(criterion=self.criterion)
        tournament = EvalTournament(players=players, game=game)
        players_scored = await tournament.run(
            agent=self.agent,
            model_settings=self.model_settings,
            strategy=adaptive_uncertainty_strategy,
            max_standard_deviation=self.max_standard_deviation,
        )

        # Players sorted by score
        players_sorted = sorted(players_scored, key=lambda p: p.score if p.score is not None else float("-inf"))
        for player in players_sorted:
            logger.debug(f"Player {player.idx:4d}   score: {player.score:7.4f}   query: {player.item}")

        # Average score for both baseline and novel queries
        scores_baseline = [tournament.get_player_by_idx(idx=i).score or 0.0 for i in range(len(players) // 2)]
        scores_novel = [tournament.get_player_by_idx(idx=i + len(players) // 2).score or 0.0 for i in range(len(players) // 2)]

        if scores_baseline and scores_novel:
            avg_baseline = np.mean(scores_baseline)
            avg_novel = np.mean(scores_novel)
            passed = bool(avg_novel > avg_baseline)
            logger.debug(f"Average score for baseline queries (Players 0 to 9): {avg_baseline:7.4f}")
            logger.debug(f"Average score for novel queries  (Players 10 to 19): {avg_novel:7.4f}")
        else:
            passed = False

        return Readout(
            passed=passed,
            details={
                "test_cases_count": len(players) // 2,  # test cases (baseline responses) + test cases (novel responses) = total players
                "scores_baseline": scores_baseline,
                "scores_novel": scores_novel,
            },
        )
