---
name: Improve bradleyterry tests
overview: Refactor and improve the Bradley-Terry evaluator tests by extracting duplicated code, splitting bloated tests, separating unit from integration tests, and adding missing coverage.
todos:
  - id: extract-duplicates
    content: Extract score-assertion helper and deterministic_rng fixture, replace all 6 occurrences
    status: pending
  - id: split-tournament-test
    content: Break test_evaltournament into construction, get_player_by_idx, get_player_by_idx_invalid, and run_default_strategy tests; remove strategy overlap
    status: pending
  - id: separate-unit-integration
    content: Reorganize into TestBradleyTerryUnit and TestBradleyTerryIntegration classes; add integration marker; clean up unused mocker and brittle assertion
    status: pending
  - id: add-missing-coverage
    content: Add test_evalgame_b_wins, invalid player lookup, two-player edge case, stronger protocol conformance; record or skip missing cassettes
    status: pending
isProject: false
---

# Improve Bradley-Terry Evaluator Tests

## Stage 1: Extract duplicated code

- Create a helper function `assert_players_scored(players: list[EvalPlayer])` in `tests/evaluators/test_bradleyterry.py` (or a local conftest) that encapsulates the repeated 6-line assertion block:

```python
def assert_players_scored(players: list[EvalPlayer]) -> None:
    assert isinstance(players, list)
    for player in players:
        assert isinstance(player, EvalPlayer)
        assert player.score is not None
        assert isinstance(player.score, float)
```

- Replace all 6 occurrences in `test_evaltournament`, `test_random_sampling_strategy`, `test_round_robin_strategy`, and `test_adaptive_uncertainty_strategy` with a call to this helper.
- Move `random.seed(42)` into a fixture (e.g. `deterministic_rng`) and apply it to the VCR tests that need it.

## Stage 2: Break up `test_evaltournament` and remove overlap

Split the current `test_evaltournament` into focused tests:

- `test_evaltournament_construction` -- verifies player count, game criterion (pure unit, no VCR).
- `test_evaltournament_get_player_by_idx` -- happy path lookup (pure unit, no VCR).
- `test_evaltournament_get_player_by_idx_invalid` -- verifies `ValueError` for missing idx (pure unit, no VCR).
- `test_evaltournament_run_default_strategy` -- only tests that the tournament dispatches to the default strategy and returns scored players (VCR).

Remove the inline random-sampling-strategy execution from this test -- it is already covered by `test_random_sampling_strategy`.

## Stage 3: Separate unit and integration tests

Reorganize the file into two classes to make the distinction explicit:

```
class TestBradleyTerryUnit:
    # All tests that need NO VCR / LLM:
    - test_evalplayer
    - test_evaltournament_construction
    - test_evaltournament_get_player_by_idx
    - test_evaltournament_get_player_by_idx_invalid
    - test_evaluator_init_defaults
    - test_evaluator_init_custom
    - test_evaluator_init_with_custom_model
    - test_evaluator_init_with_model_string
    - test_evaluator_call_no_players
    - test_evaluator_call_with_players
    - test_evaluator_protocol_conformance

class TestBradleyTerryIntegration:
    # All tests that use @pytest.mark.vcr():
    - test_evalgame
    - test_evaltournament_run_default_strategy
    - test_random_sampling_strategy
    - test_round_robin_strategy
    - test_adaptive_uncertainty_strategy
```

Add a `pytestmark = pytest.mark.integration` to `TestBradleyTerryIntegration` so CI can gate these via `-m "not integration"`.

Clean up minor issues in unit tests:

- Remove unused `mocker` parameter from `test_evaluator_init_with_custom_model`.
- Replace the brittle `"Respond with exactly one letter: A or B" in evaluator.system_prompt` assertion with a check that the system prompt is a non-empty string (or check for a more stable substring like `"A or B"`).

## Stage 4: Add missing test coverage

Add these new tests:

- `**test_evalgame_b_wins**` (integration/VCR) -- Set up a game where B is obviously better and assert `result[0]` is the B-player's idx. This covers the else-branch in `EvalGame.run`.
- `**test_evaltournament_get_player_by_idx_invalid**` (unit) -- Call `get_player_by_idx(999)` and assert `pytest.raises(ValueError)`.
- `**test_random_sampling_strategy_two_players**` (unit, mocked or VCR) -- Edge case with only 2 players.
- `**test_evaluator_protocol_conformance**` (unit) -- Strengthen beyond `callable()`: verify the `__call__` signature accepts a pytest `Item` and that the return annotation is `Readout` (use `inspect.signature`).

Record any missing VCR cassettes for the integration tests that currently lack them (`test_evaltournament`, the three strategy tests). If recording is not yet feasible, mark them `@pytest.mark.skip(reason="VCR cassette not recorded")` so the test suite stays green.