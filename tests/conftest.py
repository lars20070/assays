#!/usr/bin/env python3
from collections.abc import Generator

import pytest
from vcr.request import Request

from assays.config import config
from assays.evals import EvalGame, EvalPlayer


@pytest.fixture(autouse=True)
def config_for_testing(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """
    Override the config for unit testing.
    """
    monkeypatch.setattr(config, "logs2logfire", False)

    yield


@pytest.fixture
def ice_cream_players() -> list[EvalPlayer]:
    """
    Provide a list of EvalPlayer instances with ice cream flavours.
    """
    return [
        EvalPlayer(idx=0, item="vanilla"),
        EvalPlayer(idx=1, item="chocolate"),
        EvalPlayer(idx=2, item="strawberry"),
        EvalPlayer(idx=3, item="peach"),
        EvalPlayer(idx=4, item="toasted rice & miso caramel ice cream"),
    ]


@pytest.fixture
def ice_cream_game() -> EvalGame:
    """
    Provide an EvalGame instance for ice cream flavour comparison.
    """
    return EvalGame(criterion="Which of the two ice cream flavours A or B is more creative?")


@pytest.fixture
def vcr_config() -> dict[str, object]:
    """
    Configure VCR recordings for tests with @pytest.mark.vcr() decorator.

    When on bare metal, our host is `localhost`. When in a dev container, our host is `host.docker.internal`.
    `uri_spoofing` ensures that VCR cassettes are read or recorded as if the host was `localhost`.
    See ./tests/cassettes/*/*.yaml.

    Returns:
        dict[str, object]: VCR configuration settings.
    """

    def uri_spoofing(request: Request) -> Request:
        if request.uri and "host.docker.internal" in request.uri:
            # Replace host.docker.internal with localhost.
            request.uri = request.uri.replace("host.docker.internal", "localhost")
        return request

    return {
        "ignore_localhost": False,  # We want to record local SearXNG and Ollama requests.
        "filter_headers": ["authorization", "x-api-key"],
        "decode_compressed_response": True,
        "before_record_request": uri_spoofing,
    }
