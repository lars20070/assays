#!/usr/bin/env python3
from collections.abc import Generator

import pytest
from vcr.request import Request

from assays.config import config


@pytest.fixture(autouse=True)
def config_for_testing(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Override the config for unit testing."""
    monkeypatch.setattr(config, "logs2logfire", False)

    yield


@pytest.fixture
def vcr_config() -> dict[str, object]:
    """
    Configure VCR recordings for tests with @pytest.mark.vcr() decorator.

    When on bare metal, our host is localhost. When in a dev container, our host is host.docker.internal.
    uri_spoofing ensures that VCR cassettes are read or recorded as if the host was localhost.
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
        "ignore_localhost": False,  # We want to record local Ollama requests.
        "ignore_hosts": ["logfire-eu.pydantic.dev", "logfire.pydantic.dev"],  # We don't want to record requests to Logfire.
        "filter_headers": ["authorization", "x-api-key"],
        "decode_compressed_response": True,
        "before_record_request": uri_spoofing,
    }
