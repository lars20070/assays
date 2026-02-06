#!/usr/bin/env python3

from dotenv import load_dotenv

from assays.config import config
from assays.logger import logger

load_dotenv()


def test_config() -> None:
    """
    Test the Config class
    """
    logger.info("Testing the Config() class")

    # See values in config_for_testing() fixture
    assert config is not None
