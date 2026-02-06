#!/usr/bin/env python3

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Config(BaseSettings):
    """
    Configuration settings for the application.
    """

    logs2logfire: bool = Field(default=False, description="Post all logs to Logfire. If false, some logs are written to a local log file.")
    logfire_token: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


config = Config()
