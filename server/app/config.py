"""Centralized configuration loaded from environment variables / .env."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


@dataclass(frozen=True)
class Settings:
    model_id: str
    model_path: str | None
    host: str
    port: int
    load_in_4bit: bool
    max_new_tokens: int
    temperature: float
    top_p: float

    @property
    def model_source(self) -> str:
        """Path takes precedence over hub id when both are set."""
        return self.model_path if self.model_path else self.model_id


def load_settings() -> Settings:
    model_path = os.getenv("MODEL_PATH", "").strip() or None
    return Settings(
        model_id=os.getenv("MODEL_ID", "OMCHOKSI108/VibeThinker-3B").strip(),
        model_path=model_path,
        host=os.getenv("HOST", "127.0.0.1").strip(),
        port=_int_env("PORT", 8000),
        load_in_4bit=_bool_env("LOAD_IN_4BIT", False),
        max_new_tokens=_int_env("MAX_NEW_TOKENS", 2048),
        temperature=_float_env("TEMPERATURE", 0.6),
        top_p=_float_env("TOP_P", 0.95),
    )


settings = load_settings()
