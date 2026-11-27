from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LORE_DIR = PROJECT_ROOT / "data" / "lore"
QUIZ_BANK = PROJECT_ROOT / "data" / "quiz_bank.json"
INDEX_DIR = PROJECT_ROOT / "index"
CACHE_DIR = PROJECT_ROOT / ".cache"
EVAL_DIR = PROJECT_ROOT / "eval"
EVAL_DATASET = EVAL_DIR / "dataset.jsonl"
EVAL_RESULTS = EVAL_DIR / "results"
USER_SUBMITTED = EVAL_DIR / "user_submitted.jsonl"
HIGHSCORES = PROJECT_ROOT / "highscores.json"

Provider = Literal["ollama", "groq"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    llm_provider: Provider = "ollama"
    judge_provider: Provider | None = None
    checker_provider: Provider | None = None  # model for the checking agent; defaults to llm_provider

    ollama_model: str = "qwen2.5:7b"
    ollama_base_url: str = "http://localhost:11434"

    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    top_k: int = 4
    hybrid: bool = True
    chunk_size: int = 700
    chunk_overlap: int = 100


@lru_cache
def get_settings() -> Settings:
    return Settings()
