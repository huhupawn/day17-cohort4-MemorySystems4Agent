from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    """Shared configuration for the memory systems lab.

    Attributes:
        base_dir: Repository root directory.
        data_dir: Directory containing benchmark datasets.
        state_dir: Directory containing agent state (e.g. User.md profiles).
        compact_threshold_tokens: Token threshold before triggering compaction.
        compact_keep_messages: Number of recent messages to keep uncompressed.
        model: Provider configuration for the primary chat agent.
        judge_model: Provider configuration for benchmark evaluations.
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load configuration from environment variables and defaults.

    Resolves paths, creates required state directories, and populates LabConfig.
    """
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    # Attempt to load .env if python-dotenv is available
    env_file = root / ".env"
    if env_file.exists():
        try:
            from dotenv import load_dotenv

            load_dotenv(env_file)
        except ImportError:
            pass

    data_dir = root / "data"
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "profiles").mkdir(parents=True, exist_ok=True)

    # Provider and model resolution
    provider = normalize_provider(os.getenv("LLM_PROVIDER", "openai"))
    model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
    api_key = (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("ANTHROPIC_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
        or os.getenv("CUSTOM_API_KEY")
    )
    base_url = os.getenv("OLLAMA_BASE_URL") or os.getenv("CUSTOM_BASE_URL")

    # Compact memory parameters
    compact_threshold_tokens = int(os.getenv("COMPACT_THRESHOLD_TOKENS", "400"))
    compact_keep_messages = int(os.getenv("COMPACT_KEEP_MESSAGES", "4"))

    model_config = ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
        api_key=api_key,
        base_url=base_url,
    )

    judge_provider = normalize_provider(os.getenv("JUDGE_PROVIDER", provider))
    judge_model_config = ProviderConfig(
        provider=judge_provider,
        model_name=os.getenv("JUDGE_MODEL", model_name),
        temperature=0.0,
        api_key=os.getenv("JUDGE_API_KEY", api_key),
        base_url=base_url,
    )

    return LabConfig(
        base_dir=root,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=compact_threshold_tokens,
        compact_keep_messages=compact_keep_messages,
        model=model_config,
        judge_model=judge_model_config,
    )
