from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ProviderConfig:
    """Provider configuration shared by agents and benchmark judges.

    Required providers for this lab:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float = 0.7
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Map aliases and case variants to canonical provider names.

    Examples:
    - `anthorpic`, `claude` -> `anthropic`
    - `openai`, `gpt` -> `openai`
    - `google`, `gemini`, `google-genai` -> `gemini`
    - `ollama` -> `ollama`
    - `openrouter` -> `openrouter`
    - `custom`, `local`, `vllm` -> `custom`
    """
    cleaned = value.strip().lower()
    if cleaned in {"openai", "gpt"}:
        return "openai"
    elif cleaned in {"anthropic", "anthorpic", "claude"}:
        return "anthropic"
    elif cleaned in {"gemini", "google", "google-genai"}:
        return "gemini"
    elif cleaned in {"ollama"}:
        return "ollama"
    elif cleaned in {"openrouter"}:
        return "openrouter"
    elif cleaned in {"custom", "local", "vllm"}:
        return "custom"
    return cleaned


def build_chat_model(config: ProviderConfig) -> Any:
    """Instantiate the real chat model for the selected provider.

    Imports are performed lazily so that offline benchmark and test runs
    do not require all external LLM provider packages to be installed.
    """
    provider = normalize_provider(config.provider)

    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=config.model_name,
                temperature=config.temperature,
                api_key=config.api_key,
            )
        except ImportError:
            raise ImportError(
                "Package `langchain-openai` is required for provider 'openai'. "
                "Install it with `pip install langchain-openai`."
            )

    elif provider == "custom":
        try:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=config.model_name,
                temperature=config.temperature,
                api_key=config.api_key or "EMPTY",
                base_url=config.base_url,
            )
        except ImportError:
            raise ImportError(
                "Package `langchain-openai` is required for custom OpenAI-compatible endpoints."
            )

    elif provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=config.model_name,
                temperature=config.temperature,
                google_api_key=config.api_key,
            )
        except ImportError:
            raise ImportError(
                "Package `langchain-google-genai` is required for provider 'gemini'. "
                "Install it with `pip install langchain-google-genai`."
            )

    elif provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(
                model=config.model_name,
                temperature=config.temperature,
                api_key=config.api_key,
            )
        except ImportError:
            raise ImportError(
                "Package `langchain-anthropic` is required for provider 'anthropic'. "
                "Install it with `pip install langchain-anthropic`."
            )

    elif provider == "ollama":
        try:
            from langchain_ollama import ChatOllama

            return ChatOllama(
                model=config.model_name,
                temperature=config.temperature,
                base_url=config.base_url,
            )
        except ImportError:
            raise ImportError(
                "Package `langchain-ollama` is required for provider 'ollama'. "
                "Install it with `pip install langchain-ollama`."
            )

    elif provider == "openrouter":
        try:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=config.model_name,
                temperature=config.temperature,
                api_key=config.api_key,
                base_url=config.base_url or "https://openrouter.ai/api/v1",
            )
        except ImportError:
            raise ImportError(
                "Package `langchain-openai` is required for provider 'openrouter'."
            )

    else:
        raise ValueError(
            f"Unsupported provider: '{config.provider}'. "
            "Supported providers: openai, custom, gemini, anthropic, ollama, openrouter."
        )
