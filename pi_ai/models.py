"""
pi-ai model definitions.

Model configurations for various LLM providers including Bailian and OpenAI.
"""

from typing import Literal

from pi_ai.types import Model, Usage


# -----------------------------------------------------------------------------
# Bailian Models (Alibaba/Qwen)
# -----------------------------------------------------------------------------

BAILIAN_MODELS: dict[str, Model] = {
    "qwen-turbo": Model(
        id="qwen-turbo",
        name="Qwen Turbo",
        api="bailian",
        provider="bailian",
        base_url="https://dashscope.aliyuncs.com/api/v1",
        reasoning=False,
        input=["text", "image"],
        cost=Usage.Cost(input=0.002, output=0.006, total=0.008),
        context_window=8192,
        max_tokens=2048,
    ),
    "qwen-plus": Model(
        id="qwen-plus",
        name="Qwen Plus",
        api="bailian",
        provider="bailian",
        base_url="https://dashscope.aliyuncs.com/api/v1",
        reasoning=False,
        input=["text", "image"],
        cost=Usage.Cost(input=0.004, output=0.012, total=0.016),
        context_window=32768,
        max_tokens=8192,
    ),
    "qwen-max": Model(
        id="qwen-max",
        name="Qwen Max",
        api="bailian",
        provider="bailian",
        base_url="https://dashscope.aliyuncs.com/api/v1",
        reasoning=True,
        input=["text", "image"],
        cost=Usage.Cost(input=0.02, output=0.06, total=0.08),
        context_window=32768,
        max_tokens=8192,
    ),
    "qwen-max-longcontext": Model(
        id="qwen-max-longcontext",
        name="Qwen Max Long Context",
        api="bailian",
        provider="bailian",
        base_url="https://dashscope.aliyuncs.com/api/v1",
        reasoning=True,
        input=["text"],
        cost=Usage.Cost(input=0.02, output=0.06, total=0.08),
        context_window=100000,
        max_tokens=8192,
    ),
}


# -----------------------------------------------------------------------------
# OpenAI Models
# -----------------------------------------------------------------------------

OPENAI_MODELS: dict[str, Model] = {
    "gpt-4o": Model(
        id="gpt-4o",
        name="GPT-4o",
        api="openai",
        provider="openai",
        base_url="https://api.openai.com/v1",
        reasoning=False,
        input=["text", "image"],
        cost=Usage.Cost(input=0.005, output=0.015, total=0.02),
        context_window=128000,
        max_tokens=4096,
    ),
    "gpt-4o-mini": Model(
        id="gpt-4o-mini",
        name="GPT-4o Mini",
        api="openai",
        provider="openai",
        base_url="https://api.openai.com/v1",
        reasoning=False,
        input=["text", "image"],
        cost=Usage.Cost(input=0.00015, output=0.0006, total=0.00075),
        context_window=128000,
        max_tokens=16384,
    ),
    "gpt-4-turbo": Model(
        id="gpt-4-turbo",
        name="GPT-4 Turbo",
        api="openai",
        provider="openai",
        base_url="https://api.openai.com/v1",
        reasoning=False,
        input=["text", "image"],
        cost=Usage.Cost(input=0.01, output=0.03, total=0.04),
        context_window=128000,
        max_tokens=4096,
    ),
    "o1-preview": Model(
        id="o1-preview",
        name="O1 Preview",
        api="openai",
        provider="openai",
        base_url="https://api.openai.com/v1",
        reasoning=True,
        input=["text", "image"],
        cost=Usage.Cost(input=0.015, output=0.06, total=0.075),
        context_window=128000,
        max_tokens=32768,
    ),
    "o1-mini": Model(
        id="o1-mini",
        name="O1 Mini",
        api="openai",
        provider="openai",
        base_url="https://api.openai.com/v1",
        reasoning=True,
        input=["text"],
        cost=Usage.Cost(input=0.003, output=0.012, total=0.015),
        context_window=128000,
        max_tokens=65536,
    ),
}


# -----------------------------------------------------------------------------
# Combined Model Registry
# -----------------------------------------------------------------------------

ALL_MODELS: dict[str, Model] = {
    **BAILIAN_MODELS,
    **OPENAI_MODELS,
}


def get_model(model_id: str) -> Model:
    """
    Get a model configuration by its ID.

    Args:
        model_id: The model identifier (e.g., "gpt-4o", "qwen-max").

    Returns:
        The Model configuration.

    Raises:
        KeyError: If the model is not found.
    """
    if model_id not in ALL_MODELS:
        raise KeyError(f"Model '{model_id}' not found. Available models: {list(ALL_MODELS.keys())}")
    return ALL_MODELS[model_id]


def get_models(provider: str | None = None) -> list[Model]:
    """
    Get all available models, optionally filtered by provider.

    Args:
        provider: Optional provider filter (e.g., "openai", "bailian").

    Returns:
        List of Model configurations.
    """
    if provider is None:
        return list(ALL_MODELS.values())

    return [model for model in ALL_MODELS.values() if model.provider == provider]


def get_providers() -> list[str]:
    """
    Get all available provider names.

    Returns:
        List of provider names.
    """
    providers = set(model.provider for model in ALL_MODELS.values())
    return list(providers)