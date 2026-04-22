# tests/test_models.py
"""Tests for pi_ai models module."""

import pytest

from pi_ai.models import (
    BAILIAN_MODELS,
    OPENAI_MODELS,
    ALL_MODELS,
    get_model,
    get_models,
    get_providers,
)
from pi_ai.types import Model


class TestBailianModels:
    """Tests for BAILIAN_MODELS dict."""

    def test_bailian_models_contains_qwen_turbo(self):
        """BAILIAN_MODELS contains qwen-turbo."""
        assert "qwen-turbo" in BAILIAN_MODELS
        model = BAILIAN_MODELS["qwen-turbo"]
        assert model.api == "bailian"
        assert model.provider == "bailian"

    def test_bailian_models_contains_qwen_plus(self):
        """BAILIAN_MODELS contains qwen-plus."""
        assert "qwen-plus" in BAILIAN_MODELS
        model = BAILIAN_MODELS["qwen-plus"]
        assert model.api == "bailian"

    def test_bailian_models_contains_qwen_max(self):
        """BAILIAN_MODELS contains qwen-max."""
        assert "qwen-max" in BAILIAN_MODELS
        model = BAILIAN_MODELS["qwen-max"]
        assert model.reasoning is True

    def test_bailian_models_contains_qwen_max_longcontext(self):
        """BAILIAN_MODELS contains qwen-max-longcontext."""
        assert "qwen-max-longcontext" in BAILIAN_MODELS
        model = BAILIAN_MODELS["qwen-max-longcontext"]
        assert model.context_window == 100000


class TestOpenAIModels:
    """Tests for OPENAI_MODELS dict."""

    def test_openai_models_contains_gpt4o(self):
        """OPENAI_MODELS contains gpt-4o."""
        assert "gpt-4o" in OPENAI_MODELS
        model = OPENAI_MODELS["gpt-4o"]
        assert model.api == "openai"
        assert model.provider == "openai"

    def test_openai_models_contains_gpt4o_mini(self):
        """OPENAI_MODELS contains gpt-4o-mini."""
        assert "gpt-4o-mini" in OPENAI_MODELS
        model = OPENAI_MODELS["gpt-4o-mini"]
        assert model.reasoning is False

    def test_openai_models_contains_gpt4_turbo(self):
        """OPENAI_MODELS contains gpt-4-turbo."""
        assert "gpt-4-turbo" in OPENAI_MODELS
        model = OPENAI_MODELS["gpt-4-turbo"]
        assert model.context_window == 128000

    def test_openai_models_contains_o1_preview(self):
        """OPENAI_MODELS contains o1-preview."""
        assert "o1-preview" in OPENAI_MODELS
        model = OPENAI_MODELS["o1-preview"]
        assert model.reasoning is True

    def test_openai_models_contains_o1_mini(self):
        """OPENAI_MODELS contains o1-mini."""
        assert "o1-mini" in OPENAI_MODELS
        model = OPENAI_MODELS["o1-mini"]
        assert model.reasoning is True
        assert model.max_tokens == 65536


class TestAllModels:
    """Tests for ALL_MODELS combined dict."""

    def test_all_models_contains_bailian_models(self):
        """ALL_MODELS contains all Bailian models."""
        for model_id in BAILIAN_MODELS:
            assert model_id in ALL_MODELS

    def test_all_models_contains_openai_models(self):
        """ALL_MODELS contains all OpenAI models."""
        for model_id in OPENAI_MODELS:
            assert model_id in ALL_MODELS

    def test_all_models_count(self):
        """ALL_MODELS contains expected number of models."""
        expected_count = len(BAILIAN_MODELS) + len(OPENAI_MODELS)
        assert len(ALL_MODELS) == expected_count


class TestGetModel:
    """Tests for get_model function."""

    def test_get_model_returns_model(self):
        """get_model returns the correct Model."""
        model = get_model("gpt-4o")
        assert isinstance(model, Model)
        assert model.id == "gpt-4o"

    def test_get_model_for_bailian(self):
        """get_model returns Bailian model."""
        model = get_model("qwen-max")
        assert model.api == "bailian"
        assert model.provider == "bailian"

    def test_get_model_raises_for_unknown(self):
        """get_model raises KeyError for unknown model."""
        with pytest.raises(KeyError, match="Model 'unknown' not found"):
            get_model("unknown")

    def test_get_model_error_includes_available_models(self):
        """get_model error includes available models."""
        try:
            get_model("fake-model")
        except KeyError as e:
            error_msg = str(e)
            assert "gpt-4o" in error_msg
            assert "qwen-max" in error_msg


class TestGetModels:
    """Tests for get_models function."""

    def test_get_models_returns_all_models(self):
        """get_models returns all models without filter."""
        models = get_models()
        assert len(models) == len(ALL_MODELS)

    def test_get_models_filters_by_provider_openai(self):
        """get_models filters by provider 'openai'."""
        models = get_models(provider="openai")
        assert len(models) == len(OPENAI_MODELS)
        for model in models:
            assert model.provider == "openai"

    def test_get_models_filters_by_provider_bailian(self):
        """get_models filters by provider 'bailian'."""
        models = get_models(provider="bailian")
        assert len(models) == len(BAILIAN_MODELS)
        for model in models:
            assert model.provider == "bailian"

    def test_get_models_returns_empty_for_unknown_provider(self):
        """get_models returns empty list for unknown provider."""
        models = get_models(provider="unknown")
        assert models == []


class TestGetProviders:
    """Tests for get_providers function."""

    def test_get_providers_returns_providers(self):
        """get_providers returns list of provider names."""
        providers = get_providers()
        assert "openai" in providers
        assert "bailian" in providers

    def test_get_providers_count(self):
        """get_providers returns expected number of providers."""
        providers = get_providers()
        assert len(providers) == 2

    def test_get_providers_returns_list(self):
        """get_providers returns a list."""
        providers = get_providers()
        assert isinstance(providers, list)


class TestModelAttributes:
    """Tests for Model attribute validation."""

    def test_model_has_required_attributes(self):
        """Model has all required attributes."""
        model = get_model("gpt-4o")
        assert hasattr(model, "id")
        assert hasattr(model, "name")
        assert hasattr(model, "api")
        assert hasattr(model, "provider")
        assert hasattr(model, "base_url")
        assert hasattr(model, "reasoning")
        assert hasattr(model, "input")
        assert hasattr(model, "cost")
        assert hasattr(model, "context_window")
        assert hasattr(model, "max_tokens")

    def test_model_cost_structure(self):
        """Model.cost has correct structure."""
        model = get_model("gpt-4o")
        assert hasattr(model.cost, "input")
        assert hasattr(model.cost, "output")
        assert hasattr(model.cost, "total")

    def test_model_input_types(self):
        """Model.input contains valid input types."""
        model = get_model("gpt-4o")
        for input_type in model.input:
            assert input_type in ["text", "image"]