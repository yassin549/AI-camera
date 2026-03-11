from __future__ import annotations

import importlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _reload_config():
    import config

    return importlib.reload(config)


def test_is_api_key_valid_allows_any_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("API_KEY", raising=False)
    config = _reload_config()
    assert config.is_api_key_valid(None) is True
    assert config.is_api_key_valid("anything") is True


def test_is_api_key_valid_requires_exact_match_when_set(monkeypatch) -> None:
    monkeypatch.setenv("API_KEY", "secret-key")
    config = _reload_config()
    assert config.is_api_key_valid("secret-key") is True
    assert config.is_api_key_valid("wrong") is False
    assert config.is_api_key_valid(None) is False


def test_janus_upstream_env_applies(monkeypatch) -> None:
    custom = "http://10.0.0.8:8088/janus/"
    monkeypatch.setenv("AICAM_JANUS_UPSTREAM", custom)
    config = _reload_config()
    assert config.JANUS_UPSTREAM_URL == custom.rstrip("/")
