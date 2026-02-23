from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import is_authorized_request, to_bool


def test_to_bool_parsing() -> None:
    assert to_bool("true", False) is True
    assert to_bool("False", True) is False
    assert to_bool("1", False) is True
    assert to_bool("0", True) is False
    assert to_bool(None, True) is True


def test_api_key_auth_headers() -> None:
    req = SimpleNamespace(headers={"X-API-Key": "secret"})
    assert is_authorized_request(req, "secret") is True
    assert is_authorized_request(req, "wrong") is False

    req = SimpleNamespace(headers={"Authorization": "Bearer secret"})
    assert is_authorized_request(req, "secret") is True

    req = SimpleNamespace(headers={})
    assert is_authorized_request(req, "secret") is False
    assert is_authorized_request(req, "") is True
