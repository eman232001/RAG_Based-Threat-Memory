"""
tests/test_model_client.py
Unit tests for RotatingGeminiClient and ModelClient key rotation, cooldown, and error isolation.
"""
from __future__ import annotations

import time
import pytest
from unittest.mock import MagicMock

from src.utils.model_client import RotatingGeminiClient, KeySlot


def test_keyslot_cooldown_recovery():
    slot = KeySlot(index=0, api_key="dummy_key_123456", client=MagicMock())
    assert slot.is_available(time.time()) is True

    # Put on cooldown
    now = time.time()
    slot.status = "COOLDOWN"
    slot.cooldown_until = now + 10.0
    assert slot.is_available(now) is False

    # After cooldown expires
    future = now + 11.0
    assert slot.is_available(future) is True
    assert slot.status == "HEALTHY"


def test_keyslot_dead_stays_dead():
    slot = KeySlot(index=0, api_key="dummy_key_123456", client=MagicMock())
    slot.status = "DEAD"
    assert slot.is_available(time.time() + 1000.0) is False


def test_rotating_client_deduplicates_keys():
    keys = ["key_A", "key_B", "key_A"]
    client = RotatingGeminiClient(api_keys=keys)
    assert len(client.slots) == 2
    assert [s.api_key for s in client.slots] == ["key_A", "key_B"]


def test_rotating_client_isolates_403():
    client = RotatingGeminiClient(api_keys=["key_bad", "key_good"])

    # Simulate key_bad throwing 403 PERMISSION_DENIED, key_good succeeding
    calls = []

    def mock_fn(c):
        calls.append(c)
        if len(calls) == 1:
            raise Exception("403 PERMISSION_DENIED: project disabled")
        mock_resp = MagicMock()
        mock_resp.text = "Success from good key"
        return mock_resp

    res = client._execute_with_rotation(mock_fn)
    assert res.text == "Success from good key"
    assert client.slots[0].status == "DEAD"
    assert client.slots[1].status == "HEALTHY"


def test_rotating_client_handles_429_quota():
    client = RotatingGeminiClient(api_keys=["key_1", "key_2"])

    calls = []

    def mock_fn(c):
        calls.append(c)
        if len(calls) == 1:
            raise Exception("429 RESOURCE_EXHAUSTED: quota exceeded")
        mock_resp = MagicMock()
        mock_resp.text = "Success from key_2"
        return mock_resp

    res = client._execute_with_rotation(mock_fn)
    assert res.text == "Success from key_2"
    assert client.slots[0].status == "COOLDOWN"
    assert client.slots[0].cooldown_until > time.time()


def test_rotating_client_handles_503_high_demand():
    client = RotatingGeminiClient(api_keys=["key_1", "key_2"])

    calls = []

    def mock_fn(c):
        calls.append(c)
        if len(calls) == 1:
            raise Exception("503 UNAVAILABLE: This model is currently experiencing high demand")
        mock_resp = MagicMock()
        mock_resp.text = "Success from key_2 after 503"
        return mock_resp

    res = client._execute_with_rotation(mock_fn)
    assert res.text == "Success from key_2 after 503"
    assert client.slots[0].status == "COOLDOWN"


def test_rotating_client_retries_transient_network_error():
    client = RotatingGeminiClient(api_keys=["key_1"])

    calls = []

    def mock_fn(c):
        calls.append(c)
        if len(calls) == 1:
            raise ConnectionResetError("Server disconnected without sending a response")
        mock_resp = MagicMock()
        mock_resp.text = "Success after reconnect"
        return mock_resp

    res = client._execute_with_rotation(mock_fn)
    assert res.text == "Success after reconnect"
    assert len(calls) == 2
