"""
tests/test_run_e1.py
Unit tests for src/run_e1.py (Phase 4 — Section 4.2: Experiment E1)
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.run_e1 import (
    load_test_items,
    _create_detector,
    _load_cache,
    _append_cache,
    run_item_evaluation,
)
from src.rtm_middleware import PassthroughDetector


def test_load_test_items(tmp_path: Path):
    """Test loading test items with limit."""
    items = load_test_items(max_per_family=5)
    assert len(items) > 0
    for it in items:
        assert "text" in it
        assert "label" in it


def test_create_detector_b0():
    """B0 creates PassthroughDetector."""
    cfg = MagicMock()
    det = _create_detector("B0", cfg=cfg, seed=1)
    assert isinstance(det, PassthroughDetector)
    is_inj, score = det.detect("hello")
    assert not is_inj
    assert score == 0.0


def test_cache_io(tmp_path: Path, monkeypatch):
    """Test reading and writing cache."""
    cache_file = tmp_path / "test_cache.jsonl"
    monkeypatch.setattr("src.run_e1.CACHE_FILE", cache_file)

    rec = {"system": "B0", "sha256": "abc12345", "decision": "allow", "latency_ms": 1.0}
    _append_cache(rec)

    cache = _load_cache()
    assert "B0:abc12345" in cache
    assert cache["B0:abc12345"]["decision"] == "allow"


def test_run_item_evaluation_mock(tmp_path: Path, monkeypatch):
    """Test run_item_evaluation with mock detector and mock items."""
    cache_file = tmp_path / "cache.jsonl"
    monkeypatch.setattr("src.run_e1.CACHE_FILE", cache_file)

    mock_items = [
        {"sha256": "111", "text": "safe text", "label": 0, "family": "benign", "dataset": "test", "partition": "test"},
        {"sha256": "222", "text": "injection text", "label": 1, "family": "attack", "dataset": "test", "partition": "test"},
    ]
    monkeypatch.setattr("src.run_e1.load_test_items", lambda max_per_family=None: mock_items)

    mock_logged = []
    monkeypatch.setattr("src.run_e1._log_case_atomic", lambda row: mock_logged.append(row))

    cfg = MagicMock()
    # Run with B0 across 2 seeds
    rows_count = run_item_evaluation(systems=["B0"], seeds=[1, 2], cfg=cfg)

    # 2 items * 2 seeds = 4 rows
    assert rows_count == 4
    assert len(mock_logged) == 4
    assert mock_logged[0]["experiment_id"] == "E1"
    assert mock_logged[0]["system"] == "B0"
    assert mock_logged[0]["decision"] == "allow"
