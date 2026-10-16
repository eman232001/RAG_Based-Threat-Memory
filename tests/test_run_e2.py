"""
tests/test_run_e2.py
Unit tests for src/run_e2.py (Phase 4 — Section 4.3: Experiment E2 Adaptation)
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.run_e2 import (
    build_verified_threat_record,
    AdaptedMemoryRetriever,
    _load_cache,
    _append_cache,
)
from src.detector import MemoryRetriever
from src.utils.config_loader import load_config


def test_build_verified_threat_record():
    items = [
        {"text": "Exemplar attack 1 text.", "attacker_instruction": "Instruction 1"},
        {"text": "Exemplar attack 2 text.", "attacker_instruction": "Instruction 2"},
    ]
    rec = build_verified_threat_record("test_family", items)
    assert rec["family"] == "test_family"
    assert rec["record_id"].startswith("RTM-ADAPT-")
    assert len(rec["exemplars"]) == 2
    assert "Instruction 1" in rec["pattern_description"]


def test_adapted_retriever():
    """Verify AdaptedMemoryRetriever adds new vectors and finds them."""
    cfg = load_config(PROJECT_ROOT)
    base_retriever = MemoryRetriever(cfg)
    base_count = base_retriever._index.ntotal

    new_record = {
        "record_id": "RTM-ADAPT-TEST1",
        "family": "test_family",
        "pattern_description": "Unique pattern description for test family.",
        "exemplars": ["Unique exemplar sentence XZY987 for test family."],
    }

    adapted = AdaptedMemoryRetriever(base_retriever, new_record)
    assert adapted._index.ntotal == base_count + 2  # 1 description + 1 exemplar
    assert len(adapted._metadata) == len(base_retriever._metadata) + 2

    # Query the unique exemplar
    res = adapted.retrieve("Unique exemplar sentence XZY987 for test family.")
    assert res["best_record_id"] == "RTM-ADAPT-TEST1"
    assert res["best_score"] > 0.85


def test_e2_cache_io(tmp_path: Path, monkeypatch):
    cache_file = tmp_path / "e2_cache.jsonl"
    monkeypatch.setattr("src.run_e2.CACHE_FILE", cache_file)

    rec = {"exp": "E2_after", "system": "B3", "sha256": "hash999", "R": 0.88, "decision": "block"}
    _append_cache(rec)

    cache = _load_cache()
    key = "E2_after:B3:hash999"
    assert key in cache
    assert cache[key]["decision"] == "block"
    assert cache[key]["R"] == 0.88
