"""
tests/test_poison.py
Unit tests for src/poison.py (Phase 5 — Section 5.2: Experiment E4 Memory Poisoning)
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.poison import (
    generate_poison_records,
    build_fixed_probe_set,
    evaluate_probe_set,
    POISON_TYPES,
)
from src.utils.config_loader import load_config


def test_generate_poison_records():
    data = generate_poison_records(num_per_type=5, seed=123)
    assert set(data.keys()) == set(POISON_TYPES)
    for pt in POISON_TYPES:
        assert len(data[pt]) == 5
        for rec in data[pt]:
            assert "record_id" in rec
            assert "pattern_description" in rec
            assert len(rec["exemplars"]) > 0
            assert rec["type"] == pt


def test_build_fixed_probe_set():
    cfg = load_config(PROJECT_ROOT)
    probes = build_fixed_probe_set(cfg, sample_size=10)
    assert len(probes) > 0
    for p in probes:
        assert "text" in p
        assert "label" in p


def test_evaluate_probe_set_mock():
    mock_probes = [
        {"source_id": "probe1", "text": "benign text", "label": 0},
        {"source_id": "probe2", "text": "attack text", "label": 1},
    ]

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = {"raw_sim_k5": 0.1}

    mock_detector = MagicMock()

    cal = {
        "alpha": 0.1,
        "tau": 0.6,
        "g_det": MagicMock(predict=lambda x: [float(x[0][0])]),
        "g_sim": MagicMock(predict=lambda x: [float(x[0][0])]),
    }

    decisions = evaluate_probe_set(mock_probes, mock_retriever, mock_detector, cal)
    assert len(decisions) == 2
    assert "probe1" in decisions
    assert "probe2" in decisions
    assert decisions["probe1"] in ("allow", "block")
    assert decisions["probe2"] in ("allow", "block")
