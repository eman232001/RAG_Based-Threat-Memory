"""
tests/test_run_e6.py
Unit tests for src/run_e6_ablations.py (Phase 5 — Section 5.4: Experiment E6)
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import pytest
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.run_e6_ablations import (
    compute_latency_and_ttp,
    generate_table_vii,
    generate_table_viii,
    load_ablation_test_items,
)


def test_load_ablation_test_items():
    items = load_ablation_test_items()
    assert isinstance(items, list)
    assert len(items) > 0
    for item in items:
        assert "text" in item
        assert "label" in item


def test_compute_latency_and_ttp():
    metrics = compute_latency_and_ttp()
    assert isinstance(metrics, dict)
    assert "b3_med_lat" in metrics
    assert "b5_med_lat" in metrics
    assert "ttp_median_min" in metrics
    assert metrics["b3_med_lat"] < metrics["b5_med_lat"]
    assert metrics["ttp_median_min"] > 0


def test_generate_table_vii_and_viii():
    mock_ablations = {
        "Full RTM (B3)": {"asr": 0.0, "fpr": 0.0, "f1": 1.0, "evr_str": "94.0\\%"},
        "− memory": {"asr": 0.25, "fpr": 0.0, "f1": 0.85, "evr_str": "N/A"},
    }
    mock_lat_ttp = {
        "b3_med_lat": 5.0,
        "b3_p95_lat": 18.5,
        "b5_med_lat": 1102.9,
        "ttp_median_min": 1440.0,
        "ttp_median_hrs": 24.0,
        "ttp_count": 155,
    }
    t7 = generate_table_vii(mock_ablations, mock_lat_ttp)
    assert "Table VII: Ablation Results" in t7
    assert "Full RTM (B3)" in t7
    assert "Time-to-Protect" in t7

    t8 = generate_table_viii()
    assert "Table VIII: Pre-Specified Hypotheses" in t8
    assert "H1 (RQ1)" in t8
    assert "Supported" in t8
