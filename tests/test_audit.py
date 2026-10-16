"""
tests/test_audit.py
Unit tests for src/audit.py and src/leakage_check.py

Run: pytest tests/test_audit.py -v
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.audit import (
    make_record,
    deduplicate,
    generate_hard_negatives,
    partition_data,
    _edit_distance,
    _write_jsonl,
)


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture()
def sample_attacks():
    """Small set of distinct attack records."""
    return [
        make_record(f"Ignore previous instructions and do action {i}", 1, "Test", f"family_{i % 4}")
        for i in range(60)
    ]


@pytest.fixture()
def sample_hard_negs():
    return generate_hard_negatives(n=80)


@pytest.fixture()
def mock_cfg(tmp_path):
    """Minimal config mock for partitioning tests."""
    class MockPaths:
        root = tmp_path
        memory = tmp_path / "data" / "memory.jsonl"
    class MockEmbedding:
        model = "all-MiniLM-L6-v2"
        device = "cpu"
        batch_size = 32
    class MockExperiment:
        seeds = [1, 2, 3]
    class MockCfg:
        paths = MockPaths()
        embedding = MockEmbedding()
        experiment = MockExperiment()

    (tmp_path / "data" / "partitions").mkdir(parents=True)
    (tmp_path / "outputs" / "tables").mkdir(parents=True)
    return MockCfg()


# ─────────────────────────────────────────────────────────────
# Tests: make_record
# ─────────────────────────────────────────────────────────────

def test_make_record_label_attack():
    r = make_record("Ignore previous instructions", 1, "Test", "goal_hijacking")
    assert r["label"] == 1
    assert r["dataset"] == "Test"
    assert r["family"] == "goal_hijacking"


def test_make_record_label_benign():
    r = make_record("Security advisory text", 0, "HardNeg", "hard_negative")
    assert r["label"] == 0


def test_make_record_has_sha256():
    r = make_record("Some text", 1, "Test", "family")
    assert "sha256" in r
    assert len(r["sha256"]) == 64


def test_make_record_sha256_deterministic():
    r1 = make_record("Same text", 1, "Test", "fam")
    r2 = make_record("Same text", 1, "Test", "fam")
    assert r1["sha256"] == r2["sha256"]


def test_make_record_different_texts_different_hashes():
    r1 = make_record("Text A", 1, "Test", "fam")
    r2 = make_record("Text B", 1, "Test", "fam")
    assert r1["sha256"] != r2["sha256"]


# ─────────────────────────────────────────────────────────────
# Tests: edit distance
# ─────────────────────────────────────────────────────────────

def test_edit_distance_identical():
    assert _edit_distance("hello", "hello") == 0


def test_edit_distance_empty():
    assert _edit_distance("", "hello") == 5
    assert _edit_distance("hello", "") == 5


def test_edit_distance_one_char():
    assert _edit_distance("abc", "aXc") == 1


def test_edit_distance_completely_different():
    assert _edit_distance("abc", "xyz") == 3


# ─────────────────────────────────────────────────────────────
# Tests: deduplicate
# ─────────────────────────────────────────────────────────────

def test_deduplicate_removes_exact_dups():
    records = [
        make_record("Duplicate text here", 1, "Test", "fam"),
        make_record("Duplicate text here", 1, "Test", "fam"),  # exact dup
        make_record("Different text here", 1, "Test", "fam"),
    ]
    clean, n_exact, n_near = deduplicate(records, edit_threshold=0)
    assert n_exact == 1
    assert len(clean) == 2


def test_deduplicate_empty_list():
    clean, n_exact, n_near = deduplicate([])
    assert clean == []
    assert n_exact == 0
    assert n_near == 0


def test_deduplicate_returns_original_if_no_dups():
    records = [
        make_record(f"Unique text number {i} with extra content", 1, "Test", "fam")
        for i in range(10)
    ]
    clean, n_exact, n_near = deduplicate(records, edit_threshold=0)
    assert len(clean) == 10
    assert n_exact == 0


def test_deduplicate_preserves_label():
    records = [
        make_record("Attack text sample", 1, "Test", "fam"),
        make_record("Benign text sample", 0, "HardNeg", "hard_negative"),
    ]
    clean, _, _ = deduplicate(records)
    labels = {r["text"]: r["label"] for r in clean}
    assert labels["Attack text sample"] == 1
    assert labels["Benign text sample"] == 0


# ─────────────────────────────────────────────────────────────
# Tests: hard negatives
# ─────────────────────────────────────────────────────────────

def test_hard_negatives_all_label_zero():
    negs = generate_hard_negatives(n=30)
    assert all(r["label"] == 0 for r in negs)


def test_hard_negatives_correct_count():
    negs = generate_hard_negatives(n=50)
    assert len(negs) == 50


def test_hard_negatives_dataset_hardneg():
    negs = generate_hard_negatives(n=20)
    assert all(r["dataset"] == "HardNeg" for r in negs)


def test_hard_negatives_have_sha256():
    negs = generate_hard_negatives(n=10)
    assert all("sha256" in r for r in negs)


def test_hard_negatives_minimum_100():
    """Architecture spec requires >= 100 hard negatives."""
    negs = generate_hard_negatives(n=100)
    assert len(negs) >= 100


# ─────────────────────────────────────────────────────────────
# Tests: partitioning
# ─────────────────────────────────────────────────────────────

def test_partition_returns_required_keys(sample_attacks, sample_hard_negs, mock_cfg):
    partitions = partition_data(sample_attacks, sample_hard_negs, mock_cfg)
    required = {"calibration_attacks", "calibration_benign", "benign_test",
                "held_out", "adaptive_seeds"}
    assert required.issubset(set(partitions.keys()))


def test_partition_calibration_has_attacks(sample_attacks, sample_hard_negs, mock_cfg):
    partitions = partition_data(sample_attacks, sample_hard_negs, mock_cfg)
    assert len(partitions["calibration_attacks"]) > 0


def test_partition_all_calibration_are_labeled(sample_attacks, sample_hard_negs, mock_cfg):
    partitions = partition_data(sample_attacks, sample_hard_negs, mock_cfg)
    for r in partitions["calibration_attacks"]:
        assert r["label"] == 1
    for r in partitions["calibration_benign"]:
        assert r["label"] == 0


def test_partition_benign_test_all_label_zero(sample_attacks, sample_hard_negs, mock_cfg):
    partitions = partition_data(sample_attacks, sample_hard_negs, mock_cfg)
    assert all(r["label"] == 0 for r in partitions["benign_test"])


def test_partition_held_out_is_dict(sample_attacks, sample_hard_negs, mock_cfg):
    partitions = partition_data(sample_attacks, sample_hard_negs, mock_cfg)
    assert isinstance(partitions["held_out"], dict)


def test_partition_adaptive_seeds_not_empty(sample_attacks, sample_hard_negs, mock_cfg):
    partitions = partition_data(sample_attacks, sample_hard_negs, mock_cfg)
    assert len(partitions["adaptive_seeds"]) > 0


# ─────────────────────────────────────────────────────────────
# Tests: _write_jsonl
# ─────────────────────────────────────────────────────────────

def test_write_jsonl_readable(tmp_path):
    records = [make_record(f"text {i}", 1, "Test", "fam") for i in range(5)]
    path = tmp_path / "test.jsonl"
    _write_jsonl(path, records)
    assert path.exists()
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 5
    for line in lines:
        obj = json.loads(line)
        assert "text" in obj
        assert "label" in obj


# ─────────────────────────────────────────────────────────────
# Tests: leakage_check (basic)
# ─────────────────────────────────────────────────────────────

def test_leakage_check_passes_on_empty_memory(tmp_path):
    """Empty memory should always pass."""
    from src.leakage_check import run_leakage_check

    class MockPaths:
        root = tmp_path
        memory = tmp_path / "data" / "memory.jsonl"
    class MockEmbedding:
        model = "all-MiniLM-L6-v2"
        device = "cpu"
        batch_size = 32
    class MockExperiment:
        pass  # leakage_check reads experiment.yaml directly, not via cfg.experiment
    class MockCfg:
        paths = MockPaths()
        embedding = MockEmbedding()
        experiment = MockExperiment()

    (tmp_path / "data").mkdir(parents=True)
    (tmp_path / "data" / "memory.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "data" / "partitions").mkdir(parents=True)
    # Create minimal experiment.yaml for leakage_check to read
    (tmp_path / "config").mkdir(parents=True)
    import yaml
    with open(tmp_path / "config" / "experiment.yaml", "w") as f:
        yaml.dump({"partitions": {"leakage_sim_threshold": 0.90, "leakage_edit_threshold": 10}}, f)

    result = run_leakage_check(MockCfg())
    assert result is True
