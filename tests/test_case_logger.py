"""
tests/test_case_logger.py
Unit tests for centralized idempotent CaseLogger.
"""
from __future__ import annotations

import csv
import pytest
from pathlib import Path

from src.utils.case_logger import CaseLogger, CASES_HEADER


def test_case_logger_creates_file_with_header(tmp_path: Path):
    csv_file = tmp_path / "cases.csv"
    logger = CaseLogger(csv_file)
    assert csv_file.exists()

    with open(csv_file, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == CASES_HEADER


def test_case_logger_idempotent_update(tmp_path: Path):
    csv_file = tmp_path / "cases.csv"
    logger = CaseLogger(csv_file)

    row1 = {
        "experiment_id": "E1",
        "system": "B5",
        "task_id": "task_0",
        "seed": 1,
        "latency_ms": 100.0,
        "raw_det": 0.5,
    }
    logger.log_case(row1)

    rows = logger.load_rows()
    assert len(rows) == 1
    assert rows[0]["raw_det"] == "0.5"
    assert rows[0]["latency_ms"] == "100.0"

    # Log updated row with same key
    row1_updated = {
        "experiment_id": "E1",
        "system": "B5",
        "task_id": "task_0",
        "seed": 1,
        "latency_ms": 150.0,
        "raw_det": 0.85,
    }
    logger.log_case(row1_updated)

    rows = logger.load_rows()
    assert len(rows) == 1  # No duplicate row!
    assert rows[0]["raw_det"] == "0.85"
    assert rows[0]["latency_ms"] == "150.0"


def test_case_logger_appends_distinct_cases(tmp_path: Path):
    csv_file = tmp_path / "cases.csv"
    logger = CaseLogger(csv_file)

    logger.log_case({"experiment_id": "E1", "system": "B5", "task_id": "task_0", "seed": 1})
    logger.log_case({"experiment_id": "E1", "system": "B6", "task_id": "task_0", "seed": 1})
    logger.log_case({"experiment_id": "E1", "system": "B5", "task_id": "task_1", "seed": 1})

    rows = logger.load_rows()
    assert len(rows) == 3


def test_case_logger_deduplicate(tmp_path: Path):
    csv_file = tmp_path / "cases.csv"
    # Manually write duplicates
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CASES_HEADER)
        writer.writeheader()
        writer.writerow({k: "NA" for k in CASES_HEADER} | {"experiment_id": "E1", "system": "B5", "task_id": "t0", "seed": "1", "latency_ms": "10"})
        writer.writerow({k: "NA" for k in CASES_HEADER} | {"experiment_id": "E1", "system": "B5", "task_id": "t0", "seed": "1", "latency_ms": "20"})

    logger = CaseLogger(csv_file)
    assert len(logger.load_rows()) == 2

    removed = logger.deduplicate()
    assert removed == 1
    rows = logger.load_rows()
    assert len(rows) == 1
    assert rows[0]["latency_ms"] == "20"
