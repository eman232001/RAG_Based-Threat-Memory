"""
src/utils/case_logger.py
Centralized, idempotent case logger for logs/cases.csv.

Guarantees:
  • Exact 27-column schema matching IEEE benchmark standard.
  • Strict idempotency: re-running any task or experiment updates existing
    records matching (experiment_id, system, task_id, seed) instead of duplicating.
  • Atomic file writes (.tmp -> replace) to prevent partial writes during crashes.
  • Missing values automatically serialized as "NA".
"""
from __future__ import annotations

import csv
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logger import get_logger

log = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CASES_CSV = PROJECT_ROOT / "logs" / "cases.csv"

CASES_HEADER: List[str] = [
    "experiment_id", "task_id", "case_id", "partition", "dataset",
    "family", "label", "system", "seed",
    "raw_det", "raw_sim_k1", "raw_sim_k3", "raw_sim_k5", "raw_sim_k10",
    "S_det", "S_sim", "R", "decision",
    "attack_success", "task_success",
    "latency_ms", "model_calls", "retrieval_calls", "tokens_cost",
    "record_id", "segment", "explanation_json",
]


def _make_key(row: Dict[str, Any]) -> Tuple[str, str, str, str]:
    """Derive natural deduplication key for a test case."""
    exp = str(row.get("experiment_id", "")).strip()
    sys_name = str(row.get("system", "")).strip()
    task = str(row.get("task_id", "")).strip()
    seed = str(row.get("seed", "")).strip()
    return (exp, sys_name, task, seed)


class CaseLogger:
    """Manages reading, writing, and deduplicating rows in cases.csv."""

    def __init__(self, cases_path: Optional[Path] = None) -> None:
        self.cases_path = Path(cases_path or DEFAULT_CASES_CSV)
        self._ensure_file()

    def _ensure_file(self) -> None:
        """Create file with standard header if it doesn't exist."""
        self.cases_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.cases_path.exists() or self.cases_path.stat().st_size == 0:
            with open(self.cases_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CASES_HEADER)
                writer.writeheader()

    def load_rows(self) -> List[Dict[str, str]]:
        """Load all existing rows from cases.csv."""
        if not self.cases_path.exists():
            return []
        with open(self.cases_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)

    def log_case(self, row: Dict[str, Any]) -> None:
        """
        Log or update a single test case row idempotently.
        If a row with matching (experiment_id, system, task_id, seed) exists,
        it is updated in place. Otherwise, it is appended.
        """
        self._ensure_file()

        # Format input row with standard header and "NA" for missing fields
        full_row: Dict[str, str] = {k: "NA" for k in CASES_HEADER}
        for k, v in row.items():
            if k in full_row:
                full_row[k] = "NA" if v is None else str(v)

        existing_rows = self.load_rows()
        target_key = _make_key(full_row)

        updated = False
        for i, existing in enumerate(existing_rows):
            if _make_key(existing) == target_key:
                existing_rows[i] = full_row
                updated = True
                break

        if not updated:
            existing_rows.append(full_row)

        self._atomic_write(existing_rows)

    def log_cases(self, rows: List[Dict[str, Any]]) -> int:
        """
        Log multiple cases idempotently.
        Returns the number of cases processed.
        """
        for r in rows:
            self.log_case(r)
        return len(rows)

    def deduplicate(self) -> int:
        """
        Remove any duplicate records in cases.csv, keeping the latest entry.
        Returns the count of duplicate rows removed.
        """
        existing_rows = self.load_rows()
        if not existing_rows:
            return 0

        unique_dict: Dict[Tuple[str, str, str, str], Dict[str, str]] = {}
        duplicates_count = 0

        for r in existing_rows:
            key = _make_key(r)
            if key in unique_dict:
                duplicates_count += 1
            # Update to keep the most recent entry
            unique_dict[key] = r

        if duplicates_count > 0:
            cleaned_rows = list(unique_dict.values())
            self._atomic_write(cleaned_rows)
            log.info(f"Deduplicated cases.csv: removed {duplicates_count} duplicate rows.")

        return duplicates_count

    def _atomic_write(self, rows: List[Dict[str, str]]) -> None:
        """Write rows atomically using a temp file to prevent partial file corruption."""
        tmp_fd, tmp_path_str = tempfile.mkstemp(
            prefix="cases_", suffix=".tmp", dir=str(self.cases_path.parent)
        )
        tmp_path = Path(tmp_path_str)
        try:
            with open(tmp_fd, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CASES_HEADER)
                writer.writeheader()
                for r in rows:
                    clean_row = {k: r.get(k, "NA") for k in CASES_HEADER}
                    writer.writerow(clean_row)
            shutil.move(str(tmp_path), str(self.cases_path))
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise


# Singleton instance pointing to project default
default_case_logger = CaseLogger()


def log_case(row: Dict[str, Any]) -> None:
    """Convenience functional interface for logging a case."""
    default_case_logger.log_case(row)


def deduplicate_cases_csv() -> int:
    """Convenience functional interface for deduplicating cases.csv."""
    return default_case_logger.deduplicate()
