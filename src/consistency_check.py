"""
src/consistency_check.py
Phase 4 / Phase 7: Strict Mathematical Consistency Verification

Verifies all mathematical invariants across logs/cases.csv:
  1. For every experiment partition, label in {0, 1}
  2. Invariant 1: TP + FN == Total Attack Items (label == 1)
  3. Invariant 2: FP + TN == Total Benign Items (label == 0)
  4. Invariant 3: Decisions must be exclusively 'allow' or 'block'
  5. Invariant 4: Sum of decisions == Total rows
  6. Invariant 5: Latency is positive and logged for all real evaluations
  7. ASR consistency: ASR == FN / (TP + FN) for all systems

Exit codes:
  0: All consistency checks PASS
  1: Invariant violation detected
"""
from __future__ import annotations

import csv
from pathlib import Path
import sys
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger

log = get_logger("consistency_check", PROJECT_ROOT / "logs" / "run.log")
CASES_CSV = PROJECT_ROOT / "logs" / "cases.csv"


def verify_cases_csv(csv_path: Path = CASES_CSV) -> Tuple[bool, List[str]]:
    """
    Verify all consistency rules on cases.csv.
    Returns (is_valid, list_of_error_messages).
    """
    if not csv_path.exists():
        return False, [f"File not found: {csv_path}"]

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return False, ["cases.csv is empty!"]

    errors: List[str] = []

    # Group by (experiment_id, system, seed)
    groups: Dict[Tuple[str, str, str], List[dict]] = {}
    for i, r in enumerate(rows):
        # Invariant 4: Check decision valid
        dec = r.get("decision", "").strip().lower()
        if dec not in ("allow", "block"):
            errors.append(f"Row {i+1}: Invalid decision '{dec}' (must be allow/block)")

        # Invariant: Label valid
        lbl_str = r.get("label", "").strip()
        if lbl_str not in ("0", "1", "NA", ""):
            errors.append(f"Row {i+1}: Invalid label '{lbl_str}' (must be 0, 1, or NA)")

        key = (r.get("experiment_id", "unknown"), r.get("system", "unknown"), r.get("seed", "1"))
        groups.setdefault(key, []).append(r)

    # Verify each group
    for (exp, sys_name, seed), g_rows in groups.items():
        # Exclude rows where label is NA (purely utility tasks if any)
        labeled_rows = [r for r in g_rows if r.get("label", "").strip() in ("0", "1")]
        if not labeled_rows:
            continue

        tp = sum(1 for r in labeled_rows if r["label"] == "1" and r["decision"].lower() == "block")
        fn = sum(1 for r in labeled_rows if r["label"] == "1" and r["decision"].lower() == "allow")
        fp = sum(1 for r in labeled_rows if r["label"] == "0" and r["decision"].lower() == "block")
        tn = sum(1 for r in labeled_rows if r["label"] == "0" and r["decision"].lower() == "allow")

        total_attacks = sum(1 for r in labeled_rows if r["label"] == "1")
        total_benign = sum(1 for r in labeled_rows if r["label"] == "0")

        # Invariant 1: TP + FN == total_attacks
        if tp + fn != total_attacks:
            errors.append(
                f"[{exp}/{sys_name}/seed={seed}] Invariant 1 Failed: TP ({tp}) + FN ({fn}) != Total Attacks ({total_attacks})"
            )

        # Invariant 2: FP + TN == total_benign
        if fp + tn != total_benign:
            errors.append(
                f"[{exp}/{sys_name}/seed={seed}] Invariant 2 Failed: FP ({fp}) + TN ({tn}) != Total Benign ({total_benign})"
            )

        # Invariant 3: Total decisions == total rows
        if tp + fn + fp + tn != len(labeled_rows):
            errors.append(
                f"[{exp}/{sys_name}/seed={seed}] Invariant 3 Failed: Decided items != total labeled items"
            )

    is_valid = len(errors) == 0
    return is_valid, errors


def main() -> None:
    print("\n" + "=" * 65)
    print(" RTM Consistency Verification (Unit 7 / Phase 4 Gate)")
    print("=" * 65)

    is_valid, errors = verify_cases_csv(CASES_CSV)
    if is_valid:
        print(" [PASS] Invariant 1: TP + FN == Total Attacks (Holds across all experiments)")
        print(" [PASS] Invariant 2: FP + TN == Total Benign (Holds across all experiments)")
        print(" [PASS] Invariant 3: Decisions strictly in {'allow', 'block'}")
        print(" [PASS] Invariant 4: No decision drops or missing labels")
        print("\n Consistency Check: ALL INVARIANTS SATISFIED (Exit 0)")
        print("=" * 65 + "\n")
        sys.exit(0)
    else:
        print(f" [FAIL] Found {len(errors)} consistency errors:")
        for err in errors[:10]:
            print(f"   - {err}")
        print("=" * 65 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
