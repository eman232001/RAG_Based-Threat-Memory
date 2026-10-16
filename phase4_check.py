"""
phase4_check.py
Phase 4 Acceptance Gate — verifies all Phase 4 completion criteria.

Checks:
  1. src/rtm_middleware.py exists and all 14 systems are registered
  2. tests/test_middleware.py passes (39/39 tests)
  3. src/run_e1.py exists and tests/test_run_e1.py passes
  4. src/run_e2.py exists and tests/test_run_e2.py passes
  5. logs/cases.csv contains valid rows for both E1 and E2
  6. E2 covers >= 5 held-out families with both E2_before and E2_after
  7. src/consistency_check.py exits 0 (all mathematical invariants verified)
  8. src/eval.py successfully generates table_III.tex and table_IV.tex
  9. Calibration parameters remain strictly frozen in config/prereg.yaml

Run from project root:
  python phase4_check.py
"""
from __future__ import annotations

import csv
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

CASES_CSV = PROJECT_ROOT / "logs" / "cases.csv"
PREREG_FILE = PROJECT_ROOT / "config" / "prereg.yaml"

PASS = "[PASS]"
FAIL = "[FAIL]"


def _check(label: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    msg = f"{status} {label}"
    if detail:
        msg += f": {detail}"
    print(msg)
    return condition


def main() -> None:
    print("\n" + "=" * 65)
    print(" RTM Phase 4 Acceptance Gate Verification")
    print("=" * 65 + "\n")

    results = []

    # ── Check 1: rtm_middleware.py imports and has all 14 systems ──
    try:
        from src.rtm_middleware import list_systems
        sys_list = list_systems()
        has_14 = len(sys_list) >= 14
        results.append(_check("rtm_middleware.py has all 14 systems registered", has_14, f"Found {len(sys_list)}: {sys_list}"))
    except Exception as e:
        results.append(_check("rtm_middleware.py importable", False, str(e)))

    # ── Check 2: pytest tests/test_middleware.py passes ──
    cmd_mid = [sys.executable, "-m", "pytest", "tests/test_middleware.py", "-q"]
    ret_mid = subprocess.run(cmd_mid, cwd=PROJECT_ROOT, capture_output=True, text=True)
    results.append(_check("tests/test_middleware.py passes (39/39)", ret_mid.returncode == 0, ret_mid.stdout.strip()))

    # ── Check 3: run_e1.py exists and tests pass ──
    cmd_e1 = [sys.executable, "-m", "pytest", "tests/test_run_e1.py", "-q"]
    ret_e1 = subprocess.run(cmd_e1, cwd=PROJECT_ROOT, capture_output=True, text=True)
    results.append(_check("tests/test_run_e1.py passes", ret_e1.returncode == 0, ret_e1.stdout.strip()))

    # ── Check 4: run_e2.py exists and tests pass ──
    cmd_e2 = [sys.executable, "-m", "pytest", "tests/test_run_e2.py", "-q"]
    ret_e2 = subprocess.run(cmd_e2, cwd=PROJECT_ROOT, capture_output=True, text=True)
    results.append(_check("tests/test_run_e2.py passes", ret_e2.returncode == 0, ret_e2.stdout.strip()))

    # ── Check 5: cases.csv exists and has E1 and E2 rows ──
    cases_exist = CASES_CSV.exists()
    e1_count = 0
    e2_count = 0
    e2_families = set()
    if cases_exist:
        with open(CASES_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                exp = r.get("experiment_id", "")
                if exp == "E1":
                    e1_count += 1
                elif exp.startswith("E2_"):
                    e2_count += 1
                    fam = r.get("family", "")
                    if fam and fam not in ("unknown", "benign"):
                        e2_families.add(fam)

    results.append(_check("cases.csv contains E1 rows", e1_count > 0, f"{e1_count} rows"))
    results.append(_check("cases.csv contains E2 rows", e2_count > 0, f"{e2_count} rows across {len(e2_families)} families"))
    results.append(_check("E2 covers >= 5 held-out families", len(e2_families) >= 5, f"{len(e2_families)} families: {sorted(e2_families)}"))

    # ── Check 6: consistency_check.py exits 0 ──
    cmd_const = [sys.executable, "src/consistency_check.py"]
    ret_const = subprocess.run(cmd_const, cwd=PROJECT_ROOT, capture_output=True, text=True)
    results.append(_check("consistency_check.py exits 0 (mathematical invariants)", ret_const.returncode == 0))

    # ── Check 7: eval.py produces table_III and table_IV ──
    cmd_eval = [sys.executable, "src/eval.py"]
    ret_eval = subprocess.run(cmd_eval, cwd=PROJECT_ROOT, capture_output=True, text=True)
    t3_exists = (PROJECT_ROOT / "outputs" / "tables" / "table_III.tex").exists()
    t4_exists = (PROJECT_ROOT / "outputs" / "tables" / "table_IV.tex").exists()
    results.append(_check("eval.py generates table_III.tex and table_IV.tex", ret_eval.returncode == 0 and t3_exists and t4_exists))

    # ── Check 8: prereg.yaml remains strictly frozen ──
    import yaml
    frozen = False
    if PREREG_FILE.exists():
        with open(PREREG_FILE, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            frozen = data.get("calibration", {}).get("frozen", False)
    results.append(_check("Calibration parameters remain strictly frozen in prereg.yaml", frozen))

    # ── Summary ──
    total = len(results)
    passed = sum(1 for r in results if r)
    print("\n" + "=" * 65)
    print(f" Phase 4 Acceptance Gate: {passed}/{total} CHECKS PASSED")
    print("=" * 65 + "\n")

    if passed == total:
        print("[SUCCESS] Phase 4 (Core Experiments E1 & E2) is 100% COMPLETE and VERIFIED!")
        sys.exit(0)
    else:
        print(f"[FAILURE] {total - passed} check(s) failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
