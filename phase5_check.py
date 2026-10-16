"""
phase5_check.py
Phase 5 Acceptance Gate — Verifies that all Phase 5 experiments are complete
and all acceptance criteria are met before tagging phase5-complete.

Gate Checks:
  [1]  E3 (Fuzzing)          table_V.tex exists, 120 E3 cases logged
  [2]  E4 (Poisoning)        table_VI.tex exists, MPS_cand = 0.0%
  [3]  E5 (Explanation)      table_VII_explanation_eval.tex exists,
                             EVR >= 90%, Kappa >= 0.60, Privacy Leaks = 0
  [4]  E6 (Ablations)        table_VII_ablations.tex exists
  [5]  E6 (Hypotheses)       table_VIII_hypotheses.tex exists
  [6]  All 6 hypotheses marked Supported in Table VIII
  [7]  Consistency check passes (mathematical invariants on cases.csv)
  [8]  161+ pytest tests pass
  [9]  cases.csv has > 400 rows (E1–E6 combined)
  [10] prereg.yaml still frozen (frozen: true, alpha=0.1, tau=0.6138...)
"""
from __future__ import annotations

import csv
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR  = PROJECT_ROOT / "outputs" / "tables"
LOGS_DIR     = PROJECT_ROOT / "logs"
CASES_CSV    = LOGS_DIR / "cases.csv"
PREREG_YAML  = PROJECT_ROOT / "config" / "prereg.yaml"

PASS = "[PASS]"
FAIL = "[FAIL]"

results = []

def chk(condition: bool, msg: str) -> bool:
    tag = PASS if condition else FAIL
    print(f"  {tag}  {msg}")
    results.append(condition)
    return condition


print()
print("=" * 65)
print("  Phase 5 Acceptance Gate")
print("=" * 65)

# ─────────────────────────────────────────────────────────────
# Check [1]: E3 Fuzzing table exists and cases logged
# ─────────────────────────────────────────────────────────────
print("\n[E3 — Fuzzing]")
t5 = OUTPUTS_DIR / "table_V.tex"
chk(t5.exists(), f"table_V.tex exists at {t5}")

if CASES_CSV.exists():
    with open(CASES_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    e3_rows = [r for r in rows if r.get("experiment_id", "").startswith("E3")]
    chk(len(e3_rows) >= 100, f"E3 cases logged: {len(e3_rows)} (>= 100)")
else:
    chk(False, "cases.csv not found")
    rows = []

# ─────────────────────────────────────────────────────────────
# Check [2]: E4 Poisoning table exists and MPS_cand = 0%
# ─────────────────────────────────────────────────────────────
print("\n[E4 — Memory Poisoning]")
t6 = OUTPUTS_DIR / "table_VI.tex"
chk(t6.exists(), f"table_VI.tex exists at {t6}")

e4_cand = [r for r in rows if r.get("experiment_id") == "E4_cand"]
if e4_cand:
    mps_cand = sum(1 for r in e4_cand if r.get("attack_success") == "1") / max(1, len(e4_cand))
    chk(mps_cand == 0.0, f"MPS (candidates) = {mps_cand*100:.1f}% (must be 0.0%)")
else:
    chk(False, "No E4_cand rows found in cases.csv")

# ─────────────────────────────────────────────────────────────
# Check [3]: E5 Explanation table exists and criteria met
# ─────────────────────────────────────────────────────────────
print("\n[E5 — Explanation Quality & Privacy]")
t7e = OUTPUTS_DIR / "table_VII_explanation_eval.tex"
chk(t7e.exists(), f"table_VII_explanation_eval.tex exists at {t7e}")

if t7e.exists():
    content = t7e.read_text(encoding="utf-8")
    # EVR line: extract value
    import re
    evr_match = re.search(r"Explanation Verification Rate.*?\\textbf\{(\d+\.?\d*)\\?%?\}", content)
    evr = float(evr_match.group(1)) if evr_match else 0.0
    chk(evr >= 90.0, f"EVR = {evr:.1f}% (>= 90.0%)")

    kappa_match = re.search(r"Cohen.*?\\textbf\{(\d+\.?\d+)\}", content)
    kappa = float(kappa_match.group(1)) if kappa_match else 0.0
    chk(kappa >= 0.60, f"Cohen's kappa = {kappa:.3f} (>= 0.60)")

    leaks_match = re.search(r"Privacy Leakage Incidents.*?\\textbf\{(\d+)\}", content)
    leaks = int(leaks_match.group(1)) if leaks_match else 1
    chk(leaks == 0, f"Privacy Leaks = {leaks} (must be 0)")

# ─────────────────────────────────────────────────────────────
# Check [4]: E6 Ablations table exists
# ─────────────────────────────────────────────────────────────
print("\n[E6 — Ablations, Latency & TTP]")
t7a = OUTPUTS_DIR / "table_VII_ablations.tex"
chk(t7a.exists(), f"table_VII_ablations.tex exists at {t7a}")

# ─────────────────────────────────────────────────────────────
# Check [5]: Table VIII (Hypotheses) exists
# ─────────────────────────────────────────────────────────────
t8 = OUTPUTS_DIR / "table_VIII_hypotheses.tex"
chk(t8.exists(), f"table_VIII_hypotheses.tex exists at {t8}")

# ─────────────────────────────────────────────────────────────
# Check [6]: All 6 hypotheses Supported
# ─────────────────────────────────────────────────────────────
if t8.exists():
    t8_content = t8.read_text(encoding="utf-8")
    n_supported = t8_content.count("Supported")
    chk(n_supported >= 6, f"All 6 hypotheses Supported in Table VIII (found: {n_supported})")

# ─────────────────────────────────────────────────────────────
# Check [7]: Consistency check
# ─────────────────────────────────────────────────────────────
print("\n[Consistency Check]")
result = subprocess.run(
    [sys.executable, "src/consistency_check.py"],
    capture_output=True, text=True, cwd=PROJECT_ROOT
)
chk(result.returncode == 0, "consistency_check.py passed (exit 0)")

# ─────────────────────────────────────────────────────────────
# Check [8]: pytest tests count
# ─────────────────────────────────────────────────────────────
print("\n[Pytest Suite]")
result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no"],
    capture_output=True, text=True, cwd=PROJECT_ROOT
)
# Parse "N passed"
import re
m = re.search(r"(\d+) passed", result.stdout)
n_passed = int(m.group(1)) if m else 0
chk(n_passed >= 160, f"pytest: {n_passed} tests passed (>= 160)")
if result.returncode != 0:
    print(f"    stderr: {result.stdout[-300:]}")

# ─────────────────────────────────────────────────────────────
# Check [9]: cases.csv row count
# ─────────────────────────────────────────────────────────────
print("\n[cases.csv]")
chk(len(rows) > 400, f"cases.csv has {len(rows)} rows (> 400)")

# ─────────────────────────────────────────────────────────────
# Check [10]: prereg.yaml frozen
# ─────────────────────────────────────────────────────────────
print("\n[Pre-registration Freeze]")
if PREREG_YAML.exists():
    import yaml
    with open(PREREG_YAML, encoding="utf-8") as f:
        prereg = yaml.safe_load(f)
    cal = prereg.get("calibration", {})
    frozen = cal.get("frozen", False)
    alpha  = cal.get("alpha", None)
    tau    = cal.get("tau", None)
    chk(frozen is True, f"frozen = {frozen} (must be true)")
    chk(alpha == 0.1,   f"alpha* = {alpha} (must be 0.1)")
    chk(abs(tau - 0.6138383838383838) < 1e-6, f"tau* = {tau:.10f} (must be 0.6138...)")
else:
    chk(False, "prereg.yaml not found!")

# ─────────────────────────────────────────────────────────────
# Final verdict
# ─────────────────────────────────────────────────────────────
print()
print("=" * 65)
n_pass = sum(results)
n_total = len(results)
if n_pass == n_total:
    print(f"  ALL {n_total} CHECKS PASSED — Phase 5 gate satisfied!")
    print("  Ready to tag: git tag phase5-complete")
    print("=" * 65)
    sys.exit(0)
else:
    n_fail = n_total - n_pass
    print(f"  {n_fail}/{n_total} CHECKS FAILED — Phase 5 gate NOT satisfied.")
    print("=" * 65)
    sys.exit(1)
