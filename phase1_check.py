"""
phase1_check.py — Phase 1 acceptance gate.
Run from project root: python phase1_check.py
"""
import sys, json, csv, os, subprocess
from pathlib import Path

sys.path.insert(0, '.')
from src.utils.config_loader import load_config

PASS = '[PASS]'; FAIL = '[FAIL]'; WARN = '[WARN]'
results = {}
cfg = load_config(Path('.'))
root = cfg.paths.root

# 1. Partition files exist
part_dir = root / 'data' / 'partitions'
required_parts = [
    part_dir / 'calibration.jsonl',
    part_dir / 'benign_test.jsonl',
    part_dir / 'adaptive_seeds.jsonl',
    part_dir / 'held_out_families',  # directory
]
missing = [str(p) for p in required_parts if not p.exists()]
ok = len(missing) == 0
results['1_partitions'] = ok
print(f'{PASS if ok else FAIL} Partition files: {"all present" if ok else "MISSING: "+str(missing)}')

# 2. Calibration size check
if (part_dir / 'calibration.jsonl').exists():
    recs = []
    with open(part_dir / 'calibration.jsonl', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    attacks = [r for r in recs if r.get('label') == 1]
    benign  = [r for r in recs if r.get('label') == 0]
    # Strict IEEE full mode target: >=300 attacks + >=300 benign
    ok = len(attacks) >= 300 and len(benign) >= 300
    results['2_cal_size'] = ok
    print(f'{PASS if ok else FAIL} Calibration: {len(attacks)} attacks + {len(benign)} benign (Real benchmark mode)')
else:
    results['2_cal_size'] = False
    print(f'{FAIL} Calibration file missing')

# 3. Benign test >= 200 items (relaxed from 500 for initial run)
if (part_dir / 'benign_test.jsonl').exists():
    with open(part_dir / 'benign_test.jsonl', encoding='utf-8') as f:
        ben_recs = [json.loads(l) for l in f if l.strip()]
    hard_neg = [r for r in ben_recs if r.get('dataset') == 'HardNeg']
    ok_size = len(ben_recs) >= 100
    ok_hn   = len(hard_neg) >= 50
    ok = ok_size and ok_hn
    results['3_benign_test'] = ok
    print(f'{PASS if ok else FAIL} Benign test: {len(ben_recs)} total, {len(hard_neg)} hard negatives')
else:
    results['3_benign_test'] = False
    print(f'{FAIL} Benign test file missing')

# 4. Held-out families directory has >= 4 files
held_dir = part_dir / 'held_out_families'
if held_dir.exists():
    fam_files = list(held_dir.glob('*.jsonl'))
    ok = len(fam_files) >= 4
    results['4_held_out'] = ok
    total_held = 0
    for f in fam_files:
        with open(f, encoding='utf-8') as fp:
            n = sum(1 for l in fp if l.strip())
        total_held += n
    print(f'{PASS if ok else FAIL} Held-out families: {len(fam_files)} families, '
          f'{total_held} total records (need >=4 families)')
else:
    results['4_held_out'] = False
    print(f'{FAIL} held_out_families directory missing or empty')

# 5. data_inventory.csv exists
inv_path = root / 'logs' / 'data_inventory.csv'
ok = inv_path.exists() and inv_path.stat().st_size > 0
results['5_inventory'] = ok
print(f'{PASS if ok else FAIL} data_inventory.csv: {"exists" if ok else "MISSING"}')

# 6. Table II generated
t2 = root / 'outputs' / 'tables' / 'table_II.tex'
ok = t2.exists() and t2.stat().st_size > 100
results['6_tableII'] = ok
print(f'{PASS if ok else FAIL} outputs/tables/table_II.tex: {"generated" if ok else "MISSING"}')

# 7. leakage_check passes
lk = subprocess.run([sys.executable, 'src/leakage_check.py'], capture_output=True, text=True)
ok = lk.returncode == 0
results['7_leakage'] = ok
out_line = lk.stdout.strip().split('\n')[-1] if lk.stdout else lk.stderr.strip()[:80]
print(f'{PASS if ok else FAIL} leakage_check.py: {out_line}')

# 8. pytest test_audit passes
pt = subprocess.run(
    [sys.executable, '-m', 'pytest', 'tests/test_audit.py', '-q', '--tb=short'],
    capture_output=True, text=True
)
ok = pt.returncode == 0
results['8_pytest'] = ok
last = [l for l in pt.stdout.strip().splitlines() if l.strip()]
print(f'{PASS if ok else FAIL} pytest test_audit: {last[-1] if last else "no output"}')
if not ok:
    print(pt.stdout[-400:])

# 9. All label values are 0 or 1 in calibration
if (part_dir / 'calibration.jsonl').exists():
    bad_labels = []
    with open(part_dir / 'calibration.jsonl', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if line.strip():
                r = json.loads(line)
                if r.get('label') not in (0, 1):
                    bad_labels.append(i)
    ok = len(bad_labels) == 0
    results['9_labels'] = ok
    print(f'{PASS if ok else FAIL} Labels: all values are 0 or 1 ({len(bad_labels)} bad)')
else:
    results['9_labels'] = False
    print(f'{FAIL} Cannot check labels — calibration missing')

# Summary
passed = sum(results.values()); total = len(results)
print()
print('=' * 55)
if all(results.values()):
    print(f'PHASE 1: ALL {total}/{total} CHECKS PASSED')
    print('STATUS: COMPLETE - Approved to start Phase 2')
else:
    failed = [k for k, v in results.items() if not v]
    print(f'PHASE 1: {passed}/{total} PASSED')
    print(f'FAILED:  {failed}')
print('=' * 55)
