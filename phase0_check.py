"""phase0_check.py — Phase 0 acceptance gate. Run from project root."""
import sys, yaml, csv, os, subprocess
sys.path.insert(0, '.')
from pathlib import Path

PASS = '[PASS]'; FAIL = '[FAIL]'
results = {}

# 1. Required files
req = [
    '.gitignore','README.md','requirements.txt','prereg.yaml',
    'config/models.yaml','config/experiment.yaml','config/paths.yaml',
    'config/secrets.yaml','config/detector_prompt.txt',
    'src/__init__.py','src/utils/__init__.py','src/utils/config_loader.py',
    'src/utils/logger.py','src/utils/model_client.py','src/smoke_test_b0.py',
    'tests/__init__.py','tests/test_config_loader.py','tests/test_model_client.py',
    'logs/cases.csv','data/memory.jsonl','data/candidate_queue.jsonl',
    'data/verification_log.csv',
]
missing = [f for f in req if not os.path.exists(f)]
ok = len(missing) == 0
results['1_files'] = ok
print(f'{PASS if ok else FAIL} Files: {len(req)-len(missing)}/{len(req)} present'
      + (f'\n       MISSING: {missing}' if missing else ''))

# 2. API key
with open('config/secrets.yaml', encoding='utf-8') as f:
    sec = yaml.safe_load(f)
key = sec.get('gemini', {}).get('api_key', '')
ok = bool(key) and key != 'YOUR_GOOGLE_API_KEY_HERE' and len(key) > 10
results['2_apikey'] = ok
print(f'{PASS if ok else FAIL} Gemini API key: {key[:12]}... ({len(key)} chars)')

# 3. gitignore
ok = 'secrets.yaml' in open('.gitignore').read()
results['3_gitignore'] = ok
print(f'{PASS if ok else FAIL} secrets.yaml in .gitignore')

# 4. No actual API key hardcoded in .py files
# Note: comparing against placeholder string in code is legitimate (e.g. "if key == 'YOUR_GOOGLE...'")
# We only flag if the ACTUAL key value appears in source (not the placeholder string itself)
leaked = []
actual_key = sec.get('gemini', {}).get('api_key', '')
for root, dirs, files in os.walk('src'):
    for fn in files:
        if fn.endswith('.py'):
            c = open(os.path.join(root, fn), encoding='utf-8').read()
            # Only flag if the real key (not placeholder) appears in code
            if actual_key and actual_key in c:
                leaked.append(fn)
            # Also flag if any actual Google API key pattern (AIzaSy...) appears
            import re
            if re.search(r'AIzaSy[a-zA-Z0-9_-]{33}', c):
                leaked.append(fn + ' (AIza pattern)')
ok = len(leaked) == 0
results['4_nokey'] = ok
print(f'{PASS if ok else FAIL} No actual API key hardcoded in source{" - LEAKED: "+str(leaked) if leaked else ""}')

# 5. prereg.yaml integrity
with open('prereg.yaml', encoding='utf-8') as f:
    pr = yaml.safe_load(f)
ok = (
    pr['calibration']['frozen'] is False and
    pr['calibration']['alpha'] is None and
    pr['calibration']['tau'] is None and
    set(pr['hypotheses'].keys()) == {'H1', 'H2', 'H3', 'H4', 'H5', 'H6'} and
    pr['thresholds']['fpr_budget_calibration'] == 0.01
)
results['5_prereg'] = ok
print(f'{PASS if ok else FAIL} prereg.yaml: frozen=false, 6 hypotheses, FPR=0.01')

# 6. cases.csv columns
with open('logs/cases.csv', encoding='utf-8') as f:
    hdr = next(csv.reader(f))
expected = [
    'experiment_id','task_id','case_id','partition','dataset','family','label',
    'system','seed','raw_det','raw_sim_k1','raw_sim_k3','raw_sim_k5','raw_sim_k10',
    'S_det','S_sim','R','decision','attack_success','task_success',
    'latency_ms','model_calls','retrieval_calls','tokens_cost',
    'record_id','segment','explanation_json'
]
ok = hdr == expected
results['6_csv'] = ok
print(f'{PASS if ok else FAIL} cases.csv: {len(hdr)} columns {"correct" if ok else "MISMATCH"}')
if not ok:
    print(f'       Missing: {[c for c in expected if c not in hdr]}')
    print(f'       Extra:   {[c for c in hdr if c not in expected]}')

# 7. config_loader
try:
    from src.utils.config_loader import load_config
    cfg = load_config(Path('.'))
    ok = (
        cfg.base_detector.temperature == 0 and
        cfg.calibration.fpr_budget == 0.01 and
        cfg.retrieval.k_default == 5 and
        cfg.retrieval.k_default in cfg.retrieval.k_sensitivity
    )
    results['7_config'] = ok
    print(f'{PASS if ok else FAIL} config_loader: '
          f'model={cfg.base_detector.model}, '
          f'T={cfg.base_detector.temperature}, '
          f'k={cfg.retrieval.k_default}')
except Exception as e:
    results['7_config'] = False
    print(f'{FAIL} config_loader raised: {e}')

# 8. git tag
tags = subprocess.run(['git', 'tag'], capture_output=True, text=True).stdout.strip()
ok = 'prereg-v0' in tags
results['8_gittag'] = ok
print(f'{PASS if ok else FAIL} git tag prereg-v0 {"exists" if ok else "MISSING"}')

# 9. pytest (no real API calls — mocked)
pytest_result = subprocess.run(
    [sys.executable, '-m', 'pytest', 'tests/', '-q', '--tb=short'],
    capture_output=True, text=True
)
last_line = [l for l in pytest_result.stdout.strip().splitlines() if l.strip()][-1] if pytest_result.stdout else ''
ok = pytest_result.returncode == 0
results['9_pytest'] = ok
print(f'{PASS if ok else FAIL} pytest: {last_line}')
if not ok:
    print(pytest_result.stdout[-500:])

# === SUMMARY ===
passed = sum(results.values())
total = len(results)
print()
print('=' * 55)
if all(results.values()):
    print(f'PHASE 0: ALL {total}/{total} CHECKS PASSED')
    print('STATUS: COMPLETE - Approved to start Phase 1')
else:
    failed = [k for k, v in results.items() if not v]
    print(f'PHASE 0: {passed}/{total} PASSED')
    print(f'FAILED CHECKS: {failed}')
    print('Fix failures before starting Phase 1.')
print('=' * 55)
