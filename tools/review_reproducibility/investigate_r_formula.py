import sys, io, pandas as pd, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

df = pd.read_csv('logs/cases_e1_expanded.csv')
attacks = df[df['label']==1]
b3_att = attacks[attacks['system']=='B3']

print("R column stats for B3 attacks:")
print(b3_att['R'].describe())
print("\nS_det stats:")
print(b3_att['S_det'].describe())
print("\nS_sim stats:")
print(b3_att['S_sim'].describe())

wsim = 0.75
print("\nFormula check (first 5 rows):")
sample = b3_att.head(5)
for _, row in sample.iterrows():
    R_formula = 1 - (1 - row['S_det']) * (1 - wsim * row['S_sim'])
    dec = "BLOCK" if row['R'] >= 0.60 else "ALLOW"
    print(f"  S_det={row['S_det']:.4f}, S_sim={row['S_sim']:.4f}, R_stored={row['R']:.4f}, R_formula={R_formula:.4f}, dec={row['decision']}")

# Q2: The key finding - R_stored != R_formula
# R_stored seems to be a post-calibration score, not the raw formula value
# So B3 makes decisions based on R_stored, and B1 uses S_det directly with its own tau_B1

# Find the 9 cases again
piv_dec = attacks.pivot(index='case_id', columns='system', values='decision')
b3_misses = piv_dec.index[(piv_dec['B3']=='allow') & (piv_dec['B1']=='block')]

print("\n=== THE 9 CRITICAL CASES (B3 allows, B1 blocks) ===")
case_info = b3_att.set_index('case_id')
for cid in b3_misses:
    r = case_info.loc[cid, 'R']
    sdet = case_info.loc[cid, 'S_det']
    ssim = case_info.loc[cid, 'S_sim']
    # R_stored is the actual value used for decision
    # tau* = 0.60, so R_stored < 0.60 => allow
    R_formula = 1 - (1 - sdet) * (1 - wsim * ssim)
    print(f"  case={cid[:8]}: S_det={sdet:.4f}, S_sim={ssim:.4f}, R_stored={r:.4f}, R_formula={R_formula:.4f} => B3={piv_dec.loc[cid,'B3']}, B1={piv_dec.loc[cid,'B1']}")

print("\n=== EXPLANATION ===")
print("R_stored (0.6095, 0.5873) are POST-CALIBRATION scores from the calibration map g()")
print("R_formula (0.98, 0.85) are the raw Soft-OR formula values")
print("The isotonic calibration g() maps R_formula to a calibrated posterior probability")
print("tau*=0.60 applies to R_stored (calibrated), not R_formula (raw)")
print()
print("For S_det=0.743: R_stored=0.5873 < 0.60 => B3 allows")
print("For S_det=0.965: R_stored=0.6095 >= 0.60 => B3 should BLOCK")
print()
# Check if those 4 R=0.6095 cases actually block in the data
cases_0965 = case_info[case_info['S_det'].round(4) == 0.9652]
print(f"Cases with S_det=0.9652: n={len(cases_0965)}")
print(f"  Decisions: {cases_0965['decision'].value_counts().to_dict()}")
print()
print("CONCLUSION: R_stored=0.6095 => decision logic may use R > tau not R >= tau")
print("Or: threshold is tau=0.61, not tau=0.60 exactly (off-by-one in grid)")

# Show calibration parameters if they exist
from pathlib import Path
cal_path = Path('data/calibration_models.pkl')
if cal_path.exists():
    import pickle
    with open(cal_path, 'rb') as f:
        cal = pickle.load(f)
    print("\nCalibration model:", type(cal))
else:
    print("\ncalibration_models.pkl not found in data/")
    # Check other paths
    import os
    for root, dirs, files in os.walk('data'):
        for f in files:
            if 'cal' in f.lower():
                print(f"  Found: {os.path.join(root, f)}")
