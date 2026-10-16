import sys, io, pandas as pd, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

df = pd.read_csv('logs/cases_e1_expanded.csv')
attacks = df[df['label']==1]

piv_dec = attacks.pivot(index='case_id', columns='system', values='decision')
b3_misses = piv_dec.index[(piv_dec['B3']=='allow') & (piv_dec['B1']=='block')]
piv_r = attacks.pivot(index='case_id', columns='system', values='R')
piv_s = attacks.pivot(index='case_id', columns='system', values='S_det')
piv_sim = attacks.pivot(index='case_id', columns='system', values='S_sim')

print("=== Q2: 9 cases B3 allows, B1 blocks ===")
for cid in b3_misses:
    r = piv_r.loc[cid, 'B3']
    sdet = piv_s.loc[cid, 'B3']
    ssim = piv_sim.loc[cid, 'B3']
    verdict = "BLOCK" if r >= 0.60 else "ALLOW"
    print(f"  case_id={cid}: S_det={sdet:.4f}, S_sim={ssim:.4f}, R_B3={r:.4f} => {verdict}")

print()
# Check: are there cases where R >= 0.60 but B3 still allows?
b3_r_high_allows = piv_dec.index[(piv_dec['B3']=='allow') & (piv_r['B3'] >= 0.60)]
print(f"Cases R>=0.60 but B3 allows: {len(b3_r_high_allows)}")

print()
print("=== KEY EXPLANATION ===")
print("R = 1 - (1-Sdet)*(1-wsim*Ssim)")
print("For S_det=0.9652, S_sim=0.5699, wsim=0.75:")
wsim = 0.75
sdet = 0.9652
ssim = 0.5699
R_calc = 1 - (1-sdet)*(1-wsim*ssim)
print(f"R = 1 - (1-{sdet})*(1-{wsim}*{ssim}) = {R_calc:.4f}")
print(f"tau*=0.60, so R={R_calc:.4f} >= 0.60 means B3 SHOULD block")
print()
print("For S_det=0.743, S_sim=0.5699:")
sdet2 = 0.743
R_calc2 = 1 - (1-sdet2)*(1-wsim*ssim)
print(f"R = {R_calc2:.4f} < 0.60 => B3 correctly allows")
print()
print("CONCLUSION: The 5 cases with R=0.6095 should have been blocked by B3")
print("=> There may be a calibration scaling issue in the R computation")
print("   OR B1 uses a LOWER threshold (S_det >= 0.54 approx)")
# B1 blocks when S_det >= tau_B1
# tau_B1 = ? For S_det=0.743 to be blocked, tau_B1 <= 0.743
print()
print("B1 threshold inferred: tau_B1 <= 0.743 (since B1 blocks S_det=0.743)")
print("tau_B1 > B0 threshold (0.0) - likely tau_B1 ~ 0.54 from calibration grid")
print()

# E2 case selection Q4
main_csv = pd.read_csv('logs/cases.csv')
e2_before = main_csv[main_csv['experiment_id']=='E2_before']
e2_after  = main_csv[main_csv['experiment_id']=='E2_after']
print("=== Q4: E2 case selection ===")
print(f"E2_before: {len(e2_before)} cases")
print(f"E2_after: {len(e2_after)} cases")
print("Families in E2_before:")
print(e2_before['family'].value_counts().to_dict())
print()
print("=== Q10: n for E3 and ablations ===")
e3 = main_csv[main_csv['experiment_id'].str.startswith('E3')]
e6 = main_csv[main_csv['experiment_id']=='E6_ablation']
print(f"E3 total rows: {len(e3)}")
print(f"E3 per budget: {e3['experiment_id'].value_counts().to_dict()}")
e3_per_budget_per_system = e3.groupby(['experiment_id','system']).size()
print("E3 rows per budget+system (=> n per cell):")
print(e3_per_budget_per_system.to_dict())
print()
print(f"E6 ablation rows: {len(e6)}")
e6_per_config = e6.groupby('system').size()
print("E6 per system config:", e6_per_config.to_dict())
e6_labels = e6['label'].value_counts()
print("E6 labels:", e6_labels.to_dict())
# Infer n_attacks, n_benign in ablation
e6_b3 = e6[e6['system']=='B3']
print(f"E6 B3 attacks: {(e6_b3['label']==1).sum()}, benign: {(e6_b3['label']==0).sum()}")
