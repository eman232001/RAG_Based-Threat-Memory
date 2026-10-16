import sys, io, pandas as pd, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

df = pd.read_csv('logs/cases_e1_expanded.csv')
attacks = df[df['label']==1]
benign  = df[df['label']==0]

print("=" * 70)
print("Q1: Seeds with temperature=0 - what do they control?")
print("=" * 70)
# Seeds in the codebase are experiment iteration identifiers for data shuffling
# and random sampling (not LLM stochasticity - temp=0 is fully deterministic)
# From config_loader.py: seeds = [1, 2, 3] default, MELON uses 3 seeds
# Seeds control: (a) random order of items in evaluation loop,
#               (b) bootstrap resampling for CIs in E2,
#               (c) MELON trajectory re-execution ordering
# With temp=0, LLM outputs are fully deterministic; seeds only affect sampling order
print("Seeds control: (a) evaluation item ordering, (b) bootstrap CI resampling, (c) MELON task ordering")
print("With temperature=0, LLM outputs are deterministic - seeds do NOT affect model predictions")
print("Variance across seeds in E1 would be zero (same items, same LLM outputs)")
print()

print("=" * 70)
print("Q2: B1 threshold - why can it block attacks B3 misses?")
print("=" * 70)
# From Soft-OR: R = 1 - (1-Sdet)*(1-wsim*Ssim)
# R >= Sdet always because wsim*Ssim >= 0
# BUT: B1 uses Sdet alone with threshold tau_B1, while B3 uses R with tau*=0.60
# If tau_B1 is LOWER than tau*, then B1 can block items B3 allows
b3_df = attacks[attacks['system']=='B3']
b1_df = attacks[attacks['system']=='B1']
# Check raw detector scores for cases B3 allows but B1 blocks
piv = attacks.pivot(index='case_id', columns='system', values='S_det')
piv_dec = attacks.pivot(index='case_id', columns='system', values='decision')
b3_misses_b1_blocks = piv_dec.index[(piv_dec['B3'] == 'allow') & (piv_dec['B1'] == 'block')]
if len(b3_misses_b1_blocks) > 0:
    scores = piv.loc[b3_misses_b1_blocks]
    print(f"Cases where B3 allows but B1 blocks (n={len(b3_misses_b1_blocks)}):")
    print(f"  B1 S_det scores: {scores['B1'].values}")
    print(f"  B3 S_det scores (same): {scores['B3'].values}")
    # Now check B3's R scores for these cases
    piv_r = attacks.pivot(index='case_id', columns='system', values='R')
    r_scores = piv_r.loc[b3_misses_b1_blocks, 'B3']
    sim_scores = attacks.pivot(index='case_id', columns='system', values='S_sim').loc[b3_misses_b1_blocks, 'B3']
    print(f"  B3 R (fused) scores: {r_scores.values}")
    print(f"  B3 S_sim scores: {sim_scores.values}")
else:
    print("No cases found - checking pivot correctly...")
    print("Sample of attack decisions:", piv_dec.head())
print()

print("=" * 70)
print("Q5: Stage 1 resolution rate - what fraction goes to B1 vs FAISS only?")
print("=" * 70)
b3_all = df[df['system']=='B3']
# Stage 1 blocks when S_sim >= tau_block (0.80)
# Stage 2 needed when S_sim < tau_block  
tau_block = 0.80
b3_attacks_only = b3_all[b3_all['label']==1]
stage1_resolved = (b3_attacks_only['S_sim'] >= tau_block).sum()
total_attacks = len(b3_attacks_only)
print(f"Attacks resolved by Stage 1 (S_sim >= {tau_block}): {stage1_resolved}/{total_attacks} ({stage1_resolved/total_attacks*100:.1f}%)")
print(f"Attacks requiring Stage 2 (LLM call): {total_attacks-stage1_resolved}/{total_attacks} ({(total_attacks-stage1_resolved)/total_attacks*100:.1f}%)")
# Benign Stage 1 resolution
b3_benign_only = b3_all[b3_all['label']==0]
stage1_benign = (b3_benign_only['S_sim'] >= tau_block).sum()
print(f"Benign items reaching Stage 2: {len(b3_benign_only)-stage1_benign}/{len(b3_benign_only)}")
print()
# Live latency estimate
p1 = stage1_resolved / total_attacks
t1_local = 22.0  # ms (FAISS only)
t2_llm = 1719.9  # ms (LLM)
t_live = p1 * t1_local + (1-p1) * (t1_local + t2_llm)
print(f"Expected live latency = {p1:.3f}*{t1_local} + {(1-p1):.3f}*{t1_local+t2_llm:.1f} = {t_live:.1f} ms")
print()

print("=" * 70)
print("Q7: FPR breakdown - Hard Negatives vs Regular Benign")
print("=" * 70)
b3_benign = benign[benign['system']=='B3']
b3_hn = b3_benign[b3_benign['dataset']=='HardNeg']
b3_reg = b3_benign[b3_benign['dataset'] != 'HardNeg']
hn_fpr = (b3_hn['decision']=='block').mean()
reg_fpr = (b3_reg['decision']=='block').mean()
print(f"Hard Negatives (n={len(b3_hn)}): FPR = {hn_fpr*100:.1f}% ({(b3_hn['decision']=='block').sum()}/{len(b3_hn)})")
print(f"Regular Benign (n={len(b3_reg)}): FPR = {reg_fpr*100:.1f}% ({(b3_reg['decision']=='block').sum()}/{len(b3_reg)})")
total_fpr = (b3_benign['decision']=='block').mean()
print(f"Overall FPR (n={len(b3_benign)}): {total_fpr*100:.1f}%")
print()

print("=" * 70)
print("Q8: Reconcile 809/837 vs 1,990")
print("=" * 70)
main_csv = pd.read_csv('logs/cases.csv')
print(f"cases.csv rows: {len(main_csv)}")
print(f"cases_e1_expanded.csv rows: {len(df)}")
print(f"cases.csv unique (exp, system, task, seed) combos: {main_csv[['experiment_id','system','task_id','seed']].drop_duplicates().shape[0]}")
print(f"Experiments in cases.csv: {main_csv['experiment_id'].value_counts().to_dict()}")
print()

print("=" * 70)
print("Q9: Calibration method")
print("=" * 70)
print("From config/models.yaml - CalibrationConfig.calibration_method = 'isotonic'")
print("=> Isotonic regression is the primary calibration method (NOT Platt scaling)")
print("Note: The paper says 'Platt scaling AND isotonic regression' but code uses only isotonic")
