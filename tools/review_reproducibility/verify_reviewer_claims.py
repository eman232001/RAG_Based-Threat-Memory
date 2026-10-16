import sys, io, pandas as pd, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

df = pd.read_csv('logs/cases_e1_expanded.csv')
benign = df[df['label']==0]
attacks = df[df['label']==1]

b3_benign = benign[benign['system']=='B3']
b1_benign = benign[benign['system']=='B1']

print("=== C7: TSR vs FPR mismatch ===")
print("Benign datasets:", benign['dataset'].value_counts().to_dict())
hn_b3 = b3_benign[b3_benign['dataset']=='HardNeg']
print(f"HardNeg blocked by B3: {(hn_b3['decision']=='block').sum()}/175")
b3_fpr = (b3_benign['decision']=='block').mean()
b1_fpr = (b1_benign['decision']=='block').mean()
print(f"B3 actual FPR test = {b3_fpr*100:.2f}%")
print(f"B1 actual FPR test = {b1_fpr*100:.2f}%")
print("=> C7 VALID - TSR and FPR are measured on different sets (TSR=utility tasks, FPR=content items)")
print()

print("=== C8: Calibration gap ===")
print("Calibration target FPR <= 1%, but test FPR = 16.1%")
print("=> REVIEWER CORRECT: large unexplained calibration-test gap")
print()

print("=== C3: B5 Accuracy ===")
tp,fn,fp,tn = 21,3,1,19
acc = (tp+tn)/(tp+fn+fp+tn)
print(f"B5: TP={tp},FN={fn},FP={fp},TN={tn}")
print(f"Correct accuracy = {acc*100:.1f}%")
print(f"Table shows 90.0%, reviewer says 90.9% - REVIEWER IS CORRECT")
print()

print("=== C5: E1 vs E2 contradiction ===")
print("E1: B3 ASR on held-out attacks = 6.3% (memory zero-day rate)")
print("E2: ASR_before = 88.9% (same 6 families, 18 cases)")
print("This is because E2 evaluates B3 WITHOUT memory for those families (leave-one-out)")
print("E1 evaluates B3 WITH full memory => different conditions")
print("=> REVIEWER C5 IS PARTIALLY VALID: needs explicit clarification in paper")
print()

print("=== C1 CRITICAL SUMMARY ===")
print("McNemar result: B1 blocked 9 attacks B3 missed, B3 blocked 0 B1 missed")
print("This means B1 is STRICTLY SUPERIOR to B3 on these 9 attacks")
print("The paper claims 'B3 achieves statistically significant superiority over B1'")
print("=> This is FACTUALLY WRONG. The reviewer is 100% correct on C1.")
