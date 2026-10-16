"""
src/summarize_expanded.py
Generates official statistical report and table summary from logs/cases_e1_expanded.csv.
"""
import json
import math
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPANDED_CSV = PROJECT_ROOT / "logs" / "cases_e1_expanded.csv"
OUTPUT_JSON = PROJECT_ROOT / "outputs" / "phase1_e1_expanded_summary.json"


def wilson_ci(k: int, n: int, z: float = 1.95996):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + (z ** 2) / n
    center = (p + (z ** 2) / (2 * n)) / denom
    spread = (z / denom) * math.sqrt((p * (1 - p) / n) + ((z ** 2) / (4 * (n ** 2))))
    lo = max(0.0, center - spread)
    hi = min(1.0, center + spread)
    return round(lo * 100, 2), round(hi * 100, 2)


def main():
    if not EXPANDED_CSV.exists():
        print(f"Error: {EXPANDED_CSV} does not exist.")
        return

    df = pd.read_csv(EXPANDED_CSV)
    summary = {}

    for sys_name, group in df.groupby("system"):
        attacks = group[group["label"] == 1]
        benigns = group[group["label"] == 0]

        n_attacks = len(attacks)
        n_benign = len(benigns)

        fn = sum(1 for _, r in attacks.iterrows() if r["decision"] == "allow")
        fp = sum(1 for _, r in benigns.iterrows() if r["decision"] == "block")

        asr = (fn / n_attacks) * 100.0 if n_attacks else 0.0
        asr_ci = wilson_ci(fn, n_attacks)

        fpr = (fp / n_benign) * 100.0 if n_benign else 0.0
        fpr_ci = wilson_ci(fp, n_benign)

        correct = (n_attacks - fn) + (n_benign - fp)
        acc = (correct / len(group)) * 100.0
        acc_ci = wilson_ci(correct, len(group))

        lats = group["latency_ms"].dropna().astype(float)
        p50 = float(lats.median()) if len(lats) else 0.0
        p95 = float(lats.quantile(0.95)) if len(lats) else 0.0

        summary[sys_name] = {
            "system": sys_name,
            "n_attacks": n_attacks,
            "n_benign": n_benign,
            "asr_percent": round(asr, 2),
            "asr_ci_95": asr_ci,
            "fpr_percent": round(fpr, 2),
            "fpr_ci_95": fpr_ci,
            "accuracy_percent": round(acc, 2),
            "accuracy_ci_95": acc_ci,
            "median_latency_ms": round(p50, 1),
            "p95_latency_ms": round(p95, 1),
        }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(" SUMMARY OF PHASE 1 EXPANDED EVALUATION (n=1,673 attacks, n=317 benign)")
    print("=" * 80)
    print(f"{'System':<8} | {'ASR (%)':<9} | {'ASR 95% Wilson CI':<19} | {'FPR (%)':<9} | {'FPR 95% Wilson CI':<19} | {'Median Lat (ms)':<15}")
    print("-" * 80)
    for s in ["B0", "B2", "B6"]:
        if s in summary:
            v = summary[s]
            ci_asr = f"[{v['asr_ci_95'][0]}%, {v['asr_ci_95'][1]}%]"
            ci_fpr = f"[{v['fpr_ci_95'][0]}%, {v['fpr_ci_95'][1]}%]"
            print(f"{s:<8} | {v['asr_percent']:<9.2f} | {ci_asr:<19} | {v['fpr_percent']:<9.2f} | {ci_fpr:<19} | {v['median_latency_ms']:<15.1f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
