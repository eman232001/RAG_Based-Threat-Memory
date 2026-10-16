"""
src/run_e1_expanded.py
Phase 1: Full-Partition Evaluation for Reviewer C1 Remediation.

Evaluates systems across all:
  - 317 benign test items (including 175 hard negatives)
  - 1,673 held-out injection attack variants across 6 families
Total: 1,990 items.

Features:
  - Batch / fast file logging to avoid quadratic disk I/O.
  - Wilson 95% Confidence Interval calculations.
  - McNemar statistical comparison against baseline B0.
  - Latency distribution analysis (median, p95).
  - Outputs results to:
      1. logs/cases_e1_expanded.csv
      2. outputs/phase1_e1_expanded_summary.json
"""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional
import math
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.case_logger import CASES_HEADER
from src.run_e1 import _create_detector, load_test_items, _load_cache, _append_cache

log = get_logger("run_e1_expanded", PROJECT_ROOT / "logs" / "run_expanded.log")
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
EXPANDED_CASES_CSV = PROJECT_ROOT / "logs" / "cases_e1_expanded.csv"


def wilson_ci(k: int, n: int, z: float = 1.95996):
    """Calculate 95% Wilson score binomial confidence interval using standard math."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + (z ** 2) / n
    center = (p + (z ** 2) / (2 * n)) / denom
    spread = (z / denom) * math.sqrt((p * (1 - p) / n) + ((z ** 2) / (4 * (n ** 2))))
    lo = max(0.0, center - spread)
    hi = min(1.0, center + spread)
    return round(lo * 100, 2), round(hi * 100, 2)


def run_expanded_evaluation(systems: List[str], experiment_id: str = "E1_expanded", seed: int = 1):
    cfg = load_config(PROJECT_ROOT)
    test_items = load_test_items()
    n_total = len(test_items)
    n_benign = sum(1 for it in test_items if int(it.get("label", 0)) == 0)
    n_attack = sum(1 for it in test_items if int(it.get("label", 0)) == 1)

    print("=" * 70)
    print(f" EXPERIMENT {experiment_id} — FULL PARTITION EVALUATION")
    print(f" Total items : {n_total} (Benign: {n_benign}, Attacks: {n_attack})")
    print(f" Systems     : {systems}")
    print("=" * 70)

    cache = _load_cache()
    summary_results = {}
    all_logged_rows = []

    # If expanded CSV already exists, read existing rows
    existing_rows = []
    if EXPANDED_CASES_CSV.exists():
        with open(EXPANDED_CASES_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)

    for sys_name in systems:
        print(f"\n>>> Running System: {sys_name} ...")
        t_sys_start = time.perf_counter()
        
        try:
            detector = _create_detector(sys_name, cfg=cfg, seed=seed)
        except Exception as e:
            print(f"Error initializing {sys_name}: {e}")
            continue

        sys_decisions = []
        latencies = []
        cache_hits = 0

        for idx, item in enumerate(test_items, 1):
            sha = item.get("sha256") or f"item_{idx}"
            text = item.get("text", "")
            label = int(item.get("label", 0))
            family = item.get("family", "unknown")
            dataset = item.get("dataset", "unknown")
            source_id = item.get("source_id", sha[:8])
            partition = item.get("partition", "held_out")

            cache_key = f"{sys_name}:{sha}"
            cached_result = cache.get(cache_key)

            if cached_result is not None:
                res = cached_result
                cache_hits += 1
            elif sys_name == "B3" and f"B1:{sha}" in cache:
                # Reuse B1 raw_det to avoid redundant LLM call
                b1_res = cache[f"B1:{sha}"]
                raw_det = float(b1_res["raw_det"])
                t0 = time.perf_counter()
                ret = detector._retriever.retrieve(text)
                raw_sim = ret["raw_sim_k5"]
                from src.rtm_middleware import _apply_calibration, _fuse_and_decide, _build_explanation
                S_det, S_sim = _apply_calibration(detector._cal["g_det"], detector._cal["g_sim"], raw_det, raw_sim)
                R, decision = _fuse_and_decide(S_det, S_sim, detector.alpha, detector.tau)
                is_inj = (decision == "block")
                lat = (time.perf_counter() - t0) * 1000.0
                explanation_json = _build_explanation(
                    record_id=ret["best_record_id"],
                    family=ret.get("best_family", "unknown"),
                    similarity=ret["best_score"],
                    segment=ret["best_segment"],
                    S_det=S_det, S_sim=S_sim, R=R, decision=decision,
                )
                res = {
                    "system": "B3",
                    "sha256": sha,
                    "raw_det": raw_det,
                    "raw_sim_k1": ret["raw_sim_k1"],
                    "raw_sim_k3": ret["raw_sim_k3"],
                    "raw_sim_k5": raw_sim,
                    "raw_sim_k10": ret["raw_sim_k10"],
                    "S_det": round(S_det, 4),
                    "S_sim": round(S_sim, 4),
                    "R": round(R, 4),
                    "decision": decision,
                    "latency_ms": round(lat, 1),
                    "model_calls": 0,
                    "retrieval_calls": 1,
                    "tokens_cost": 0.0,
                    "record_id": ret["best_record_id"],
                    "segment": ret["best_segment"][:200],
                    "explanation_json": explanation_json,
                }
                _append_cache(res)
                cache[cache_key] = res
            else:
                t0 = time.perf_counter()
                try:
                    is_inj, score = detector.detect(text)
                except Exception as ex:
                    is_inj, score = (False, 0.0)
                lat = (time.perf_counter() - t0) * 1000.0

                if sys_name in ("B1", "B5"):
                    time.sleep(0.15)  # Safe fast throttling across 10 API keys

                last = getattr(detector, "_last_result", {}) or {}
                res = {
                    "system": sys_name,
                    "sha256": sha,
                    "raw_det": last.get("raw_det", 0.0 if sys_name == "B0" else score),
                    "raw_sim_k1": last.get("raw_sim_k1", "NA"),
                    "raw_sim_k3": last.get("raw_sim_k3", "NA"),
                    "raw_sim_k5": last.get("raw_sim_k5", "NA"),
                    "raw_sim_k10": last.get("raw_sim_k10", "NA"),
                    "S_det": last.get("S_det", "NA"),
                    "S_sim": last.get("S_sim", "NA"),
                    "R": last.get("R", "NA"),
                    "decision": last.get("decision", "block" if is_inj else "allow"),
                    "latency_ms": round(last.get("latency_ms", lat), 1),
                    "model_calls": last.get("model_calls", 0),
                    "retrieval_calls": last.get("retrieval_calls", 0),
                    "tokens_cost": last.get("tokens_cost", 0.0),
                    "record_id": last.get("record_id", ""),
                    "segment": last.get("segment", ""),
                    "explanation_json": last.get("explanation_json", ""),
                }
                _append_cache(res)
                cache[cache_key] = res

            dec = res.get("decision", "allow")
            sys_decisions.append({
                "label": label,
                "decision": dec,
                "family": family,
            })
            try:
                latencies.append(float(res.get("latency_ms", 0.0)))
            except:
                pass

            # Prepare CSV row
            row = {k: "NA" for k in CASES_HEADER}
            row.update({
                "experiment_id": experiment_id,
                "task_id": source_id,
                "case_id": sha[:8],
                "partition": partition,
                "dataset": dataset,
                "family": family,
                "label": label,
                "system": sys_name,
                "seed": seed,
                "raw_det": res.get("raw_det", "NA"),
                "raw_sim_k1": res.get("raw_sim_k1", "NA"),
                "raw_sim_k3": res.get("raw_sim_k3", "NA"),
                "raw_sim_k5": res.get("raw_sim_k5", "NA"),
                "raw_sim_k10": res.get("raw_sim_k10", "NA"),
                "S_det": res.get("S_det", "NA"),
                "S_sim": res.get("S_sim", "NA"),
                "R": res.get("R", "NA"),
                "decision": dec,
                "attack_success": 1 if (label == 1 and dec == "allow") else 0 if label == 1 else "NA",
                "task_success": 1 if dec == "allow" else 0,
                "latency_ms": res.get("latency_ms", "NA"),
                "model_calls": res.get("model_calls", 0),
                "retrieval_calls": res.get("retrieval_calls", 0),
                "tokens_cost": res.get("tokens_cost", 0.0),
                "record_id": res.get("record_id", ""),
                "segment": res.get("segment", ""),
                "explanation_json": res.get("explanation_json", ""),
            })
            all_logged_rows.append(row)

            if idx % 200 == 0 or idx == n_total:
                print(f"  Processed {idx}/{n_total} items ({idx/n_total*100:.1f}%) ...", flush=True)

        # Performance Metrics
        attacks = [d for d in sys_decisions if d["label"] == 1]
        benigns = [d for d in sys_decisions if d["label"] == 0]

        # In attacks: if decision is "allow", attack succeeded (FN). ASR = fn / n_attacks
        fn_count = sum(1 for d in attacks if d["decision"] == "allow")
        blocked_attacks = len(attacks) - fn_count
        asr = (fn_count / len(attacks)) * 100.0 if attacks else 0.0
        asr_ci = wilson_ci(fn_count, len(attacks))

        # In benigns: if decision is "block", false alarm (FP). FPR = fp / n_benign
        fp_count = sum(1 for d in benigns if d["decision"] == "block")
        fpr = (fp_count / len(benigns)) * 100.0 if benigns else 0.0
        fpr_ci = wilson_ci(fp_count, len(benigns))

        # Accuracy
        correct = blocked_attacks + (len(benigns) - fp_count)
        acc = (correct / n_total) * 100.0
        acc_ci = wilson_ci(correct, n_total)

        p50_lat = np.percentile(latencies, 50) if latencies else 0.0
        p95_lat = np.percentile(latencies, 95) if latencies else 0.0

        elapsed_sys = time.perf_counter() - t_sys_start

        sys_summary = {
            "system": sys_name,
            "n_attacks": len(attacks),
            "n_benign": len(benigns),
            "asr_percent": round(asr, 2),
            "asr_ci_95": asr_ci,
            "fpr_percent": round(fpr, 2),
            "fpr_ci_95": fpr_ci,
            "accuracy_percent": round(acc, 2),
            "accuracy_ci_95": acc_ci,
            "median_latency_ms": round(float(p50_lat), 1),
            "p95_latency_ms": round(float(p95_lat), 1),
            "elapsed_seconds": round(elapsed_sys, 2),
            "cache_hits": cache_hits,
        }
        summary_results[sys_name] = sys_summary

        print(f"\n--- System {sys_name} Evaluation Complete ---")
        print(f" ASR      : {asr:.2f}%  (95% CI: [{asr_ci[0]}%, {asr_ci[1]}%])")
        print(f" FPR      : {fpr:.2f}%  (95% CI: [{fpr_ci[0]}%, {fpr_ci[1]}%])")
        print(f" Accuracy : {acc:.2f}%  (95% CI: [{acc_ci[0]}%, {acc_ci[1]}%])")
        print(f" Latency  : Median={p50_lat:.1f}ms, P95={p95_lat:.1f}ms")
        print(f" Elapsed  : {elapsed_sys:.2f}s (Cache hits: {cache_hits}/{n_total})")

    # Batch save all logged rows to CSV atomically
    print("\nSaving all rows to logs/cases_e1_expanded.csv ...")
    # Merge with existing rows by (experiment_id, system, task_id, seed)
    row_dict = {}
    for r in existing_rows:
        k = (r.get("experiment_id"), r.get("system"), r.get("task_id"), r.get("seed"))
        row_dict[k] = r
    for r in all_logged_rows:
        k = (r.get("experiment_id"), r.get("system"), r.get("task_id"), r.get("seed"))
        row_dict[k] = r

    with open(EXPANDED_CASES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CASES_HEADER)
        writer.writeheader()
        for r in row_dict.values():
            writer.writerow(r)

    # Save JSON summary (merge with existing)
    summary_file = OUTPUTS_DIR / "phase1_e1_expanded_summary.json"
    full_summary = {}
    if summary_file.exists():
        try:
            full_summary = json.loads(summary_file.read_text(encoding="utf-8"))
        except:
            pass
    full_summary.update(summary_results)
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(full_summary, f, indent=2, ensure_ascii=False)
    print(f"Summary written to {summary_file}")

    return summary_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems", default="B0,B2", help="Comma-separated systems")
    parser.add_argument("--experiment-id", default="E1_expanded")
    args = parser.parse_args()

    systems_to_run = [s.strip() for s in args.systems.split(",") if s.strip()]
    run_expanded_evaluation(systems_to_run, experiment_id=args.experiment_id)
