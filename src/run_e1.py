"""
src/run_e1.py
Phase 4 — Section 4.2: Experiment E1 (Main Comparison)

Executes Experiment E1:
  1. Agent-level evaluation on AgentDojo (interactive benchmark tasks with injection attacks)
  2. Item-level static evaluation on held-out test partitions (benign_test + held_out_families)
  3. Supports all 7 primary systems (B0–B6) and all 7 ablation variants
  4. Supports 3 random seeds (1, 2, 3)
  5. Logs all rows atomically to logs/cases.csv with experiment_id="E1"
  6. Employs persistent caching (data/e1_scores_cache.jsonl) for zero redundant API calls

Usage:
  python src/run_e1.py --systems B0,B1,B2,B3,B4,B5,B6 --mode all --seeds 1 2 3
  python src/run_e1.py --systems B0,B2,B3,B6 --mode item --max-items-per-family 50
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional
import uuid

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.case_logger import log_case as _log_case_atomic, deduplicate_cases_csv
from src.rtm_middleware import (
    build_pipeline,
    list_systems,
    PassthroughDetector,
    LLMOnlySystem,
    MemoryOnlySystem,
    RTMFullSystem,
    B4AdaptiveSystem,
    MelonReplicationSystem,
    DebertaSystem,
    B3RandSystem,
    B3LexSystem,
    B3NoCalSystem,
    B3NoSegSystem,
    B3KSystem,
)

log = get_logger("run_e1", PROJECT_ROOT / "logs" / "run.log")

CASES_CSV = PROJECT_ROOT / "logs" / "cases.csv"
CACHE_FILE = PROJECT_ROOT / "data" / "e1_scores_cache.jsonl"
PARTITIONS_DIR = PROJECT_ROOT / "data" / "partitions"
BENIGN_TEST_FILE = PARTITIONS_DIR / "benign_test.jsonl"
HELD_OUT_DIR = PARTITIONS_DIR / "held_out_families"

DEFAULT_SYSTEMS = ["B0", "B1", "B2", "B3", "B4", "B5", "B6"]
ABLATION_SYSTEMS = ["B3_rand", "B3_lex", "B3_no_cal", "B3_no_seg", "B3_k1", "B3_k3", "B3_k10"]
ALL_SYSTEMS = DEFAULT_SYSTEMS + ABLATION_SYSTEMS

AGENT_USER_TASKS = ["user_task_0", "user_task_1", "user_task_2"]
AGENT_INJECTION_TASKS = ["injection_task_0"]
AGENT_SUITE = "banking"
AGENT_SUITE_VERSION = "v1.2.2"


# ─────────────────────────────────────────────────────────────
# Cache management
# ─────────────────────────────────────────────────────────────

def _load_cache() -> Dict[str, dict]:
    cache: Dict[str, dict] = {}
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        k = f"{record.get('system')}:{record.get('sha256')}"
                        cache[k] = record
                    except json.JSONDecodeError:
                        pass
        log.info(f"Loaded {len(cache)} cached item scores from {CACHE_FILE}")
    return cache


def _append_cache(record: dict) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────────
# Detector instantiation helper
# ─────────────────────────────────────────────────────────────

def _create_detector(system: str, cfg, seed: int = 1, user_instruction: str = ""):
    """Instantiate a detector instance for static or agent testing."""
    if system == "B0":
        return PassthroughDetector(experiment_id="E1", seed=seed)
    elif system == "B1":
        return LLMOnlySystem(cfg=cfg, experiment_id="E1", seed=seed, user_instruction=user_instruction)
    elif system == "B2":
        return MemoryOnlySystem(cfg=cfg, experiment_id="E1", seed=seed)
    elif system == "B3":
        return RTMFullSystem(cfg=cfg, experiment_id="E1", seed=seed, user_instruction=user_instruction)
    elif system == "B4":
        return B4AdaptiveSystem(cfg=cfg, experiment_id="E1", seed=seed, user_instruction=user_instruction)
    elif system == "B5":
        return MelonReplicationSystem(cfg=cfg, experiment_id="E1", seed=seed)
    elif system == "B6":
        return DebertaSystem(cfg=cfg, experiment_id="E1", seed=seed)
    elif system == "B3_rand":
        return B3RandSystem(cfg=cfg, experiment_id="E1", seed=seed, user_instruction=user_instruction)
    elif system == "B3_lex":
        return B3LexSystem(cfg=cfg, experiment_id="E1", seed=seed, user_instruction=user_instruction)
    elif system == "B3_no_cal":
        return B3NoCalSystem(cfg=cfg, experiment_id="E1", seed=seed, user_instruction=user_instruction)
    elif system == "B3_no_seg":
        return B3NoSegSystem(cfg=cfg, experiment_id="E1", seed=seed, user_instruction=user_instruction)
    elif system in ("B3_k1", "B3_k3", "B3_k10"):
        k_map = {"B3_k1": 1, "B3_k3": 3, "B3_k10": 10}
        return B3KSystem(cfg=cfg, experiment_id="E1", seed=seed, user_instruction=user_instruction, k=k_map[system])
    else:
        raise ValueError(f"Unknown system: {system}")


# ─────────────────────────────────────────────────────────────
# 1. Agent-level evaluation on AgentDojo
# ─────────────────────────────────────────────────────────────

def run_agent_evaluation(
    systems: List[str],
    seeds: List[int],
    cfg,
    user_tasks: List[str] = AGENT_USER_TASKS,
    injection_tasks: List[str] = AGENT_INJECTION_TASKS,
) -> int:
    """
    Run requested systems on AgentDojo benchmark tasks across seeds.
    Returns number of rows logged.
    """
    log.info("Starting AgentDojo interactive evaluation (E1)...")
    try:
        from agentdojo.task_suite.load_suites import get_suite
        from agentdojo.attacks.baseline_attacks import DirectAttack
        from agentdojo.benchmark import benchmark_suite_with_injections
        from agentdojo.logging import NullLogger
        NullLogger.logdir = None
        from google import genai
        genai_client = genai.Client(api_key=cfg.secrets.gemini_api_key)
    except Exception as e:
        log.warning(f"Could not initialize AgentDojo environment: {e}. Skipping agent evaluation.")
        return 0

    suite = get_suite(AGENT_SUITE_VERSION, AGENT_SUITE)
    attack = DirectAttack()
    rows_logged = 0

    for seed in seeds:
        for sys_name in systems:
            log.info(f"Running AgentDojo for system={sys_name}, seed={seed}...")
            try:
                pipeline, detector = build_pipeline(
                    system=sys_name,
                    cfg=cfg,
                    genai_client=genai_client,
                    experiment_id="E1",
                    seed=seed,
                )
                results = benchmark_suite_with_injections(
                    pipeline=pipeline,
                    suite=suite,
                    attack=attack,
                    user_tasks=user_tasks,
                    injection_tasks=injection_tasks,
                    logger=NullLogger(),
                )

                for ut in user_tasks:
                    for it in injection_tasks:
                        task_id = f"banking_{ut}_{it}"
                        task_res = results.get((ut, it), {})
                        attack_succ = 1 if task_res.get("security", False) else 0
                        task_succ = 1 if task_res.get("utility", False) else 0

                        # Call log_case on detector if it has last_result
                        detector.task_id = task_id
                        detector.log_case(
                            task_id=task_id,
                            attack_success=attack_succ,
                            task_success=task_succ,
                            extra={
                                "partition": "agentdojo_banking",
                                "dataset": "AgentDojo",
                                "family": "banking_injection",
                                "label": 1,
                            },
                        )
                        rows_logged += 1
                        log.info(f"Logged agent row: sys={sys_name}, task={task_id}, atk_succ={attack_succ}")
            except Exception as ex:
                log.error(f"Error running agentdojo for {sys_name} seed {seed}: {ex}")

    log.info(f"AgentDojo evaluation complete: {rows_logged} rows logged.")
    return rows_logged


# ─────────────────────────────────────────────────────────────
# 2. Item-level static evaluation
# ─────────────────────────────────────────────────────────────

def load_test_items(max_per_family: Optional[int] = None) -> List[dict]:
    """Load items from benign_test.jsonl and all held_out_families/*.jsonl."""
    items: List[dict] = []

    # Benign items
    if BENIGN_TEST_FILE.exists():
        with open(BENIGN_TEST_FILE, "r", encoding="utf-8") as f:
            benign_items = [json.loads(line) for line in f if line.strip()]
            if max_per_family is not None and max_per_family > 0:
                benign_items = benign_items[:max_per_family]
            items.extend(benign_items)
            log.info(f"Loaded {len(benign_items)} items from benign_test.jsonl")

    # Held-out attack families
    if HELD_OUT_DIR.exists():
        for jsonl_path in sorted(HELD_OUT_DIR.glob("*.jsonl")):
            with open(jsonl_path, "r", encoding="utf-8") as f:
                family_items = [json.loads(line) for line in f if line.strip()]
                if max_per_family is not None and max_per_family > 0:
                    family_items = family_items[:max_per_family]
                items.extend(family_items)
                log.info(f"Loaded {len(family_items)} items from {jsonl_path.name}")

    return items


def run_item_evaluation(
    systems: List[str],
    seeds: List[int],
    cfg,
    max_per_family: Optional[int] = None,
) -> int:
    """
    Run item-level evaluation for all systems across seeds on held-out test partitions.
    Uses caching to avoid duplicate inference.
    """
    test_items = load_test_items(max_per_family=max_per_family)
    if not test_items:
        log.warning("No test items found. Skipping item evaluation.")
        return 0

    log.info(f"Starting item-level static evaluation (E1) on {len(test_items)} items...")
    cache = _load_cache()
    rows_logged = 0

    for sys_name in systems:
        log.info(f"--- Running static item evaluation for system={sys_name} ---")
        try:
            detector = _create_detector(sys_name, cfg=cfg, seed=1)
        except Exception as e:
            log.error(f"Could not initialize detector for {sys_name}: {e}")
            continue

        for item in test_items:
            sha = item.get("sha256") or str(uuid.uuid4())
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
            else:
                try:
                    t0 = time.perf_counter()
                    is_inj, score = detector.detect(text)
                    lat = (time.perf_counter() - t0) * 1000.0

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
                except Exception as e:
                    log.warning(f"Detection failed on item {source_id} for {sys_name}: {e}")
                    continue

            # Log a row for each seed
            for seed in seeds:
                row = {
                    "experiment_id": "E1",
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
                    "decision": res.get("decision", "allow"),
                    "attack_success": "NA",
                    "task_success": "NA",
                    "latency_ms": res.get("latency_ms", "NA"),
                    "model_calls": res.get("model_calls", 0),
                    "retrieval_calls": res.get("retrieval_calls", 0),
                    "tokens_cost": res.get("tokens_cost", 0.0),
                    "record_id": res.get("record_id", ""),
                    "segment": res.get("segment", ""),
                    "explanation_json": res.get("explanation_json", ""),
                }
                _log_case_atomic(row)
                rows_logged += 1

    log.info(f"Static item evaluation complete: {rows_logged} rows logged.")
    return rows_logged


# ─────────────────────────────────────────────────────────────
# Main execution
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 — Section 4.2: Experiment E1 Runner")
    parser.add_argument(
        "--systems",
        type=str,
        default=",".join(DEFAULT_SYSTEMS),
        help=f"Comma-separated list of systems to run. 'all' runs all 14 systems. (Default: {','.join(DEFAULT_SYSTEMS)})",
    )
    parser.add_argument(
        "--mode",
        choices=["agent", "item", "all"],
        default="all",
        help="Evaluation mode: 'agent', 'item', or 'all' (Default: all)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="Random seeds to run (Default: 1 2 3)",
    )
    parser.add_argument(
        "--max-items-per-family",
        type=int,
        default=None,
        help="Max items per family for static evaluation (useful for quick evaluation or testing)",
    )
    args = parser.parse_args()

    cfg = load_config(PROJECT_ROOT)

    if args.systems.strip().lower() == "all":
        target_systems = ALL_SYSTEMS
    else:
        target_systems = [s.strip() for s in args.systems.split(",") if s.strip()]

    print("\n" + "=" * 65)
    print(" Phase 4 — Experiment E1: Main Comparison")
    print("=" * 65)
    print(f" Systems : {target_systems}")
    print(f" Mode    : {args.mode}")
    print(f" Seeds   : {args.seeds}")
    if args.max_items_per_family:
        print(f" Max/Fam : {args.max_items_per_family}")
    print("=" * 65 + "\n")

    t_start = time.perf_counter()
    agent_rows = 0
    item_rows = 0

    if args.mode in ("agent", "all"):
        agent_rows = run_agent_evaluation(
            systems=target_systems,
            seeds=args.seeds,
            cfg=cfg,
        )

    if args.mode in ("item", "all"):
        item_rows = run_item_evaluation(
            systems=target_systems,
            seeds=args.seeds,
            cfg=cfg,
            max_per_family=args.max_items_per_family,
        )

    deduplicate_cases_csv()
    elapsed = time.perf_counter() - t_start

    print("\n" + "=" * 65)
    print(" Experiment E1 Run Summary")
    print("=" * 65)
    print(f" Total Agent Rows Logged : {agent_rows}")
    print(f" Total Item Rows Logged  : {item_rows}")
    print(f" Total Rows Logged       : {agent_rows + item_rows}")
    print(f" Elapsed Time            : {elapsed:.2f}s")
    print(f" cases.csv path          : {CASES_CSV}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
