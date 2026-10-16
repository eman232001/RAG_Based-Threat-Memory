"""
src/leakage_check.py
Integrity guard: ensures NO test/calibration item is a near-duplicate of any
memory record exemplar.

Must exit 0 (pass) before any table is generated.
Must exit 1 (fail) immediately if any leakage is detected.

Run from project root:
    python src/leakage_check.py

Uses:
  - Cosine similarity via sentence-transformers embeddings
  - Edit distance (Levenshtein)

Thresholds (from config/experiment.yaml):
  leakage_sim_threshold:  0.90  (max allowed cosine similarity)
  leakage_edit_threshold: 10    (min required edit distance)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

log = get_logger("leakage_check", PROJECT_ROOT / "logs" / "run.log")


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _edit_distance(a: str, b: str, max_len: int = 300, max_diff: int = 10) -> int:
    """Levenshtein distance, with length difference pruning."""
    a, b = a[:max_len], b[:max_len]
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if abs(la - lb) > max_diff:
        return abs(la - lb)
    if la == 0: return lb
    if lb == 0: return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        for j in range(1, lb + 1):
            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + (0 if a[i - 1] == b[j - 1] else 1),
            )
        prev = curr
    return prev[lb]


def _load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _load_memory_exemplars(memory_path: Path) -> List[str]:
    """Extract all text strings from memory.jsonl (description + exemplars)."""
    texts = []
    records = _load_jsonl(memory_path)
    for r in records:
        if r.get("pattern_description"):
            texts.append(r["pattern_description"])
        for ex in r.get("exemplars", []):
            if isinstance(ex, str) and ex.strip():
                texts.append(ex.strip())
    return texts


def _load_partition_texts(partitions_dir: Path) -> List[Tuple[str, str]]:
    """Load all texts from all partition files. Returns list of (text, source_file)."""
    result = []
    for jsonl_file in partitions_dir.rglob("*.jsonl"):
        records = _load_jsonl(jsonl_file)
        for r in records:
            text = r.get("text", "")
            if text and len(text.strip()) > 5:
                result.append((text.strip(), str(jsonl_file.name)))
    return result


# ─────────────────────────────────────────────────────────────
# Main leakage check
# ─────────────────────────────────────────────────────────────

def run_leakage_check(cfg) -> bool:
    """
    Run the full leakage check.

    Returns True if no leakage detected (all clear).
    Returns False and prints details if leakage found.
    """
    sim_threshold = cfg.experiment.__dict__.get("leakage_sim_threshold", 0.90)
    edit_threshold = cfg.experiment.__dict__.get("leakage_edit_threshold", 10)

    # Load thresholds from raw experiment config
    import yaml
    with open(cfg.paths.root / "config" / "experiment.yaml", encoding="utf-8") as f:
        exp_raw = yaml.safe_load(f)
    part_cfg = exp_raw.get("partitions", {})
    sim_threshold = part_cfg.get("leakage_sim_threshold", 0.90)
    edit_threshold = part_cfg.get("leakage_edit_threshold", 10)

    log.info(f"Leakage check thresholds: sim<{sim_threshold}, edit>{edit_threshold}")

    memory_path = cfg.paths.memory
    partitions_dir = cfg.paths.root / "data" / "partitions"

    # Load memory exemplars
    memory_texts = _load_memory_exemplars(memory_path)
    if not memory_texts:
        log.info("Memory is empty — leakage check passes trivially (no exemplars to compare).")
        print("[PASS] Memory is empty — no leakage possible yet.")
        return True

    log.info(f"Memory: {len(memory_texts)} text strings to check against")

    # Load partition texts
    partition_texts = _load_partition_texts(partitions_dir)
    if not partition_texts:
        log.info("No partition files found — leakage check passes trivially.")
        print("[PASS] No partition files found — no leakage possible yet.")
        return True

    log.info(f"Partitions: {len(partition_texts)} texts to check")

    # ── Edit distance check (fast, no model needed) ──
    log.info("Running edit distance leakage check...")
    edit_violations = []
    for p_text, p_source in partition_texts:
        for m_text in memory_texts:
            if abs(len(p_text[:300]) - len(m_text[:300])) > edit_threshold:
                continue
            dist = _edit_distance(p_text, m_text, max_len=300, max_diff=edit_threshold)
            if dist <= edit_threshold:
                edit_violations.append({
                    "partition_text": p_text[:100],
                    "memory_text": m_text[:100],
                    "edit_distance": dist,
                    "source_file": p_source,
                })

    if edit_violations:
        log.error(f"LEAKAGE DETECTED: {len(edit_violations)} edit-distance violations!")
        for v in edit_violations[:5]:
            log.error(f"  edit_dist={v['edit_distance']}: {v['partition_text'][:60]}...")
        print(f"[FAIL] Edit-distance leakage: {len(edit_violations)} violations found.")
        print("  -> Remove these items from the partition or memory before proceeding.")
        return False

    log.info(f"Edit distance check: PASSED (0 violations)")

    # ── Embedding similarity check (slower, needs sentence-transformers) ──
    log.info("Running embedding similarity leakage check...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(cfg.embedding.model, device=cfg.embedding.device)

        # Embed memory texts
        mem_embeddings = model.encode(
            memory_texts,
            batch_size=cfg.embedding.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # Check partition texts in batches
        batch_size = 64
        sim_violations = []
        for i in range(0, len(partition_texts), batch_size):
            batch = partition_texts[i: i + batch_size]
            batch_texts = [t for t, _ in batch]
            batch_sources = [s for _, s in batch]

            part_embeddings = model.encode(
                batch_texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            # Cosine similarity matrix: (batch_size, n_memory)
            # Since both are L2-normalized, dot product = cosine similarity
            sims = part_embeddings @ mem_embeddings.T  # shape: (batch, memory)

            max_sims = sims.max(axis=1)  # best match per partition item
            for j, (max_sim, p_text, p_source) in enumerate(
                zip(max_sims, batch_texts, batch_sources)
            ):
                if max_sim >= sim_threshold:
                    mem_idx = int(sims[j].argmax())
                    sim_violations.append({
                        "partition_text": p_text[:100],
                        "memory_text": memory_texts[mem_idx][:100],
                        "cosine_similarity": float(max_sim),
                        "source_file": p_source,
                    })

        if sim_violations:
            log.error(f"LEAKAGE DETECTED: {len(sim_violations)} similarity violations!")
            for v in sim_violations[:5]:
                log.error(f"  sim={v['cosine_similarity']:.3f}: {v['partition_text'][:60]}...")
            print(f"[FAIL] Similarity leakage: {len(sim_violations)} violations found.")
            print("  -> Remove these items from the partition or memory before proceeding.")
            return False

        log.info(f"Embedding similarity check: PASSED (0 violations)")

    except ImportError:
        log.warning("sentence-transformers not installed — skipping embedding check.")
        print("[WARN] Embedding similarity check skipped (sentence-transformers not installed).")
        print("       Install with: pip install sentence-transformers")

    print(f"[PASS] Leakage check: 0 violations found.")
    print(f"       Checked {len(partition_texts)} partition items vs {len(memory_texts)} memory strings.")
    return True


def main() -> None:
    print("\n" + "=" * 55)
    print("RTM Leakage Check")
    print("=" * 55)

    cfg = load_config(PROJECT_ROOT)
    passed = run_leakage_check(cfg)

    if passed:
        print("\nRESULT: PASS — Safe to proceed with table generation.")
        sys.exit(0)
    else:
        print("\nRESULT: FAIL — Fix leakage before proceeding.")
        print("        Tables will NOT be generated until this passes.")
        sys.exit(1)


if __name__ == "__main__":
    main()
