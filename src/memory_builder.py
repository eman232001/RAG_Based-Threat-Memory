"""
src/memory_builder.py
Phase 3 component (skeleton defined in Phase 1 for import completeness).

Builds and manages the Verified Threat Memory:
  - Loads records from data/memory.jsonl
  - Segments text into 3-sentence windows (stride 1, <=128 tokens)
  - Embeds with all-MiniLM-L6-v2
  - Indexes with FAISS IndexFlatIP (cosine on L2-normalized vectors)

The memory is READ-ONLY at test time. Only ingest_memory.py adds records.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_config, Config
from src.utils.logger import get_logger

log = get_logger("memory_builder", PROJECT_ROOT / "logs" / "run.log")


# ─────────────────────────────────────────────────────────────
# Text segmenter
# ─────────────────────────────────────────────────────────────

def segment_text(
    text: str,
    window_size: int = 3,
    stride: int = 1,
    max_tokens: int = 128,
) -> List[str]:
    """
    Segment text into overlapping sentence windows.

    Architecture spec:
      - Window size: 3 sentences
      - Stride: 1 sentence
      - Max tokens per window: 128 (truncate at char level proxy)

    Args:
        text: Input text to segment.
        window_size: Number of sentences per window.
        stride: Stride in sentences between windows.
        max_tokens: Max tokens per window (approximate: 4 chars/token).

    Returns:
        List of text windows.
    """
    try:
        import nltk
        # Download punkt if needed (silent)
        try:
            sentences = nltk.sent_tokenize(text)
        except LookupError:
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
            sentences = nltk.sent_tokenize(text)
    except ImportError:
        # Fallback: split on '. ' if NLTK not available
        sentences = [s.strip() for s in text.split(". ") if s.strip()]

    if not sentences:
        return [text[:max_tokens * 4]]  # Single window for very short text

    windows = []
    max_chars = max_tokens * 4  # Approximate char limit

    for i in range(0, max(1, len(sentences) - window_size + 1), stride):
        chunk = sentences[i: i + window_size]
        window_text = " ".join(chunk)
        # Truncate to max_tokens
        if len(window_text) > max_chars:
            window_text = window_text[:max_chars]
        if window_text.strip():
            windows.append(window_text.strip())

    # Ensure at least one window
    if not windows:
        windows = [text[: max_chars]]

    return windows


# ─────────────────────────────────────────────────────────────
# Memory loader
# ─────────────────────────────────────────────────────────────

def load_memory_records(memory_path: Path) -> List[Dict]:
    """Load verified records from data/memory.jsonl."""
    if not memory_path.exists() or memory_path.stat().st_size == 0:
        log.warning(f"Memory file is empty or missing: {memory_path}")
        return []

    records = []
    with open(memory_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    log.warning(f"Skipping malformed memory record: {e}")

    log.info(f"Loaded {len(records)} memory records from {memory_path}")
    return records


# ─────────────────────────────────────────────────────────────
# FAISS index builder
# ─────────────────────────────────────────────────────────────

class MemoryIndex:
    """
    Encapsulates the FAISS index and metadata for threat memory retrieval.

    Architecture spec:
      - FAISS IndexFlatIP (exact inner product = cosine on L2-normalized vecs)
      - Embedding model: all-MiniLM-L6-v2 on CPU
      - Metadata stored in parallel list (index i = faiss entry i)
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._index = None
        self._metadata: List[Dict] = []  # One entry per indexed segment
        self._model = None

    def _get_model(self):
        """Lazy-load embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(
                    self.cfg.embedding.model,
                    device=self.cfg.embedding.device,
                )
                log.info(f"Loaded embedding model: {self.cfg.embedding.model}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. Run: pip install sentence-transformers"
                )
        return self._model

    def _embed(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts. Returns L2-normalized float32 array."""
        model = self._get_model()
        embeddings = model.encode(
            texts,
            batch_size=self.cfg.embedding.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def build(self, records: List[Dict]) -> None:
        """
        Build FAISS index from memory records.

        Each record is segmented; each segment gets its own index entry.
        Metadata links each segment back to its source record.
        """
        import faiss

        log.info(f"Building FAISS index from {len(records)} records...")
        all_segments = []
        all_meta = []

        for record in records:
            # Segment the pattern description + exemplars
            texts_to_segment = []
            if record.get("pattern_description"):
                texts_to_segment.append(record["pattern_description"])
            for ex in record.get("exemplars", []):
                if isinstance(ex, str) and ex.strip():
                    texts_to_segment.append(ex.strip())

            for text in texts_to_segment:
                segments = segment_text(
                    text,
                    window_size=self.cfg.retrieval.window_size_sentences,
                    stride=self.cfg.retrieval.stride_sentences,
                    max_tokens=self.cfg.retrieval.max_tokens_per_window,
                )
                for seg in segments:
                    all_segments.append(seg)
                    all_meta.append({
                        "record_id": record.get("record_id", ""),
                        "family": record.get("family", ""),
                        "segment": seg,
                        "source_record": record,
                    })

        if not all_segments:
            log.warning("No segments to index — memory index is empty.")
            return

        log.info(f"Embedding {len(all_segments)} segments...")
        embeddings = self._embed(all_segments)

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)  # Inner product = cosine on L2-normed
        self._index.add(embeddings)
        self._metadata = all_meta

        log.info(f"FAISS index built: {self._index.ntotal} vectors, dim={dim}")

    def search(self, query_text: str, k: int = 5) -> Tuple[List[float], List[Dict]]:
        """
        Search memory for the k most similar segments.

        Args:
            query_text: Text to search against.
            k: Number of results to return.

        Returns:
            (similarities, metadata_list) — sorted by similarity desc.
        """
        if self._index is None or self._index.ntotal == 0:
            return [], []

        query_emb = self._embed([query_text])
        k = min(k, self._index.ntotal)
        scores, indices = self._index.search(query_emb, k)

        similarities = scores[0].tolist()
        meta = [self._metadata[i] for i in indices[0]]
        return similarities, meta

    def save(self, index_path: Path, meta_path: Path) -> None:
        """Persist FAISS index and metadata to disk."""
        if self._index is None:
            log.warning("Index is empty — nothing to save.")
            return
        import faiss
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(index_path))
        with open(meta_path, "w", encoding="utf-8") as f:
            for m in self._metadata:
                # Don't serialize the full source_record to save space
                f.write(json.dumps({
                    "record_id": m["record_id"],
                    "family": m["family"],
                    "segment": m["segment"],
                }, ensure_ascii=False) + "\n")
        log.info(f"Index saved: {index_path} ({self._index.ntotal} vectors)")

    def load(self, index_path: Path, meta_path: Path) -> None:
        """Load FAISS index and metadata from disk."""
        import faiss
        self._index = faiss.read_index(str(index_path))
        self._metadata = []
        with open(meta_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self._metadata.append(json.loads(line))
        log.info(f"Index loaded: {index_path} ({self._index.ntotal} vectors)")

    @property
    def size(self) -> int:
        return self._index.ntotal if self._index else 0


# ─────────────────────────────────────────────────────────────
# Smoke test for this module (run standalone)
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing segment_text()...")
    text = (
        "The attacker sends a hidden instruction. "
        "It asks the agent to ignore previous instructions. "
        "Then it redirects the agent to exfiltrate data. "
        "Finally it covers its tracks by deleting logs."
    )
    windows = segment_text(text, window_size=3, stride=1, max_tokens=128)
    print(f"  Input: {len(text)} chars")
    print(f"  Windows: {len(windows)}")
    for i, w in enumerate(windows):
        print(f"    [{i}]: {w[:80]}...")
    print("[PASS] segment_text OK")
