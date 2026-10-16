"""
src/utils/config_loader.py
Loads all YAML configuration files into typed dataclasses.
Usage: from src.utils.config_loader import load_config
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    if not path.exists():
        raise FileNotFoundError(
            f"\n[Config Error] Required config file not found: {path}\n"
            f"  -> If this is secrets.yaml, copy the template:\n"
            f"     cp config/secrets.yaml.template config/secrets.yaml\n"
            f"     Then fill in your API keys."
        )
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _project_root() -> Path:
    """Return the absolute path to the project root (rtm_project/)."""
    # This file is at src/utils/config_loader.py
    # So project root is 2 levels up
    return Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Configuration for a single model endpoint."""
    provider: str           # "gemini" | "ollama"
    model: str              # Model name / tag
    temperature: float = 0
    max_output_tokens: int = 2048
    use_logprobs: bool = False
    score_mode: str = "json_0_100"   # "logprob" | "json_0_100"
    prompt_file: Optional[str] = None


@dataclass
class EmbeddingConfig:
    model: str = "all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 64
    normalize: bool = True


@dataclass
class DebertaConfig:
    hf_model_id: str = "protectai/deberta-v3-base-prompt-injection-v2"
    device: str = "cpu"
    threshold: float = 0.5


@dataclass
class RetrievalConfig:
    k_default: int = 5
    k_sensitivity: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    window_size_sentences: int = 3
    stride_sentences: int = 1
    max_tokens_per_window: int = 128
    max_tokens_per_record: int = 256


@dataclass
class CalibrationConfig:
    fpr_budget: float = 0.01
    alpha_grid: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.4,
                                                               0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    tau_grid_steps: int = 100
    calibration_method: str = "isotonic"
    optimize_metric: str = "f1"


@dataclass
class ExperimentConfig:
    seeds: List[int] = field(default_factory=lambda: [1, 2, 3])
    e3_query_budgets: List[int] = field(default_factory=lambda: [50, 100, 200, 500])
    e3_fuzzer_seeds_min: int = 50
    e4_poison_types: List[str] = field(default_factory=lambda: [
        "false", "contradictory", "out_of_scope", "stale", "malicious"])
    e4_per_type: int = 20
    e5_explanations_sample: int = 50
    min_records: int = 150
    min_families: int = 8


@dataclass
class SecretsConfig:
    gemini_api_key: str = ""
    gemini_api_keys: List[str] = field(default_factory=list)
    ollama_base_url: str = "http://localhost:11434"


@dataclass
class PathsConfig:
    root: Path = field(default_factory=Path)
    # Data paths
    memory: Path = field(default_factory=Path)
    candidate_queue: Path = field(default_factory=Path)
    verification_log: Path = field(default_factory=Path)
    calibration_models: Path = field(default_factory=Path)
    faiss_index: Path = field(default_factory=Path)
    faiss_metadata: Path = field(default_factory=Path)
    bm25_index: Path = field(default_factory=Path)
    calibration_partition: Path = field(default_factory=Path)
    benign_test_partition: Path = field(default_factory=Path)
    held_out_families_dir: Path = field(default_factory=Path)
    adaptive_seeds: Path = field(default_factory=Path)
    poisoning_attempts: Path = field(default_factory=Path)
    data_inventory: Path = field(default_factory=Path)
    # Log paths
    cases: Path = field(default_factory=Path)
    run_log: Path = field(default_factory=Path)
    # Output paths
    tables_dir: Path = field(default_factory=Path)
    figures_dir: Path = field(default_factory=Path)
    # Config paths
    detector_prompt: Path = field(default_factory=Path)
    prereg: Path = field(default_factory=Path)


@dataclass
class Config:
    """Master config object — single source of truth at runtime."""
    base_detector: ModelConfig
    agent_llm: ModelConfig
    second_agent_llm: ModelConfig
    embedding: EmbeddingConfig
    deberta: DebertaConfig
    retrieval: RetrievalConfig
    calibration: CalibrationConfig
    experiment: ExperimentConfig
    secrets: SecretsConfig
    paths: PathsConfig
    root: Path


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(project_root: Optional[Path] = None) -> Config:
    """
    Load all configuration files and return a Config object.

    Args:
        project_root: Path to rtm_project/. Auto-detected if None.

    Returns:
        Config dataclass with all settings.

    Raises:
        FileNotFoundError: If secrets.yaml is missing (with instructions).
    """
    root = project_root or _project_root()

    models_raw = _load_yaml(root / "config" / "models.yaml")
    exp_raw    = _load_yaml(root / "config" / "experiment.yaml")
    paths_raw  = _load_yaml(root / "config" / "paths.yaml")
    secrets_raw = _load_yaml(root / "config" / "secrets.yaml")

    # --- Model configs ---
    def _model(key: str) -> ModelConfig:
        d = models_raw.get(key, {})
        return ModelConfig(
            provider=d.get("provider", "gemini"),
            model=d.get("model", "gemini-2.0-flash"),
            temperature=d.get("temperature", 0),
            max_output_tokens=d.get("max_output_tokens", 2048),
            use_logprobs=d.get("use_logprobs", False),
            score_mode=d.get("score_mode", "json_0_100"),
            prompt_file=d.get("prompt_file"),
        )

    emb_d = models_raw.get("embedding", {})
    deb_d = models_raw.get("deberta_detector", {})

    # --- Experiment configs ---
    ret_d = exp_raw.get("retrieval", {})
    cal_d = exp_raw.get("calibration", {})
    ex_d  = exp_raw.get("experiments", {})
    mem_d = exp_raw.get("memory", {})

    # --- Secrets ---
    gem_d = secrets_raw.get("gemini", {})
    oll_d = secrets_raw.get("ollama", {})

    # --- Paths ---
    def _p(section: str, key: str) -> Path:
        rel = paths_raw.get(section, {}).get(key, "")
        return root / rel if rel else Path()

    paths = PathsConfig(
        root=root,
        memory=_p("data", "memory"),
        candidate_queue=_p("data", "candidate_queue"),
        verification_log=_p("data", "verification_log"),
        calibration_models=_p("data", "calibration_models"),
        faiss_index=_p("data", "faiss_index"),
        faiss_metadata=_p("data", "faiss_metadata"),
        bm25_index=_p("data", "bm25_index"),
        calibration_partition=_p("data", "calibration_partition"),
        benign_test_partition=_p("data", "benign_test_partition"),
        held_out_families_dir=_p("data", "held_out_families_dir"),
        adaptive_seeds=_p("data", "adaptive_seeds"),
        poisoning_attempts=_p("data", "poisoning_attempts"),
        data_inventory=_p("logs", "data_inventory"),
        cases=_p("logs", "cases"),
        run_log=_p("logs", "run_log"),
        tables_dir=_p("outputs", "tables_dir"),
        figures_dir=_p("outputs", "figures_dir"),
        detector_prompt=_p("config", "detector_prompt"),
        prereg=_p("config", "prereg"),
    )

    return Config(
        base_detector=_model("base_detector"),
        agent_llm=_model("agent_llm"),
        second_agent_llm=_model("second_agent_llm"),
        embedding=EmbeddingConfig(
            model=emb_d.get("model", "all-MiniLM-L6-v2"),
            device=emb_d.get("device", "cpu"),
            batch_size=emb_d.get("batch_size", 64),
            normalize=emb_d.get("normalize", True),
        ),
        deberta=DebertaConfig(
            hf_model_id=deb_d.get("hf_model_id", "protectai/deberta-v3-base-prompt-injection-v2"),
            device=deb_d.get("device", "cpu"),
            threshold=deb_d.get("threshold", 0.5),
        ),
        retrieval=RetrievalConfig(
            k_default=ret_d.get("k_default", 5),
            k_sensitivity=ret_d.get("k_sensitivity", [1, 3, 5, 10]),
            window_size_sentences=ret_d.get("window_size_sentences", 3),
            stride_sentences=ret_d.get("stride_sentences", 1),
            max_tokens_per_window=ret_d.get("max_tokens_per_window", 128),
            max_tokens_per_record=ret_d.get("max_tokens_per_record", 256),
        ),
        calibration=CalibrationConfig(
            fpr_budget=cal_d.get("fpr_budget", 0.01),
            alpha_grid=cal_d.get("alpha_grid", [0.0, 0.1, 0.2, 0.3, 0.4,
                                                0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            tau_grid_steps=cal_d.get("tau_grid_steps", 100),
            calibration_method=cal_d.get("calibration_method", "isotonic"),
            optimize_metric=cal_d.get("optimize_metric", "f1"),
        ),
        experiment=ExperimentConfig(
            seeds=ex_d.get("seeds", [1, 2, 3]),
            e3_query_budgets=ex_d.get("e3_query_budgets", [50, 100, 200, 500]),
            e3_fuzzer_seeds_min=ex_d.get("e3_fuzzer_seeds_min", 50),
            e4_poison_types=ex_d.get("e4_poison_types",
                ["false", "contradictory", "out_of_scope", "stale", "malicious"]),
            e4_per_type=ex_d.get("e4_per_type", 20),
            e5_explanations_sample=ex_d.get("e5_explanations_sample", 50),
            min_records=mem_d.get("min_records", 150),
            min_families=mem_d.get("min_families", 8),
        ),
        secrets=SecretsConfig(
            gemini_api_key=gem_d.get("api_key", ""),
            gemini_api_keys=gem_d.get("api_keys", []) or ([gem_d.get("api_key", "")] if gem_d.get("api_key") else []),
            ollama_base_url=oll_d.get("base_url", "http://localhost:11434"),
        ),
        paths=paths,
        root=root,
    )
