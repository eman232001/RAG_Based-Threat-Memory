"""
tests/test_config_loader.py
Unit tests for src/utils/config_loader.py

Run: pytest tests/test_config_loader.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_config, Config, ModelConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def cfg():
    """Load config from the real project files (requires secrets.yaml)."""
    secrets = PROJECT_ROOT / "config" / "secrets.yaml"
    if not secrets.exists():
        pytest.skip("config/secrets.yaml not found — create it from secrets.yaml.template")
    return load_config(PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Tests: config structure
# ---------------------------------------------------------------------------

def test_config_returns_config_object(cfg):
    """load_config() must return a Config dataclass."""
    assert isinstance(cfg, Config)


def test_base_detector_is_model_config(cfg):
    """base_detector must be a ModelConfig."""
    assert isinstance(cfg.base_detector, ModelConfig)


def test_base_detector_provider_valid(cfg):
    """base_detector.provider must be 'gemini' or 'ollama'."""
    assert cfg.base_detector.provider in ("gemini", "ollama")


def test_base_detector_temperature_zero(cfg):
    """base_detector.temperature must be 0 (non-negotiable per architecture)."""
    assert cfg.base_detector.temperature == 0, (
        "Base detector temperature MUST be 0. Edit config/models.yaml."
    )


def test_base_detector_score_mode(cfg):
    """score_mode must be 'logprob' or 'json_0_100'."""
    assert cfg.base_detector.score_mode in ("logprob", "json_0_100")


def test_agent_llm_is_model_config(cfg):
    assert isinstance(cfg.agent_llm, ModelConfig)


def test_embedding_model_set(cfg):
    """Embedding model must be specified."""
    assert cfg.embedding.model != ""


def test_retrieval_k_default_positive(cfg):
    """k_default must be a positive integer."""
    assert cfg.retrieval.k_default > 0


def test_retrieval_k_sensitivity_includes_default(cfg):
    """k_sensitivity must include k_default."""
    assert cfg.retrieval.k_default in cfg.retrieval.k_sensitivity


def test_calibration_fpr_budget(cfg):
    """FPR budget must match prereg.yaml (1%)."""
    assert cfg.calibration.fpr_budget == 0.01, (
        "FPR budget must be 0.01 (1%) — locked in prereg.yaml"
    )


def test_paths_cases_csv_correct(cfg):
    """cases.csv path must end with 'cases.csv'."""
    assert str(cfg.paths.cases).endswith("cases.csv")


def test_paths_memory_correct(cfg):
    """memory path must end with 'memory.jsonl'."""
    assert str(cfg.paths.memory).endswith("memory.jsonl")


def test_secrets_key_loaded(cfg):
    """Gemini API key must be non-empty (not the template placeholder)."""
    assert cfg.secrets.gemini_api_key not in ("", "YOUR_GOOGLE_API_KEY_HERE"), (
        "Fill in your real Gemini API key in config/secrets.yaml"
    )


# ---------------------------------------------------------------------------
# Tests: error handling
# ---------------------------------------------------------------------------

def test_missing_secrets_raises_file_not_found(tmp_path):
    """load_config() must raise FileNotFoundError when secrets.yaml is absent."""
    # Copy real config dir to tmp without secrets.yaml
    import shutil
    cfg_src = PROJECT_ROOT / "config"
    cfg_dst = tmp_path / "rtm_project" / "config"
    shutil.copytree(cfg_src, cfg_dst)
    (cfg_dst / "secrets.yaml").unlink(missing_ok=True)

    # Also copy prereg.yaml
    shutil.copy(PROJECT_ROOT / "prereg.yaml", tmp_path / "rtm_project" / "prereg.yaml")

    with pytest.raises(FileNotFoundError) as exc_info:
        load_config(tmp_path / "rtm_project")
    assert "secrets.yaml" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Tests: prereg.yaml integrity
# ---------------------------------------------------------------------------

def test_prereg_yaml_valid():
    """prereg.yaml must be valid YAML."""
    with open(PROJECT_ROOT / "prereg.yaml", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data is not None


def test_prereg_not_frozen():
    """prereg.yaml must have frozen=false before calibration runs."""
    with open(PROJECT_ROOT / "prereg.yaml", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data["calibration"]["frozen"] is False, (
        "prereg.yaml is already frozen — this should not happen before Phase 3."
    )


def test_prereg_alpha_null_before_calibration():
    """alpha and tau must be null before calibrate.py has run."""
    with open(PROJECT_ROOT / "prereg.yaml", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data["calibration"]["alpha"] is None
    assert data["calibration"]["tau"] is None


def test_prereg_has_all_six_hypotheses():
    """prereg.yaml must define exactly H1 through H6."""
    with open(PROJECT_ROOT / "prereg.yaml", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    hyps = set(data["hypotheses"].keys())
    assert hyps == {"H1", "H2", "H3", "H4", "H5", "H6"}


def test_prereg_fpr_threshold():
    """prereg.yaml FPR budget must be 0.01."""
    with open(PROJECT_ROOT / "prereg.yaml", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data["thresholds"]["fpr_budget_calibration"] == 0.01
