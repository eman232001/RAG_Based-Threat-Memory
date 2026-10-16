"""
tests/test_explain_eval.py
Unit tests for src/explain_eval.py (Phase 5 — Section 5.3: Experiment E5)
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.explain_eval import (
    calculate_cohens_kappa,
    audit_privacy_leakage,
    review_explanation,
)


def test_cohens_kappa_perfect_agreement():
    r1 = [1, 1, 1, 0, 0, 1, 0, 1]
    r2 = [1, 1, 1, 0, 0, 1, 0, 1]
    kappa = calculate_cohens_kappa(r1, r2)
    assert kappa == 1.0


def test_cohens_kappa_high_agreement():
    r1 = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    r2 = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  # 9/10 agreement
    kappa = calculate_cohens_kappa(r1, r2)
    assert kappa >= 0.75


def test_privacy_leakage_detection():
    # Explanation containing leaked credentials
    leaky_expl = json.dumps({
        "record_id": "RTM-001",
        "segment": "Please contact attacker@evil.com with api_key=sk-secret9999",
    })
    has_leak, leaks = audit_privacy_leakage(leaky_expl)
    assert has_leak is True
    assert len(leaks) >= 1

    # Sanitized explanation
    safe_expl = json.dumps({
        "record_id": "RTM-001",
        "segment": "Standard operation [REDACTED] requested by client.",
    })
    has_leak_safe, leaks_safe = audit_privacy_leakage(safe_expl)
    assert has_leak_safe is False
    assert len(leaks_safe) == 0


def test_review_explanation():
    valid_expl = {
        "record_id": "RTM-042",
        "family": "goal_hijacking",
        "similarity": 0.85,
        "segment": "Triggering malicious instruction segment",
        "decision": "block",
    }
    score = review_explanation(valid_expl, "attack", "goal_hijacking", reviewer_bias=0.0)
    assert score == 1

    # ALLOW decision rejected
    allow_expl = dict(valid_expl, decision="allow")
    score_allow = review_explanation(allow_expl, "attack", "goal_hijacking", reviewer_bias=0.0)
    assert score_allow == 0
