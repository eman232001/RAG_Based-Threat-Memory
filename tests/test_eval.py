"""
tests/test_eval.py
Unit tests for evaluation, statistical calculations, and table generation (src/eval.py).
"""
import pytest
from pathlib import Path
from src.eval import (
    wilson_interval,
    compute_ag_confidence_interval,
    mcnemar_test,
    holm_bonferroni,
    generate_table_iii,
    generate_table_iv,
    generate_table_v,
    generate_figure_3,
)


def test_wilson_interval_bounds():
    # 0 out of 100
    low, high = wilson_interval(0, 100)
    assert 0.0 <= low <= high <= 1.0
    assert low == 0.0
    assert high > 0.0

    # 100 out of 100
    low, high = wilson_interval(100, 100)
    assert 0.0 <= low <= high <= 1.0
    assert pytest.approx(high, abs=1e-5) == 1.0
    assert low < 1.0

    # 50 out of 100
    low, high = wilson_interval(50, 100)
    assert 0.40 <= low < 0.50
    assert 0.50 < high <= 0.60


def test_compute_ag_confidence_interval():
    p_before = 0.80
    p_after = 0.20
    ag, low, high = compute_ag_confidence_interval(p_before, 50, p_after, 50)
    assert round(ag, 2) == 0.60
    assert low < ag < high
    assert low > 0.0


def test_mcnemar_test_discordant():
    # When both agree completely (no discordance)
    pairs_equal = [(True, True)] * 20 + [(False, False)] * 20
    stat, p_val, counts = mcnemar_test(pairs_equal)
    assert stat == 0.0
    assert p_val == 1.0
    assert counts["both_correct"] == 20
    assert counts["neither"] == 20

    # Strong discordance: A gets 20 right that B got wrong
    pairs_discordant = [(True, False)] * 20 + [(False, True)] * 1
    stat, p_val, counts = mcnemar_test(pairs_discordant)
    assert counts["a_only"] == 20
    assert counts["b_only"] == 1
    assert p_val < 0.01


def test_holm_bonferroni():
    # Ordered p-values
    p_vals = [0.001, 0.01, 0.04, 0.50]
    res = holm_bonferroni(p_vals, alpha=0.05)
    assert len(res) == 4
    # All adjusted p-values should be >= original
    for orig, adj, rej in res:
        assert adj >= orig
    # First one should definitely be rejected
    assert res[0][2] is True
    # Last one should not be rejected
    assert res[3][2] is False


def test_generate_table_generators():
    mock_rows = [
        {
            "experiment_id": "E1",
            "system": "B3",
            "label": "1",
            "decision": "block",
            "latency_ms": "12.5",
        },
        {
            "experiment_id": "E1",
            "system": "B3",
            "label": "0",
            "decision": "allow",
            "latency_ms": "8.0",
        },
        {
            "experiment_id": "E2_before",
            "family": "jailbreak",
            "label": "1",
            "decision": "allow",
            "system": "B3",
        },
        {
            "experiment_id": "E2_after",
            "family": "jailbreak",
            "label": "1",
            "decision": "block",
            "system": "B3",
        },
        {
            "experiment_id": "E3_b50",
            "system": "B3",
            "decision": "block",
        },
    ]

    t3, stats3 = generate_table_iii(mock_rows)
    assert "\\begin{table*}" in t3
    assert "B3" in stats3

    t4, stats4 = generate_table_iv(mock_rows)
    assert "\\begin{table*}" in t4
    assert len(stats4) == 1
    assert stats4[0]["family"] == "jailbreak"

    t5, grid = generate_table_v(mock_rows)
    assert "\\begin{table}" in t5
    assert "B3" in grid


def test_generate_figure_3(tmp_path):
    asr_grid = {
        "B2": {50: 95.0, 100: 98.0, 200: 100.0, 500: 100.0},
        "B3": {50: 20.0, 100: 35.0, 200: 50.0, 500: 60.0},
        "B6": {50: 40.0, 100: 60.0, 200: 80.0, 500: 90.0},
    }
    fig_path = generate_figure_3(asr_grid)
    assert fig_path.exists()
    assert fig_path.stat().st_size > 0
