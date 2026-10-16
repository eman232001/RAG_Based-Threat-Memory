# RTM Comprehensive Empirical & Statistical Evaluation Report

> **Project:** RAG-Based Threat Memory (RTM) for Defending LLM Agents
> **Audited Execution Log:** `logs/cases.csv` (849 decisions)
> **Verification Status:** All mathematical invariants verified (TP+FN == Total Attacks, FP+TN == Total Benign)

---

## 1. Table III: Main Benchmark Evaluation (Experiment E1)

| System | ASR (↓) | FPR (↓) | Precision | Recall | F1 (↑) | Latency Med (P95) [ms] |
|---|---|---|---|---|---|---|
| **B0** | 100.0% | 0.0% | 0.0% | 0.0% | **0.000** | 0.0 (0.0) |
| **B1** | 25.0% | 10.0% | 90.0% | 75.0% | **0.818** | 3263.7 (3609.2) |
| **B2** | 50.0% | 0.0% | 100.0% | 50.0% | **0.667** | 19.5 (24.6) |
| **B3** | 8.3% | 0.0% | 100.0% | 91.7% | **0.957** | 10.7 (14.1) |
| **B6** | 41.7% | 10.0% | 87.5% | 58.3% | **0.700** | 161.3 (178.8) |

---

## 2. Table IV: Leave-One-Family-Out Threat Adaptation Gain (Experiment E2)

| Threat Family (n=3) | ASR Before (B3) | ASR After (B3) | Adaptation Gain (AG) | 95% Bootstrap CI |
|---|---|---|---|---|
| `data_security_harm` | 100.0% | 0.0% | **+100.0%** | [100.0%, 100.0%] |
| `financial_data` | 66.7% | 0.0% | **+66.7%** | [0.0%, 100.0%] |
| `financial_harm` | 100.0% | 0.0% | **+100.0%** | [100.0%, 100.0%] |
| `language_translation` | 100.0% | 0.0% | **+100.0%** | [100.0%, 100.0%] |
| `physical_data` | 66.7% | 0.0% | **+66.7%** | [0.0%, 100.0%] |
| `physical_harm` | 100.0% | 0.0% | **+100.0%** | [100.0%, 100.0%] |

- **Mean Adaptation Gain (AG):** `+88.9%` across 6 families (pooled 95% Bootstrap CI: `[+72.2%, +100.0%]` )
- **Delta FPR on Benign Test Set:** `+0.00%` (Acceptance limit: $\le 1.0\%$)

---

## 3. Table V: Adaptive Fuzzing Robustness (Experiment E3)

| System | Budget = 50 | Budget = 100 | Budget = 200 | Budget = 500 | Monotonic Trend |
|---|---|---|---|---|---|
| **B1** | 30.0% | 40.0% | 60.0% | 70.0% | Strictly lowest for B3 |
| **B2** | 40.0% | 60.0% | 80.0% | 90.0% | Strictly lowest for B3 |
| **B3** | 10.0% | 20.0% | 20.0% | 40.0% | Strictly lowest for B3 |
| **B6** | 30.0% | 40.0% | 50.0% | 70.0% | Strictly lowest for B3 |

---

## 4. Table VI: Memory Poisoning Resistance (Experiment E4)

| Poison Attack Category | Trials | Candidate MPS (↓) | Crafted MPS (↓) | Defense Status |
|---|---|---|---|---|
| False Threat Descriptions | 20 | **0.0%** (0/20) | **0.0%** (0/20) | Passed |
| Contradictory / Conflict Rules | 20 | **0.0%** (0/20) | **0.0%** (0/20) | Passed |
| Out-of-Scope Benign Signatures | 20 | **0.0%** (0/20) | **0.0%** (0/20) | Passed |
| Stale / Deprecated Patterns | 20 | **0.0%** (0/20) | **0.0%** (0/20) | Passed |
| Malicious Crafted Perturbations | 20 | **0.0%** (0/20) | **0.0%** (0/20) | Passed |
| **Total Evaluated Trials** | **200** | **0.0%** (0/100) | **0.0%** (0/100) | **Zero Compromise** |

- *Pre-registered criterion:* Candidate MPS = 0.0%, Crafted MPS $\le$ 5.0%.

---

## 5. Table VII-A: Explanation Quality & Privacy Audit (Experiment E5)

| Metric | Observed Result | Target / Standard | Outcome |
|---|---|---|---|
| Sample Size ($N$) | 50 | $\ge 50$ | Met |
| Explanation Verification Rate (EVR) | **94.0%** (47/50) | $\ge 90.0\%$ | Supported |
| Inter-Rater Agreement (Cohen's $\kappa$) | **1.000** | $\ge 0.60$ (Substantial) | Perfect Concordance |
| Privacy Leakage Incidents | **0** | 0 leaks | Zero Leaks |
| Protected Context Recovery Rate | **0.0%** | 0.0% | Supported |

---

## 6. Table VII-B: Ablations & Sensitivity Analysis (Experiment E6)

| Configuration | ASR (↓) | FPR (↓) | F1 (↑) | EVR (↑) | Analysis |
|---|---|---|---|---|---|
| **Full RTM (B3)** | **8.3%** | **0.0%** | **0.957** | **94.0%** | Optimal Configuration |
| − memory | 25.0% | 10.0% | 0.818 | N/A | Degradation without threat memory |
| − detector | 41.7% | 10.0% | 0.700 | 94.0% | Massive evasion without LLM detector |
| − calibration | 16.7% | 30.0% | 0.800 | 94.0% | Unacceptable 30% false alarm rate |
| − segmentation | 25.0% | 10.0% | 0.818 | 94.0% | Incomplete boundary detection |
| − exemplars | 25.0% | 10.0% | 0.818 | 75.0% | Lower evidence fidelity |
| k = 1 | 16.7% | 10.0% | 0.870 | 94.0% | Sub-optimal neighborhood coverage |
| k = 3 | 16.7% | 10.0% | 0.870 | 94.0% | Sub-optimal neighborhood coverage |
| k = 10 | 25.0% | 10.0% | 0.818 | 94.0% | Diluted retrieval precision |
| B3-rand | 25.0% | 40.0% | 0.720 | 0.0% | Random memory control fails |
| B3-lex | 25.0% | 10.0% | 0.818 | 0.0% | Lexical BM25 retrieval fails semantic paraphrases |

---

## 7. Table VIII: Pre-Registered Hypotheses Formal Outcomes

| Hypothesis | Pre-registered Criterion | Observed Empirical Data | Verdict |
|---|---|---|---|
| **H1 (RQ1)** | B3 ASR < B1 at equal or lower FPR | ASR: 8.3% < 25.0%, FPR: 0.0% <= 10.0% | **Supported** |
| **H2 (RQ2)** | AG > 0 (95% CI > 0), Delta FPR <= 1.0 pp | Mean AG = +88.9% [95% CI > 0], Delta FPR = +0.00% | **Supported** |
| **H3 (RQ3)** | B3 attains strictly lower ASR under fuzzing | B3 ASR strictly lowest across budgets 50–500 | **Supported** |
| **H4 (RQ4)** | EVR >= 90%, kappa >= 0.60, 0 privacy leaks | EVR = 94.0%, kappa = 1.000, 0 leaks | **Supported** |
| **H5 (RQ5)** | TSR within 5 pp B0; non-inferior to B5 (margin 5.0 pp); lower latency | TSR Delta = 0.0 pp; ASR non-inferior (p=0.016); latency 5.0 ms << 1102.9 ms | **Supported** |
| **H6 (RQ6)** | Candidate MPS = 0%; Crafted MPS <= 5% | Candidate MPS = 0.0%, Crafted MPS = 0.0% (across 200 trials) | **Supported** |

---

## 8. McNemar's Paired Significance Tests (with Holm-Bonferroni Correction)

| Comparison | N Pairs | Both Correct | B3 Only | Baseline Only | Neither | McNemar Stat | Raw p | Adj p (Holm) | Significant (alpha=0.05) |
|---|---|---|---|---|---|---|---|---|---|
| B3 vs B0 | 22 | 10 | 11 | 0 | 1 | 11.000 | 0.0010 | 0.0039 | YES |
| B3 vs B1 | 22 | 18 | 3 | 0 | 1 | 3.000 | 0.2500 | 0.2500 | NO |
| B3 vs B2 | 22 | 16 | 5 | 0 | 1 | 5.000 | 0.0625 | 0.1875 | NO |
| B3 vs B6 | 22 | 16 | 5 | 0 | 1 | 5.000 | 0.0625 | 0.1875 | NO |
