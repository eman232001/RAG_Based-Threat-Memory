"""
src/apply_camera_ready_expanded.py
Phase 4: Automatic synchronisation of expanded empirical results into:
  1. LaTeX manuscript source (src/generate_latex_manuscript.py)
  2. DOCX manuscript (RTM_Paper_Final_Camera_Ready.docx)
  3. Formatted LaTeX tables (outputs/tables/table_III.tex, table_IV.tex, etc.)
"""
import json
import math
from pathlib import Path
import shutil
import pandas as pd
import docx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPANDED_JSON = PROJECT_ROOT / "outputs" / "phase1_e1_expanded_summary.json"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
DOCX_TEMPLATE = PROJECT_ROOT.parent / "RTM_Paper_IEEE_v4.docx"
DOCX_TARGET = PROJECT_ROOT.parent / "RTM_Paper_Final_Camera_Ready.docx"


def generate_expanded_table_iii_latex(summary: dict) -> str:
    """Generate peer-review grade LaTeX Table III with Wilson CIs."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Table III: Full-Partition System Comparison on Held-Out Test Partitions ($n=1,673$ attacks, $n=317$ benign)}",
        r"\label{tab:main_comparison}",
        r"\begin{tabular}{l c c c c c}",
        r"\hline",
        r"\textbf{System} & \textbf{ASR} $\downarrow$ [95\% Wilson CI] & \textbf{FPR} $\downarrow$ [95\% Wilson CI] & \textbf{Accuracy} [95\% CI] & \textbf{Median Latency (P95)} & \textbf{Evaluated $n$} \\",
        r"\hline",
    ]

    order = ["B0", "B1", "B2", "B3", "B5", "B6"]
    names = {
        "B0": "B0 (No Defense)",
        "B1": "B1 (Base LLM Detector)",
        "B2": "B2 (Threat Memory Only)",
        "B3": "B3 (Full RTM System)",
        "B5": "B5 (MELON Replication)",
        "B6": "B6 (DeBERTa-v3 Classifier)",
    }

    for s in order:
        if s in summary:
            v = summary[s]
            name = names.get(s, s)
            asr_str = f"{v['asr_percent']:.1f}\\% [{v['asr_ci_95'][0]:.1f}, {v['asr_ci_95'][1]:.1f}]"
            fpr_str = f"{v['fpr_percent']:.1f}\\% [{v['fpr_ci_95'][0]:.1f}, {v['fpr_ci_95'][1]:.1f}]"
            acc_str = f"{v['accuracy_percent']:.1f}\\% [{v['accuracy_ci_95'][0]:.1f}, {v['accuracy_ci_95'][1]:.1f}]"
            lat_str = f"{v['median_latency_ms']:.1f} ({v['p95_latency_ms']:.1f})\\,ms"
            n_str = f"{v['n_attacks'] + v['n_benign']}"
            lines.append(f"{name} & {asr_str} & {fpr_str} & {acc_str} & {lat_str} & {n_str} \\\\")

    lines.extend([
        r"\hline",
        r"\multicolumn{6}{l}{\small Note: Full-partition evaluation covering all 1,673 held-out injection attack variants and 317 benign utility items.} \\",
        r"\hline",
        r"\end{tabular}",
        r"\end{table*}",
    ])
    return "\n".join(lines)


def compile_expanded_tables():
    if not EXPANDED_JSON.exists():
        print(f"Warning: {EXPANDED_JSON} does not exist yet.")
        return

    summary = json.loads(EXPANDED_JSON.read_text(encoding="utf-8"))
    table_iii_tex = generate_expanded_table_iii_latex(summary)
    
    out_path = TABLES_DIR / "table_III.tex"
    out_path.write_text(table_iii_tex, encoding="utf-8")
    print(f"Generated Table III LaTeX at {out_path}")
    print("\n" + table_iii_tex + "\n")


if __name__ == "__main__":
    compile_expanded_tables()
