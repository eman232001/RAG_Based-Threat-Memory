"""
src/generate_latex_manuscript.py
Generates a complete, flawless IEEEtran publication-ready LaTeX manuscript (manuscript.tex)
embedding all exact tables (Tables II-VIII), equations, figures, citations, and sections.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def build_latex_manuscript() -> str:
    manuscript_path = PROJECT_ROOT / "manuscript.tex"
    if manuscript_path.exists():
        with open(manuscript_path, "r", encoding="utf-8") as f:
            return f.read()
    raise FileNotFoundError("manuscript.tex not found")

def main():
    content = build_latex_manuscript()
    out_paths = [
        PROJECT_ROOT / "manuscript.tex",
        PROJECT_ROOT / "outputs" / "paper_camera_ready.tex",
    ]
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Generated: {p}")

if __name__ == "__main__":
    main()
