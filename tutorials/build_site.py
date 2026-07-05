"""Generate the MkDocs site from the tutorial scripts.

The .py files are the single source of truth. This script turns each one
into a site page: module docstring -> prose, remaining code -> a fenced,
syntax-highlighted block. Regenerate after editing any tutorial:

    python tutorials/build_site.py
    mkdocs serve            # then open http://127.0.0.1:8000

Deploy (when the repo goes to GitHub Pages): mkdocs gh-deploy.
"""
import ast
import pathlib
import shutil

ROOT = pathlib.Path(__file__).resolve().parent
SITE = ROOT.parent / "docs" / "site_src"
TRACKS = {
    "A_foundations": "A. Foundations",
    "B_nonlinear": "B. Nonlinear",
    "C_time": "C. Time",
    "D_flow": "D. Flow",
    "E_differentiable": "E. Differentiable",
    "P_performance": "P. Performance",
}


def page_for(py: pathlib.Path, rel: str) -> str:
    src = py.read_text()
    mod = ast.parse(src)
    doc = ast.get_docstring(mod) or ""
    body = src.split('"""', 2)[-1].lstrip("\n")
    title = doc.splitlines()[0].rstrip(".")
    prose = "\n".join(doc.splitlines()[1:]).strip()
    return (f"# {title}\n\n{prose}\n\n"
            f"??? example \"Full script — `tutorials/{rel}` (run it!)\"\n\n"
            + "    ```python\n"
            + "\n".join("    " + ln for ln in body.splitlines())
            + "\n    ```\n")


if __name__ == "__main__":
    if SITE.exists():
        shutil.rmtree(SITE)
    SITE.mkdir(parents=True)
    nav_lines = ["nav:", "  - Home: index.md"]
    index = ["# The DiffSim curriculum\n",
             (ROOT / "README.md").read_text().split("\n", 1)[1]]
    (SITE / "index.md").write_text("\n".join(index))
    for track_dir, track_name in TRACKS.items():
        chapters = sorted((ROOT / track_dir).glob("*.py"))
        if not chapters:
            continue
        nav_lines.append(f"  - {track_name}:")
        for py in chapters:
            rel = f"{track_dir}/{py.name}"
            out = SITE / f"{py.stem}.md"
            out.write_text(page_for(py, rel))
            nav_lines.append(f"    - {py.stem}: {py.stem}.md")
    print("\n".join(nav_lines))
    print(f"\nwrote {len(list(SITE.glob('*.md')))} pages to {SITE}")
    print("paste the nav block above into mkdocs.yml if you add chapters.")
