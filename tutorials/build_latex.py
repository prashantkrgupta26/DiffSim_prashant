"""Generate latex/chapters/*.tex from the tutorial scripts.

Each chapter .tex renders the chapter contract: docstring prose split into
the outcome / background / expected-results boxes, then the full script as
a listing (the EXPLORE block lives at the end of each script's output and
is extracted from the trailing print). Regeneration is MANUAL by design —
generated files are committed and may be hand-polished; rerunning this
overwrites them, so polish deliberately or fork the chapter file.

    python tutorials/build_latex.py            # writes latex/chapters/
"""
import ast
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT.parent / "latex" / "chapters"
TRACKS = ["A_foundations", "B_nonlinear", "C_time", "D_flow",
          "E_differentiable", "P_performance"]

ESCAPES = [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
           ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
           ("}", r"\}"), ("~", r"\textasciitilde{}"),
           ("^", r"\textasciicircum{}")]


def esc(t: str) -> str:
    for a, b in ESCAPES:
        t = t.replace(a, b)
    return t


def split_doc(doc: str):
    """title, and the {OUTCOME, BACKGROUND, EXPECTED, RUN} sections."""
    lines = doc.splitlines()
    title = lines[0].strip().rstrip(".")
    body = "\n".join(lines[1:])
    sec = {"pre": "", "outcome": "", "background": "", "expected": ""}
    cur = "pre"
    for ln in body.splitlines():
        up = ln.strip()
        if up.startswith("LEARNING OUTCOME"):
            cur = "outcome"
            ln = re.sub(r"^\s*LEARNING OUTCOME\.?\s*", "", ln)
        elif up.startswith("BACKGROUND"):
            cur = "background"
            ln = re.sub(r"^\s*BACKGROUND[^\n]*?\.\s*", "", ln, count=1) \
                if up.startswith("BACKGROUND.") else re.sub(
                    r"^\s*BACKGROUND", "", ln)
        elif up.startswith("EXPECTED RESULTS"):
            cur = "expected"
            ln = re.sub(r"^\s*EXPECTED RESULTS[^:\n]*[:.]?", "", ln)
        elif up.startswith("Run:"):
            cur = "run"
            continue
        if cur in sec:
            sec[cur] += ln + "\n"
    return title, sec


def extract_explore(code: str):
    m = re.search(r'print\("""\s*\nEXPLORE\n(.*?)"""\)', code, re.S)
    return m.group(1).strip() if m else None


def chapter_tex(py: pathlib.Path, rel: str) -> str:
    src = py.read_text()
    doc = ast.get_docstring(ast.parse(src)) or ""
    code = src.split('"""', 2)[-1].lstrip("\n")
    title, sec = split_doc(doc)
    explore = extract_explore(code)
    # strip the explore print from the displayed listing (it is boxed)
    if explore:
        code = re.sub(r'\s*print\("""\s*\nEXPLORE\n.*?"""\)', "\n", code,
                      flags=re.S)
    parts = [f"\\chapter{{{esc(title.split('—', 1)[-1].strip())}}}",
             f"\\label{{ch:{py.stem}}}",
             f"\\noindent\\texttt{{{esc('tutorials/' + rel)}}}\\\\[0.75em]"]
    if sec["outcome"].strip():
        parts.append("\\begin{outcome}\n" + esc(sec["outcome"].strip())
                     + "\n\\end{outcome}\n")
    if sec["background"].strip():
        parts.append(esc(sec["background"].strip()) + "\n")
    if sec["expected"].strip():
        parts.append("\\begin{expected}\n\\begin{verbatim}\n"
                     + sec["expected"].strip()
                     + "\n\\end{verbatim}\n\\end{expected}\n")
    parts.append("\\section*{The script}")
    parts.append("\\begin{lstlisting}\n" + code.rstrip()
                 + "\n\\end{lstlisting}\n")
    if explore:
        parts.append("\\begin{explore}\n\\begin{verbatim}\n" + explore
                     + "\n\\end{verbatim}\n\\end{explore}\n")
    return "\n".join(parts)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for track in TRACKS:
        for py in sorted((ROOT / track).glob("*.py")):
            rel = f"{track}/{py.name}"
            (OUT / f"{py.stem}.tex").write_text(chapter_tex(py, rel))
            n += 1
    print(f"wrote {n} chapters to {OUT}")
