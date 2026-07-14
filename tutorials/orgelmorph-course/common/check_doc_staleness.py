"""Doc-vs-data staleness gate: do the numbers PRINTED in the course match the
numbers the code GENERATES today?

Every figure/number in the course document is generated from saved data, never
hand-copied (spec cross-cutting principle). This checker enforces that: it
parses the ``\\newcommand{\\NAME}{VALUE}`` macros a chapter cites in
``latex/numbers/<ch>.tex`` and compares each against a freshly generated
``results.json``, within a documented tolerance. If the code drifts from the
documented number (or someone edits the doc without regenerating), CI fails.

A chapter provides a ``doc_numbers.yaml`` mapping:

    tex: ../../latex/numbers/p1.tex     # relative to the mapping file
    macros:
      POLYFend:      {path: poly.Fend,       rtol: 0.05}
      POLYcrange:    {path: poly.c_max,      rtol: 0.05, extract: high}
      POLYmass:      {path: poly.mass_drift, max: 1.0e-10}

``extract`` (optional) pulls a number out of a LaTeX range/expression macro:
``low``/``high`` take the first/last number in ``[a, b]``; default is the sole
number in the macro body.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import yaml

_NEWCMD = re.compile(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}")
_NUM = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_macros(tex_path):
    """Return ``{macro_name: raw_body}`` from a ``\\newcommand`` file."""
    out = {}
    with open(tex_path) as fh:
        for line in fh:
            m = _NEWCMD.search(line)
            if m:
                out[m.group(1)] = m.group(2)
    return out


def _numbers_in(body):
    return [float(x) for x in _NUM.findall(body)]


def macro_value(body, extract=None):
    """Extract a float from a macro body (optionally a range endpoint)."""
    nums = _numbers_in(body)
    if not nums:
        raise ValueError(f"no number in macro body {body!r}")
    if extract == "low":
        return nums[0]
    if extract == "high":
        return nums[-1]
    return nums[0]


def _get_path(obj, dotted):
    cur = obj
    for part in dotted.split("."):
        cur = cur[int(part)] if isinstance(cur, list) else cur[part]
    return cur


def check(mapping, base_dir, results):
    """Compare each mapped macro against the results. Returns ``(ok, rows)``.

    ``rows`` are ``(macro, status, doc_value, data_value, detail)``.
    """
    tex_path = os.path.normpath(os.path.join(base_dir, mapping["tex"]))
    macros = parse_macros(tex_path)
    rows, ok = [], True
    for name, spec in mapping.get("macros", {}).items():
        if name not in macros:
            rows.append((name, "MISSING-MACRO", None, None,
                         f"{name} not in {tex_path}"))
            ok = False
            continue
        try:
            doc = macro_value(macros[name], spec.get("extract"))
            data = float(_get_path(results, spec["path"]))
        except (KeyError, IndexError, ValueError, TypeError) as exc:
            rows.append((name, "ERROR", None, None, str(exc)))
            ok = False
            continue
        if "max" in spec:
            passed = data <= float(spec["max"])
            detail = f"data {data:.3g} <= {float(spec['max']):.3g}"
        else:
            rtol = float(spec.get("rtol", 0.05))
            atol = float(spec.get("atol", 0.0))
            tol = atol + rtol * abs(doc)
            passed = abs(doc - data) <= tol
            detail = f"|doc {doc:.4g} - data {data:.4g}| <= {tol:.3g}"
        rows.append((name, "PASS" if passed else "STALE", doc, data, detail))
        ok = ok and passed
    return ok, rows


def format_table(rows):
    w = max((len(r[0]) for r in rows), default=5)
    lines = [f"{'macro'.ljust(w)}  status        doc        data     detail",
             f"{'-'*w}  ------------  ---------  --------  ------"]
    for name, status, doc, data, detail in rows:
        ds = "" if doc is None else f"{doc:.4g}"
        da = "" if data is None else f"{data:.4g}"
        lines.append(f"{name.ljust(w)}  {status:<12}  {ds:>9}  {da:>8}  {detail}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mapping", required=True,
                    help="doc_numbers.yaml for the chapter")
    ap.add_argument("--results", required=True, help="freshly generated results.json")
    args = ap.parse_args(argv)
    with open(args.mapping) as fh:
        mapping = yaml.safe_load(fh)
    with open(args.results) as fh:
        results = json.load(fh)
    ok, rows = check(mapping, os.path.dirname(os.path.abspath(args.mapping)),
                     results)
    print(format_table(rows))
    print(f"\n{'PASS' if ok else 'STALE'}: doc-vs-data staleness check")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
