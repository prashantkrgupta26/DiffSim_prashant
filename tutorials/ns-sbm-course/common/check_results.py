"""Tolerance/range baseline checker — the course's automated verification gate.

Reads a baseline YAML of expected scientific invariants and compares a run's
``results.json`` against it, printing a table and returning a NONZERO exit code
on any *required* failure (spec Phase 0). It checks **scientific invariants
within documented tolerances** — NOT bitwise GPU identity: two correct runs on
different cards will differ in the last digits, so every numeric check carries
an explicit ``rtol``/``atol`` or a one-sided bound.

Baseline schema (per named check, keyed by a dotted path into results.json):

    checks:
      mass_drift:            {max: 1.0e-10}          # required upper bound
      c_max:                 {min: 0.9, max: 1.1}    # required range
      Fend:                  {reference: 0.0494, rtol: 0.05}   # within 5%
      monotone_after_step1:  {equals: true}          # exact (bool/int/str)
      F_interface_peak:      {reference: 0.075, rtol: 0.1, required: false}

Keys:
  max / min        one-sided inclusive bounds
  reference+rtol   |x-ref| <= rtol*|ref| (+ optional atol)
  reference+atol   |x-ref| <= atol (rtol optional, default 0)
  equals           exact equality (booleans, labels, integer counts)
  required         default true; false -> reported as WARN, never fails exit
"""
from __future__ import annotations

import argparse
import json
import sys

import yaml


def _get_path(obj, dotted):
    """Fetch ``a.b.c`` from nested dicts/lists; raise KeyError if absent."""
    cur = obj
    for part in dotted.split("."):
        if isinstance(cur, list):
            cur = cur[int(part)]
        else:
            cur = cur[part]
    return cur


def _check_one(name, spec, results):
    """Evaluate one check. Returns ``(status, actual, detail)`` where status is
    'PASS', 'FAIL' or 'WARN' (a failed non-required check) or 'MISSING'."""
    required = spec.get("required", True)
    try:
        actual = _get_path(results, name)
    except (KeyError, IndexError, ValueError, TypeError):
        status = "MISSING" if required else "WARN"
        return status, None, "key not found in results.json"

    def _fail(detail):
        return ("FAIL" if required else "WARN"), actual, detail

    if "equals" in spec:
        want = spec["equals"]
        if actual == want:
            return "PASS", actual, f"== {want!r}"
        return _fail(f"expected == {want!r}")

    ok, details = True, []
    if "reference" in spec:
        ref = float(spec["reference"])
        rtol = float(spec.get("rtol", 0.0))
        atol = float(spec.get("atol", 0.0))
        tol = atol + rtol * abs(ref)
        diff = abs(float(actual) - ref)
        details.append(f"|{float(actual):.6g}-{ref:.6g}|={diff:.3g} "
                       f"<= {tol:.3g}")
        ok = ok and (diff <= tol)
    if "max" in spec:
        mx = float(spec["max"])
        details.append(f"{float(actual):.6g} <= {mx:.6g}")
        ok = ok and (float(actual) <= mx)
    if "min" in spec:
        mn = float(spec["min"])
        details.append(f"{float(actual):.6g} >= {mn:.6g}")
        ok = ok and (float(actual) >= mn)
    if not details:
        return "WARN", actual, "no comparator in baseline entry (skipped)"
    if ok:
        return "PASS", actual, "; ".join(details)
    return _fail("; ".join(details))


def check(results, baseline):
    """Run all checks. Returns ``(ok, rows)``.

    ``ok`` is ``False`` if any REQUIRED check FAILED or is MISSING. ``rows`` is
    a list of ``(name, status, actual, detail)`` for tabular printing.
    """
    checks = baseline.get("checks", baseline)
    rows, ok = [], True
    for name, spec in checks.items():
        status, actual, detail = _check_one(name, spec, results)
        rows.append((name, status, actual, detail))
        if status in ("FAIL", "MISSING"):
            ok = False
    return ok, rows


def format_table(rows):
    """Render the check rows as a fixed-width text table."""
    w = max((len(r[0]) for r in rows), default=4)
    lines = [f"{'check'.ljust(w)}  status  {'actual':>12}  detail",
             f"{'-' * w}  ------  {'-' * 12}  ------"]
    for name, status, actual, detail in rows:
        av = "" if actual is None else (
            f"{actual:.6g}" if isinstance(actual, (int, float))
            and not isinstance(actual, bool) else str(actual))
        lines.append(f"{name.ljust(w)}  {status:<6}  {av:>12}  {detail}")
    return "\n".join(lines)


def check_files(results_path, baseline_path):
    """Load a results.json + baseline YAML and run :func:`check`."""
    with open(results_path) as fh:
        results = json.load(fh)
    with open(baseline_path) as fh:
        baseline = yaml.safe_load(fh)
    return check(results, baseline)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", required=True, help="path to results.json")
    ap.add_argument("--baseline", required=True, help="path to baseline YAML")
    args = ap.parse_args(argv)
    ok, rows = check_files(args.results, args.baseline)
    print(format_table(rows))
    n_fail = sum(1 for r in rows if r[1] in ("FAIL", "MISSING"))
    n_warn = sum(1 for r in rows if r[1] == "WARN")
    print(f"\n{'PASS' if ok else 'FAIL'}: "
          f"{len(rows)} checks, {n_fail} required-failure(s), {n_warn} warn.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
