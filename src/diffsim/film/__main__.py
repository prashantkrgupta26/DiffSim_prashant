"""CLI: python -m diffsim.film <config.yaml> [options]

Options:
  --outdir DIR        output directory (default runs/<name>)
  --device DEV        cuda:0 / cuda:1 / cpu (overrides config)
  --set key=value     dotted override into the YAML tree, repeatable,
                      value parsed as YAML (e.g. --set evaporation.Bi=0.1
                      --set domain.resolution=[125,125,250]
                      --set domain.dim=3)
  --preflight-only    print the preflight report and exit (no compute)
  --no-strict         continue past preflight FAILs (preflight: warn)
"""
import argparse
import json
import sys

from .params import FilmParams
from .preflight import run_preflight, PreflightError
from .run import FilmRun


def _apply_set(tree, spec):
    key, _, val = spec.partition("=")
    if not _:
        raise SystemExit(f"--set needs key=value, got {spec!r}")
    try:
        import yaml
        val = yaml.safe_load(val)
    except ImportError:
        try:
            val = json.loads(val)
        except json.JSONDecodeError:
            pass                                  # keep the raw string
    node = tree
    parts = key.split(".")
    for k in parts[:-1]:
        node = node.setdefault(k, {})
    node[parts[-1]] = val


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m diffsim.film",
        description=__doc__.split("\n")[0])
    ap.add_argument("config", help="YAML config (see diffsim/film/configs)")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--set", action="append", default=[], dest="sets",
                    metavar="KEY=VALUE")
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--no-strict", action="store_true")
    args = ap.parse_args(argv)

    with open(args.config) as fh:
        text = fh.read()
    try:
        import yaml
        tree = yaml.safe_load(text)
    except ImportError:
        tree = json.loads(text)
    for spec in args.sets:
        _apply_set(tree, spec)
    params = FilmParams.from_dict(tree)
    if args.device:
        params = params.replace(device=args.device)
    if args.no_strict:
        params = params.replace(preflight="warn")

    if args.preflight_only:
        report = run_preflight(params, params.resolve())
        print(report.format(params, params.resolve()))
        return 1 if report.failed else 0

    outdir = args.outdir or f"runs/{params.name}"
    try:
        summary = FilmRun(params).run(outdir)
    except PreflightError as e:
        print(f"ABORT: {e}", flush=True)
        return 1
    return 2 if "autopsy" in summary else 0


if __name__ == "__main__":
    sys.exit(main())
