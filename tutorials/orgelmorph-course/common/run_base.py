"""Standardized run harness every tutorial ``run.py`` adopts.

Provides the common CLI (``--config --device --solver --output --seed --mode
--overwrite --resume --log-level``), the standard output-directory layout, and
the glue that ties config + provenance + results + tolerance-check together so
each tutorial is a *repeatable workflow*, not a bespoke script (spec Phase 0).

Output layout (``outputs/<run>/``):
    config.resolved.yaml   the canonical, re-runnable config record
    metadata.json          environment + run fingerprint (provenance)
    results.json           the scalar results the tutorial reports/checks
    history.npz            time-series / field arrays for figures
    run.log                the run's log
    checkpoints/           solver checkpoints (resume)
    figures/              generated figures

A tutorial writes a ``run_fn(cfg, ctx) -> results_dict`` and calls
:func:`run_tutorial`. Everything else (dirs, logging, provenance, solver
resolution, saving, checking) is handled here.

`--solver auto`: cuDSS when available (the measured stepping default at scale),
otherwise a documented small-problem fallback to scipy SuperLU (``splu``) — the
CPU host direct solver that is unbeatable at the tutorial's small sizes but
pays a full factorization every step. The choice + rationale is logged and
recorded in metadata.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys

import numpy as np

from . import config as _config
from .provenance import RunProvenance
from . import check_results as _check

# CLI solver name -> the linsolver string the diffsim steppers expect.
SOLVER_ALIASES = {
    "cudss": "cudss",
    "splu": "splu",
    "blockch": "blockch",
    "amgx": "amgx",
    "matrix_free": "fused",
}
SOLVER_CHOICES = ["auto", "cudss", "splu", "blockch", "amgx", "matrix_free"]
MODE_CHOICES = ["quick", "reference", "research"]


# --------------------------------------------------------------------------
# solver availability probes + resolution
# --------------------------------------------------------------------------
def cudss_available():
    """Whether the GPU direct solver cuDSS (via nvmath) can be used here."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        import nvmath.sparse.advanced  # noqa: F401
        return True
    except Exception:
        return False


def amgx_available():
    """Whether NVIDIA AMGX (via pyamgx) is importable."""
    try:
        import pyamgx  # noqa: F401
        return True
    except Exception:
        return False


def resolve_solver(name, dofs=None):
    """Map a CLI ``--solver`` choice to a concrete backend + rationale.

    Returns ``(linsolver_string, rationale)``. ``auto`` probes cuDSS and falls
    back to ``splu`` for small problems, explaining the choice. A requested
    backend that is unavailable is reported honestly in the rationale (the
    caller may still proceed and let the stepper raise a clear error).
    """
    if name == "auto":
        if cudss_available():
            return "cudss", "auto -> cuDSS (GPU direct solver available)"
        return ("splu", "auto -> scipy SuperLU (cuDSS unavailable; host "
                "direct solver, fine for the small tutorial sizes)")
    backend = SOLVER_ALIASES[name]
    if name == "cudss" and not cudss_available():
        return backend, "cudss requested but nvmath/CUDA not available here"
    if name == "amgx" and not amgx_available():
        return backend, "amgx requested but pyamgx not built here"
    return backend, f"{name} (explicit)"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser(description="OrgElMorph tutorial", parents=None):
    """The standard tutorial parser. Tutorials can pass extra ``parents`` or
    add their own arguments to the returned parser."""
    ap = argparse.ArgumentParser(description=description,
                                 parents=parents or [])
    ap.add_argument("--config", default=None,
                    help="YAML config (canonical run record)")
    ap.add_argument("--device", default="cuda:0",
                    help="compute device, e.g. cuda:0 or cpu")
    ap.add_argument("--solver", default="auto", choices=SOLVER_CHOICES,
                    help="linear solver backend (auto picks cuDSS/splu)")
    ap.add_argument("--output", default=None,
                    help="output run directory (default outputs/<name>)")
    ap.add_argument("--seed", type=int, default=None,
                    help="RNG seed (overrides the config)")
    ap.add_argument("--mode", default="quick", choices=MODE_CHOICES,
                    help="cost tier: quick <2min / reference 5-30min / research")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an existing output directory")
    ap.add_argument("--resume", action="store_true",
                    help="resume from checkpoints/ if present")
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return ap


# --------------------------------------------------------------------------
# output directory + logging
# --------------------------------------------------------------------------
def prepare_output_dir(path, overwrite=False, resume=False):
    """Create the standard run directory layout and return the paths dict."""
    if os.path.exists(path) and os.listdir(path):
        if overwrite:
            shutil.rmtree(path)
        elif not resume:
            raise SystemExit(
                f"output dir '{path}' exists and is non-empty; pass "
                f"--overwrite to replace or --resume to continue.")
    os.makedirs(path, exist_ok=True)
    sub = {
        "root": path,
        "config": os.path.join(path, "config.resolved.yaml"),
        "metadata": os.path.join(path, "metadata.json"),
        "results": os.path.join(path, "results.json"),
        "history": os.path.join(path, "history.npz"),
        "log": os.path.join(path, "run.log"),
        "checkpoints": os.path.join(path, "checkpoints"),
        "figures": os.path.join(path, "figures"),
    }
    os.makedirs(sub["checkpoints"], exist_ok=True)
    os.makedirs(sub["figures"], exist_ok=True)
    return sub


def setup_logging(log_path, level="INFO", name="orgelmorph"):
    """Logger that tees to the console and to ``run.log``."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s",
                            "%H:%M:%S")
    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


# --------------------------------------------------------------------------
# results / history IO (JSON-safe)
# --------------------------------------------------------------------------
def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_results(path, results):
    """Write ``results.json`` (JSON-safe: numpy scalars/arrays coerced)."""
    with open(path, "w") as fh:
        json.dump(_json_safe(results), fh, indent=2, sort_keys=True)


def save_history(path, history):
    """Write ``history.npz`` from a dict of arrays (skipped if empty)."""
    if not history:
        return
    np.savez(path, **{k: np.asarray(v) for k, v in history.items()})


# --------------------------------------------------------------------------
# the run context passed to a tutorial's run_fn
# --------------------------------------------------------------------------
class RunContext:
    """Everything a tutorial's ``run_fn`` needs, plus a provenance channel.

    Attributes
    ----------
    cfg : dict            the resolved config
    paths : dict          output-directory paths (see prepare_output_dir)
    device, solver : str  resolved device / linsolver string
    solver_rationale : str  why that solver was chosen
    mode, seed : ...      run mode and seed
    logger : Logger       tees to console + run.log
    provenance : RunProvenance   call .update(**fields) to record mesh/tols

    A run_fn returns ``results`` (dict, saved to results.json). It may store
    time series for figures by assigning ``ctx.history`` (dict of arrays).
    """

    def __init__(self, cfg, paths, device, solver, solver_rationale,
                 mode, seed, logger, provenance):
        self.cfg = cfg
        self.paths = paths
        self.device = device
        self.solver = solver
        self.solver_rationale = solver_rationale
        self.mode = mode
        self.seed = seed
        self.logger = logger
        self.provenance = provenance
        self.history = {}

    def log(self, msg):
        self.logger.info(msg)


def run_tutorial(run_fn, schema=None, args=None, description="OrgElMorph tutorial",
                 default_output=None, baseline=None, cli_overrides=None,
                 extra_run_info=None, default_solver=None):
    """Drive one tutorial run end-to-end.

    Parameters
    ----------
    run_fn : callable
        ``run_fn(cfg, ctx) -> results_dict``. Fills ``ctx.history`` for figures
        and ``ctx.provenance.update(...)`` for mesh/tolerance provenance.
    schema : ConfigSchema, optional
        Config schema to validate against.
    args : argparse.Namespace, optional
        Pre-parsed args; if ``None`` the standard parser is used on ``sys.argv``.
    default_output : str, optional
        Output dir when ``--output`` is not given.
    baseline : str, optional
        Path to a baseline YAML; if present, results are checked and the
        harness exits non-zero on a required-check failure.
    cli_overrides : dict, optional
        Extra config overrides derived from tutorial-specific CLI flags.
    extra_run_info : dict, optional
        Additional provenance fields to record.
    default_solver : str, optional
        Backend to use for ``--solver auto`` in place of the global auto
        (cuDSS-else-splu) policy — the chapter's documented small-problem
        choice. An explicit ``--solver`` on the CLI still overrides it.

    Returns
    -------
    dict
        The results dict (also written to results.json).
    """
    if args is None:
        args = build_parser(description).parse_args()

    # Device/solver are RUN-level arguments (available via ctx.device /
    # ctx.solver and recorded in metadata), not config fields — do NOT inject
    # them into the config overrides, or a schema that doesn't declare them
    # would reject the run. A tutorial that genuinely wants device IN its
    # config declares the field and sets it in the YAML.
    overrides = dict(cli_overrides or {})
    if args.seed is not None:
        overrides["seed"] = args.seed

    if args.config:
        cfg = _config.resolve_config(args.config, schema=schema,
                                     overrides=overrides, mode=args.mode)
    else:
        # no config file: validate the overrides/defaults through the schema
        cfg = (schema or _config.ConfigSchema()).validate(overrides)

    out = args.output or default_output or os.path.join("outputs", "run")
    paths = prepare_output_dir(out, overwrite=args.overwrite,
                               resume=args.resume)
    logger = setup_logging(paths["log"], args.log_level)

    # A chapter may declare its own documented `auto` choice: some tutorials
    # solve a small INDEFINITE saddle-point system (e.g. the binary CH (c, mu)
    # block) where cuDSS's lack of partial pivoting diverges but scipy
    # SuperLU's pivoting is exact. `default_solver` lets the chapter pin that
    # documented fallback for `--solver auto` while still honouring an explicit
    # --solver from an advanced user.
    dofs = cfg.get("dofs")
    requested = args.solver
    if requested == "auto" and default_solver:
        solver, rationale = resolve_solver(default_solver, dofs=dofs)
        rationale = f"auto -> {rationale} [chapter default_solver]"
    else:
        solver, rationale = resolve_solver(requested, dofs=dofs)
    logger.info("solver: %s", rationale)

    _config.save_resolved(cfg, paths["config"])
    logger.info("resolved config -> %s", paths["config"])

    seed = cfg.get("seed", args.seed)
    run_info = {
        "device": args.device,
        "solver": solver,
        "solver_rationale": rationale,
        "mode": args.mode,
        "seeds": [seed] if seed is not None else None,
        "precision": cfg.get("precision", "fp64"),
        "config_path": os.path.abspath(paths["config"]),
    }
    if extra_run_info:
        run_info.update(extra_run_info)

    results = None
    with RunProvenance(paths["root"], run_info) as prov:
        ctx = RunContext(cfg, paths, args.device, solver, rationale,
                         args.mode, seed, logger, prov)
        results = run_fn(cfg, ctx)
        save_results(paths["results"], results)
        save_history(paths["history"], ctx.history)
        prov.set_exit("completed")
    logger.info("results -> %s", paths["results"])
    logger.info("metadata -> %s", paths["metadata"])

    if baseline:
        ok, rows = _check.check_files(paths["results"], baseline)
        logger.info("baseline check:\n%s", _check.format_table(rows))
        if not ok:
            logger.error("baseline check FAILED")
            raise SystemExit(2)
        logger.info("baseline check PASSED")

    return results


__all__ = [
    "SOLVER_CHOICES", "MODE_CHOICES", "SOLVER_ALIASES",
    "cudss_available", "amgx_available", "resolve_solver",
    "build_parser", "prepare_output_dir", "setup_logging",
    "save_results", "save_history", "RunContext", "run_tutorial",
]
