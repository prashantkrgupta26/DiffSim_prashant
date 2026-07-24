"""Smoke tests for the .py tutorial track (Phase-1 visualization rollout).

Strategy: run each touched tutorial's main() (or a named demo function) at
a TINY problem size with _viz.HAS_VIZ monkeypatched to False — so:
  - No GPU / Warp is needed for *this* test file (all GPU imports happen
    inside the tutorial modules, which are not imported here; we import
    them lazily so the collector can run without GPU too).
  - No viz library is needed (HAS_VIZ=False path).
  - The test verifies the tutorial runs to completion and exits without
    raising.

All GPU-using tutorials skip when Warp is not available.

Size knobs used (fast on CPU if Warp is present):
  A1: levels (2, 3) × p=1 only          — 2 small solves
  D1: levels (2, 3) × nu=0.1            — 2 small solves (no advection-dom run)
  D2: level=3, max_steps=5              — 5 pseudo-steps only
  D3: level=3, max_steps=5              — 5 pseudo-steps only
  F1: level=3, nsteps=3 / level=3, nsteps=3
  F2: level=3, nsteps=3
  F3: level=3, n_epochs=1, steps_before=2, steps_per_epoch=2
  E1: n_iters=2                          — 2 optimization steps
  P1: levels=(3, 4), warm_level=3
  P2: level=4                            — tiny CG run
"""
import sys
import types
import importlib
import pathlib

import pytest

# ── helpers ──────────────────────────────────────────────────────────────────

# Warp availability — tutorials call warp.init() at module import time via
# diffsim.default_device().  Skip all GPU-touching tests when Warp is absent.
try:
    import warp as _wp
    _wp.init()
    HAS_WARP = True
except Exception:
    HAS_WARP = False

needs_warp = pytest.mark.skipif(not HAS_WARP,
                                reason="Warp not installed (CPU-only tier)")


def _force_no_viz(monkeypatch, tutorial_module):
    """Monkeypatch _viz.HAS_VIZ=False on the _viz module the tutorial imported."""
    viz_mod = getattr(tutorial_module, "_viz", None)
    if viz_mod is None:
        # Try looking it up by name (the tutorial did `import _viz as _viz`)
        viz_mod = sys.modules.get("_viz")
    if viz_mod is not None:
        monkeypatch.setattr(viz_mod, "HAS_VIZ", False)


def _add_tutorials_path():
    """Ensure the tutorials/ directory is on sys.path so `import _viz` works."""
    tut_dir = str(pathlib.Path(__file__).parent.parent / "tutorials")
    if tut_dir not in sys.path:
        sys.path.insert(0, tut_dir)


# ── A1: MMS convergence ──────────────────────────────────────────────────────

@needs_warp
def test_A1_mms_convergence_noviz(monkeypatch):
    """A1 main() at tiny size: exits 0, viz forced off."""
    _add_tutorials_path()
    import importlib.util, importlib
    spec = importlib.util.spec_from_file_location(
        "A1_mms_convergence",
        str(pathlib.Path(__file__).parent.parent
            / "tutorials/A_foundations/A1_mms_convergence.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _force_no_viz(monkeypatch, mod)
    # Two tiny levels for p=1 only (no p=2 in tiny mode)
    result = mod.main(levels_p1=(2, 3), levels_p2=(2, 3))
    assert result is not None


# ── D1: NS MMS ───────────────────────────────────────────────────────────────

@needs_warp
def test_D1_ns_mms_noviz(monkeypatch):
    """D1 main() at tiny size: exits 0, viz forced off."""
    _add_tutorials_path()
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "D1_ns_mms",
        str(pathlib.Path(__file__).parent.parent
            / "tutorials/D_flow/D1_ns_mms.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _force_no_viz(monkeypatch, mod)
    eu, ep = mod.main(levels=(2, 3))
    assert len(eu) == 2


# ── D2: Lid-driven cavity ────────────────────────────────────────────────────

@needs_warp
def test_D2_lid_driven_cavity_noviz(monkeypatch):
    """D2 main() at tiny size: exits 0, viz forced off."""
    _add_tutorials_path()
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "D2_lid_driven_cavity",
        str(pathlib.Path(__file__).parent.parent
            / "tutorials/D_flow/D2_lid_driven_cavity.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _force_no_viz(monkeypatch, mod)
    u_c = mod.main(level=3, max_steps=5)
    assert u_c is not None


# ── D3: Cylinder ─────────────────────────────────────────────────────────────

@needs_warp
def test_D3_cylinder_noviz(monkeypatch):
    """D3 main() at tiny size: exits 0, viz forced off."""
    _add_tutorials_path()
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "D3_cylinder",
        str(pathlib.Path(__file__).parent.parent
            / "tutorials/D_flow/D3_cylinder.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _force_no_viz(monkeypatch, mod)
    # level=4 minimum: at level=3 the cylinder (r=0.07) is too large relative
    # to the coarse mesh — GeometryData.evaluate gets an empty surrogate face
    # set and hits a max() on an empty tensor.
    Cd, Cl = mod.main(level=4, max_steps=5)
    assert isinstance(Cd, float)


# ── F1: Allen-Cahn ───────────────────────────────────────────────────────────

@needs_warp
def test_F1_allen_cahn_noviz(monkeypatch):
    """F1 demo functions at tiny size: exits 0, viz forced off."""
    _add_tutorials_path()
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "F1_allen_cahn",
        str(pathlib.Path(__file__).parent.parent
            / "tutorials/F_phasefield/F1_allen_cahn.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _force_no_viz(monkeypatch, mod)
    mod.demo_1_coarsening(level=3, nsteps=3, print_every=1)
    mod.demo_2_shrinking_circle(level=3, nsteps=3)


# ── F2: Cahn-Hilliard ────────────────────────────────────────────────────────

@needs_warp
def test_F2_cahn_hilliard_noviz(monkeypatch):
    """F2 main() at tiny size: exits 0, viz forced off."""
    _add_tutorials_path()
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "F2_cahn_hilliard",
        str(pathlib.Path(__file__).parent.parent
            / "tutorials/F_phasefield/F2_cahn_hilliard.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _force_no_viz(monkeypatch, mod)
    mod.main(level=3, nsteps=3)


# ── F3: Adaptivity ───────────────────────────────────────────────────────────

@needs_warp
def test_F3_adaptivity_noviz(monkeypatch):
    """F3 main() at tiny size: exits 0, viz forced off."""
    _add_tutorials_path()
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "F3_adaptivity",
        str(pathlib.Path(__file__).parent.parent
            / "tutorials/F_phasefield/F3_adaptivity.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _force_no_viz(monkeypatch, mod)
    mod.main(n_epochs=1, steps_before=2, steps_per_epoch=2, level=3)


# ── E1: Shape optimization ────────────────────────────────────────────────────

@needs_warp
def test_E1_shape_optimization_noviz(monkeypatch):
    """E1 main() at 2-iteration tiny run: exits 0, viz forced off."""
    _add_tutorials_path()
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "E1_shape_optimization",
        str(pathlib.Path(__file__).parent.parent
            / "tutorials/E_differentiable/E1_shape_optimization.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _force_no_viz(monkeypatch, mod)
    found = mod.main(n_iters=2)
    assert found is not None and len(found) == 3


# ── P1: Cost model ────────────────────────────────────────────────────────────

@needs_warp
def test_P1_cost_model_noviz(monkeypatch):
    """P1 main() at tiny levels: exits 0, viz forced off."""
    _add_tutorials_path()
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "P1_cost_model_and_scaling",
        str(pathlib.Path(__file__).parent.parent
            / "tutorials/P_performance/P1_cost_model_and_scaling.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _force_no_viz(monkeypatch, mod)
    rows = mod.main(levels=(3, 4), warm_level=3)
    assert len(rows) == 2


# ── P2: Solver showdown ───────────────────────────────────────────────────────

@needs_warp
def test_P2_solver_showdown_noviz(monkeypatch):
    """P2 main() at level 4: exits 0, viz forced off."""
    _add_tutorials_path()
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "P2_solver_showdown",
        str(pathlib.Path(__file__).parent.parent
            / "tutorials/P_performance/P2_solver_showdown.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _force_no_viz(monkeypatch, mod)
    t_direct, t_host, t_fused = mod.main(level=4)
    assert t_direct > 0 and t_host > 0 and t_fused > 0
