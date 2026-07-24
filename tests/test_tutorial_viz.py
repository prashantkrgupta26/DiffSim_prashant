"""Tests for tutorials/_viz.py (Phase-1 shared helper).

Test strategy
-------------
(a) No-viz path (always runs on base venv): monkeypatch HAS_VIZ=False.
    Every emitter must return None, print a hint, and never raise.
(b) vtu export on CPU (when meshio is available): deterministic, parseable.
(c) figures_dir: creates the directory on demand.
(d) GL-guarded render: skipped if pyvista / GL is unavailable.
"""
import pathlib
import importlib
import types

import numpy as np
import pytest


# ── fixture helpers ──────────────────────────────────────────────────────────

@pytest.fixture()
def viz_mod():
    """Fresh re-import of tutorials._viz so monkeypatching is clean."""
    import tutorials._viz as m
    return m


@pytest.fixture()
def no_viz(viz_mod, monkeypatch):
    """Force HAS_VIZ=False on the already-imported module."""
    monkeypatch.setattr(viz_mod, "HAS_VIZ", False)
    return viz_mod


@pytest.fixture()
def tmp_tutorial_file(tmp_path):
    """A fake tutorial __file__ inside a temp directory."""
    fake = tmp_path / "some_tutorial" / "tutorial.py"
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_text("# fake tutorial")
    return fake


# ── (a) No-viz path ──────────────────────────────────────────────────────────

def test_noviz_convergence_returns_none_no_raise(no_viz, tmp_tutorial_file,
                                                  capsys):
    result = no_viz.convergence(
        tmp_tutorial_file, [3, 4, 5], [1e-3, 5e-4, 2.5e-4], "conv"
    )
    assert result is None
    assert "pip install" in capsys.readouterr().out.lower()


def test_noviz_history_returns_none_no_raise(no_viz, tmp_tutorial_file,
                                              capsys):
    result = no_viz.history(
        tmp_tutorial_file, [0.0, 1.0], {"Cd": [1.0, 1.1]}, "history"
    )
    assert result is None
    assert "pip install" in capsys.readouterr().out.lower()


def test_noviz_surface_profile_returns_none_no_raise(no_viz, tmp_tutorial_file,
                                                      capsys):
    result = no_viz.surface_profile(
        tmp_tutorial_file, [0.0, 0.5, 1.0], {"u": [0.0, 1.0, 0.0]}, "profile"
    )
    assert result is None
    assert "pip install" in capsys.readouterr().out.lower()


def test_noviz_field_returns_none_no_raise(no_viz, tmp_tutorial_file, capsys):
    result = no_viz.field(tmp_tutorial_file, "dummy.vtu", "field")
    assert result is None
    assert "pip install" in capsys.readouterr().out.lower()


def test_noviz_vtu_returns_none_no_raise(no_viz, tmp_tutorial_file, capsys):
    result = no_viz.vtu(tmp_tutorial_file, "dummy", "mesh")
    assert result is None
    assert "pip install" in capsys.readouterr().out.lower()


def test_noviz_vtu_sbm_returns_none_no_raise(no_viz, tmp_tutorial_file, capsys):
    result = no_viz.vtu_sbm(
        tmp_tutorial_file, "dummy", "sbm",
        retained_tree=None, frac=np.array([1.0]),
    )
    assert result is None
    assert "pip install" in capsys.readouterr().out.lower()


def test_noviz_body_vtp_returns_none_no_raise(no_viz, tmp_tutorial_file, capsys):
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    tris = np.array([[0, 1, 2]], dtype=int)
    result = no_viz.body_vtp(tmp_tutorial_file, verts, tris, "body")
    assert result is None
    assert "pip install" in capsys.readouterr().out.lower()


def test_noviz_paraview_state_returns_none_no_raise(no_viz, tmp_tutorial_file,
                                                     capsys):
    result = no_viz.paraview_state(
        tmp_tutorial_file, "dummy.vtu", "state"
    )
    assert result is None
    assert "pip install" in capsys.readouterr().out.lower()


# ── (c) figures_dir ──────────────────────────────────────────────────────────

def test_figures_dir_creates_directory(viz_mod, tmp_tutorial_file):
    d = viz_mod.figures_dir(tmp_tutorial_file)
    assert d.is_dir()
    assert d.name == "figures"
    assert d.parent == tmp_tutorial_file.parent


def test_figures_dir_idempotent(viz_mod, tmp_tutorial_file):
    d1 = viz_mod.figures_dir(tmp_tutorial_file)
    d2 = viz_mod.figures_dir(tmp_tutorial_file)   # second call: dir already exists
    assert d1 == d2
    assert d1.is_dir()


# ── (b) vtu export on CPU (requires meshio) ──────────────────────────────────

try:
    import meshio as _meshio_probe  # noqa: F401
    _MESHIO_OK = True
except ImportError:
    _MESHIO_OK = False

try:
    import diffsim.viz  # noqa: F401
    _VIZ_OK = True
except Exception:
    _VIZ_OK = False


@pytest.mark.skipif(not _MESHIO_OK or not _VIZ_OK,
                    reason="meshio / diffsim.viz not installed")
def test_vtu_writes_parseable_file(tmp_tutorial_file):
    """export_vtu from a tiny point cloud → readable .vtu on CPU."""
    import tutorials._viz as viz

    # Use the .npz-path code-path (no GPU Mesh needed) — CPU-deterministic.
    import tempfile, os
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=np.float64)
    npz_path = tmp_tutorial_file.parent / "tiny.npz"
    np.savez(str(npz_path), coords=coords, u=np.array([0.0, 1.0, 0.5]))

    out = viz.vtu(tmp_tutorial_file, str(npz_path), "tiny_mesh")
    assert out is not None
    assert out.exists()
    assert out.suffix == ".vtu"
    # Verify it is parseable by meshio
    m = _meshio_probe.read(str(out))
    assert m.points.shape[0] == 3


# ── (d) GL-guarded render ────────────────────────────────────────────────────

@pytest.mark.skipif(not _VIZ_OK, reason="diffsim.viz not installed")
def test_field_skips_cleanly_without_gl(tmp_tutorial_file, monkeypatch):
    """If offscreen_gl_ok() returns False, field() returns None gracefully."""
    import tutorials._viz as viz
    from diffsim.viz import renders as rnd
    monkeypatch.setattr(rnd, "offscreen_gl_ok", lambda: False)
    result = viz.field(tmp_tutorial_file, "dummy.vtu", "vel",
                       field_name="velocity_magnitude", physics="ns")
    assert result is None
