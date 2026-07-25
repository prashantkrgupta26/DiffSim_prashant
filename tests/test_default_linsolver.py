"""Contract tests for diffsim.device.default_linsolver.

Mirrors default_device's precedence design: explicit arg > env > auto-by-device.
CPU-only (no Warp/CUDA needed) — these lock the onboarding auto-fallback that
lets CPU machines run the GPU-defaulted benchmarks without --linsolver.
"""
import importlib

import pytest

from diffsim import default_linsolver


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("DIFFSIM_LINSOLVER", raising=False)
    monkeypatch.delenv("DIFFSIM_DEVICE", raising=False)


def test_cpu_device_picks_host_direct():
    assert default_linsolver(device="cpu") == "splu"


def test_cuda_device_picks_cudss():
    assert default_linsolver(device="cuda:0") == "cudss"
    assert default_linsolver(device="cuda:1") == "cudss"


def test_prefer_arg_wins_over_everything(monkeypatch):
    monkeypatch.setenv("DIFFSIM_LINSOLVER", "blockch")
    assert default_linsolver(device="cpu", prefer="cudss") == "cudss"


def test_env_override_honored(monkeypatch):
    monkeypatch.setenv("DIFFSIM_LINSOLVER", "blockch")
    assert default_linsolver(device="cpu") == "blockch"
    assert default_linsolver(device="cuda:0") == "blockch"


def test_auto_follows_default_device_when_no_device(monkeypatch):
    monkeypatch.setenv("DIFFSIM_DEVICE", "cpu")
    assert default_linsolver() == "splu"


def test_exported_from_package():
    mod = importlib.import_module("diffsim")
    assert "default_linsolver" in mod.__all__
    assert mod.default_linsolver is default_linsolver
