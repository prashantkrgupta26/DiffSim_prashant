"""2-rank gloo distributed CG integration test.

Validates the N>=2 distributed path end-to-end on CPU:
  - TorchDistComm.exchange_halo / allreduce (halo loop)
  - padded all_gather in the driver (gather path)
  - β-recurrence / p-update over many iterations (--rhs random)

Both cases launch `scripts/nccl_cg_proof.py` under 2 gloo ranks via
``torchrun --standalone --nproc_per_node=2`` as a subprocess.  The gloo
backend is CPU-only; torch 2.x CPU is available in the project venv, so
these tests are Mac-runnable and close the coverage gap identified in the
final review (the distributed path had zero automated tests).

Skip conditions (never a hard failure on torch-less machines):
  - ``torch`` not installed
  - ``torchrun`` not on PATH / not in venv
  - ``torch.distributed.is_gloo_available()`` is False
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import pathlib

import pytest

# ---------------------------------------------------------------------------
# Module-level guards
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch", reason="torch not installed")


def _check_gloo():
    """Return True iff gloo backend is available in the installed torch."""
    try:
        import torch.distributed as dist
        return dist.is_gloo_available()
    except Exception:
        return False


def _find_torchrun() -> str | None:
    """Return path to torchrun, preferring the venv next to sys.executable."""
    # Primary: torchrun in the same venv bin dir as the running python
    venv_bin = pathlib.Path(sys.executable).parent
    candidate = venv_bin / "torchrun"
    if candidate.exists():
        return str(candidate)
    # Fallback: PATH
    return shutil.which("torchrun")


TORCHRUN = _find_torchrun()

if TORCHRUN is None:
    pytest.skip("torchrun not found; skipping distributed CG tests", allow_module_level=True)

if not _check_gloo():
    pytest.skip("torch.distributed gloo backend unavailable", allow_module_level=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DRIVER = str(pathlib.Path(__file__).parent.parent / "scripts" / "nccl_cg_proof.py")
TIMEOUT = 180  # seconds


def _run_2rank_gloo(*extra_args: str) -> subprocess.CompletedProcess:
    """Launch the driver under 2 gloo ranks via torchrun and return the result."""
    cmd = [
        TORCHRUN,
        "--standalone",
        "--nproc_per_node=2",
        DRIVER,
        "--backend", "gloo",
        "--precond", "jacobi",
        "--rhs", "random",
        *extra_args,
    ]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
    )


def _parse_pass_line(stdout: str) -> tuple[float, int]:
    """Extract (rel_err, iters) from the driver's PASS output line.

    The driver prints (rank 0):
        PASS  rel_err=<float> < 1e-8
    and earlier:
        iters      : <int>

    Returns
    -------
    rel_err : float
    iters   : int

    Raises
    ------
    AssertionError if the expected lines are not found.
    """
    # rel_err from the PASS line
    pass_match = re.search(r"PASS\s+rel_err=([0-9eE+\-\.]+)\s*<\s*1e-8", stdout)
    assert pass_match is not None, (
        "Could not find 'PASS  rel_err=...' in stdout.\n"
        f"stdout was:\n{stdout}"
    )
    rel_err = float(pass_match.group(1))

    # iters from the VALIDATION RESULTS block
    iters_match = re.search(r"iters\s*:\s*(\d+)", stdout)
    assert iters_match is not None, (
        "Could not find 'iters : <n>' in stdout.\n"
        f"stdout was:\n{stdout}"
    )
    iters = int(iters_match.group(1))

    return rel_err, iters


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDist2RankGloo:
    """End-to-end distributed CG tests: 2 ranks, gloo backend, CPU."""

    def test_synthetic_2rank_gloo(self):
        """Synthetic 7-point FD Poisson, 16x8x8 grid, 2 ranks, random RHS.

        Assertions:
          - process exits 0
          - stdout contains PASS with rel_err < 1e-8
          - iters > 5 (multi-iteration β-recurrence / halo loop actually ran)
        """
        result = _run_2rank_gloo("--stage", "synthetic", "--dims", "16,8,8")

        assert result.returncode == 0, (
            f"torchrun exited with code {result.returncode}.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

        rel_err, iters = _parse_pass_line(result.stdout)

        assert rel_err < 1e-8, (
            f"synthetic 2-rank gloo: rel_err={rel_err:.3e} >= 1e-8"
        )
        assert iters > 5, (
            f"synthetic 2-rank gloo: iters={iters} <= 5 — the multi-iteration "
            "halo/β-recurrence loop may not have run (random RHS should need >5 iters)"
        )

    def test_real_2rank_gloo(self):
        """Real octree K_p stiffness, level 3 (9^3=729 nodes), 2 ranks.

        Assertions:
          - process exits 0
          - stdout contains PASS with rel_err < 1e-8
          - iters > 5 (multi-iteration β-recurrence / halo loop actually ran)
        """
        result = _run_2rank_gloo("--stage", "real", "--level", "3")

        assert result.returncode == 0, (
            f"torchrun exited with code {result.returncode}.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

        rel_err, iters = _parse_pass_line(result.stdout)

        assert rel_err < 1e-8, (
            f"real 2-rank gloo: rel_err={rel_err:.3e} >= 1e-8"
        )
        assert iters > 5, (
            f"real 2-rank gloo: iters={iters} <= 5 — the multi-iteration "
            "halo/β-recurrence loop may not have run"
        )
