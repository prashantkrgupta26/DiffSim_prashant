import pathlib, subprocess
import subprocess as _sp   # alias used in integration tests: _sp.run(["ssh", ...])
import pytest
REPO = pathlib.Path(__file__).resolve().parents[1]
RD = REPO / "scripts" / "remote"

def _source_var(var):
    out = subprocess.run(
        ["bash", "-c", f'source "{RD}/config.sh"; printf "%s" "${{{var}}}"'],
        capture_output=True, text=True)
    assert out.returncode == 0, f"sourcing config.sh failed: {out.stderr}"
    return out.stdout.strip()

def _sh(script, *args):
    # Shared helper: run a toolkit script, capture output. Used by many tasks.
    return subprocess.run(["bash", str(RD / script), *args],
                          capture_output=True, text=True)

def test_config_exports_core_vars():
    assert _source_var("GPUBOX_HOST") == "gpubox"
    assert _source_var("GPUBOX_REPO_ABS").endswith("Baskar/DiffSim")
    assert _source_var("GPUBOX_VENV_PY").endswith(".venv/bin/python")
    assert _source_var("GPUBOX_LOCK").endswith(".remote-run.lock")
    assert _source_var("GPUBOX_GIT_URL") == "gpubox:Baskar/DiffSim"
    assert _source_var("BOX_CLAUDE_FLAGS") == "--dangerously-skip-permissions"

def test_lib_log_path_and_tmux_name():
    out = subprocess.run(
        ["bash", "-c",
         f'source "{RD}/lib.sh"; '
         f'remote_log_path /logs 20260717-1 solve; echo; '
         f'remote_tmux_name "20260717/1"'],
        capture_output=True, text=True).stdout.split("\n")
    assert out[0] == "/logs/solve-20260717-1.log"
    assert out[1] == "diffsim-20260717-1"   # '/' sanitized to '-'

def test_nova_stubs_are_guarded():
    for s in ("nova-sync-submit.sh", "nova-poll.sh"):
        r = _sh(s)
        assert r.returncode == 64, f"{s} rc={r.returncode}"
        assert "not yet wired" in r.stderr

# NOTE: _sh() is defined once at the top — do not redefine in later task additions.
def _gpubox_up():
    return _sp.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                    "gpubox", "true"]).returncode == 0

needs_box = pytest.mark.skipif(not _gpubox_up(), reason="gpubox unreachable")

@needs_box
def test_doctor_green():
    r = _sh("remote-doctor.sh", "--fix")
    assert r.returncode == 0, f"doctor failed:\n{r.stdout}\n{r.stderr}"

@needs_box
def test_sync_refuses_under_lock():
    lock = _source_var("GPUBOX_LOCK")
    _sp.run(["ssh", "gpubox", f"touch '{lock}'"], check=True)
    try:
        r = _sh("gpubox-sync.sh", "--dry-run")
        assert r.returncode == 3, f"expected refusal (3), got {r.returncode}\n{r.stderr}"
        assert "run-lock" in r.stderr
    finally:
        _sp.run(["ssh", "gpubox", f"rm -f '{lock}'"], check=True)
