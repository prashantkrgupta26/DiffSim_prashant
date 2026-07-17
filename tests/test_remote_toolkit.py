import pathlib, subprocess
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
