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
