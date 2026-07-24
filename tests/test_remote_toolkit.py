import pathlib, subprocess, time
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

def test_nova_scripts_require_args():
    # The Nova scripts are now WIRED (git-bundle ship + sbatch + <=4-GPU cap /
    # squeue/sacct poll). Run with no args they must still be guarded: print a
    # usage line and exit 64 (EX_USAGE) rather than doing anything unguarded.
    for s in ("nova-sync-submit.sh", "nova-poll.sh"):
        r = _sh(s)
        assert r.returncode == 64, f"{s} rc={r.returncode}"
        assert "usage:" in r.stderr.lower()

# NOTE: _sh() is defined once at the top — do not redefine in later task additions.
def _gpubox_up():
    return _sp.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                    "gpubox", "true"]).returncode == 0

needs_box = pytest.mark.skipif(not _gpubox_up(), reason="gpubox unreachable")

def _box_claude_authed():
    # Headless dispatch needs the box's Claude CLI logged in. Mirrors _gpubox_up:
    # evaluated at import to gate the dispatch test with a skip (not a failure)
    # when the OAuth session is absent/expired.
    if not _gpubox_up():
        return False
    claude_bin = _source_var("CLAUDE_BIN")
    r = _sp.run(["ssh", "-o", "BatchMode=yes", "gpubox",
                 f"{claude_bin} auth status 2>/dev/null"],
                capture_output=True, text=True)
    return '"loggedIn": true' in r.stdout

needs_box_auth = pytest.mark.skipif(
    not _box_claude_authed(),
    reason="box-Claude not authenticated — run `claude auth login` on gpubox")

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

@needs_box
def test_run_poll_lock_lifecycle():
    r = _sh("gpubox-run.sh", "echo hello-from-box; sleep 3", "smoke")
    assert r.returncode == 0, r.stderr
    sess, log = r.stdout.strip().split("\t")
    assert sess.startswith("diffsim-")
    lock = _source_var("GPUBOX_LOCK")
    # lock exists while running
    assert _sp.run(["ssh", "gpubox", f"test -e '{lock}'"]).returncode == 0
    # poll until DONE (cap ~30s)
    done = False
    for _ in range(15):
        p = _sh("gpubox-poll.sh", log)
        if "status=DONE" in p.stdout + p.stderr:
            done = True; break
        time.sleep(2)
    assert done, "run never reported DONE"
    # log captured output, lock cleared
    p = _sh("gpubox-poll.sh", log)
    assert "hello-from-box" in p.stdout
    assert _sp.run(["ssh", "gpubox", f"test -e '{lock}'"]).returncode != 0

@needs_box_auth
def test_dispatch_roundtrip():
    r = _sh("gpubox-dispatch.sh", "Reply with exactly the token PONG and nothing else.")
    assert "PONG" in r.stdout.upper(), f"stdout={r.stdout!r} stderr={r.stderr!r}"

@needs_box
def test_run_refuses_when_locked():
    lock = _source_var("GPUBOX_LOCK")
    _sp.run(["ssh", "gpubox", f"touch '{lock}'"], check=True)
    try:
        r = _sh("gpubox-run.sh", "echo nope", "smoke")
        assert r.returncode == 3, f"expected refusal (3), got {r.returncode}\n{r.stderr}"
        assert "run-lock" in r.stderr
    finally:
        _sp.run(["ssh", "gpubox", f"rm -f '{lock}'"], check=True)

@needs_box
def test_fetch_noop_clean():
    r = _sh("gpubox-fetch.sh")
    assert r.returncode == 0, f"stderr={r.stderr}"
    assert _sp.run(["git", "-C", str(REPO), "remote", "get-url", "gpubox"]).returncode == 0

@needs_box
def test_fetch_artifacts_pulls_files(tmp_path):
    # Create a marker artifact in a temp subdir on the box, pull it, verify contents.
    repo_abs = _source_var("GPUBOX_REPO_ABS")
    sub = "_artifact_test"
    _sp.run(["ssh", "gpubox",
             f"mkdir -p '{repo_abs}/{sub}' && echo hello-artifact > '{repo_abs}/{sub}/marker.txt'"],
            check=True)
    try:
        dest = tmp_path / "pulled"
        r = _sh("fetch-artifacts.sh", sub, str(dest))
        assert r.returncode == 0, f"stderr={r.stderr}"
        assert (dest / "marker.txt").read_text().strip() == "hello-artifact"
    finally:
        _sp.run(["ssh", "gpubox", f"rm -rf '{repo_abs}/{sub}'"], check=True)
