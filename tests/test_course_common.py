"""Warp-free tests for the OrgElMorph course common/ foundation.

Covers config load+validate+resolve+save, the tolerance baseline checker, and
provenance metadata assembly. No GPU required, so these gate in the CPU CI
(the course tutorial-smoke workflow points pytest at this file).
"""
import json
import os
import sys

import pytest
import yaml

_COURSE = os.path.join(os.path.dirname(__file__), os.pardir,
                       "tutorials", "orgelmorph-course")
sys.path.insert(0, os.path.abspath(_COURSE))

from common import check_results, config, run_base       # noqa: E402
from common.provenance import write_run_metadata          # noqa: E402


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------
def _schema():
    return config.ConfigSchema(name="p1", fields={
        "energy": config.Field(str, required=True,
                               choices=("poly", "fh", "both")),
        "level": config.Field(int, default=6, min=2, max=10),
        "steps": config.Field(int, default=250, min=1),
        "kappa": config.Field(float, default=5e-4, min=0.0),
        "seed": config.Field(int, default=3),
    })


def test_schema_defaults_and_coercion():
    cfg = _schema().validate({"energy": "poly", "level": 7})
    assert cfg["energy"] == "poly"
    assert cfg["level"] == 7
    assert cfg["steps"] == 250            # default filled
    assert isinstance(cfg["kappa"], float)


def test_missing_required_key_rejected():
    with pytest.raises(config.ConfigError) as e:
        _schema().validate({"level": 6})
    assert "energy" in str(e.value)


def test_bad_choice_and_range_rejected():
    with pytest.raises(config.ConfigError):
        _schema().validate({"energy": "nope"})
    with pytest.raises(config.ConfigError):
        _schema().validate({"energy": "poly", "level": 99})


def test_unknown_key_rejected():
    with pytest.raises(config.ConfigError):
        _schema().validate({"energy": "poly", "typo_key": 1})


def test_resolve_with_mode_and_overrides(tmp_path):
    cfg_yaml = tmp_path / "c.yaml"
    cfg_yaml.write_text(yaml.safe_dump({
        "energy": "poly", "level": 6, "steps": 250,
        "modes": {"quick": {"level": 5, "steps": 50},
                  "reference": {"level": 7, "steps": 250}},
    }))
    quick = config.resolve_config(str(cfg_yaml), schema=_schema(),
                                  mode="quick")
    assert quick["level"] == 5 and quick["steps"] == 50
    assert "modes" not in quick
    # CLI override wins over the mode block
    ref = config.resolve_config(str(cfg_yaml), schema=_schema(),
                                mode="reference", overrides={"steps": 10})
    assert ref["level"] == 7 and ref["steps"] == 10
    # None overrides are ignored
    keep = config.resolve_config(str(cfg_yaml), schema=_schema(),
                                 mode="quick", overrides={"steps": None})
    assert keep["steps"] == 50


def test_save_resolved_roundtrip(tmp_path):
    cfg = _schema().validate({"energy": "fh", "level": 6})
    p = tmp_path / "config.resolved.yaml"
    config.save_resolved(cfg, str(p))
    back = yaml.safe_load(p.read_text())
    assert back["energy"] == "fh" and back["level"] == 6


# --------------------------------------------------------------------------
# check_results
# --------------------------------------------------------------------------
def _results():
    return {"mass_drift": 4e-16, "Fend": 0.0494, "c_max": 0.99,
            "monotone": True, "nested": {"peak": 0.075}}


def test_checks_pass():
    baseline = {"checks": {
        "mass_drift": {"max": 1e-10},
        "Fend": {"reference": 0.0494, "rtol": 0.05},
        "c_max": {"min": 0.9, "max": 1.1},
        "monotone": {"equals": True},
        "nested.peak": {"reference": 0.075, "rtol": 0.1},
    }}
    ok, rows = check_results.check(_results(), baseline)
    assert ok
    assert all(r[1] == "PASS" for r in rows)


def test_required_failure_flags_nonzero():
    baseline = {"checks": {"mass_drift": {"max": 1e-20}}}   # too tight
    ok, rows = check_results.check(_results(), baseline)
    assert not ok and rows[0][1] == "FAIL"


def test_optional_failure_is_warn_not_fail():
    baseline = {"checks": {
        "Fend": {"reference": 999.0, "rtol": 0.01, "required": False}}}
    ok, rows = check_results.check(_results(), baseline)
    assert ok and rows[0][1] == "WARN"


def test_missing_required_key_fails():
    baseline = {"checks": {"does_not_exist": {"max": 1.0}}}
    ok, rows = check_results.check(_results(), baseline)
    assert not ok and rows[0][1] == "MISSING"


def test_check_files_and_main(tmp_path):
    rp = tmp_path / "results.json"
    bp = tmp_path / "baseline.yaml"
    rp.write_text(json.dumps(_results()))
    bp.write_text(yaml.safe_dump({"checks": {"mass_drift": {"max": 1e-10}}}))
    ok, rows = check_results.check_files(str(rp), str(bp))
    assert ok
    assert check_results.main(["--results", str(rp), "--baseline", str(bp)]) == 0


# --------------------------------------------------------------------------
# provenance + solver resolution (no GPU assumptions)
# --------------------------------------------------------------------------
def test_write_metadata_has_spec_fields(tmp_path):
    meta = write_run_metadata(str(tmp_path), run_info={"solver": "splu",
                                                       "seeds": [3]})
    assert os.path.exists(tmp_path / "metadata.json")
    for k in ("diffsim_commit", "timestamp_utc", "python_version",
              "wall_time_seconds", "exit_reason", "solver"):
        assert k in meta
    assert meta["solver"] == "splu"


def test_resolve_solver_maps_and_probes():
    backend, why = run_base.resolve_solver("splu")
    assert backend == "splu" and "explicit" in why
    backend, why = run_base.resolve_solver("matrix_free")
    assert backend == "fused"
    backend, why = run_base.resolve_solver("auto")
    assert backend in ("cudss", "splu")     # depends on box; both valid
    assert "auto ->" in why
