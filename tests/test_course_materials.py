"""Warp-free tests for the OrgElMorph course materials loader.

Covers the schema-v2 materials.yaml loader (systems + crystallization, the
production/accelerated_tutorial honesty rule, provenance records) and the
original P4 ternary-blend loader. No GPU required -- gates in CPU CI next to
tests/test_course_common.py.
"""
import json
import os
import sys

import pytest
import yaml

_COURSE = os.path.join(os.path.dirname(__file__), os.pardir,
                       "tutorials", "orgelmorph-course")
sys.path.insert(0, os.path.abspath(os.path.join(_COURSE, "materials")))

import loader  # noqa: E402
from loader import (MaterialError, load_system, load_crystallization,  # noqa: E402
                    load_blend, available_systems, save_resolved_materials)

_MAT = os.path.join(_COURSE, "materials", "materials.yaml")


# --------------------------------------------------------------------------
# materials.yaml -- systems
# --------------------------------------------------------------------------
def test_real_systems_present():
    names = available_systems()
    for s in ("P3HT_PCBM", "PDPP5T_PCBM", "isoindigo_PCBM",
              "drying_demo_ternary"):
        assert s in names


def test_p3ht_values_and_provenance():
    s = load_system("P3HT_PCBM")
    assert s.value("chi_polymer_fullerene") == 0.86
    p = s.param("chi_polymer_fullerene")
    assert p.status == "production"
    assert p.units == "dimensionless"
    assert p.definition                       # non-empty
    assert p.source["citation"] == "wodo2012morphology"
    # polymer N is honestly null (a range, not a single number)
    assert s.value("N")[0] is None


def test_drying_demo_matches_p5_baseline_numbers():
    """The P5 default blend must load the EXACT numbers P5 hard-coded, so the
    loader wiring does not change the run."""
    s = load_system("drying_demo_ternary")
    assert s.value("chi") == [1.5, 0.3, 0.3]
    assert s.value("N") == [5.0, 5.0, 1.0]
    assert s.param("chi").status == "accelerated_tutorial"


def test_accelerated_null_value_allowed():
    """isoindigo chi is intentionally null (recovered later); allowed only
    because it is marked accelerated_tutorial."""
    s = load_system("isoindigo_PCBM")
    assert s.value("chi_polymer_fullerene") is None
    assert s.param("chi_polymer_fullerene").status == "accelerated_tutorial"


def test_crystallization_pcbm_class_values():
    c = load_crystallization("PCBM_class")
    assert c.value("dsig") == 2.6355
    assert c.value("dh") == 1.3072
    assert c.value("Tm") == 558.0
    assert c.value("eps2") == 1.0e-3
    assert c.value("chi_aa") == 0.7248
    assert c.value("chi_ca") == 1.0836
    assert c.value("N") == [5.0298, 1.0]
    # Tm is physical -> production; requires units+definition
    assert c.param("Tm").status == "production"
    assert c.param("Tm").units == "K"


# --------------------------------------------------------------------------
# honesty rule (production requires units/definition/non-null value)
# --------------------------------------------------------------------------
def _write(tmp_path, block):
    p = tmp_path / "m.yaml"
    p.write_text(yaml.safe_dump({"systems": {"S": {
        "description": "x", "species": ["a"], "parameters": block}}}))
    return str(p)


def test_production_missing_units_rejected(tmp_path):
    path = _write(tmp_path, {"chi": {
        "value": 1.0, "definition": "d", "status": "production"}})
    with pytest.raises(MaterialError) as e:
        load_system("S", path=path)
    assert "units" in str(e.value)


def test_production_missing_definition_rejected(tmp_path):
    path = _write(tmp_path, {"chi": {
        "value": 1.0, "units": "dimensionless", "status": "production"}})
    with pytest.raises(MaterialError) as e:
        load_system("S", path=path)
    assert "definition" in str(e.value)


def test_missing_status_defaults_to_production_strict(tmp_path):
    # no status + no units -> treated as production -> rejected
    path = _write(tmp_path, {"chi": {"value": 1.0, "definition": "d"}})
    with pytest.raises(MaterialError):
        load_system("S", path=path)


def test_production_null_value_rejected(tmp_path):
    path = _write(tmp_path, {"chi": {
        "value": None, "units": "u", "definition": "d",
        "status": "production"}})
    with pytest.raises(MaterialError):
        load_system("S", path=path)


def test_accelerated_allows_missing_units_and_definition(tmp_path):
    path = _write(tmp_path, {"chi": {
        "value": None, "status": "accelerated_tutorial"}})
    s = load_system("S", path=path)          # must not raise
    assert s.value("chi") is None


def test_bad_status_and_source_type_rejected(tmp_path):
    bad_status = _write(tmp_path, {"chi": {"value": 1.0, "status": "guess"}})
    with pytest.raises(MaterialError):
        load_system("S", path=bad_status)
    bad_src = tmp_path / "m2.yaml"
    bad_src.write_text(yaml.safe_dump({"systems": {"S": {
        "parameters": {"chi": {
            "value": 1.0, "units": "u", "definition": "d",
            "status": "production", "source": {"type": "vibes"}}}}}}))
    with pytest.raises(MaterialError):
        load_system("S", path=str(bad_src))


def test_missing_value_key_rejected(tmp_path):
    path = _write(tmp_path, {"chi": {"units": "u", "definition": "d"}})
    with pytest.raises(MaterialError):
        load_system("S", path=path)


def test_unknown_system_rejected():
    with pytest.raises(MaterialError):
        load_system("does_not_exist")


# --------------------------------------------------------------------------
# provenance record + run-output archival
# --------------------------------------------------------------------------
def test_record_is_json_serializable_and_complete():
    s = load_system("P3HT_PCBM")
    rec = s.record()
    txt = json.dumps(rec)                    # must serialize
    assert "wodo2012morphology" in txt
    p = rec["parameters"]["chi_polymer_fullerene"]
    for k in ("value", "units", "definition", "nondimensionalization",
              "status", "source", "uncertainty", "validity", "notes"):
        assert k in p


def test_save_resolved_materials_roundtrip(tmp_path):
    s = load_system("drying_demo_ternary")
    c = load_crystallization("PCBM_class")
    out = tmp_path / "materials.resolved.json"
    save_resolved_materials([s, c], str(out))
    back = json.loads(out.read_text())
    names = [m["name"] for m in back["materials"]]
    assert names == ["drying_demo_ternary", "PCBM_class"]


# --------------------------------------------------------------------------
# P4 ternary blends (original API) still works
# --------------------------------------------------------------------------
def test_load_blend_still_works():
    b = load_blend("synthetic_demix")
    assert len(b.chi) == 3 and len(b.N) == 3
    assert b.chi12 == b.chi[0]


def test_load_blend_range_check():
    with pytest.raises(MaterialError):
        load_blend("does_not_exist")


# --------------------------------------------------------------------------
# the real materials.yaml validates end-to-end (every declared system loads)
# --------------------------------------------------------------------------
def test_all_shipped_systems_load():
    for name in available_systems():
        load_system(name)
    for name in available_systems(section="crystallization"):
        load_crystallization(name)
