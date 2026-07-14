"""Unit tests for diffsim.diagnostics — Warp-free, analytic ground truth.

Every diagnostic in the library gets a test against a case whose answer is
known in closed form (a constant field integrates to value*volume, a pure
sinusoid has a known wavelength, errors that quarter give observed order 2,
fields on the simplex have zero residual, ...). These run on CPU with numpy
only, so they gate in the GitHub-hosted CPU CI (no GPU required).
"""
import numpy as np
import pytest

from diffsim.diagnostics import (admissibility, conservation, convergence,
                                  energy, morphology, profiling, provenance,
                                  stochastic)


# --------------------------------------------------------------------------
# conservation
# --------------------------------------------------------------------------
def test_quadrature_mass_constant_field():
    # constant field c over volume V integrates to c*V
    w = np.full(100, 0.01)            # sum(w) = 1.0  -> unit volume
    c = np.full(100, 3.5)
    assert conservation.quadrature_mass(c, w) == pytest.approx(3.5)
    assert conservation.domain_volume(w) == pytest.approx(1.0)
    assert conservation.mean_field(c, w) == pytest.approx(3.5)


def test_component_content_and_drift():
    w = np.full(10, 0.1)
    fields = [np.full(10, 0.2), np.full(10, 0.3), np.full(10, 0.5)]
    content = conservation.component_content(fields, w)
    assert content == pytest.approx([0.2, 0.3, 0.5])
    series = [1.0, 1.0 + 1e-15, 1.0 - 2e-15]
    assert conservation.mass_drift(series) == pytest.approx(2e-15, abs=1e-16)
    assert conservation.relative_mass_drift(series) == pytest.approx(2e-15, abs=1e-16)


def test_boundary_flux_sign():
    # uniform outward flux J.n = 2 over a boundary of measure 3 -> content
    # decreases at rate -6
    fn = np.full(6, 2.0)
    wb = np.full(6, 0.5)              # sum = 3
    assert conservation.boundary_flux(fn, wb) == pytest.approx(-6.0)


def test_moving_domain_balance_closes():
    # content on fixed domain constant at 1; height shrinks 1->0.5; with no
    # flux the raw scaled-content change is h*C - h0*C0
    content = np.array([1.0, 1.0, 1.0])
    height = np.array([1.0, 0.75, 0.5])
    resid = conservation.moving_domain_balance(content, height)
    assert resid == pytest.approx([0.0, -0.25, -0.5])
    # supplying the matching loss flux closes it to ~0
    times = np.array([0.0, 1.0, 2.0])
    flux = np.gradient(height * content, times)
    resid2 = conservation.moving_domain_balance(content, height, flux, times)
    assert np.max(np.abs(resid2)) < 0.06


def test_transfer_mass_change():
    assert conservation.transfer_mass_change(2.0, 2.0 + 1e-12) == pytest.approx(1e-12)


# --------------------------------------------------------------------------
# energy
# --------------------------------------------------------------------------
def test_gradient_energy_uniform():
    # |grad phi|^2 = 1 everywhere, kappa=2 -> density 1; unit volume -> F=1
    npts = 50
    w = np.full(npts, 1.0 / npts)
    grad = np.zeros((npts, 2))
    grad[:, 0] = 1.0
    assert energy.gradient_energy(grad, w, kappa=2.0) == pytest.approx(1.0)


def test_energy_split_and_total():
    w = np.full(4, 0.25)
    dens = {"bulk": np.full(4, 2.0), "grad": np.full(4, 1.0)}
    out = energy.energy_split(dens, w)
    assert out["bulk"] == pytest.approx(2.0)
    assert out["grad"] == pytest.approx(1.0)
    assert out["total"] == pytest.approx(3.0)
    tot = energy.assemble_total(bulk=2.0, grad=1.0, wall=0.5)
    assert tot["total"] == pytest.approx(3.5)


def test_monotone_and_increment():
    F = np.array([1.0, 5.0, 4.0, 3.0, 2.5])   # step-0 transient then decreasing
    assert energy.is_monotone_decreasing(F, skip=1)
    assert not energy.is_monotone_decreasing(F, skip=0)
    assert energy.largest_positive_increment(F) == pytest.approx(4.0)  # 1->5
    assert energy.largest_positive_increment(F[1:]) == pytest.approx(0.0)


# --------------------------------------------------------------------------
# morphology
# --------------------------------------------------------------------------
def test_peak_wavelength_sinusoid():
    # phi = sin(2 pi x / L) with L = 16 cells on a 64-grid: peak wavelength ~16
    N, L = 64, 16.0
    x = np.arange(N)
    field = np.sin(2 * np.pi * x[:, None] / L) * np.ones((1, N))
    lam = morphology.peak_wavelength(field, dx=1.0)
    assert lam == pytest.approx(L, rel=0.15)


def test_first_moment_wavelength_finite():
    rng = np.random.default_rng(0)
    field = rng.standard_normal((32, 32))
    lam = morphology.first_moment_wavelength(field)
    assert np.isfinite(lam) and lam > 0


def test_two_point_correlation_zero_lag_is_variance():
    rng = np.random.default_rng(1)
    field = rng.standard_normal((48, 48))
    r, C = morphology.two_point_correlation(field)
    assert r[0] == 0.0
    assert C[0] == pytest.approx(field.var(), rel=1e-6)


def test_interfacial_area_and_phase_fractions():
    # half-filled field: one straight interface across a 40-wide domain
    field = np.zeros((40, 40))
    field[:, 20:] = 1.0
    area = morphology.interfacial_area(field, dx=1.0, threshold=0.5)
    assert area > 0
    lo, hi = morphology.phase_fractions(field, threshold=0.5)
    assert lo == pytest.approx(0.5) and hi == pytest.approx(0.5)


def test_anisotropy_isotropic_vs_striped():
    rng = np.random.default_rng(2)
    iso = rng.standard_normal((64, 64))
    x = np.arange(64)
    striped = np.sin(2 * np.pi * x[:, None] / 8.0) * np.ones((1, 64))
    a_iso = morphology.anisotropy(iso)
    a_str = morphology.anisotropy(striped)
    assert a_str > a_iso
    assert 0.0 <= a_iso < 0.5


# --------------------------------------------------------------------------
# admissibility
# --------------------------------------------------------------------------
def test_bounds_and_projection():
    f = np.array([-0.1, 0.5, 1.2, 0.3])
    lo, hi = admissibility.field_bounds(f)
    assert lo == pytest.approx(-0.1) and hi == pytest.approx(1.2)
    v = admissibility.bound_violation(f, 0.0, 1.0)
    assert v["below"] == pytest.approx(0.1) and v["above"] == pytest.approx(0.2)
    assert v["violating_fraction"] == pytest.approx(0.5)
    rep = admissibility.projection_report(f, 0.0, 1.0)
    assert rep["projected_dofs"] == 2
    assert rep["max_correction"] == pytest.approx(0.2)
    assert admissibility.clipped_fraction(f) == pytest.approx(0.5)


def test_simplex_residual():
    a = np.array([0.2, 0.5, 0.1])
    b = np.array([0.3, 0.5, 0.6])
    c = np.array([0.5, 0.0, 0.3])       # a+b+c = 1 exactly
    res = admissibility.simplex_residual([a, b, c])
    assert res["max_abs"] < 1e-12
    res2 = admissibility.simplex_residual([a, b])   # sums < 1
    assert res2["max_abs"] > 0.1


# --------------------------------------------------------------------------
# convergence
# --------------------------------------------------------------------------
def test_observed_order_second_order():
    h = np.array([1.0, 0.5, 0.25, 0.125])
    err = 3.0 * h ** 2                    # exact 2nd order
    assert convergence.observed_order(h, err) == pytest.approx(2.0, abs=1e-9)
    po = convergence.pairwise_orders(h, err)
    assert np.allclose(po, 2.0)


def test_richardson_and_gci():
    # f(h) = f_exact + C h^2 ; r=2, p=2 recovers f_exact
    f_exact, C = 5.0, 0.4
    f_coarse = f_exact + C * (0.2 ** 2)
    f_fine = f_exact + C * (0.1 ** 2)
    ext = convergence.richardson_extrapolation(f_coarse, f_fine, 2.0, 2.0)
    assert ext == pytest.approx(f_exact, abs=1e-9)
    est = convergence.richardson_error_estimate(f_coarse, f_fine, 2.0, 2.0)
    assert est == pytest.approx(abs(f_fine - f_exact), rel=1e-6)
    assert convergence.gci(f_coarse, f_fine, 2.0, 2.0) > 0


# --------------------------------------------------------------------------
# stochastic
# --------------------------------------------------------------------------
def test_ensemble_aggregate():
    samples = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    agg = stochastic.ensemble_aggregate(samples, axis=0)
    assert agg["mean"] == pytest.approx([3.0, 4.0])
    assert agg["n"] == 3
    assert np.all(agg["sd"] > 0)


def test_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(3)
    data = rng.normal(10.0, 2.0, size=200)
    ci = stochastic.bootstrap_ci(data, seed=0)
    assert ci["low"] < ci["estimate"] < ci["high"]
    assert ci["low"] < 10.0 < ci["high"]


def test_first_passage_and_event_probability():
    series = np.array([0.0, 0.1, 0.4, 0.9])
    t = stochastic.first_passage(series, 0.5, direction="up")
    assert 2.0 < t < 3.0
    assert stochastic.first_passage(series, 5.0) is None
    ep = stochastic.event_probability([True, True, False, True])
    assert ep["p"] == pytest.approx(0.75)
    assert 0.0 <= ep["low"] <= ep["p"] <= ep["high"] <= 1.0


# --------------------------------------------------------------------------
# profiling + provenance (Warp-free, best-effort)
# --------------------------------------------------------------------------
def test_stage_timer_records():
    st = profiling.StageTimer(sync=False)
    with st.stage("a"):
        _ = sum(range(1000))
    with st.stage("a"):
        _ = sum(range(1000))
    s = st.summary()
    assert s["a"]["calls"] == 2 and s["a"]["total_s"] >= 0.0
    with profiling.timer("blk", sync=False) as rec:
        pass
    assert rec["seconds"] is not None


def test_provenance_has_all_spec_fields():
    meta = provenance.collect_metadata({"solver": "cudss", "seeds": [3]})
    for key in ("diffsim_commit", "git_dirty", "timestamp_utc", "hostname",
                "python_version", "warp_version", "torch_version",
                "nvmath_version", "gpu", "gpu_memory_bytes", "cuda_runtime",
                "cuda_driver", "precision", "device", "solver", "mesh",
                "time_integrator", "nonlinear_tol", "linear_tol", "seeds",
                "wall_time_seconds", "peak_device_memory_bytes", "exit_reason"):
        assert key in meta, f"missing provenance field {key}"
    assert meta["solver"] == "cudss"
    assert meta["seeds"] == [3]
