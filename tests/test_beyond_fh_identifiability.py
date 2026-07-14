r"""Beyond-Flory-Huggins gauge-identifiability gate.

Asserts the two claims of benchmarks/phase-field/beyond_fh_identifiability.py:

  (i)  including the T1 (linear) gauge mode makes the parameter-Jacobian
       numerically singular -- chi(=B) and the linear correction are exactly
       aliased (the recorded M4 finding) -- while the gauge-anchored head
       (P2+) is well conditioned: a >=1e8x conditioning gap.
  (ii) on the anchored (identifiable) parameterisation the beyond-FH
       coefficients are recovered from a single interior trajectory.

Small size (level 3, 4 steps); the conditioning claim is fit-free and fast,
the recovery uses a short Adam run."""
import pytest

pytestmark = pytest.mark.ad


def _demo(fit_iters):
    from importlib.machinery import SourceFileLoader
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "benchmarks", "phase-field",
                        "beyond_fh_identifiability.py")
    return SourceFileLoader("bfh_demo", path).load_module()


def test_gauge_conditioning_gap():
    """Un-anchored (with the linear mode) is singular; anchored is well posed."""
    m = _demo(0)
    r = m.run_demo(level=3, n_steps=4, fit_iters=120, verbose=False)
    print(f"cond un-anchored={r['cond_unanchored']:.2e}  "
          f"anchored={r['cond_anchored']:.2e}  "
          f"gap={r['cond_unanchored'] / r['cond_anchored']:.2e}")
    # the linear mode aliases chi exactly -> the un-anchored map is
    # rank-deficient (cond ~ 1/eps); the anchored map is O(10-1e3).
    assert r["cond_anchored"] < 1e3, r["cond_anchored"]
    assert r["cond_unanchored"] / r["cond_anchored"] > 1e8, r


def test_anchored_recovery():
    """The gauge-anchored beyond-FH coefficients are recovered from one
    interior trajectory (well-posed inverse problem)."""
    m = _demo(0)
    r = m.run_demo(level=3, n_steps=4, fit_iters=250, verbose=False)
    print(f"anchored coeff recovery rel_err = {r['coeff_err']:.2e}")
    assert r["coeff_err"] < 1e-3, r["coeff_err"]
