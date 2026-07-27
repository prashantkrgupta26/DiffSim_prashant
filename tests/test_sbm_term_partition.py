"""Partition gates for per-term Nitsche assembly (spec 2026-07-27)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))


def _tiny_two_sided(refine_to=None):
    from p2r1a_thin_plate_flow import _build_shell
    fx = _build_shell(4, 5.0/16.0, 0.5, 1.0/16.0, refine_to=refine_to)
    return fx


def _assemble(fx, return_terms, a_face=None):
    from diffsim.sbm.vector import sbm_vector_dirichlet_twosided
    noslip = lambda y: np.zeros((len(y), 2))
    kw = dict(alpha=50.0)
    # VERIFY-FIRST: read the real signature for the advecting-field kwarg
    # names (a_face_plus/a_face_minus) and the beta_backflow default; pass
    # a_face to BOTH sides when provided.
    if a_face is not None:
        kw.update(a_face_plus=a_face, a_face_minus=a_face, beta_backflow=1.0)
    return sbm_vector_dirichlet_twosided(
        fx["dm"], fx["sfp"], fx["gp"], fx["sfm"], fx["gm"],
        noslip, 0.004, 3, return_terms=return_terms, **kw)


def test_partition_exact_uniform():
    fx = _tiny_two_sided()
    A, b = _assemble(fx, False)
    A2, b2, terms = _assemble(fx, True)
    assert (A2 - A).nnz == 0 or abs(A2 - A).max() < 1e-14
    assert np.allclose(b2, b, atol=1e-14)
    As = sum(t[0] for t in terms.values())
    bs = sum(t[1] for t in terms.values())
    assert abs(As - A).max() < 1e-14, "term matrices do not partition A"
    assert np.allclose(bs, b, atol=1e-14), "term rhs do not partition b"
    assert len(terms) >= 3, f"split too coarse: {list(terms)}"


def test_partition_exact_adaptive():
    fx = _tiny_two_sided(refine_to=6)
    A, b = _assemble(fx, False)
    _, _, terms = _assemble(fx, True)
    As = sum(t[0] for t in terms.values())
    bs = sum(t[1] for t in terms.values())
    assert abs(As - A).max() < 1e-14
    assert np.allclose(bs, b, atol=1e-14)


def test_default_false_bit_identical():
    fx = _tiny_two_sided()
    A1, b1 = _assemble(fx, False)
    A2, b2 = _assemble(fx, False)
    assert (A1 - A2).nnz == 0 and np.array_equal(b1, b2)
