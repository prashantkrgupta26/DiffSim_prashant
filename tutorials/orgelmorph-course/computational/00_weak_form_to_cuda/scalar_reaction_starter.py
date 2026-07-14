"""C0 hands-on STARTER -- the reaction term is in the residual but its analytic
Jacobian contribution is MISSING.  This is the state of the code *before* you do
the exercise.

Run the regression test against this module and it FAILS::

    RD_MODULE=scalar_reaction_starter pytest -q test_reaction_jacobian.py

Then look at ``jacobian`` below: it assembles ONLY the diffusion stiffness and
never adds ``reaction_jacobian_elem`` (the 3*alpha*u^2 mass matrix).  Your task
is to add that one analytic term -- the completed version is
``scalar_reaction.py`` (``jacobian`` there), against which the SAME test
PASSES::

    pytest -q test_reaction_jacobian.py

Everything else (mesh, residual, Newton loop, error) is reused unchanged from
``scalar_reaction`` -- only the Jacobian is wrong here, on purpose.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from scalar_reaction import (                      # reuse the correct pieces
    Problem, residual, gp_value, l2_error, newton_solve, stiffness_csr,
    reaction_residual_elem, reaction_jacobian_elem, manufactured, source,
    ALPHA_DEFAULT,
)


def jacobian(prob, u):
    """BROKEN tangent: diffusion stiffness only -- the reaction Jacobian
    ``reaction_jacobian_elem`` (3*alpha*u^2 mass matrix) is NOT added.

    >>> HANDS-ON: add the reaction Jacobian here to match scalar_reaction.py:
    >>>     Ke = prob._Ke + reaction_jacobian_elem(prob, gp_value(prob, u))
    """
    Ke = prob._Ke                                  # <-- reaction term omitted
    rows = np.repeat(prob.conn, prob.nbf, axis=1).ravel()
    cols = np.tile(prob.conn, (1, prob.nbf)).ravel()
    J = sp.coo_matrix((np.broadcast_to(Ke, (prob.ne, prob.nbf, prob.nbf)).ravel(),
                       (rows, cols)), shape=(prob.n, prob.n)).tocsr()
    J = J.tolil()
    for i in np.where(prob.bnd)[0]:
        J.rows[i] = [i]
        J.data[i] = [1.0]
    return J.tocsr()
