"""M3 rung 2: epoch-continuation recovery.

The measured requirements (M2-D v2 + tutorial 4f arc): cell-scale edits
are observable but per-alpha re-carving gives a carve-jump-dominated
landscape; within-epoch objectives are smooth. Protocol: OPTIMIZE
WITHIN an epoch's trust region; when the boundary drift approaches the
trust bound, RE-ANCHOR (re-carve at the current alpha, transfer
warm state via the rung-1 P operator when the caller carries state,
re-anchor feet) and continue. The TARGET is fixed data; only trial
evaluations live in epochs.

Caller contract:
  epoch_fn(alpha)          -> opaque epoch handle (carve+anchor at alpha)
  resid_fn(epoch, alpha)   -> residual vector r(alpha) within the epoch
  jac_fn(epoch, alpha, r)  -> J columns (FD or adjoint) within the epoch
"""
import numpy as np


def continuation_recover(epoch_fn, resid_fn, jac_fn, alpha0, trust,
                         n_epochs=8, inner_iters=4, lm=1e-3,
                         verbose=True):
    alpha = np.asarray(alpha0, float).copy()
    history = []
    for ep in range(n_epochs):
        epoch = epoch_fn(alpha)                     # re-anchor HERE
        anchor = alpha.copy()
        for it in range(inner_iters):
            r = resid_fn(epoch, alpha)
            J = 0.5 * float(r @ r)
            Jac = jac_fn(epoch, alpha, r)
            lamb = lm * np.trace(Jac.T @ Jac) / max(len(alpha), 1)
            step = np.linalg.solve(Jac.T @ Jac + lamb * np.eye(len(alpha)),
                                   -(Jac.T @ r))
            # confine to the epoch's trust region around the anchor
            excess = np.linalg.norm(alpha + step - anchor)
            if excess > trust:
                step *= max(0.0, trust - np.linalg.norm(alpha - anchor)
                            ) / max(np.linalg.norm(step), 1e-30)
            alpha = alpha + step
            history.append((ep, it, J, alpha.copy()))
            if verbose:
                print(f"  ep{ep} it{it}: J={J:.4e} "
                      f"|alpha|={np.linalg.norm(alpha):.4f}", flush=True)
            if np.linalg.norm(alpha - anchor) >= trust * 0.98:
                break                               # epoch exhausted
        if history and len(history) > 1 and \
                np.linalg.norm(alpha - anchor) < 1e-12:
            break                                   # converged (no move)
    return alpha, history
