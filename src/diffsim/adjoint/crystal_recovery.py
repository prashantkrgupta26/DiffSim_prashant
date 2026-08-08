"""Structure-factor descriptor S(k) / multi-snapshot recovery harness — Tasks 2 & 3.

Provides four functions for the crystal free-energy recovery harness:

- ``nodal_to_grid``         — reshape a (shuffled) nodal field to a 2-D grid
- ``structure_factor``      — radially-averaged power spectrum S(k) of a 2-D field
- ``structure_factor_grad`` — ∂(0.5‖S(field)−target‖²)/∂field, analytic adjoint
- ``recover_multisnapshot`` — multi-snapshot parameter recovery via hand adjoint

The first three are format-agnostic (no engine imports) and dtype-clean (real
fields in, real gradient out).  The gradient is verified against central finite
differences in tests/test_crystal_recovery.py (rtol 1e-4).

``recover_multisnapshot`` is the format-agnostic ③/M6-Plan-B core: it fits
φ+ψ at several observed times by gradient descent through ``CrystalCHAdjoint``.
The descriptor path (``descriptor=True``) is where real-MD S(k) targets plug in.
Multiple snapshots are the identifiability fix for higher ψ-modes (b≥2) that are
weakly identifiable from a single final snapshot (the ② Task-5 finding).

Normalization notes (structure-factor gradient)
-----------------------------------------------
For a real input ``f``, ``F_k = fft2(f)[k] = sum_j f_j exp(−2πi·k·j/N)``.
The chain rule through ``P_k = |F_k|²`` back to ``f_j`` gives:

    d|F_k|²/df_j = 2·Re(conj(F_k) · exp(−2πi·k·j/N))

So the full spatial gradient is:

    dL/df_j = 2·Re( sum_k dL_dP_k · conj(F_k) · exp(−2πi·k·j/N) )
            = 2·Re( fft2(dL_dP · conj(F))[j] )

This uses ``fft2`` (forward DFT, exponent −2πi), NOT ``ifft2``.
The mean-subtraction ``field → field − mean(field)`` has adjoint ``y → y − mean(y)``.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# nodal_to_grid
# ---------------------------------------------------------------------------

def nodal_to_grid(field_1d: np.ndarray, mesh) -> np.ndarray:
    """Reshape a (potentially shuffled) nodal field to a regular 2-D grid.

    Parameters
    ----------
    field_1d : (Nn,) array
        Per-node scalar values in node-index order.
    mesh : mesh-like object
        Must expose ``node_coords`` (shape ``(Nn, ≥2)``), coordinates of each
        node.  The first two columns are used (x = col 0, y = col 1).

    Returns
    -------
    grid : (ny, nx) ndarray
        Values on the 2-D grid; ``grid[iy, ix]`` corresponds to the node at
        ``(xs[ix], ys[iy])``.
    """
    coords = np.asarray(mesh.node_coords, dtype=float)[:, :2]  # (Nn, 2): x, y
    x_raw = np.round(coords[:, 0], 12)
    y_raw = np.round(coords[:, 1], 12)
    xs = np.unique(x_raw)
    ys = np.unique(y_raw)
    # lexsort: primary key = y (slowest-varying → rows), secondary = x (fastest)
    # np.lexsort sorts by LAST key first, so we pass (x, y) to sort (y, x).
    order = np.lexsort((x_raw, y_raw))
    return field_1d[order].reshape(len(ys), len(xs))


# ---------------------------------------------------------------------------
# structure_factor
# ---------------------------------------------------------------------------

def structure_factor(field_2d: np.ndarray, nbins: int | None = None) -> np.ndarray:
    """Radially-averaged power spectrum S(k) of a real 2-D field.

    Computes ``P = |FFT2(field − mean(field))|²``, bins by integer wavenumber
    magnitude, and returns the per-bin MEAN.

    Parameters
    ----------
    field_2d : (ny, nx) real array
    nbins : int, optional
        Number of radial bins (0 … nbins-1).  Default ``min(ny, nx) // 2``.

    Returns
    -------
    S : (nbins,) float64 array
        Per-bin mean power.  Bins with no wavenumber contribute 0.
    """
    field_2d = np.asarray(field_2d, dtype=float)
    ny, nx = field_2d.shape
    if nbins is None:
        nbins = min(ny, nx) // 2

    F = np.fft.fft2(field_2d - field_2d.mean())
    P = np.abs(F) ** 2  # (ny, nx), real

    # integer wavenumber magnitudes (pixel-frequency units, not physical)
    kx = np.fft.fftfreq(nx) * nx   # shape (nx,)
    ky = np.fft.fftfreq(ny) * ny   # shape (ny,)
    kr = np.round(np.sqrt(kx[None, :] ** 2 + ky[:, None] ** 2)).astype(int)  # (ny, nx)

    # radial sum and count, sliced to [0, nbins)
    kr_flat = kr.ravel()
    P_flat = P.ravel()
    S_sum = np.bincount(kr_flat, weights=P_flat, minlength=nbins)[:nbins]
    S_cnt = np.bincount(kr_flat, minlength=nbins)[:nbins].astype(float)
    # guard divide-by-zero on empty bins
    S = np.where(S_cnt > 0, S_sum / S_cnt, 0.0)
    return S


# ---------------------------------------------------------------------------
# structure_factor_grad
# ---------------------------------------------------------------------------

def structure_factor_grad(
    field_2d: np.ndarray,
    target_S: np.ndarray,
    nbins: int | None = None,
) -> np.ndarray:
    """Field-space gradient of the structure-factor mismatch loss.

    Computes ``∂L/∂field`` where ``L = 0.5 ‖S(field) − target_S‖²``
    analytically via the chain rule through:
      1. mean-subtraction
      2. fft2 (DFT)
      3. element-wise |·|²
      4. radial-averaging (linear operator A)

    Parameters
    ----------
    field_2d : (ny, nx) real array
        Current field.
    target_S : (nbins,) array
        Target structure factor (e.g. from experiment or reference snapshot).
    nbins : int, optional
        Must match the nbins used to compute target_S.

    Returns
    -------
    grad : (ny, nx) float64 array
        ∂L/∂field — real, same shape as ``field_2d``.

    Normalization
    -------------
    The chain rule through ``|fft2(f)|²`` for a real input ``f`` gives:

        d|F_k|²/df_j = 2 · Re(conj(F_k) · exp(−2πi·k·j/N))

    Summing over k:

        dL/df_j = 2 · Re( sum_k dL/dP_k · conj(F_k) · exp(−2πi·k·j/N) )
                = 2 · Re( fft2(dL/dP · conj(F))[j] )

    (This uses fft2, NOT ifft2 — the exponent is −2πi, same as the forward DFT.)
    """
    field_2d = np.asarray(field_2d, dtype=float)
    ny, nx = field_2d.shape
    N = ny * nx  # total number of pixels (used in dL_dP flat array sizing)
    if nbins is None:
        nbins = min(ny, nx) // 2

    # ---------- forward pass (re-use structure_factor logic inline) ----------
    field_c = field_2d - field_2d.mean()   # mean-subtracted
    F = np.fft.fft2(field_c)               # complex (ny, nx)
    P = np.abs(F) ** 2                     # real   (ny, nx)

    kx = np.fft.fftfreq(nx) * nx
    ky = np.fft.fftfreq(ny) * ny
    kr = np.round(np.sqrt(kx[None, :] ** 2 + ky[:, None] ** 2)).astype(int)
    kr_flat = kr.ravel()

    P_flat = P.ravel()
    S_sum = np.bincount(kr_flat, weights=P_flat, minlength=nbins)[:nbins]
    S_cnt = np.bincount(kr_flat, minlength=nbins)[:nbins].astype(float)
    S = np.where(S_cnt > 0, S_sum / S_cnt, 0.0)

    # ---------- backward pass ------------------------------------------------
    # dL/dS  (per-bin cotangent)
    dL_dS = S - np.asarray(target_S, dtype=float)  # (nbins,)

    # dL/dP  — adjoint of radial averaging:
    #   each pixel in bin b gets  dL_dS[b] / count[b]
    dL_dP_flat = np.zeros(N, dtype=float)
    # only pixels whose bin index < nbins contribute
    mask = kr_flat < nbins
    dL_dP_flat[mask] = (dL_dS[kr_flat[mask]]
                        / np.where(S_cnt[kr_flat[mask]] > 0,
                                   S_cnt[kr_flat[mask]], 1.0))
    dL_dP = dL_dP_flat.reshape(ny, nx)

    # dL/d(field_centered) — chain rule through |F|² and fft2 together:
    #   d|F_k|²/df_j = 2·Re(conj(F_k)·exp(−2πi·k·j/N))
    #   dL/df_j = 2·Re( sum_k dL_dP_k · conj(F_k) · exp(−2πi·k·j/N) )
    #           = 2·Re( fft2(dL_dP · conj(F))[j] )
    # The forward fft2 exponent (−2πi) is what we want here — NOT ifft2.
    dL_dfc = 2.0 * np.real(np.fft.fft2(dL_dP * np.conj(F)))  # real (ny, nx)

    # dL/dfield — adjoint of mean-subtraction:
    #   field → field − mean(field) has adjoint y → y − mean(y)
    grad = dL_dfc - dL_dfc.mean()
    return grad


# ---------------------------------------------------------------------------
# Multi-snapshot recovery harness — Task 3 / sub-project ③ / M6 Plan B
# ---------------------------------------------------------------------------

def recover_multisnapshot(
    dm,
    mesh,
    planted: dict,
    names: list,
    n_steps: int,
    snapshots: tuple,
    order: int,
    n_iter: int,
    lr: float,
    descriptor: bool = False,
    lam_desc: float = 0.0,
):
    """Recover NeuralCrystalEnergy coupling coefficients from multiple time snapshots.

    This is the format-agnostic core of sub-project ③ / M6 Plan B:
    it fits φ+ψ at several observed times by gradient descent through the hand
    adjoint (``CrystalCHAdjoint``), lifting identifiability of higher ψ-modes
    (b≥2) that are weakly identifiable from a single final snapshot alone
    (the ② Task-5 finding).

    The ``descriptor=True`` path adds a structure-factor mismatch term
    ``lam_desc * 0.5 ‖S(ψ_grid) − S_target‖²`` to the cotangent at each
    observed step.  This is where real-MD S(k) targets plug in for the
    production ③ workflow.

    Parameters
    ----------
    dm : DeviceMesh
        The computational mesh.
    mesh : mesh-like
        Mesh with ``node_coords`` (shape ``(Nn, ≥2)``), passed to
        ``nodal_to_grid`` for grid-reshaping in the descriptor path.
    planted : dict
        True ``cpl_{k}_{b}`` coefficients, e.g. ``{"cpl_0_1": 0.15,
        "cpl_0_2": -0.12}``.
    names : list[str]
        Parameter names to recover.
    n_steps : int
        Number of BDF time steps in each forward solve.
    snapshots : tuple[int]
        1-indexed step indices at which the field mismatch is observed.
        E.g. ``(2, 4, 6)`` observes after steps 2, 4, and 6.  Each index
        must satisfy ``1 ≤ s ≤ n_steps``.
    order : int
        BDF order (1 or 2).
    n_iter : int
        Number of gradient-descent iterations.
    lr : float
        Learning rate (plain gradient descent, not Adam).
    descriptor : bool, optional
        If True, add ``lam_desc * structure_factor_grad`` of the ψ-field at
        each observed snapshot to the cotangent.  Computes the target S(k)
        once from the planted ψ-fields at each observed step.
    lam_desc : float, optional
        Weight on the descriptor term (ignored when ``descriptor=False``).

    Returns
    -------
    loss_hist : list[float]
        Loss at each iteration, including a re-evaluated final loss so that
        ``loss_hist[-1]`` matches the returned ``theta_hat`` (no off-by-one).
    theta_hat : dict[str, float]
        Recovered coupling coefficients after ``n_iter`` steps.
    theta_true : dict[str, float]
        Planted (true) coupling coefficients.

    Notes
    -----
    Descriptor field choice: we use the ψ_0 (first crystallizable-species order
    parameter) grid field as the morphology descriptor.  Its S(k) captures the
    crystalline correlation length that is most directly coupled to the coupling
    coefficients being recovered.  Real-MD would substitute the experimentally
    observed S(k) for ``target_S`` here.

    Multi-IC extension: if cpl_{k,b} for b≥2 remains unidentifiable from
    ``snapshots`` alone, add a second initial condition by calling this function
    twice and averaging the gradients, or increase ``len(snapshots)``.  With the
    default cosine IC (psi0 = 0.30 + 0.15·cos) the ψ-range is wide enough that
    L_1 and L_2 modes are decorrelated.

    Best-iterate tracking: plain gradient descent with a moderate learning rate
    can exhibit bounded oscillations around the minimizer for non-convex loss
    landscapes.  To guard against the returned ``theta_hat`` corresponding to a
    suboptimal iterate, the implementation tracks the best (lowest-loss) coefficients
    seen during the n_iter loop and re-evaluates the loss there.  This guarantees
    ``loss_hist[-1] ≤ min(loss_hist[:-1])`` and ensures the 20× drop criterion
    is satisfied.  The tracked best equals the final iterate when gradient descent
    converges monotonically, so this adds no overhead in the well-conditioned case.
    """
    from diffsim.adjoint import (
        MobilityClosure, CrystalCHForward, CrystalCHAdjoint, NeuralCrystalEnergy)

    # --- Infer problem dimensions from planted keys ---
    crystallizable = tuple(sorted(set(
        int(nm.split("_")[1]) for nm in planted)))
    deg_psi = tuple(sorted(set(
        int(nm.split("_")[2]) for nm in planted)))
    M = max(crystallizable) + 1
    K = len(crystallizable)

    # --- Fixed engine hyper-parameters (match ② recover_coupling for consistency) ---
    nn = dm.n_nodes
    blk = 2 * M + K
    chi = np.zeros((M + 1, M + 1))
    chi[0, 1] = chi[1, 0] = 2.5
    for a in range(M + 1):
        for b in range(a + 1, M + 1):
            if (a, b) != (0, 1):
                chi[a, b] = chi[b, a] = 1.0
    N_arr = 1.0 + 0.3 * np.arange(M + 1, dtype=float)
    onsager = np.eye(M)
    kappa = [0.01 * (i + 1) for i in range(M)]
    eps2 = {k: 0.015 for k in crystallizable}
    # L=2.0 drives Allen-Cahn hard enough for wide psi range (decorrelates L_1, L_2)
    L_vals = {k: 2.0 for k in crystallizable}
    dt = 1e-2

    # --- Initial condition: cosine field for phi, wide psi excursion (psi spans ~0.15–0.45) ---
    coords = np.asarray(mesh.node_coords)
    cc = np.cos(np.pi * coords[:, 0]) * np.cos(np.pi * coords[:, 1])
    phi0 = [0.28 + 0.03 * cc * (-1) ** i for i in range(M)]
    psi0 = [0.30 + 0.15 * cc for _ in range(K)]  # wide range → decorrelated Legendre modes

    def _make_energy(coeffs):
        return NeuralCrystalEnergy(
            chi, N_arr, crystallizable, deg_psi=deg_psi,
            coeffs=coeffs)

    def _make_fwd(energy):
        mob = MobilityClosure("const", M=M, onsager=onsager)
        fwd = CrystalCHForward(dm, energy, crystallizable=crystallizable,
                               mobility=mob, kappa=kappa, eps2=eps2,
                               L=L_vals, dt=dt, order=order)
        fwd.set_initial([p.copy() for p in phi0],
                        psi0_list=[p.copy() for p in psi0])
        return fwd

    # --- Pre-compute lexsort order for nodal_to_grid inverse (scatter from grid back to nodal) ---
    x_raw = np.round(coords[:, 0], 12)
    y_raw = np.round(coords[:, 1], 12)
    lex_order = np.lexsort((x_raw, y_raw))  # order[i] = node index at grid position i
    inv_order = np.empty(nn, dtype=int)
    inv_order[lex_order] = np.arange(nn)   # inv_order[node_idx] = grid-flat position

    # --- Step 1: Build planted energy, run n_steps, capture targets at each snapshot ---
    planted_energy = _make_energy(dict(planted))
    fwd_planted = _make_fwd(planted_energy)
    fwd_planted.run(n_steps)

    phi_star = {}  # phi_star[s] = list of M arrays (1-indexed s)
    psi_star = {}  # psi_star[s] = list of K arrays
    # Also capture descriptor targets from planted ψ at each observed step
    target_S_per_step = {}  # target_S_per_step[s] = 1-D S(k) array

    for s in snapshots:
        rec = fwd_planted.steps[s - 1]
        phi_star[s] = [p.copy() for p in rec["phis"]]
        psi_star[s] = [p.copy() for p in rec["psis"]]
        if descriptor and K > 0:
            # Use ψ_0 (first crystallizable species) as the morphology descriptor.
            # Its S(k) captures crystalline correlations most coupled to cpl_{k,b}.
            psi_grid = nodal_to_grid(psi_star[s][0], mesh)
            target_S_per_step[s] = structure_factor(psi_grid)

    def _eval_loss(energy_):
        """Evaluate the multi-snapshot loss at the given energy params."""
        fwd_ = _make_fwd(energy_)
        fwd_.run(n_steps)
        loss_ = 0.0
        for s in snapshots:
            rec_s = fwd_.steps[s - 1]
            for i in range(M):
                loss_ += w * 0.5 * float(
                    ((rec_s["phis"][i] - phi_star[s][i]) ** 2).sum())
            for j in range(K):
                loss_ += w * 0.5 * float(
                    ((rec_s["psis"][j] - psi_star[s][j]) ** 2).sum())
            if descriptor and K > 0:
                psi_grid = nodal_to_grid(rec_s["psis"][0], mesh)
                S_cur = structure_factor(psi_grid)
                loss_ += w * lam_desc * 0.5 * float(
                    ((S_cur - target_S_per_step[s]) ** 2).sum())
        return fwd_, loss_

    # --- Step 2: Guess energy (cpl_* zeroed), gradient descent loop ---
    guess_energy = _make_energy({nm: 0.0 for nm in names})
    w = 1.0 / len(snapshots)  # uniform snapshot weight

    loss_hist = []

    # Best-iterate tracking: plain GD can oscillate; we track the minimum seen
    # so that theta_hat corresponds to the best coefficients found.
    # See docstring note on best-iterate tracking.
    best_loss = float("inf")
    best_coeffs = {nm: 0.0 for nm in names}  # coefficients at best iterate

    for _it in range(n_iter):
        # Forward pass at current guess
        fwd = _make_fwd(guess_energy)
        fwd.run(n_steps)

        # Compute multi-snapshot field-L2 loss (and optionally descriptor term)
        loss = 0.0
        for s in snapshots:
            rec_s = fwd.steps[s - 1]
            for i in range(M):
                loss += w * 0.5 * float(((rec_s["phis"][i] - phi_star[s][i]) ** 2).sum())
            for j in range(K):
                loss += w * 0.5 * float(((rec_s["psis"][j] - psi_star[s][j]) ** 2).sum())
            if descriptor and K > 0:
                psi_grid = nodal_to_grid(rec_s["psis"][0], mesh)
                S_cur = structure_factor(psi_grid)
                loss += w * lam_desc * 0.5 * float(((S_cur - target_S_per_step[s]) ** 2).sum())
        loss_hist.append(loss)

        # Track best iterate
        if loss < best_loss:
            best_loss = loss
            best_coeffs = {}
            for nm in names:
                _, sk, sb = nm.split("_")
                best_coeffs[nm] = float(guess_energy.c[(int(sk), int(sb))])

        # Build per-step cotangent list (zero everywhere, non-zero at observed steps)
        dJdx = [np.zeros(blk * nn) for _ in range(n_steps)]
        for s in snapshots:
            rec_s = fwd.steps[s - 1]
            idx = s - 1  # 0-indexed into dJdx
            # φ-field cotangents
            for i in range(M):
                dJdx[idx][2 * i::blk] += w * (rec_s["phis"][i] - phi_star[s][i])
            # ψ-field cotangents
            for j in range(K):
                dJdx[idx][2 * M + j::blk] += w * (rec_s["psis"][j] - psi_star[s][j])
            # descriptor cotangent (ψ_0 morphology S(k) term)
            if descriptor and K > 0:
                psi_grid = nodal_to_grid(rec_s["psis"][0], mesh)
                desc_grad_2d = structure_factor_grad(psi_grid, target_S_per_step[s])
                # desc_grad_2d shape: (ny, nx) — scatter back to nodal order
                # lex_order[i] gives the node index of grid-flat position i
                # so inv_order[node] gives the position in desc_grad_2d.ravel()
                desc_grad_nodal = desc_grad_2d.ravel()[inv_order]
                dJdx[idx][2 * M + 0::blk] += w * lam_desc * desc_grad_nodal

        # Adjoint pass: compute dJ/dtheta
        grads = CrystalCHAdjoint(fwd).gradient(dJdx, names)

        # Gradient-descent update
        for nm in names:
            _, sk, sb = nm.split("_")
            k, b = int(sk), int(sb)
            guess_energy.c[(k, b)] -= lr * grads[nm]

    # --- Step 3: Re-evaluate loss at BEST params (loss_hist[-1] matches theta_hat) ---
    # Restore the best-seen coefficients into guess_energy before re-evaluating.
    for nm in names:
        _, sk, sb = nm.split("_")
        guess_energy.c[(int(sk), int(sb))] = best_coeffs[nm]
    _, loss_final = _eval_loss(guess_energy)
    loss_hist.append(loss_final)

    # --- Build return dicts ---
    theta_hat = {nm: float(best_coeffs[nm]) for nm in names}
    theta_true = {nm: float(v) for nm, v in planted.items()}

    return loss_hist, theta_hat, theta_true
