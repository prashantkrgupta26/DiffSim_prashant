"""SP-1 B1+B2+B3: XDD Poisson brick, carrier drift-diffusion bricks, and
exciton diffusion-reaction bricks.

WEAK FORMS (SP-1 brick family — keep in sync as B4 adds terms)
-----------------------------------------------------------------
B1  Poisson (this brick):
    Strong (nondim): −∇·(λ² ε̂(x) ∇φ̂) − (p̂ − n̂) = 0
    Weak:   λ² (ε̂ ∇φ̂, ∇w) = (p̂ − n̂, w)
    NO stabilization — operator is elliptic; SUPG is a deliberate drop vs the
    CPU code (recorded deviation: CPU SUPG-on-Poisson removed per B1 spec).

    λ² from XDDParams.scales().lambda2  (= φ₀ ε_m / (x₀² C₀ q), Debye² scale)
    ε̂(x) = ε_r(x) / max(ε_A, ε_D)  — supplied as a GP field by the caller
    (same GP-field contract as kappa in scalar_transport.py).

B2  Carrier drift-diffusion (n̂ electrons / p̂ holes) — NONCONSERVATIVE form:
    Strong:  ∂_t n̂  + a·∇n̂  − μ̂∇²n̂  = f
    with advection velocity  a = sign · μ̂(x) · ∇φ̂  (frozen at GPs from the
    M2 one-way-coupler contract).
      sign = −1  for electrons: a_n = −μ̂_n ∇φ̂
      sign = +1  for holes:     a_p = +μ̂_p ∇φ̂  (holes drift DOWN potential)

    FORM CHOICE — NONCONSERVATIVE  (matching scalar_transport.py's convention):
    The kernel accumulates (a·∇n̂, w) + μ̂(∇n̂, ∇w), NOT ∇·(an̂).
    The difference is n̂(∇·a); for the MMS tests (σ = BDF coefficient) the
    source must match this convention — see _carrier_source() in
    tests/test_exciton_dd.py for the explicit hand derivation.

    Equivalence to the CPU residual convention μ(−n∇φ·∇w + ∇n·∇w):
    Integrating the conservative flux −μn∇φ + μ∇n by parts gives exactly
    the Galerkin terms above (boundary terms zero for homogeneous Dirichlet)
    ONLY when ∇·(μ∇φ) = 0.  For a manufactured φ̂ this does not hold in
    general, so the source differs between conservative/nonconservative forms
    by n̂·∇·a.  The kernel implements nonconservative, matching scalar_transport.

    Galerkin weak form (σ = BDF coefficient, f includes history term):
      σ(n̂, w) + (a·∇n̂, w) + μ̂(∇n̂, ∇w) + SUPG = (f, w) + SUPG_rhs

    SUPG: Tezduyar-class τ_M via tau_m_metric(|a|, h, μ̂, sig²τ, dim) exactly
    as scalar_transport.py.  Complete VMS residual (-μ̂ lapN) for p2 exactness:
      res_b = σ N_b + a·∇N_b − μ̂ lapN_b      (on the trial function N_b)
      SUPG contribution: τ_M (a·∇w, res_b)    (test function augmentation)

    ONE factory pair parameterised by the `sign` kernel arg (float).  The
    caller computes aq_gp = sign * mu_gp * grad_phi_gp and passes it directly;
    sign is absorbed into aq — the kernel itself is sign-agnostic (aq is used
    as-is).  See assemble_xdd_carrier for the calling convention.

B3  Exciton diffusion-reaction (X̂_D donor / X̂_A acceptor) — NO SUPG:
    Strong (nondim, per species i ∈ {D, A}):
      ∂_t X̂_i − ∇·(μ̂_{X,i}(x) ∇X̂_i) + σ_tot X̂_i = Ĝ_i + R̂_{feed,i}
      σ_tot = σ_BDF + 1/τ̂_{x,i} + k̂_{diss,i}(x)
    Galerkin weak form (NO SUPG — excitons are diffusion-dominated):
      σ_tot (X̂_i, w) + μ̂_{X,i}(∇X̂_i, ∇w) = (f̂_i, w)
      f̂_i = Ĝ_i + R̂_{feed,i} + σ_BDF X̂_i^n [+ BDF2 correction]

    DRY DESIGN DECISION — REUSE of B2 carrier kernel:
    The exciton weak form is the B2 carrier form with aq_gp = 0 (zero
    advection) and supg = 0.0 (no stabilization).  Substituting aq = 0 and
    supg = 0 into the carrier kernel gives exactly:
        σ_tot (N_b, N_a) + μ̂_X (∇N_b, ∇N_a)
    which is the exciton stiffness.  The reaction sum σ_tot = σ_BDF + 1/τ̂_x
    + k̂_diss is passed as the effective mass coefficient; because it may be
    spatially varying (k̂_diss(x)), it is supplied as a GP field (sigma_gp)
    rather than a scalar.  The assembly wrapper computes the effective scalar
    sigma = mean(sigma_gp) for the carrier assembler's SUPG path (which is
    disabled), and folds the spatial σ_tot variation into fq_gp:

        fq_eff = fq_raw + (sigma_gp − sigma_scalar) * X̂_prev_gp  [NOT used]

    Actually the cleanest approach is: pass sigma=0 to the carrier assembler
    and supply σ_tot(x)·X̂ implicitly through the stiffness — but the carrier
    kernel adds σ·M as a scalar, not spatially varying.  Therefore
    assemble_xdd_exciton builds an AUXILIARY mass-weighted contribution for
    the spatially-varying reaction and adds it to the RHS, with the carrier
    kernel handling only the spatially-uniform part of σ_tot.

    Implementation (see assemble_xdd_exciton below):
    For spatially-varying σ_tot(x), split:
      σ_tot(x) = σ_scalar + δσ(x),  σ_scalar ≥ 0 (e.g., 0 or min)
    The carrier kernel handles σ_scalar·M exactly.  The spatially-varying
    δσ(x)·X̂ residual is moved to the RHS as part of f̂:
      f̂_eff = f̂ + δσ(x)·X̂_prev   — only for the linearised solve
    For MMS (σ_tot appears in the source), σ_scalar = 0 and σ_tot(x) is
    entirely in the source; the kernel handles diffusion + zero mass, and the
    source carries the full reaction.

    SIMPLIFICATION for the gates: since σ_tot is the EFFECTIVE coefficient
    (already includes 1/τ̂ + k̂), assemble_xdd_exciton accepts sigma_gp as
    a GP field, maps it to a scalar for the carrier assembler (σ = 0, supg=0),
    and adds the reaction term via a separate mass-weighted RHS pass.  But
    that requires a new mass kernel.

    CLEANEST IMPLEMENTATION (chosen):  Pass sigma_gp mean as the scalar sigma
    to the carrier assembler (handles the uniform part), and add the residual
    δσ·X̂ contribution via the RHS by computing an element-wise mass-weighted
    correction.  However, for the B3 gates the source already encodes the full
    reaction (via the MMS derivation), so sigma_gp is passed directly as a
    spatially-varying effective coefficient using the carrier kernel's sigma
    argument — WHICH IS A SCALAR.

    FINAL DECISION (cleanest for correctness and DRY compliance):
    assemble_xdd_exciton loops over p-bins, and for each Gauss point uses the
    LOCAL value of sigma_tot to build the element matrices.  Since the carrier
    kernel only accepts a scalar sigma, we cannot directly pass sigma_gp.
    Instead we use a TWO-FIELD SPLIT: pass sigma=0 to the carrier assembler
    (handles diffusion only) and add the mass-reaction term (sigma_tot * M)
    separately.  The mass matrix M is computed via the Poisson load kernel
    with rho_gp = sigma_tot_gp and f_src_gp = 0 (= mass-weighted reaction
    stiffness contribution).

    ACTUALLY: the simplest clean DRY path that avoids a new kernel entirely:
    For the B3 gates, σ_tot is supplied as a GP field; we assemble the
    reaction contribution as an additional "rho" load through the carrier be
    kernel (by factoring σ_tot·X̂ into f̂), and use the carrier Ae kernel with
    sigma=0 (diffusion only).  This works exactly because the carrier be with
    aq=0 and supg=0 is just (f̂, N_a)dV, so we can fold everything into f̂.

    SUMMARY OF CHOSEN APPROACH (the one actually coded):
    - assemble_xdd_carrier(dm, aq_gp=zeros, mu_gp=mu_X_gp, fq_gp, sigma=0, supg=0)
      gives: μ̂_X·K  (diffusion stiffness, no mass, no advection)
    - Additional MASS-REACTION contribution from sigma_tot:
      Added via assemble_xdd_poisson Ae kernel with eps_gp=sigma_tot_gp, lam2=0
      PLUS a separate Ae built with lam2=0 and mass term.
      NOT clean — requires a 3rd kernel invocation.

    FINAL FINAL DECISION (reading the carrier Ae signature more carefully):
    The carrier Ae takes a SCALAR sigma.  For spatially-uniform σ_tot (Gates
    1, 3, 4), passing sigma=sigma_tot_scalar to carrier works perfectly.
    For spatially-varying σ_tot (Gate 2), we observe that the MMS source
    already encodes the reaction, so we assemble with sigma=sigma_mean and
    add a correction Δf = (sigma_gp - sigma_mean)*X̂.  However, the B3 MMS
    DOES NOT use X̂ on the RHS during assembly — the source f̂ is the full
    manufactured source from the strong PDE, and σ_tot(x)·X̂ is already baked
    into f̂.  So for MMS tests the correct approach is sigma=0 in the carrier
    (no mass term) and f̂ encodes the full reaction.  For the transient and
    decay gates, σ_tot IS uniform (constant τ̂ and k̂), so sigma=σ_total works.
    For Gate 2 (variable k̂(x)), the assemble call uses sigma=0 and the source
    term (which the test constructs correctly via _exciton_source_var) carries
    the full variable reaction.  This means assemble_xdd_exciton must support
    BOTH modes transparently.

    PRAGMATIC IMPLEMENTATION (what assemble_xdd_exciton actually does):
    Accept sigma_gp as a GP field.  Pass sigma = 0 to carrier (diffusion only).
    Fold σ_tot(x)·X̂ into f̂_eff by adding it to fq_gp EXTERNALLY (the caller
    constructs f̂ = Ĝ + R̂_feed + σ_BDF·X̂ⁿ + σ_react(x)·X̂ⁿ... NO — that
    conflates stiffness and RHS.

    RESOLUTION: For the exciton STIFFNESS, σ_tot(x) appears as a spatially-
    varying mass coefficient.  Use the Poisson Ae kernel (which has eps_gp
    as a GP field multiplying ∇N·∇N dV): set lam2=0 and add the mass-reaction
    term (σ_tot * N_b * N_a) via a DEDICATED PASS using the Poisson be kernel
    as: rho_gp = sigma_tot_gp, f_src_gp = 0, which gives (σ_tot, N_a) dV.
    But (σ_tot, N_a)dV is NOT the same as σ_tot*(N_b, N_a)dV — it's a load
    vector, not a stiffness contribution.

    CORRECT APPROACH FOR SPATIALLY-VARYING MASS:
    Need (σ_tot(x) N_b, N_a) dV per element.  This is NOT one of the existing
    kernels.  Options:
    (A) Add a new kernel — violates DRY goal.
    (B) Use the scalar-mean trick: sigma_scalar = GP-mean of σ_tot_gp.
        Error is O(variation of σ_tot) — not zero, will degrade convergence.
    (C) For the exciton gates: sigma_gp feeds directly into the assembly by
        having the CALLER pre-multiply it: pass sigma_tot_gp to the carrier
        assembler as a SCALAR sigma (using a GP-mean approximation) and accept
        the O(δσ h²) mass-lump error.  This degrades the MMS orders.
    (D) Express the spatially-varying mass THROUGH the RHS: the exciton solve
        at each Newton/BDF step has X̂ from the previous step available.  For
        linearised BDF: A_σ X̂^{n+1} = f - (σ_tot(x) - σ_scalar)·(N_b, N_a)
        ... still needs the same mass kernel.

    DEFINITIVE DRY SOLUTION (actually chosen — re-reading the brief):
    The brief says: "the exciton brick may be expressible by REUSING the
    carrier factories with zero advection — evaluate that first".
    Carrier Ae with aq=0 and supg=0 gives: sigma·(N_b, N_a) + μ̂(∇N_b, ∇N_a)
    This is the exciton stiffness EXACTLY when sigma = σ_tot_scalar.
    For the MMS gates: σ_tot is constant per gate (const τ̂ and const k̂ in
    Gate 1; variable k̂ in Gate 2).  For Gate 2, the source encodes the full
    variable reaction, and we pass sigma=0 to carrier (diffusion only), since
    the reaction is in the source.  For Gate 3 (decay identity), σ_tot IS
    spatially uniform — perfect for the scalar sigma path.
    For the transient gate (Gate 4): σ_tot is also constant.
    For Gate 5 (source-coupling smoke): σ_tot=1 (structure test).

    Therefore: assemble_xdd_exciton accepts sigma_gp AS A SCALAR FIELD (GP
    array), uses its element-by-element MEAN (per element) as σ_scalar, and
    calls the carrier assembler.  Because the carrier kernel uses one scalar
    sigma per kernel call (not per element), we compute sigma_scalar as the
    global GP mean.  For the specific gates designed:
      Gate 1: σ_tot = const → exact (zero error from the mean)
      Gate 2: σ_tot = σ_bdf + τ_inv + k̂(x) with k̂ variable; the test
              source encodes -∇·(μ̂∇X̂) + σ_tot X̂ = f̂; the assembly uses
              sigma_mean ≈ σ_tot_mean plus a residual absorbed into f̂ via the
              test's source term.  BUT the test uses sigma_gp as the stiffness
              and fq_gp from _exciton_source_var which already bakes in σ_tot.
              MISMATCH: if we use sigma_mean in the stiffness but the source
              was derived for the exact σ_tot(x) stiffness, the solution will
              be wrong.

    GENUINE RESOLUTION — accept spatially-varying sigma via a new kernel:
    Rather than contorting the API, write a thin new kernel
    make_xdd_exciton_Ae that replaces the scalar sigma with a GP-array sigma:
    Ae += (sigma_gp[gp] * N_b * N_a + mu_X[gp] * grad_Nb·grad_Na) dV
    This IS a new factory (< 40 lines), is clean, and the reuse path for be
    still holds (be = carrier_be with aq=0, supg=0 = (f̂, N_a)dV exactly).
    The brief explicitly allows new factories if "reuse contorts the API".
    Conclusion: spatially-varying σ_tot CONTORTS the carrier API (the scalar
    sigma argument). Write make_xdd_exciton_Ae; reuse carrier_be.

NO SUPG on excitons (recorded deviation from the carrier brick; excitons are
diffusion-dominated and the CPU code does not stabilise them).
"""
import numpy as np
import scipy.sparse as sp
import warp as wp

from ..assembly.femelm import FEMElm, fe_dN_s, fe_detJxW_s, fe_N
from ..assembly.operators import _kernel_cache
from ..api.ns_bricks import tau_m_metric

wp.set_module_options({"enable_backward": False})


# ──────────────────────────────────────────────────────────────────────────────
# Stiffness kernel: Ke[e,a,b] += λ² ε̂_q (∇Na · ∇Nb) dJxW
# ──────────────────────────────────────────────────────────────────────────────

def make_xdd_poisson_Ae(nbf: int, nqp: int, dim: int):
    """Element stiffness for the XDD Poisson operator  λ² (ε̂ ∇φ, ∇w).

    Kernel signature (matches house idiom in scalar_transport / operators):
        conn   [ne, nbf]   int32   — local→global DOF map
        h      [ne]        f64     — element size
        dNtab  [nqp,nbf,dim] f64  — reference basis gradients
        wtab   [nqp]       f64     — quadrature weights
        eps_gp [ne*nqp]    f64     — ε̂ at every Gauss point
        lam2               f64     — λ² scalar
        Ae     [ne,nbf,nbf] f64   — output (pre-zeroed by caller)

    dim is a Python compile-time constant closed over from the factory;
    jac = (he/2)^dim via a power loop (house idiom).
    """
    key = ("xdd_poisson_Ae", nbf, nqp, dim)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @wp.kernel(module="unique", enable_backward=False)
    def xdd_poisson_Ae(
        conn:   wp.array2d(dtype=wp.int32),
        h:      wp.array(dtype=wp.float64),
        dNtab:  wp.array3d(dtype=wp.float64),
        wtab:   wp.array(dtype=wp.float64),
        eps_gp: wp.array(dtype=wp.float64),   # [ne*nqp]
        lam2:   wp.float64,
        Ae:     wp.array3d(dtype=wp.float64),  # [ne, nbf, nbf]
    ):
        e = wp.tid()
        he = h[e]
        half = he * wp.float64(0.5)
        # jac = (he/2)^dim — power loop (house idiom, see operators.py)
        jac = wp.float64(1.0)
        for _ in range(dim):
            jac = jac * half
        dscale = wp.float64(2.0) / he
        fe = FEMElm()
        fe.e = e
        fe.he = he
        for q in range(nqp):
            fe.q = q
            dJxW = wtab[q] * jac
            gp = e * nqp + q
            eps_q = eps_gp[gp]
            coeff = lam2 * eps_q * dJxW
            for a in range(nbf):
                for b in range(nbf):
                    v = wp.float64(0.0)
                    for d in range(dim):
                        v = v + (fe_dN_s(dNtab, fe, a, d, dscale)
                                 * fe_dN_s(dNtab, fe, b, d, dscale))
                    wp.atomic_add(Ae, e, a, b, v * coeff)

    _kernel_cache[key] = xdd_poisson_Ae
    return xdd_poisson_Ae


# ──────────────────────────────────────────────────────────────────────────────
# Load kernel: be[e,a] += (ρ_q + f_q) N_a dJxW
# ──────────────────────────────────────────────────────────────────────────────

def make_xdd_poisson_be(nbf: int, nqp: int, dim: int):
    """Element load for the XDD Poisson RHS  (p̂ − n̂, w) + (f_src, w).

    Kernel signature:
        conn   [ne, nbf]   int32
        h      [ne]        f64
        Ntab   [nqp,nbf]   f64   — basis values
        wtab   [nqp]       f64   — quadrature weights
        rho_gp [ne*nqp]    f64   — charge density p̂−n̂ at GPs
        f_src  [ne*nqp]    f64   — extra body load (zeros if not used)
        be     [ne, nbf]   f64   — output (pre-zeroed by caller)
    """
    key = ("xdd_poisson_be", nbf, nqp, dim)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @wp.kernel(module="unique", enable_backward=False)
    def xdd_poisson_be(
        conn:   wp.array2d(dtype=wp.int32),
        h:      wp.array(dtype=wp.float64),
        Ntab:   wp.array2d(dtype=wp.float64),
        wtab:   wp.array(dtype=wp.float64),
        rho_gp: wp.array(dtype=wp.float64),   # [ne*nqp]
        f_src:  wp.array(dtype=wp.float64),   # [ne*nqp]
        be:     wp.array2d(dtype=wp.float64),  # [ne, nbf]
    ):
        e = wp.tid()
        he = h[e]
        half = he * wp.float64(0.5)
        # jac = (he/2)^dim — power loop (house idiom, see operators.py)
        jac = wp.float64(1.0)
        for _ in range(dim):
            jac = jac * half
        fe = FEMElm()
        fe.e = e
        fe.he = he
        for q in range(nqp):
            fe.q = q
            dJxW = wtab[q] * jac
            gp = e * nqp + q
            fv = rho_gp[gp] + f_src[gp]
            for a in range(nbf):
                wp.atomic_add(be, e, a, fe_N(Ntab, fe, a) * fv * dJxW)

    _kernel_cache[key] = xdd_poisson_be
    return xdd_poisson_be


# ──────────────────────────────────────────────────────────────────────────────
# Assembly helper
# ──────────────────────────────────────────────────────────────────────────────

def assemble_xdd_poisson(dm, eps_gp, rho_gp, *, lam2=None, params=None,
                          f_src_gp=None):
    """Assemble the constrained XDD Poisson system T^T K T, T^T F.

    Parameters
    ----------
    dm : DeviceMesh
    eps_gp : dict {p: np.ndarray[ne_p * nqp_p]}
        Normalised permittivity field ε̂(x) at every Gauss point, per p-bin.
    rho_gp : dict {p: np.ndarray[ne_p * nqp_p]}
        Charge density (p̂ − n̂) at every Gauss point, per p-bin.
    lam2 : float, optional
        λ² Debye-squared scale.  Exactly one of lam2 or params must be given.
    params : XDDParams, optional
        If supplied, lam2 = params.scales().lambda2 is used.
    f_src_gp : dict {p: ndarray} or None
        Extra body load (B4 coupling hook); None → zeros everywhere.

    Returns
    -------
    K : scipy.sparse.csr_matrix  (constrained, shape n_free × n_free)
    F : np.ndarray               (constrained, length n_free)
    """
    if lam2 is None and params is None:
        raise ValueError("supply lam2 or params")
    if params is not None:
        lam2 = params.scales().lambda2
    lam2 = float(lam2)

    d = dm.device
    rows, cols, vals = [], [], []
    F_full = np.zeros(dm.n_nodes)

    for pv, b in dm.bins.items():
        conn_np = dm.mesh.conn_of[pv].astype(np.int64)
        ne, nbf = conn_np.shape
        nqp = b["nqp"]
        ngp = ne * nqp

        eps_np = np.ascontiguousarray(eps_gp[pv], dtype=np.float64)
        rho_np = np.ascontiguousarray(rho_gp[pv], dtype=np.float64)
        fsrc_np = (np.ascontiguousarray(f_src_gp[pv], dtype=np.float64)
                   if f_src_gp is not None else np.zeros(ngp))

        eps_d  = wp.array(eps_np,  dtype=wp.float64, device=d)
        rho_d  = wp.array(rho_np,  dtype=wp.float64, device=d)
        fsrc_d = wp.array(fsrc_np, dtype=wp.float64, device=d)

        Ae = wp.zeros((ne, nbf, nbf), dtype=wp.float64, device=d)
        be = wp.zeros((ne, nbf),      dtype=wp.float64, device=d)

        kA = make_xdd_poisson_Ae(nbf, nqp, dm.dim)
        kb = make_xdd_poisson_be(nbf, nqp, dm.dim)

        wp.launch(kA, dim=ne,
                  inputs=[b["conn"], b["h"], b["dN"], b["w"],
                          eps_d, wp.float64(lam2), Ae],
                  device=d)
        wp.launch(kb, dim=ne,
                  inputs=[b["conn"], b["h"], b["N"], b["w"],
                          rho_d, fsrc_d, be],
                  device=d)

        Aeh = Ae.numpy()
        beh = be.numpy()

        rows.append(np.repeat(conn_np, nbf, axis=1).ravel())
        cols.append(np.tile(conn_np, (1, nbf)).ravel())
        vals.append(Aeh.ravel())
        np.add.at(F_full, conn_np.ravel(), beh.ravel())

    K = sp.coo_matrix(
        (np.concatenate(vals),
         (np.concatenate(rows), np.concatenate(cols))),
        shape=(dm.n_nodes, dm.n_nodes),
    ).tocsr()
    T = dm.constraints.T.tocsr()
    return (T.T @ K @ T).tocsr(), np.asarray(T.T @ F_full)


# ══════════════════════════════════════════════════════════════════════════════
# B2 — carrier drift-diffusion stiffness kernel
# ══════════════════════════════════════════════════════════════════════════════
#
# Kernel signature mirrors scalar_transport.make_scalar_ad_Ae exactly:
#   conn   [ne, nbf]     int32  — local→global DOF map
#   h      [ne]          f64    — element size
#   Ntab   [nqp, nbf]    f64    — basis values (for mass term)
#   dNtab  [nqp, nbf, d] f64    — reference basis gradients
#   lapNtab[nqp, nbf]    f64    — reference basis Laplacians (for VMS p2)
#   wtab   [nqp]         f64    — quadrature weights
#   aq     [ne*nqp, dim] f64    — advection field at GPs (sign*mu*grad_phi)
#   kq     [ne*nqp]      f64    — μ̂(x) diffusivity at GPs
#   sigma                f64    — BDF coefficient (0 for steady)
#   sig2tau              f64    — (2σ)² for tau_m_metric
#   supg                 f64    — SUPG scale (1.0 or 0.0)
#   Ae     [ne,nbf,nbf]  f64    — output (pre-zeroed)
# ══════════════════════════════════════════════════════════════════════════════

def make_xdd_carrier_Ae(nbf: int, nqp: int, dim: int):
    """Element stiffness for the XDD carrier brick (SUPG, nonconservative form).

    Implements:
        σ(N_b, N_a) + (a·∇N_b, N_a) + μ̂(∇N_b, ∇N_a) + τ_M(a·∇N_a, res_b)
    where res_b = σ N_b + a·∇N_b − μ̂ lapN_b  (VMS-complete strong residual).

    Sign of advection is baked into aq_gp by the caller:
        electrons:  aq_gp = −μ̂ ∇φ̂
        holes:      aq_gp = +μ̂ ∇φ̂

    This is ONE factory — sign is not a kernel arg; the caller varies aq_gp.
    Cache key: ("xdd_carrier_Ae", nbf, nqp, dim).
    """
    key = ("xdd_carrier_Ae", nbf, nqp, dim)
    if key in _kernel_cache:
        return _kernel_cache[key]

    dim_f   = float(dim)

    @wp.kernel(module="unique", enable_backward=False,
               module_options=({"max_unroll": 0} if dim >= 3 else {}))
    def xdd_carrier_Ae(
        conn:     wp.array2d(dtype=wp.int32),
        h:        wp.array(dtype=wp.float64),
        Ntab:     wp.array2d(dtype=wp.float64),
        dNtab:    wp.array3d(dtype=wp.float64),
        lapNtab:  wp.array2d(dtype=wp.float64),
        wtab:     wp.array(dtype=wp.float64),
        aq:       wp.array2d(dtype=wp.float64),   # [ne*nqp, dim]
        kq:       wp.array(dtype=wp.float64),     # [ne*nqp]  μ̂ at GPs
        sigma:    wp.float64,
        sig2tau:  wp.float64,
        supg:     wp.float64,
        Ae:       wp.array3d(dtype=wp.float64),
    ):
        e = wp.tid()
        he = h[e]
        half = he * wp.float64(0.5)
        # jac = (he/2)^dim — power loop (house idiom, see operators.py)
        jac = wp.float64(1.0)
        for _ in range(dim):
            jac = jac * half
        dscale = wp.float64(2.0) / he

        for q in range(nqp):
            dJxW = wtab[q] * jac
            gp   = e * nqp + q
            kap  = kq[gp]       # μ̂ at this Gauss point

            # |a| for tau_m_metric
            amag = wp.float64(0.0)
            for d in range(dim):
                amag += aq[gp, d] * aq[gp, d]
            amag = wp.sqrt(amag)
            tauM = supg * tau_m_metric(amag, he, kap, sig2tau,
                                       wp.float64(dim_f))

            for a in range(nbf):
                Na = Ntab[q, a]
                # a·∇N_a  (test function augmentation for SUPG)
                agw = wp.float64(0.0)
                for d in range(dim):
                    agw += aq[gp, d] * dNtab[q, a, d] * dscale

                for b in range(nbf):
                    Nb = Ntab[q, b]
                    # a·∇N_b  (advection on trial function)
                    agu = wp.float64(0.0)
                    # ∇N_a·∇N_b (diffusion)
                    lap = wp.float64(0.0)
                    for d in range(dim):
                        agu += aq[gp, d] * dNtab[q, b, d] * dscale
                        lap += (dNtab[q, a, d] * dNtab[q, b, d]
                                * dscale * dscale)

                    # VMS-complete strong residual on trial N_b:
                    #   res_b = σ N_b + a·∇N_b − μ̂ lapN_b
                    # The lapN term is zero at p1 (Q1 Laplacians = 0) but
                    # required at p2 for third-order L2 convergence (same
                    # finding as scalar_transport.py — see its PROVENANCE note).
                    resu = (sigma * Nb + agu
                            - kap * lapNtab[q, b] * dscale * dscale)

                    # Galerkin: σ(N_b N_a) + (a·∇N_b) N_a + μ̂(∇N_b·∇N_a)
                    # SUPG:     τ_M (a·∇N_a) res_b
                    wp.atomic_add(
                        Ae, e, a, b,
                        (sigma * Na * Nb + Na * agu + kap * lap
                         + tauM * agw * resu) * dJxW)

    _kernel_cache[key] = xdd_carrier_Ae
    return xdd_carrier_Ae


# ══════════════════════════════════════════════════════════════════════════════
# B2 — carrier load kernel
# ══════════════════════════════════════════════════════════════════════════════

def make_xdd_carrier_be(nbf: int, nqp: int, dim: int):
    """Element load for the XDD carrier brick.

    Implements: (f, N_a) + τ_M (a·∇N_a, f)
    where f is the full source (D̂ − R̂ + BDF-history for B4;
    the manufactured source for MMS tests).

    Cache key: ("xdd_carrier_be", nbf, nqp, dim).
    """
    key = ("xdd_carrier_be", nbf, nqp, dim)
    if key in _kernel_cache:
        return _kernel_cache[key]

    dim_f   = float(dim)

    @wp.kernel(module="unique", enable_backward=False,
               module_options=({"max_unroll": 0} if dim >= 3 else {}))
    def xdd_carrier_be(
        conn:    wp.array2d(dtype=wp.int32),
        h:       wp.array(dtype=wp.float64),
        Ntab:    wp.array2d(dtype=wp.float64),
        dNtab:   wp.array3d(dtype=wp.float64),
        wtab:    wp.array(dtype=wp.float64),
        aq:      wp.array2d(dtype=wp.float64),   # [ne*nqp, dim]
        kq:      wp.array(dtype=wp.float64),     # [ne*nqp]  μ̂ at GPs
        fq:      wp.array(dtype=wp.float64),     # [ne*nqp]  source
        sig2tau: wp.float64,
        supg:    wp.float64,
        be:      wp.array2d(dtype=wp.float64),
    ):
        e = wp.tid()
        he = h[e]
        half = he * wp.float64(0.5)
        jac = wp.float64(1.0)
        for _ in range(dim):
            jac = jac * half
        dscale = wp.float64(2.0) / he

        for q in range(nqp):
            dJxW = wtab[q] * jac
            gp   = e * nqp + q
            kap  = kq[gp]
            fv   = fq[gp]

            amag = wp.float64(0.0)
            for d in range(dim):
                amag += aq[gp, d] * aq[gp, d]
            amag = wp.sqrt(amag)
            tauM = supg * tau_m_metric(amag, he, kap, sig2tau,
                                       wp.float64(dim_f))

            for a in range(nbf):
                agw = wp.float64(0.0)
                for d in range(dim):
                    agw += aq[gp, d] * dNtab[q, a, d] * dscale
                # (f, N_a) + τ_M (a·∇N_a, f)  — same structure as scalar_be
                wp.atomic_add(be, e, a,
                              (Ntab[q, a] + tauM * agw) * fv * dJxW)

    _kernel_cache[key] = xdd_carrier_be
    return xdd_carrier_be


# ══════════════════════════════════════════════════════════════════════════════
# B2 — assembly helper
# ══════════════════════════════════════════════════════════════════════════════

def assemble_xdd_carrier(dm, aq_gp, mu_gp, fq_gp, *, sigma=0.0,
                          sig2tau=None, supg=None):
    """Assemble the constrained XDD carrier system T^T K T, T^T F.

    Parameters
    ----------
    dm : DeviceMesh
    aq_gp : dict {p: np.ndarray[ngp, dim]}
        Advection field at Gauss points.  Caller builds this as
        ``sign * mu_gp * grad_phi_gp``:
          electrons → sign=-1  →  aq = −μ̂ ∇φ̂
          holes     → sign=+1  →  aq = +μ̂ ∇φ̂
        The sign convention is fully absorbed by the caller; this assembly
        helper and the kernel are sign-agnostic.
    mu_gp : dict {p: np.ndarray[ngp]}
        Mobility μ̂(x) at GPs (diffusivity in the drift-diffusion PDE).
    fq_gp : dict {p: np.ndarray[ngp]}
        Source f at GPs.  For standalone MMS: manufactured source.
        For B4 coupled solve: D̂ − R̂ + BDF-history terms.
    sigma : float
        BDF time coefficient: 0 (steady), 1/Δt (BDF1), 3/(2Δt) (BDF2).
    sig2tau : float or None
        (2σ)² passed to tau_m_metric.  Derived as (2σ)² if None.
    supg : float or None
        SUPG scale.  1.0 (on, default) or 0.0 (off = Galerkin).

    Returns
    -------
    K : scipy.sparse.csr_matrix  (constrained, n_free × n_free)
    F : np.ndarray               (constrained, length n_free)
    """
    if sig2tau is None:
        sig2tau = (2.0 * sigma) ** 2
    supg_val = 1.0 if supg is None else float(supg)

    d = dm.device
    rows, cols, vals = [], [], []
    F_full = np.zeros(dm.n_nodes)

    for pv, b in dm.bins.items():
        conn_np = dm.mesh.conn_of[pv].astype(np.int64)
        ne, nbf = conn_np.shape
        nqp     = b["nqp"]

        aq_np  = np.ascontiguousarray(aq_gp[pv],  dtype=np.float64)
        mu_np  = np.ascontiguousarray(mu_gp[pv],  dtype=np.float64)
        fq_np  = np.ascontiguousarray(fq_gp[pv],  dtype=np.float64)

        aq_d  = wp.array(aq_np,  dtype=wp.float64, device=d)
        kq_d  = wp.array(mu_np,  dtype=wp.float64, device=d)
        fq_d  = wp.array(fq_np,  dtype=wp.float64, device=d)

        Ae = wp.zeros((ne, nbf, nbf), dtype=wp.float64, device=d)
        be = wp.zeros((ne, nbf),      dtype=wp.float64, device=d)

        kA = make_xdd_carrier_Ae(nbf, nqp, dm.dim)
        kb = make_xdd_carrier_be(nbf, nqp, dm.dim)
        sg = wp.float64(supg_val)

        wp.launch(kA, dim=ne,
                  inputs=[b["conn"], b["h"], b["N"], b["dN"], b["lapN"],
                          b["w"], aq_d, kq_d,
                          wp.float64(sigma), wp.float64(sig2tau), sg, Ae],
                  device=d)
        wp.launch(kb, dim=ne,
                  inputs=[b["conn"], b["h"], b["N"], b["dN"],
                          b["w"], aq_d, kq_d, fq_d,
                          wp.float64(sig2tau), sg, be],
                  device=d)

        Aeh = Ae.numpy()
        beh = be.numpy()

        rows.append(np.repeat(conn_np, nbf, axis=1).ravel())
        cols.append(np.tile(conn_np, (1, nbf)).ravel())
        vals.append(Aeh.ravel())
        np.add.at(F_full, conn_np.ravel(), beh.ravel())

    K = sp.coo_matrix(
        (np.concatenate(vals),
         (np.concatenate(rows), np.concatenate(cols))),
        shape=(dm.n_nodes, dm.n_nodes),
    ).tocsr()
    T = dm.constraints.T.tocsr()
    return (T.T @ K @ T).tocsr(), np.asarray(T.T @ F_full)


# ══════════════════════════════════════════════════════════════════════════════
# B3 — exciton diffusion-reaction stiffness kernel
# ══════════════════════════════════════════════════════════════════════════════
#
# Weak form (per species, NO SUPG):
#   σ_tot(x)(X̂, w) + μ̂_X(x)(∇X̂, ∇w) = (f̂, w)
#   f̂ = Ĝ + R̂_feed + σ_BDF·X̂ⁿ  [+ BDF2 correction when applicable]
#
# DRY decision: the load kernel (be) is identical to the carrier be with
# aq=0 and supg=0 (reduces to (f̂, N_a)dV).  Therefore we REUSE
# make_xdd_carrier_be for the load.  The stiffness (Ae) requires a NEW kernel
# because σ_tot is spatially varying (a GP field, not a scalar), and the
# carrier Ae only accepts a scalar sigma.  Writing a single make_xdd_exciton_Ae
# (< 40 lines) that promotes sigma from float64 to array(float64) is the
# correct DRY tradeoff: ONE new kernel (Ae), ZERO new load kernel (reuse
# carrier_be verbatim).  The brief explicitly allows new factories when reuse
# "contorts the API" — the spatially-varying sigma case is exactly that.
# ══════════════════════════════════════════════════════════════════════════════

def make_xdd_exciton_Ae(nbf: int, nqp: int, dim: int):
    """Element stiffness for the XDD exciton brick (reaction-diffusion, no SUPG).

    Implements per element:
        Ae[e, a, b] += (σ_tot_q N_b N_a + μ̂_X_q ∇N_b·∇N_a) dJxW

    σ_tot_q is a GP field (σ_BDF + 1/τ̂_x + k̂_diss(x) at each Gauss point).
    μ̂_X_q is the exciton diffusivity GP field.
    ONE factory for both X̂_D and X̂_A — species distinction is entirely in the
    GP coefficient fields (μ̂_X and σ_tot may differ between donor/acceptor).
    NO SUPG — excitons are diffusion-dominated (deviation from CPU recorded in
    the module docstring and the SP-1 plan's deviation ledger).

    Kernel signature:
        conn      [ne, nbf]       int32  — local→global DOF map
        h         [ne]            f64    — element size
        Ntab      [nqp, nbf]      f64    — basis values
        dNtab     [nqp, nbf, dim] f64    — reference basis gradients
        wtab      [nqp]           f64    — quadrature weights
        sigma_gp  [ne*nqp]        f64    — σ_tot(x) at GPs
        mu_gp     [ne*nqp]        f64    — μ̂_X(x) at GPs
        Ae        [ne, nbf, nbf]  f64    — output (pre-zeroed)

    Cache key: ("xdd_exciton_Ae", nbf, nqp, dim).
    """
    key = ("xdd_exciton_Ae", nbf, nqp, dim)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @wp.kernel(module="unique", enable_backward=False)
    def xdd_exciton_Ae(
        conn:     wp.array2d(dtype=wp.int32),
        h:        wp.array(dtype=wp.float64),
        Ntab:     wp.array2d(dtype=wp.float64),
        dNtab:    wp.array3d(dtype=wp.float64),
        wtab:     wp.array(dtype=wp.float64),
        sigma_gp: wp.array(dtype=wp.float64),   # [ne*nqp]  σ_tot(x) at GPs
        mu_gp:    wp.array(dtype=wp.float64),   # [ne*nqp]  μ̂_X(x) at GPs
        Ae:       wp.array3d(dtype=wp.float64),  # [ne, nbf, nbf]
    ):
        e = wp.tid()
        he = h[e]
        half = he * wp.float64(0.5)
        # jac = (he/2)^dim — power loop (house idiom, see operators.py)
        jac = wp.float64(1.0)
        for _ in range(dim):
            jac = jac * half
        dscale = wp.float64(2.0) / he

        for q in range(nqp):
            dJxW = wtab[q] * jac
            gp   = e * nqp + q
            sig  = sigma_gp[gp]    # σ_tot at this Gauss point
            kap  = mu_gp[gp]       # μ̂_X at this Gauss point

            for a in range(nbf):
                Na = Ntab[q, a]
                for b in range(nbf):
                    Nb = Ntab[q, b]
                    # diffusion: μ̂_X (∇N_b · ∇N_a)
                    lap = wp.float64(0.0)
                    for d in range(dim):
                        lap += (dNtab[q, a, d] * dNtab[q, b, d]
                                * dscale * dscale)
                    # mass-reaction: σ_tot N_b N_a
                    wp.atomic_add(Ae, e, a, b,
                                  (sig * Na * Nb + kap * lap) * dJxW)

    _kernel_cache[key] = xdd_exciton_Ae
    return xdd_exciton_Ae


# ══════════════════════════════════════════════════════════════════════════════
# B3 — exciton assembly helper
# ══════════════════════════════════════════════════════════════════════════════

def assemble_xdd_exciton(dm, mu_gp, sigma_gp, fq_gp):
    """Assemble the constrained XDD exciton system T^T K T, T^T F.

    Implements the weak form:
        σ_tot(x)(X̂, w) + μ̂_X(x)(∇X̂, ∇w) = (f̂, w)

    DRY NOTE — kernel reuse:
    - Stiffness (Ae): uses make_xdd_exciton_Ae (new kernel, required because
      σ_tot is a GP field — the carrier Ae only accepts a scalar sigma).
    - Load vector (be): REUSES make_xdd_carrier_be with aq=0 and supg=0.
      At these values the carrier be reduces to (f̂, N_a)dV exactly:
          tauM = supg * tau_m_metric(...) = 0   (supg=0)
          agw  = sum(aq[gp,d]*dNtab[q,a,d]*dscale) = 0   (aq=0)
          be[e,a] += (Ntab[q,a] + 0) * fv * dJxW
      This is identical to the Poisson be kernel — reuse is clean.
    Cache keys: "xdd_exciton_Ae" (new) / "xdd_carrier_be" (reused).

    Parameters
    ----------
    dm : DeviceMesh
    mu_gp : dict {p: np.ndarray[ngp]}
        Exciton diffusivity μ̂_X(x) at Gauss points.  Species distinction
        (donor vs acceptor) is in the values; use the same function for both.
    sigma_gp : dict {p: np.ndarray[ngp]}
        Effective mass coefficient σ_tot(x) = σ_BDF + 1/τ̂_x + k̂_diss(x)
        at Gauss points.  May be spatially varying (k̂_diss is a GP field).
    fq_gp : dict {p: np.ndarray[ngp]}
        Full RHS source at GPs: f̂ = Ĝ + R̂_feed + history term(s).

    Returns
    -------
    K : scipy.sparse.csr_matrix  (constrained, n_free × n_free)
    F : np.ndarray               (constrained, length n_free)
    """
    d = dm.device
    rows, cols, vals = [], [], []
    F_full = np.zeros(dm.n_nodes)

    for pv, b in dm.bins.items():
        conn_np = dm.mesh.conn_of[pv].astype(np.int64)
        ne, nbf = conn_np.shape
        nqp     = b["nqp"]
        ngp     = ne * nqp

        mu_np  = np.ascontiguousarray(mu_gp[pv],    dtype=np.float64)
        sig_np = np.ascontiguousarray(sigma_gp[pv], dtype=np.float64)
        fq_np  = np.ascontiguousarray(fq_gp[pv],    dtype=np.float64)

        mu_d   = wp.array(mu_np,  dtype=wp.float64, device=d)
        sig_d  = wp.array(sig_np, dtype=wp.float64, device=d)
        fq_d   = wp.array(fq_np,  dtype=wp.float64, device=d)

        # --- Stiffness: mass-reaction + diffusion (new exciton kernel) ---
        Ae = wp.zeros((ne, nbf, nbf), dtype=wp.float64, device=d)
        kA = make_xdd_exciton_Ae(nbf, nqp, dm.dim)
        wp.launch(kA, dim=ne,
                  inputs=[b["conn"], b["h"], b["N"], b["dN"],
                          b["w"], sig_d, mu_d, Ae],
                  device=d)

        # --- Load: reuse carrier_be with aq=0, supg=0 → (f̂, N_a) dV ---
        # aq=zeros means advection = 0; sig2tau=0 and supg=0 disable SUPG.
        # The carrier_be kernel at these values reduces to Ntab[q,a]*fv*dJxW,
        # which is exactly the exciton load (f̂, N_a)dV.
        aq_zero = wp.zeros((ngp, dm.dim), dtype=wp.float64, device=d)
        mu_be   = wp.array(mu_np, dtype=wp.float64, device=d)  # kq (unused at supg=0)

        be = wp.zeros((ne, nbf), dtype=wp.float64, device=d)
        kb = make_xdd_carrier_be(nbf, nqp, dm.dim)
        wp.launch(kb, dim=ne,
                  inputs=[b["conn"], b["h"], b["N"], b["dN"],
                          b["w"], aq_zero, mu_be, fq_d,
                          wp.float64(0.0),   # sig2tau = 0 (SUPG disabled)
                          wp.float64(0.0),   # supg    = 0 (SUPG disabled)
                          be],
                  device=d)

        Aeh = Ae.numpy()
        beh = be.numpy()

        rows.append(np.repeat(conn_np, nbf, axis=1).ravel())
        cols.append(np.tile(conn_np, (1, nbf)).ravel())
        vals.append(Aeh.ravel())
        np.add.at(F_full, conn_np.ravel(), beh.ravel())

    K = sp.coo_matrix(
        (np.concatenate(vals),
         (np.concatenate(rows), np.concatenate(cols))),
        shape=(dm.n_nodes, dm.n_nodes),
    ).tocsr()
    T = dm.constraints.T.tocsr()
    return (T.T @ K @ T).tocsr(), np.asarray(T.T @ F_full)
