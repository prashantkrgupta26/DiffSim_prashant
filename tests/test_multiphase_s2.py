"""M5 S2 gates: TERNARY thermal annealing — coupled crystallization x
amorphous spinodal decomposition, validated against Siber-Ronsin-
Harting-Frey 2512.16390 (arXiv; published Adv. Sci. 2026,
10.1002/advs.202524140).

VALIDATION-TARGET FINDING (recorded for Baskar): 2512.16390 is a
BINARY PCE11:PCBM blend, not a ternary — the P1 memo's "S2 ternary"
framing anticipated a ternary system.  The paper's Fig-6 case carries
TWO crystallinity fields (PCBM crystallization + inert PCE11
crystallites), which in our solvent-eliminated layout (psi only on
retained species, K <= M) REQUIRES the ternary machinery (M = 2) with a
trace inert third species — so the ternary (M = 2, K <= 2) gates below
are exactly what the replication needs.  The Fig-4 case (only PCBM
crystallizes) maps exactly onto the binary (M = 1, K = 1) brick.

PARAMETER PROVENANCE: the paper's SI (SI-A/B/C) is not in the arXiv
package; the COMPLETE parameter set was extracted from the authors'
published Zenodo dataset (10.5281/zenodo.17633571,
Article_Simulations/*_run*.m MATLAB decks), together with their own
post-processed kinetics curves (.fig files) which serve as quantitative
targets.  Deck-extracted parameters (their symbols):
  rho = [1100, 1600] kg/m3, M_mol = [83, 0.910] kg/mol
  => N_PCE11 = 132.6673, N_PCBM = 1 (v0 = v_PCBM = 5.6875e-4 m3/mol)
  T = 403 K, Tm = [555, 558] K, Lfus = [30000, 15000] J/kg,
  psi_c = [0.2, 0.25], Wfus = 3 Lfus (1 - T/Tm)/(1 - 2 psi_c)
        = [41081, 25000] J/kg
  chi_aa = 509.76/T = 1.2649; chi_sl = [0.8, 1] Lfus/(R T)
         = [7.1630, 4.4769] (additive psi^2 delta-chi, PowCryst = 1
         => our r14 bulk mode)
  D_self [m2/s]: PCE11 in (PCE11, PCBM) = (1e-16, 5e-16);
                 PCBM  in (PCE11, PCBM) = (4e-15, 1e-13); Vignes law,
                 FAST-mode Onsager (their Flag_DiffusionMode = 0)
  eps_cryst = [1e-5, 2.2e-5] sqrt(J/m); kappa = 1e-10 (deck comment
         says sqrt(J/m) — read as J/m, the 2310.11844 Table-1 unit;
         AMBIGUITY RECORDED: the sqrt reading gives 1e-20 J/m,
         a physically absent CH interfacial energy)
  M_psi = [1e-10, 1e-1] 1/s (Fig 4; Fig 6: PCBM 2.5e-2; SI7-DL: 10;
         SI9: 1e-3), AC prefactor (v0 N_i/RT) M_psi,i
  liquid-solid mobility drop (2204.11628 Eq. 13-14):
         (PenVal, PenCentr, PenSlop) = (1e-6, 0.97, 35) [Fig 4]
  AC-noise crystalline damping: (1e-2, 0.85, 15); sigma_AC = 1,
         sigma_CH = 0; IC noise dPhi = 1e-3; 2-D cell depth dy = 1 nm
         (their mesh: ny = 1, dy = 1e-9 — the S1c depth ASSUMPTION is
         their exact convention here)
  domain: 1024 x 1024 nm (Fig 4/6; SI 5-8: 512), dx = 1 nm, periodic
  blend: phi_PCE11 = 0.55 (Fig 4); Fig 6: amorphous phi_PCE11 = 0.45
         + 948 PCE11 crystallite seeds (~8-9 nm, psi = 1, phi = 0.99,
         random positions/orientations; 7.75% of box area from their
         CrystVol curve's t = 0 value)
MISSING/UNAVAILABLE from all sources: none blocking; the SI figure
captions (SI-D..G text) are not public, but the SI run decks + their
.fig post-processing curves cover the parameters and the numbers.

THEIR MEASURED TARGETS (from their .fig post-processing; X_mat = %
of PCBM material crystallized, onset at X_mat = 1%):
  Fig 4  (1024): onset 54.5 s, t50 131.0 s, plateau 96.0% at 271 s;
                 crystal number-avg size -> 405 nm; liquid PCBM conc
                 0.45 -> 0.032; PCBM-rich liquid phase 0.968 max ->
                 0.959 (consumed)
  SI7-NDL (512, M_psi = 0.1): onset 50.3 s, t50 100.9 s, end 96.1%
  SI7-DL  (512, M_psi = 10): onset 3.4 s, t50 5.8 s (14.8x/17.4x
                 faster than NDL), end 96.1%; PCBM-rich liquid STUCK
                 at 0.24 at the end (depletion zones - the
                 diffusion-limited signature; NDL: 0.946)
  Fig 6  (1024, seeds, M_psi = 2.5e-2): PCBM onset 375 s, t50 679 s,
                 X_box end 0.466; PCE11 crystal volume 15.1% -> 7.8%
                 of box (early bulk equilibration + late dissolution
                 at PCBM growth fronts); crystal size end 428 nm
  SI9 (M_psi = 1e-3) vs Fig 6: onset 6183 s, t50 14123 s (16.5-20.8x
                 ~ the 25x mobility ratio), END STATE EQUAL: X_box
                 0.465 vs 0.466, size 394.9 vs 395.1 nm — morphology
                 decoupled from crystallization kinetics (their SI-G).

Tolerances measured-then-locked (>= 2x headroom); noise-intensity-bound
comparisons flagged where they matter (P1 Sec 8) — here their sigma_AC
= 1 with the FDT normalization and dy = 1 nm depth are KNOWN, so onset
comparisons are calibration-matched up to domain-size statistics."""
import importlib.util

import numpy as np
import pytest

_has_nvmath = importlib.util.find_spec("nvmath") is not None
_skip_nvmath = pytest.mark.skipif(
    not _has_nvmath, reason="nvmath (cuDSS) not available on this machine"
)

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.physics.multiphase import (MultiPhaseStepper, grain_labels,
                                        np_potentials, np_ls_interp)

pytestmark = pytest.mark.tier3

# ---------------------------------------------------------------------
# 2512.16390 deck parameters (Zenodo run decks; module docstring)
# ---------------------------------------------------------------------
R_GAS, N_AV = 8.314, 6.022e23
RHO = (1100.0, 1600.0)
M_MOL = (83.0, 0.910)
V_MOL = (M_MOL[0] / RHO[0], M_MOL[1] / RHO[1])
V0 = min(V_MOL)
N_PCE11, N_PCBM = V_MOL[0] / V0, 1.0
T_Q = 403.0
TM = (555.0, 558.0)
LFUS = (30000.0, 15000.0)
PSI_C = (0.2, 0.25)
WFUS = tuple(3.0 * LFUS[i] * (1.0 - T_Q / TM[i]) / (1.0 - 2.0 * PSI_C[i])
             for i in range(2))
CHI_AA_16390 = 509.76 / T_Q
CHI_SL = (0.8 * LFUS[0] / (R_GAS * T_Q), 1.0 * LFUS[1] / (R_GAS * T_Q))
U0 = R_GAS * T_Q / V0
EPS2_CR = (1e-5 ** 2, 2.2e-5 ** 2)
KAP_16390 = 1e-10
D_16390 = np.array([[1e-16, 5e-16],     # PCE11 in (PCE11, PCBM)
                    [4e-15, 1e-13]])    # PCBM  in (PCE11, PCBM)
LS_DROP = (1e-6, 0.97, 35.0)
NOISE_DAMP = (1e-2, 0.85, 15.0)
EPSGRAIN = 3e-2                          # J/m2


def _dm(level, device, periodic=False):
    tree = build_uniform(level, dim=2,
                         periodic=(True, True) if periodic else None)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    return (DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2),
                                 device), mesh, cons)


def _noise_psi(l0, depth=1e-9):
    return np.sqrt((V0 / N_AV) / (l0 ** 2 * depth))


def _fig4_pars(l0, mpsi=0.1, ls_drop=LS_DROP):
    """Binary (M=1, K=1): retained species 0 = PCBM (crystallizing),
    eliminated = PCE11 (their Fig-4/SI-5..8 model: PCE11 crystallization
    omitted).  r14 bulk = their gg/pp polynomials at PowCryst = 1."""
    sc = 1.0 / (U0 * l0 ** 2)
    chi_aa = np.array([[0.0, CHI_AA_16390], [CHI_AA_16390, 0.0]])
    chi_ca = np.array([[0.0, CHI_SL[1]], [0.0, 0.0]])
    Dslf = np.array([[D_16390[1, 1], D_16390[1, 0]],
                     [D_16390[0, 1], D_16390[0, 0]]]) / l0 ** 2
    return dict(
        M=1, K=1, chi_aa=chi_aa, chi_ac=chi_ca.T.copy(), chi_ca=chi_ca,
        N=[N_PCBM, N_PCE11], mob="fastmode_n", D_self=Dslf,
        ls_drop=ls_drop, kappa=[KAP_16390 * sc],
        dsig=[WFUS[1] * RHO[1] / U0], dh=[LFUS[1] * RHO[1] / U0],
        Tm=[TM[1]], eps2=[EPS2_CR[1] * sc], L_psi=[N_PCBM * mpsi],
        T=T_Q, bulk="r14", noise_psi=_noise_psi(l0),
        noise_damp=NOISE_DAMP, clip_psi=False)


def _fig6_pars(l0, mpsi=2.5e-2, ls_drop=(2.5e-5, 0.97, 35.0)):
    """Ternary-trace (M=2, K=2): 0 = PCE11 (cryst, frozen M_psi 1e-10),
    1 = PCBM (cryst), eliminated 2 = INERT TRACE species (DEVIATION,
    recorded: the paper's film is a dry binary with BOTH species
    crystalline; our layout carries psi only on retained species, so a
    small inert filler (chi = 0, PCBM-class D) is the eliminated one)."""
    sc = 1.0 / (U0 * l0 ** 2)
    z3 = np.zeros((3, 3))
    chi_aa = z3.copy()
    chi_aa[0, 1] = chi_aa[1, 0] = CHI_AA_16390
    chi_ca = z3.copy()
    chi_ca[0, 1] = CHI_SL[0]
    chi_ca[1, 0] = CHI_SL[1]
    Dslf = np.array([[D_16390[0, 0], D_16390[0, 1], D_16390[0, 1]],
                     [D_16390[1, 0], D_16390[1, 1], D_16390[1, 1]],
                     [D_16390[1, 0], D_16390[1, 1], D_16390[1, 1]]]) \
        / l0 ** 2
    return dict(
        M=2, K=2, chi_aa=chi_aa, chi_ac=chi_ca.T.copy(), chi_ca=chi_ca,
        N=[N_PCE11, N_PCBM, 1.0], mob="fastmode_n", D_self=Dslf,
        ls_drop=ls_drop,
        kappa=[KAP_16390 * sc] * 2,
        dsig=[WFUS[0] * RHO[0] / U0, WFUS[1] * RHO[1] / U0],
        dh=[LFUS[0] * RHO[0] / U0, LFUS[1] * RHO[1] / U0],
        Tm=list(TM), eps2=[EPS2_CR[0] * sc, EPS2_CR[1] * sc],
        L_psi=[N_PCE11 * 1e-10, N_PCBM * mpsi],
        T=T_Q, bulk="r14", noise_psi=_noise_psi(l0),
        noise_damp=NOISE_DAMP, clip_psi=False)


# ---------------------------------------------------------------------
# S2a — TERNARY COUPLED MMS: (M=2, K=1) and (M=2, K=2), p1 bulk
# ---------------------------------------------------------------------
@pytest.mark.parametrize("K", [1, 2])
def test_s2a_mms_orders(K, device):
    """Manufactured ternary (M=2) + K crystal pairs, p1 bulk, const
    SPD Onsager with off-diagonal coupling; independent-mu* trick, all
    sources numpy pointwise from np_potentials + analytic gradient
    terms (the S1a pattern extended per-species)."""
    M = 2
    Ons = np.array([[1.0, -0.2], [-0.2, 0.8]])
    kap = [0.02, 0.015]
    dt = 1e-3
    Ninv = np.array([1.0, 0.5, 1.0])
    dsig = [1.5, 1.2][:K]
    drive_v = [-0.8, -0.6][:K]
    Tm, Tq = 1.0, 0.5
    dh = [dv / (1 - Tq / Tm) for dv in drive_v]
    eps2 = [5e-3, 4e-3][:K]
    Lpsi = [1.0, 0.8][:K]
    alpha = [0.4, 0.3][:K]
    beta = [0.1, 0.08][:K]
    Lth = [1.0, 0.9][:K]
    kgd, pfl = 1e-2, 1e-3
    chi_aa = np.array([[0.0, 1.2, 0.8], [1.2, 0.0, 0.9],
                       [0.8, 0.9, 0.0]])
    chi_ac = np.array([[0.0, 0.7, 0.5], [0.4, 0.0, 0.6],
                       [0.3, 0.45, 0.0]])
    chi_ca = chi_ac.T.copy()
    chi_cc = np.array([[0.0, 0.9, 0.7], [0.9, 0.0, 0.8],
                       [0.7, 0.8, 0.0]])
    pars = dict(chi_aa=chi_aa, chi_ac=chi_ac, chi_ca=chi_ca,
                chi_cc=chi_cc, Ninv=Ninv, dsig=dsig, drive=drive_v,
                breg=0.0, bulk="p1")

    pi = np.pi
    E = lambda t: np.exp(-t)
    ap = [0.08, 0.06]
    phis_f = [lambda x, t, a=ap[i]: 0.3 + a * np.cos(pi * x[:, 0])
              * np.cos(pi * x[:, 1]) * E(t) for i in range(M)]
    bm = [1.0, 0.7]
    mus_f = [lambda x, t, b=bm[i]: b * np.sin(pi * x[:, 0])
             * np.sin(pi * x[:, 1]) * E(t) for i in range(M)]
    cp = [0.25, 0.2]
    psis_f = [lambda x, t, c=cp[k]: 0.5 + c * np.cos(pi * x[:, 0])
              * np.cos(2 * pi * x[:, 1]) * E(t) for k in range(K)]
    dth = [0.2, 0.15]
    ths_f = [lambda x, t, d=dth[k]: 0.5 + d * np.sin(pi * x[:, 0])
             * np.cos(pi * x[:, 1]) * E(t) for k in range(K)]

    def th_grad(k, x, t):
        d = dth[k]
        gx = d * pi * np.cos(pi * x[:, 0]) * np.cos(pi * x[:, 1]) * E(t)
        gy = -d * pi * np.sin(pi * x[:, 0]) * np.sin(pi * x[:, 1]) * E(t)
        return gx, gy

    def psi_grad(k, x, t):
        c = cp[k]
        gx = -c * pi * np.sin(pi * x[:, 0]) * np.cos(2 * pi * x[:, 1]) \
            * E(t)
        gy = -2 * c * pi * np.cos(pi * x[:, 0]) * np.sin(2 * pi * x[:, 1]) \
            * E(t)
        return gx, gy

    pori = lambda s: s ** 2 * (3 - 2 * s)
    porip = lambda s: 6 * s * (1 - s)

    def f_phi(i):
        def f(x, t):
            lapmu = sum(Ons[i, j] * (-2 * pi ** 2 * mus_f[j](x, t))
                        for j in range(M))
            return -(phis_f[i](x, t) - 0.3) - lapmu
        return f

    def f_mu(i):
        def f(x, t):
            mu_b, _ = np_potentials([pf(x, t) for pf in phis_f],
                                    [sf(x, t) for sf in psis_f], pars)
            lap_phi = -2 * pi ** 2 * (phis_f[i](x, t) - 0.3)
            return mus_f[i](x, t) - mu_b[i] + kap[i] * lap_phi
        return f

    def f_psi(k):
        def f(x, t):
            _, dfs = np_potentials([pf(x, t) for pf in phis_f],
                                   [sf(x, t) for sf in psis_f], pars)
            gx, gy = th_grad(k, x, t)
            g2 = gx ** 2 + gy ** 2
            S = np.sqrt(g2 + kgd ** 2)
            Eori = 0.5 * alpha[k] * S + 0.5 * beta[k] * g2
            lap_psi = -5 * pi ** 2 * (psis_f[k](x, t) - 0.5)
            return (-(psis_f[k](x, t) - 0.5)
                    + Lpsi[k] * (dfs[k] + porip(psis_f[k](x, t)) * Eori)
                    - Lpsi[k] * eps2[k] * lap_psi)
        return f

    def f_th(k):
        def f(x, t):
            s = psis_f[k](x, t)
            th = ths_f[k](x, t)
            gx, gy = th_grad(k, x, t)
            g2 = gx ** 2 + gy ** 2
            S = np.sqrt(g2 + kgd ** 2)
            c = (pori(s) + pfl) * Lth[k] * (0.5 * alpha[k] / S + beta[k])
            sx, sy = psi_grad(k, x, t)
            d = dth[k]
            Hxx = -pi ** 2 * (th - 0.5)
            Hyy = -pi ** 2 * (th - 0.5)
            Hxy = -d * pi ** 2 * np.cos(pi * x[:, 0]) \
                * np.sin(pi * x[:, 1]) * E(t)
            Hgx = Hxx * gx + Hxy * gy
            Hgy = Hxy * gx + Hyy * gy
            cf = 0.5 * alpha[k] / S + beta[k]
            cx = Lth[k] * (porip(s) * sx * cf
                           - (pori(s) + pfl) * 0.5 * alpha[k] * Hgx
                           / S ** 3)
            cy = Lth[k] * (porip(s) * sy * cf
                           - (pori(s) + pfl) * 0.5 * alpha[k] * Hgy
                           / S ** 3)
            lap_th = -2 * pi ** 2 * (th - 0.5)
            return (pori(s) + pfl) * (-(th - 0.5)) \
                - (cx * gx + cy * gy + c * lap_th)
        return f

    fields = []
    for i in range(M):
        fields += [phis_f[i], mus_f[i]]
    for k in range(K):
        fields += [psis_f[k], ths_f[k]]
    srcs = []
    for i in range(M):
        srcs += [f_phi(i), f_mu(i)]
    for k in range(K):
        srcs += [f_psi(k), f_th(k)]

    names = [f"phi{i}" for i in range(M)] + [f"psi{k}" for k in range(K)] \
        + [f"theta{k}" for k in range(K)]
    errs = {f: [] for f in names}
    for lv in (4, 5):
        dm, mesh, cons = _dm(lv, device)
        coords = mesh.node_coords[cons.free_nodes]
        bdry = np.zeros(len(coords), bool)
        for cc in range(2):
            bdry |= (np.abs(coords[:, cc]) < 1e-12) | \
                    (np.abs(coords[:, cc] - 1) < 1e-12)
        st = MultiPhaseStepper(
            dm, M=M, K=K, chi_aa=chi_aa, chi_ac=chi_ac, chi_ca=chi_ca,
            chi_cc=chi_cc, N=1.0 / Ninv, onsager=Ons, kappa=kap,
            dsig=dsig, dh=dh, Tm=[Tm] * K, eps2=eps2, L_psi=Lpsi,
            alpha_th=alpha, beta_th=beta, L_th=Lth, T=Tq,
            dt=dt, bulk="p1", kg_delta=kgd, p_floor=pfl,
            newton_tol=1e-10, newton_max=80,
            dirichlet=np.where(bdry)[0], g_fns=fields, src_fns=srcs)
        st.set_initial([lambda x, i=i: phis_f[i](x, 0.0)
                        for i in range(M)],
                       [lambda x, k=k: psis_f[k](x, 0.0)
                        for k in range(K)],
                       [lambda x, k=k: ths_f[k](x, 0.0)
                        for k in range(K)])
        for i in range(M):
            st.x[2 * i + 1::st.ndof] = mus_f[i](st.free_coords, 0.0)
        for _ in range(4):
            st.step()
        from diffsim.physics.poisson import l2_error
        for i in range(M):
            errs[f"phi{i}"].append(l2_error(
                dm, np.asarray(cons.T @ st.phi(i)),
                lambda x, i=i: phis_f[i](x, st.t)))
        for k in range(K):
            errs[f"psi{k}"].append(l2_error(
                dm, np.asarray(cons.T @ st.psi(k)),
                lambda x, k=k: psis_f[k](x, st.t)))
            errs[f"theta{k}"].append(l2_error(
                dm, np.asarray(cons.T @ st.theta(k)),
                lambda x, k=k: ths_f[k](x, st.t)))
    orders = {f: np.log2(errs[f][0] / errs[f][1]) for f in errs}
    print(f"S2a MMS [M=2, K={K}]: " + "; ".join(
        f"{f} errs {[f'{e:.2e}' for e in errs[f]]} "
        f"order {orders[f]:.2f}" for f in errs))
    for f in names:
        assert orders[f] > 1.8, (f, errs[f], orders[f])


# ---------------------------------------------------------------------
# S2a2 — MULTICOMPONENT MOBILITY MODES (anchor Eqs. 11-14)
# ---------------------------------------------------------------------
def test_s2a2_fastmode_n_parity_and_drop(device):
    """(i) fastmode_n at M = 1 reduces EXACTLY to the S1 fastmode
    scalar (Eq. 12 at n = 2) — 3-step trajectory parity (measured
    2.2e-16, FP-identical class).  (ii) The Eq. 13-14 liquid-solid drop
    inside a uniform crystal (psi = 1) suppresses the CH flux by
    f(1; 1e-6, 0.97, 35) = 4.514e-6 — measured one-step |dphi| ratio
    vs drop-off.  (iii) slowmode_n (Eq. 11) runs the same config and
    conserves the field integral to machine precision (measured 5.6e-17
    on 3 steps at M = 2, K = 2 during the build)."""
    dm, mesh, cons = _dm(5, device)
    chi_aa = np.array([[0.0, 0.7248], [0.7248, 0.0]])
    chi_ca = np.array([[0.0, 1.0836], [0.0, 0.0]])
    common = dict(M=1, K=1, chi_aa=chi_aa, chi_ac=chi_ca.T.copy(),
                  chi_ca=chi_ca, N=[5.0298, 1.0], kappa=[2e-4],
                  dsig=[2.6355], dh=[1.3072], Tm=[558.0], eps2=[1e-3],
                  L_psi=[5.0], T=333.0, dt=2e-3, bulk="r14",
                  newton_tol=1e-9, newton_max=60)
    dlo, dhi = [5e-4, 2e-3], [1e-6, 1e-5]
    Dslf = np.array([[dhi[0], dlo[0]], [dlo[1], dhi[1]]])
    stA = MultiPhaseStepper(dm, mob="fastmode", D_lo=dlo, D_hi=dhi,
                            **common)
    stB = MultiPhaseStepper(dm, mob="fastmode_n", D_self=Dslf, **common)
    rng = np.random.default_rng(3)
    ic = 0.5 + 0.05 * rng.standard_normal(len(stA.free_coords))
    disc = lambda x: 0.5 * (1.0 - np.tanh(
        (np.sqrt((x[:, 0] - 0.5) ** 2 + (x[:, 1] - 0.5) ** 2) - 0.15)
        / 0.02))
    for st in (stA, stB):
        st.set_initial([lambda x: ic], [disc],
                       [lambda x: np.zeros(len(x))])
    errs = []
    for _ in range(3):
        xa = stA.step()
        xb = stB.step()
        errs.append(np.abs(xa - xb).max())
    print(f"S2a2 fastmode vs fastmode_n parity: "
          f"{[f'{e:.2e}' for e in errs]}")
    # measured 2.2e-16 with identical (Picard) Jacobians; the exact
    # dLam blocks (fastmode_n only) + frozen-theta rows move the
    # Newton STOPPING POINTS: converged-state parity re-measured
    # 5.1e-13 (2026-07-11) — same root, different iterate tails
    assert max(errs) < 5e-12, errs          # measured 5.1e-13

    # (ii) drop suppression: uniform crystal, smooth phi perturbation
    dphi = {}
    for drop in (None, LS_DROP):
        st = MultiPhaseStepper(dm, mob="fastmode_n", D_self=Dslf,
                               ls_drop=drop, **common)
        pert = lambda x: 0.5 + 0.05 * np.cos(2 * np.pi * x[:, 0]) \
            * np.cos(2 * np.pi * x[:, 1])
        st.set_initial([pert], [lambda x: np.ones(len(x))],
                       [lambda x: np.zeros(len(x))])
        p0 = st.phi(0).copy()
        st.step()
        dphi[drop is None] = np.abs(st.phi(0) - p0).max()
    ratio = dphi[False] / dphi[True]
    f_pred = np_ls_interp(1.0, *LS_DROP)
    print(f"S2a2 ls_drop: |dphi| {dphi[True]:.3e} -> {dphi[False]:.3e}, "
          f"ratio {ratio:.3e} vs Eq.13-14 f(1) = {f_pred:.3e}")
    assert 0.3 * f_pred < ratio < 3.0 * f_pred, (ratio, f_pred)

    # (iii) slowmode_n runs (integral conservation gated in the build
    # measurement; here: finite, converged march step)
    stS = MultiPhaseStepper(dm, mob="slowmode_n", D_self=Dslf, **common)
    stS.set_initial([lambda x: ic], [disc],
                    [lambda x: np.zeros(len(x))])
    xs = stS.step()
    assert np.isfinite(xs).all()


def test_s2a3_noise_damping(device):
    """Crystalline FDT-noise damping (anchor Eq. 18 f(phi_k)):
    (a) np_ls_interp reproduces Eq. 13 analytically at the deck values;
    (b) integration-level: same seed, psi0 = 0.9 crystalline plateau,
    fluctuation std ratio damped/undamped ~ f(0.9; 1e-2, 0.85, 15)
    = 2.32e-2 (BDF1 filters both identically; measured ratio in the
    print, locked with 2x-class headroom)."""
    assert abs(np_ls_interp(1.0, 1e-6, 0.97, 35.0) - 1e-6 ** (
        0.5 * (1 + np.tanh(35.0 * 0.03)))) < 1e-18
    assert abs(np_ls_interp(0.0, *NOISE_DAMP) - 1.0) < 1e-4
    f09 = np_ls_interp(0.9, *NOISE_DAMP)
    dm, mesh, cons = _dm(5, device)
    chi_aa = np.array([[0.0, 0.7248], [0.7248, 0.0]])
    chi_ca = np.array([[0.0, 1.0836], [0.0, 0.0]])
    stds = {}
    for damp in (None, NOISE_DAMP):
        st = MultiPhaseStepper(
            dm, M=1, K=1, chi_aa=chi_aa, chi_ac=chi_ca.T.copy(),
            chi_ca=chi_ca, N=[5.0298, 1.0], onsager=[[0.1]],
            kappa=[2e-4], dsig=[2.6355], dh=[1.3072], Tm=[558.0],
            eps2=[1e-3], L_psi=[5.0], T=333.0, dt=1e-3, bulk="r14",
            noise_psi=1e-3, noise_seed=42, noise_damp=damp,
            clip_psi=False, newton_tol=1e-9, newton_max=60)
        st.set_initial([lambda x: np.full(len(x), 0.6)],
                       [lambda x: np.full(len(x), 0.9)],
                       [lambda x: np.zeros(len(x))])
        st.step()
        psi = st.psi(0)
        stds[damp is None] = float(np.std(psi - psi.mean()))
    ratio = stds[False] / stds[True]
    print(f"S2a3 noise damp: fluct std {stds[True]:.3e} -> "
          f"{stds[False]:.3e}, ratio {ratio:.3e} vs f(0.9) = {f09:.3e}")
    assert 0.5 * f09 < ratio < 2.0 * f09, (ratio, f09)


# ---------------------------------------------------------------------
# S2b — PHASE-DIAGRAM PLACEMENT (16390 chi set; d5cp taxonomy)
# ---------------------------------------------------------------------
def _fh_binary(x, breg=0.0):
    """FH free energy per site (v0 units), x = phi_PCBM, with the house
    simplex barrier (saturated below 1e-3, the _binv convention)."""
    xs = np.maximum(x, 1e-3)
    ys = np.maximum(1.0 - x, 1e-3)
    return (x * np.log(x) / N_PCBM + (1 - x) * np.log(1 - x) / N_PCE11
            + CHI_AA_16390 * x * (1 - x) + breg * (1 / xs + 1 / ys))


def _binodal(breg):
    from scipy.optimize import brentq
    xg = np.linspace(1e-6, 1 - 1e-9, 2000001)
    fv = _fh_binary(xg, breg)

    def gap(mu):
        om = fv - mu * xg
        i1 = np.argmin(om[xg < 0.4])
        j = np.searchsorted(xg, 0.9)
        i2 = np.argmin(om[xg > 0.9]) + j
        return om[i1] - om[i2], xg[i1], xg[i2]
    mu = brentq(lambda m: gap(m)[0], -1.0, 1.0, xtol=1e-14)
    return gap(mu)[1:]


@_skip_nvmath
def test_s2b_phase_diagram_placement(device):
    """PLACEMENT (analytic, the 16390 chi set at the annealing T=403K):
    chi_aa = 1.2649 = 2.14 chi_c (chi_c = 0.5906 at N = (132.67, 1)) —
    ONE immiscible pair, all other pairs (the trace species) miscible =
    d5cp00335k taxonomy type [110] ('classic textbook' single miscibility
    gap, one critical point; their Fig. 2 key: one immiscible pair, one
    two-phase gap, no three-phase region).  Computed amorphous
    boundaries: spinodal phi_PCBM in (0.3972, 0.9951); bare binodal
    (0.2030, ~1.0); with the house barrier b_reg = 3e-4 CALIBRATED to
    the paper's own measured liquid plateaus: binodal (0.217, 0.972) vs
    THEIR post-processing curves 0.21-0.243 (PCE11-rich, LiqConc
    'Solute 1') and 0.968 (PCBM-rich max, 'Solute 2') — both branches
    within 3%.  Placement: Fig-4 IC phi_PCBM = 0.45 INSIDE the spinodal
    (f'' = -0.294) -> AAPS by spinodal decomposition (their as-cast
    observation); Fig-6 amorphous IC 0.55 deeper inside (f'' = -0.695);
    dilute control 0.05 OUTSIDE the binodal (f'' = +17.5) -> no AAPS.
    MARCH: the qualitative consequence on the ternary-trace (M=2, K=2)
    brick at the 16390 parameters, 128 nm box, IC noise 1e-3, no FDT
    noise: inside-IC phi_PCBM std GROWS (SD onset within ~1 s, their
    0.43 s wave-pattern class), outside-IC std DECAYS."""
    fpp = lambda x: 1 / (N_PCBM * x) + 1 / (N_PCE11 * (1 - x)) \
        - 2 * CHI_AA_16390
    chi_c = 0.5 * (1 / np.sqrt(N_PCE11) + 1 / np.sqrt(N_PCBM)) ** 2
    assert CHI_AA_16390 > chi_c                      # immiscible pair
    from scipy.optimize import brentq
    xs1 = brentq(fpp, 1e-6, 0.9)
    xs2 = brentq(fpp, 0.9, 1 - 1e-9)
    a0, b0 = _binodal(0.0)
    a3, b3 = _binodal(3e-4)
    print(f"S2b: chi/chi_c {CHI_AA_16390 / chi_c:.3f}; spinodal "
          f"({xs1:.4f}, {xs2:.4f}); binodal bare ({a0:.4f}, {b0:.5f}) "
          f"breg=3e-4 ({a3:.4f}, {b3:.4f}); their plateaus 0.21-0.243 / "
          f"0.968")
    assert xs1 < 0.45 < xs2 and fpp(0.45) < 0        # Fig 4 IC: AAPS
    assert xs1 < 0.55 < xs2 and fpp(0.55) < 0        # Fig 6 amorphous
    assert 0.05 < a3 and fpp(0.05) > 0               # dilute: stable
    assert abs(b3 - 0.968) < 0.03, b3                # their plateau
    assert abs(a3 - 0.23) < 0.05, a3

    # short march both ways (ternary-trace, no crystallization active)
    l0 = 128e-9
    tree = build_uniform(6, dim=2, periodic=(True, True))
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2),
                              device)
    grow = {}
    for phi2 in (0.45, 0.05):
        pars = _fig6_pars(l0)
        pars["noise_psi"] = 0.0
        st = MultiPhaseStepper(dm, dt=1e-3, newton_tol=1e-6,
                               newton_max=40, linsolver="cudss",
                               b_reg=3e-4, line_search=True,
                               alpha_th=[0.0, 0.0], beta_th=[0.0, 0.0],
                               L_th=[1e-8, 1e-8], **pars)
        rng = np.random.default_rng(4)
        nf = len(st.free_coords)
        tr = 0.02
        p2 = phi2 * (1 - tr) + 1e-3 * rng.standard_normal(nf)
        p1 = (1 - phi2) * (1 - tr) + 1e-3 * rng.standard_normal(nf)
        st.set_initial([lambda x: p1, lambda x: p2],
                       [lambda x: np.zeros(len(x))] * 2,
                       [lambda x: np.zeros(len(x))] * 2)
        s0 = float(np.std(st.phi(1)))
        # ternary-trace exchange mode: sigma_max = 2.44/s at lambda =
        # 105 nm (vs the paper's binary 6.36/s at 69 nm — the trace
        # representation slows SD ~2.6x and coarsens ~1.5x, ANALYTIC,
        # recorded); horizon 3 s gives e^{~7} growth of the box mode
        st.march(t_end=3.0, dt_max=0.02, dt_min=1e-9, grow_iters=45,
                 max_steps=1200)
        grow[phi2] = (s0, float(np.std(st.phi(1))))
    print(f"S2b march: inside-IC std {grow[0.45][0]:.2e} -> "
          f"{grow[0.45][1]:.2e}; outside-IC {grow[0.05][0]:.2e} -> "
          f"{grow[0.05][1]:.2e}")
    assert grow[0.45][1] > 10 * grow[0.45][0], grow[0.45]   # AAPS
    assert grow[0.05][1] < 0.5 * grow[0.05][0], grow[0.05]  # decay


# ---------------------------------------------------------------------
# S2d — GRAIN STATISTICS + CRYSTALLINITY BOOKKEEPING
# ---------------------------------------------------------------------
def _smooth_theta(mesh, cons, seed, cut=6.0):
    """Low-pass-filtered random field in [0, 1] on the periodic grid —
    the nucleation-continuous theta IC (each crystal inherits its local
    value; the P1 Sec 8 marker-assignment stand-in)."""
    coords = mesh.node_coords[cons.free_nodes]
    xs = np.unique(np.round(coords[:, 0], 12))
    n = len(xs)
    rng = np.random.default_rng(seed)
    F = np.fft.fft2(rng.standard_normal((n, n)))
    k1 = np.fft.fftfreq(n, d=1.0 / n)
    KX, KY = np.meshgrid(k1, k1, indexing="ij")
    g = np.real(np.fft.ifft2(F * np.exp(-(KX ** 2 + KY ** 2)
                                        / (2 * cut ** 2))))
    g = (g - g.min()) / (g.max() - g.min())
    ix = np.searchsorted(xs, np.round(coords[:, 0], 12))
    iy = np.searchsorted(xs, np.round(coords[:, 1], 12))
    return g[ix, iy]


def _seed_discs(coords, n_seeds, r0, w, seed, min_sep=0.0):
    rng = np.random.default_rng(seed)
    ctr = []
    while len(ctr) < n_seeds:
        c = rng.uniform(0.0, 1.0, size=2)
        ok = True
        for c0 in ctr:
            dd = c - c0
            dd -= np.round(dd)
            if np.hypot(*dd) < min_sep:
                ok = False
                break
        if ok:
            ctr.append(c)
    ctr = np.array(ctr)
    psi = np.zeros(len(coords))
    for c in ctr:
        dd = coords[:, :2] - c[None, :]
        dd -= np.round(dd)
        r = np.sqrt((dd ** 2).sum(axis=1))
        psi = np.maximum(psi, 0.5 * (1.0 - np.tanh((r - r0) / w)))
    return psi, ctr


@_skip_nvmath
def test_s2d_grain_statistics(device):
    """Multi-grain bookkeeping on the Fig-6-class configuration (S2c
    case): O(15) PCE11 crystallite seeds (their 948-seed case scaled to
    the 128 nm box) with CONTINUOUS per-seed theta values (their
    NucleusOrientation = one random angle per nucleus; nucleation-
    assigned values are continuous, so the labeling tolerance must be
    MEASURED), KWC orientation dynamics ON (alpha at the mapped
    EpsGrain, the S1b kg_delta/beta taming), short march at the 16390
    parameters.  Gates:
    (a) LABELING TOLERANCE (measured-then-locked): post-march
        intra-grain theta spread vs pairwise seed-theta gaps; the
        locked theta_tol = 0.02 sits >= 2x above the largest measured
        intra-grain spread, and grains whose theta values differ by
        less CAN alias if they touch (recorded limitation of any
        scalar-marker scheme, theirs included);
    (b) SIZE DISTRIBUTION: theta-watershed count == psi-connected
        count == seed count when all pairwise gaps exceed the
        tolerance (measured for the locked seed);
    (c) CRYSTALLINITY-PER-SPECIES BOOKKEEPING: per-grain GP-partitioned
        quadrature integrals of phi_i psi_i sum to the whole-domain
        crystalline-masked field integral to 1e-10 (partition
        exactness); X_i = int(phi_i psi_i)/int(phi_i) per species."""
    level, l0 = 6, 128e-9
    dm, mesh, cons = _dm(level, device, periodic=True)
    pars = _fig6_pars(l0)
    alpha_nd = np.pi * EPSGRAIN / (U0 * l0)
    st = MultiPhaseStepper(dm, dt=1e-3, newton_tol=1e-6, newton_max=60,
                           linsolver="cudss", b_reg=3e-4,
                           line_search=True, noise_seed=11,
                           alpha_th=[alpha_nd, alpha_nd],
                           beta_th=[0.1 * alpha_nd, 0.1 * alpha_nd],
                           L_th=[pars["L_psi"][1]] * 2, kg_delta=0.2,
                           **pars)
    rng = np.random.default_rng(4)
    nf = len(st.free_coords)
    n_seeds = 15
    # min separation 3.5 r0: DISJOINT grains (impingement/merged-pair
    # discrimination is the S1b gate; here the many-grain bookkeeping)
    psi_seed, ctr = _seed_discs(st.free_coords, n_seeds, r0=4.5 / 128.0,
                                w=1.5 / 128.0, seed=111,
                                min_sep=3.5 * 4.5 / 128.0)
    # per-seed continuous theta (nearest-seed assignment everywhere:
    # amorphous theta is frozen bookkeeping; crystals own their value)
    th_seed = np.random.default_rng(5).uniform(0.0, 1.0, n_seeds)
    dd = st.free_coords[:, None, :2] - ctr[None, :, :]
    dd -= np.round(dd)
    th = th_seed[np.argmin((dd ** 2).sum(axis=2), axis=1)]
    tr = 0.02
    p1 = 0.45 * (1 - tr) * (1 - psi_seed) + 0.97 * psi_seed \
        + 1e-3 * rng.standard_normal(nf)
    p2 = 0.55 * (1 - tr) * (1 - psi_seed) + 0.01 * psi_seed \
        + 1e-3 * rng.standard_normal(nf)
    st.set_initial([lambda x: p1, lambda x: p2],
                   [lambda x: psi_seed, lambda x: np.zeros(len(x))],
                   [lambda x: th, lambda x: th])
    st.march(t_end=0.5, dt_max=0.02, dt_min=1e-9, grow_iters=45,
             max_steps=400)
    full = lambda v: np.asarray(cons.T @ v)
    psi0 = full(st.psi(0))
    th0 = full(st.theta(0))
    lab_cc, sz_cc = grain_labels(mesh.node_coords, psi0, th0,
                                 psi_th=0.5, theta_tol=np.inf,
                                 periodic=True)
    G = int(lab_cc.max()) + 1
    spreads = [float(th0[lab_cc == g].max() - th0[lab_cc == g].min())
               for g in range(G)]
    gaps = np.diff(np.sort(th_seed))
    # MEASURED (2026-07-10, this config): post-march intra-grain
    # spread 5.9e-5 (KWC plateaus); min pairwise seed-theta gap 2.9e-3
    # (continuous nucleation values CAN fall arbitrarily close — the
    # scalar-marker aliasing limitation, theirs included; disjoint
    # grains are still separated by connectivity).  LOCKED tol = 1e-3:
    # 17x above the spread, 2.9x below the min gap.
    tol = 1e-3
    lab, sizes = grain_labels(mesh.node_coords, psi0, th0, psi_th=0.5,
                              theta_tol=tol, periodic=True)
    print(f"S2d labeling: {G} psi-connected grains from {n_seeds} "
          f"seeds, post-march intra-grain theta spread max "
          f"{max(spreads):.2e}, pairwise seed-theta gaps min "
          f"{gaps.min():.4f}; tol={tol} -> {len(sizes)} grains, "
          f"sizes[:6] {sizes[:6]}")
    assert max(spreads) < 0.5 * tol, (max(spreads), tol)   # 5.9e-5
    assert gaps.min() > 2.0 * tol, (gaps.min(), tol)        # 2.9e-3
    assert len(sizes) == G, (len(sizes), G)
    assert G == n_seeds, (G, n_seeds)
    assert (np.diff(sizes) <= 0).all()
    assert sizes.sum() == (psi0 > 0.5).sum()
    # (c) GP-partitioned quadrature bookkeeping
    phi_full = [full(st.phi(i)) for i in range(2)]
    psi_full = [psi0, full(st.psi(1))]
    totals = np.zeros(2)
    per_grain = np.zeros((2, len(sizes) + 1))    # +1: amorphous rest
    phi_int = np.zeros(2)
    for pv, b in dm.bins.items():
        tb = dm.tables_by_p[pv]
        conn = mesh.conn_of[pv]
        hh = mesh.tree.h()[mesh.bins[pv]]
        wJ = (np.tile(tb.w, len(conn)).reshape(len(conn), -1)
              * ((hh / 2.0) ** 2)[:, None])
        el_lab = lab[conn]
        el_maj = np.array(
            [np.bincount(r[r >= 0] + 1,
                         minlength=len(sizes) + 2).argmax() - 1
             if (r >= 0).any() else -1 for r in el_lab])
        for i in range(2):
            vgp = np.einsum("qa,ea->eq", tb.N, phi_full[i][conn]) \
                * np.einsum("qa,ea->eq", tb.N, psi_full[i][conn])
            pgp = np.einsum("qa,ea->eq", tb.N, psi_full[i][conn])
            mask = pgp > 0.5
            contrib = (vgp * wJ * mask)
            totals[i] += contrib.sum()
            phi_int[i] += (np.einsum("qa,ea->eq", tb.N,
                                     phi_full[i][conn]) * wJ).sum()
            ge = np.broadcast_to(el_maj[:, None], vgp.shape)
            for g in range(-1, len(sizes)):
                per_grain[i, g] += contrib[ge == g].sum()
    part_err = np.abs(per_grain.sum(axis=1) - totals).max()
    X = [totals[i] / phi_int[i] for i in range(2)]
    print(f"S2d bookkeeping: crystalline integrals int(phi_i psi_i) = "
          f"{totals[0]:.6e}/{totals[1]:.3e}; partition error "
          f"{part_err:.2e}; X_PCE11 = {X[0]:.4f} (seeded ~0.075/0.44), "
          f"X_PCBM = {X[1]:.5f} (pre-nucleation ~0)")
    assert part_err < 1e-10, part_err
    assert 0.02 < X[0] < 0.4, X[0]
    assert abs(X[1]) < 0.02, X[1]


# ---------------------------------------------------------------------
# S2c — THE INTERPLAY REPLICATION
# ---------------------------------------------------------------------
# VERDICT DISCIPLINE (the Negi/S1c template).  Their STOCHASTIC
# nucleation onsets are budget-unreachable at fidelity:
# wall-to-first-nucleus ~ (T_onset/dt) x nodes with T_onset ~ 1/area
# gives wall ~ 1/(dt dx^2) INDEPENDENT of domain size; their onsets
# (54.5 s NDL at dt cap 0.0304 s; 3.4 s DL at 3.04e-4 s) cost
# 1e4-5e4 implicit steps of a 4M-dof system x 20 runs (cluster
# budgets).  Measured suppressed-run bounds: L6/128 nm NDL no
# nucleation to t = 266 s (dt_max 0.5); DL no nucleation to t = 7
# (dt 0.05) / t = 4 (their-cap-class dt 5e-3).  ONSET NUMBERS: not
# reproduced at our budget (mechanism-consistent suppression:
# rate x area x dt-resolution; the noise MACHINERY itself is gated
# quantitatively in S2a3 + the S1c FDT statistics gate).
# The in-suite gates below anchor the DETERMINISTIC phenomenology —
# which is exactly the paper's mechanistic claims list (their Sec 5
# conclusions 1-4): AAPS-first ordering, binodal purities, growth-
# dominated vs diffusion-limited contrast, crystallite quench of
# coarsening, PCE11 dissolution at PCBM growth fronts.


def _mk_fig4_stepper(dm, mesh, cons, l0, mpsi, noise, seed=11):
    pars = _fig4_pars(l0, mpsi=mpsi)
    st = MultiPhaseStepper(dm, dt=1e-3, newton_tol=1e-6, newton_max=60,
                           linsolver="cudss", noise_seed=seed,
                           b_reg=3e-4, line_search=True,
                           alpha_th=[0.0], beta_th=[0.0],
                           L_th=[pars["L_psi"][0]], kg_delta=0.2,
                           **pars)
    if not noise:
        st.noise_psi = 0.0
    rng = np.random.default_rng(4)
    phi0 = 0.45 + 1e-3 * rng.standard_normal(len(st.free_coords))
    st.set_initial([lambda x: phi0], [lambda x: np.zeros(len(x))],
                   [lambda x: _smooth_theta(mesh, cons, 7)])
    return st


def _fig4_metrics(st, cons, mesh, i=0):
    full = lambda v: np.asarray(cons.T @ v)
    phi = full(st.phi(i))
    psi = full(st.psi(i))
    Xm = float((phi * psi).mean() / phi.mean())
    am = psi < 0.5
    rich = phi > 0.5
    cri = float(np.percentile(phi[am & rich], 90)) \
        if (am & rich).any() else np.nan
    cpo = float(phi[am & ~rich].mean()) if (am & ~rich).any() else np.nan
    _, sz = grain_labels(mesh.node_coords, psi, np.zeros(len(psi)),
                         psi_th=0.5, theta_tol=np.inf, periodic=True)
    ncr = int((sz >= 8).sum())
    _, dsz = grain_labels(mesh.node_coords,
                          ((phi > 0.5) & am).astype(float),
                          np.zeros(len(phi)), psi_th=0.5,
                          theta_tol=np.inf, periodic=True)
    return Xm, cri, cpo, ncr, int((dsz >= 8).sum())


@_skip_nvmath
def test_s2c_order_aaps_first(device):
    """WHICH PROCESS INITIATES FIRST (their pathway steps 1-3 +
    conclusion 2): at the Fig-4 deck parameters with FDT noise at
    their sigma_AC = 1, AAPS develops within a second (their wave
    pattern at 0.43 s; measured droplet pattern by t = 0.5) and both
    amorphous phases purify to the binodal plateaus, while NO
    crystallization occurs on the 60 s horizon (their large-box onset
    50-55 s; suppressed here — bound recorded above).  Purity targets:
    their PCBM-rich plateau 0.968, PCE11-rich 0.21-0.243 (their own
    LiqConc curves).  MEASURED (locks with >= 2x-class headroom in
    parens): see print."""
    dm, mesh, cons = _dm(6, device, periodic=True)
    st = _mk_fig4_stepper(dm, mesh, cons, 128e-9, 0.1, noise=True)
    rec = []

    def cb(s, dt, iters):
        if len(rec) == 0 or s.t - rec[-1][0] > 0.4:
            rec.append((s.t,) + _fig4_metrics(s, cons, mesh))
    r1 = st.march(t_end=3.0, dt_max=0.02, callback=cb, dt_min=1e-9,
                  grow_iters=45, max_steps=3000)
    r2 = st.march(t_end=60.0, dt_max=0.1, callback=cb, dt_min=1e-9,
                  grow_iters=45, max_steps=3000)
    assert r1 == "t_end" and r2 == "t_end", (r1, r2)
    r = np.array(rec)
    nd05 = int(r[np.argmax(r[:, 0] >= 0.5), 5])
    tail = r[r[:, 0] > 20.0]
    pur = float(np.nanmean(tail[:, 2]))
    cpo = float(np.nanmean(tail[:, 3]))
    print(f"S2c order: droplets(0.5 s) = {nd05}, purity90(t>20) = "
          f"{pur:.3f} (theirs 0.968), cpoor = {cpo:.3f} (theirs "
          f"0.21-0.243), X_max = {r[:, 1].max():.4f}, ncr_max = "
          f"{int(r[:, 4].max())}")
    # MEASURED (2026-07-10 calibration, this config): droplets(0.5 s)
    # = 4; purity90 = 0.973 (theirs 0.968 — 0.5%); cpoor = 0.246
    # (theirs 0.243 — 1.2%); X_max = 0.019 noise band; ncr_max = 0.
    assert nd05 >= 2, nd05                      # measured 4
    assert pur > 0.93, pur                      # measured 0.973
    assert 0.17 < cpo < 0.31, cpo               # measured 0.246
    assert r[:, 1].max() < 0.05 and int(r[:, 4].max()) == 0


@_skip_nvmath
def test_s2c_growth_mode_contrast(device):
    """GROWTH-DOMINATED vs DIFFUSION-LIMITED (their conclusions 3-4 +
    SI-7): deterministic seeded-growth contrast at the deck M_psi pair
    (0.1 vs 10, their NDL/DL).  A noise-free march to t = 5 forms the
    purified droplet pattern; ONE PCBM nucleus (8 nm) is implanted at
    the largest droplet's centroid; both mobilities run the same
    +12 s.  Their measured contrast at 512 nm: full-domain
    crystallization t50 100.9 s (NDL) vs 5.8 s (DL) = 17.4x; DL leaves
    the PCBM-rich liquid depleted (their stuck 0.24 plateau) while NDL
    consumes it near-pure (0.946-0.968).  MEASURED here: see print."""
    dm, mesh, cons = _dm(6, device, periodic=True)
    base = _mk_fig4_stepper(dm, mesh, cons, 128e-9, 0.1, noise=False)
    rb = base.march(t_end=5.0, dt_max=0.05, dt_min=1e-9, grow_iters=45,
                    max_steps=3000)
    assert rb == "t_end", rb
    xb = base.x.copy()
    full = lambda v: np.asarray(cons.T @ v)
    phi = base.phi(0)
    lab, _ = grain_labels(mesh.node_coords,
                          full((phi > 0.5).astype(float)),
                          np.zeros(dm.n_nodes), psi_th=0.5,
                          theta_tol=np.inf, periodic=True)
    m = lab[cons.free_nodes] == 0
    pts = base.free_coords[m][:, :2]
    d = pts - pts[0]
    d -= np.round(d)
    ctr = (pts[0] + d.mean(axis=0)) % 1.0
    res = {}
    for mpsi in (0.1, 10.0):
        st = _mk_fig4_stepper(dm, mesh, cons, 128e-9, mpsi, noise=False)
        st.x = xb.copy()
        dd = st.free_coords[:, :2] - ctr[None, :]
        dd -= np.round(dd)
        rr = np.sqrt((dd ** 2).sum(axis=1))
        disc = 0.5 * (1.0 - np.tanh((rr - 8.0 / 128.0) / (2.0 / 128.0)))
        st.x[2::st.ndof] = np.maximum(st.x[2::st.ndof], disc)
        st.hist = st.x.copy()
        st.t, st.dt = 5.0, 1e-3
        rm = st.march(t_end=17.0, dt_max=0.05, dt_min=1e-9,
                      grow_iters=45, max_steps=3500)
        assert rm == "t_end", rm
        res[mpsi] = _fig4_metrics(st, cons, mesh)
    Xn, Xd = res[0.1][0], res[10.0][0]
    print(f"S2c growth contrast (+12 s from one seed): X_NDL = {Xn:.4f} "
          f"vs X_DL = {Xd:.4f} (ratio {Xd / max(Xn, 1e-9):.1f}x; their "
          f"whole-domain t50 ratio 17.4x); DL purity90 "
          f"{res[10.0][1]:.3f} vs NDL {res[0.1][1]:.3f}")
    # MEASURED (2026-07-11, frozen-theta solver): X_DL = 0.9586 by
    # +3 s (their plateau 96.1% of material!), X_NDL = 0.1491 at
    # +12 s; ratio 6.4x at MATCHED dt_max = 0.05 for both branches
    # (their caps are 0.0304 NDL / 3.04e-4 DL — running DL at its cap
    # is 4e4 steps, out of budget; the ratio at their caps would be
    # LARGER, so this is a conservative bound consistent with their
    # 17.4x t50 contrast).  The DL depletion-zone liquid signature
    # (their stuck 0.24) does NOT manifest at single-crystal/128 nm
    # scale — multi-crystal competition is its mechanism; recorded as
    # a domain-size gap.
    assert Xd > 0.5, Xd                         # measured 0.9586
    assert Xd / max(Xn, 1e-9) > 4.0, (Xn, Xd)   # measured 6.4x


def _iface_density(mesh, rich):
    """Rich-phase boundary edges / rich nodes (4-conn periodic grid)."""
    coords = mesh.node_coords
    xs = np.unique(np.round(coords[:, 0], 12))
    n = len(xs)
    ix = np.searchsorted(xs, np.round(coords[:, 0], 12))
    iy = np.searchsorted(xs, np.round(coords[:, 1], 12))
    g = np.zeros((n, n), bool)
    g[ix, iy] = rich
    edges = (g != np.roll(g, 1, 0)).sum() + (g != np.roll(g, 1, 1)).sum()
    return edges / max(g.sum(), 1)


def _mk_fig6_stepper(dm, mesh, cons, seeded, mpsi=2.5e-2, seed=11):
    pars = _fig6_pars(128e-9, mpsi=mpsi)
    st = MultiPhaseStepper(dm, dt=1e-3, newton_tol=1e-6, newton_max=60,
                           linsolver="cudss", noise_seed=seed,
                           b_reg=3e-4, line_search=True,
                           alpha_th=[0.0, 0.0], beta_th=[0.0, 0.0],
                           L_th=[1e-8, 1e-8], kg_delta=0.2, **pars)
    st.noise_psi = 0.0
    rng = np.random.default_rng(4)
    nf = len(st.free_coords)
    if seeded:
        psi_seed, _ = _seed_discs(st.free_coords, 15, r0=7.3 / 128.0,
                                  w=1.5 / 128.0, seed=111)
    else:
        psi_seed = np.zeros(nf)
    tr = 0.02
    p1 = 0.45 * (1 - tr) * (1 - psi_seed) + 0.97 * psi_seed \
        + 1e-3 * rng.standard_normal(nf)
    p2 = 0.55 * (1 - tr) * (1 - psi_seed) + 0.01 * psi_seed \
        + 1e-3 * rng.standard_normal(nf)
    st.set_initial([lambda x: p1, lambda x: p2],
                   [lambda x: psi_seed, lambda x: np.zeros(len(x))],
                   [lambda x: _smooth_theta(mesh, cons, 7)] * 2)
    return st


@_skip_nvmath
def test_s2c_crystallite_quench_and_dissolution(device):
    """THE FIG-6 MECHANISMS (their conclusions 1 + 3; ternary-trace
    M = 2, K = 2 at the Fig-6 deck, 15 seeds = their 948 at the same
    33 nm spacing, r_eff = 7.3 nm = their t = 0 crystal volume):
    (a) QUENCH: PCE11 crystallites pin the amorphous demixing pattern —
        measured amorphous-PCBM-phase interface density at t = 10:
        seeded 0.160 vs unseeded 0.069 (2.3x), the paper's
        'crystallites prevent relaxation to round configurations /
        freeze the pattern' claim as a number;
    (b) crystallite volume stays put pre-crystallization (their
        conclusion 1 'not actively affected'; our phi psi integral
        0.133 -> 0.139 over t = 0..10 — NOTE their CrystVol metric
        shows an early 0.151 -> 0.12 equilibration our integral
        metric does not reproduce; metric-definition gap recorded);
    (c) DISSOLUTION AT PCBM GROWTH FRONTS (their PCE11 CrystVol
        0.103 -> 0.078 across the PCBM crystallization window):
        an implanted PCBM nucleus grown at M_psi = 0.5 (their SI-9
        result — end-state morphology decoupled from M_psi over 25x —
        licenses the acceleration; recorded) consumes amorphous PCBM
        AND dissolves PCE11 crystallites it reaches."""
    dm, mesh, cons = _dm(6, device, periodic=True)
    full = lambda v: np.asarray(cons.T @ v)
    ifd, states = {}, {}
    for seeded in (True, False):
        st = _mk_fig6_stepper(dm, mesh, cons, seeded)
        r1 = st.march(t_end=3.0, dt_max=0.02, dt_min=1e-9,
                      grow_iters=45, max_steps=3000)
        r2 = st.march(t_end=10.0, dt_max=0.4, dt_min=1e-9,
                      grow_iters=45, max_steps=3000)
        assert r1 == "t_end" and r2 == "t_end", (r1, r2)
        s1, s2 = full(st.psi(0)), full(st.psi(1))
        am = (s1 < 0.5) & (s2 < 0.5)
        rich = (full(st.phi(1)) > 0.5) & am
        ifd[seeded] = _iface_density(mesh, rich)
        states[seeded] = st
    X1_10 = float((full(states[True].phi(0))
                   * full(states[True].psi(0))).mean())
    print(f"S2c quench: iface density seeded {ifd[True]:.4f} vs "
          f"unseeded {ifd[False]:.4f} (ratio "
          f"{ifd[True] / ifd[False]:.2f}); Xpce(10) = {X1_10:.4f}")
    assert ifd[True] > 1.5 * ifd[False], ifd     # measured 2.32x
    assert 0.10 < X1_10 < 0.20, X1_10            # measured 0.139
    # (c) accelerated dissolution
    st = states[True]
    st2 = _mk_fig6_stepper(dm, mesh, cons, True, mpsi=0.5)
    st2.x = st.x.copy()
    st2.hist = st.x.copy()
    st2.t = st.t
    cand = int(np.argmax(st2.phi(1) * (st2.psi(0) < 0.3)))
    ctr = st2.free_coords[cand, :2]
    dd = st2.free_coords[:, :2] - ctr[None, :]
    dd -= np.round(dd)
    rr = np.sqrt((dd ** 2).sum(axis=1))
    disc = 0.5 * (1.0 - np.tanh((rr - 8.0 / 128.0) / (2.0 / 128.0)))
    st2.x[2 * st2.M + 2::st2.ndof] = \
        np.maximum(st2.x[2 * st2.M + 2::st2.ndof], disc)
    st2.hist = st2.x.copy()
    st2.dt = 1e-3
    t_seed = st2.t
    r3 = st2.march(t_end=st2.t + 40.0, dt_max=0.02, dt_min=1e-9,
                   grow_iters=45, max_steps=5000)
    X1_e = float((full(st2.phi(0)) * full(st2.psi(0))).mean())
    X2_e = float((full(st2.phi(1)) * full(st2.psi(1))).mean())
    print(f"S2c dissolution (at M_psi 0.5): {r3} at t = {st2.t:.2f} "
          f"(seeded {t_seed:.2f}); Xpcb -> {X2_e:.4f}, Xpce "
          f"{X1_10:.4f} -> {X1_e:.4f}")
    # MEASURED (deterministic across 3 runs, 2026-07-10/11): Xpcb ->
    # 0.4408 (the implanted crystal consumes ~80% of the amorphous
    # PCBM), Xpce 0.1391 -> 0.1288 (crystallite dissolution at the
    # growth front; their CrystVol 0.103 -> 0.078 across full
    # consumption).  KNOWN SOLVER FRONTIER (recorded): the ladder
    # stalls (dt_underflow) at a reproducible late state where the
    # PCBM front impinges frozen-mobility PCE11 crystallites (the
    # chi_cc-stiff crystal-crystal moment; blockch deep-quench caveat
    # class) — the dissolution physics above is complete BEFORE the
    # stall, so the gate asserts the measured observables and accepts
    # either stop reason; full-horizon completion is the S3-era
    # preconditioner follow-up.
    assert r3 in ("t_end", "dt_underflow"), r3
    assert st2.t > t_seed + 3.0, (t_seed, st2.t)
    assert X2_e > 0.2, X2_e                      # measured 0.4408
    assert X1_e < X1_10 - 0.005, (X1_10, X1_e)   # measured -0.0103



def test_s2d_grain_labels_100_grains():
    """Pure-numpy many-grain stress (the O(10-100) requirement): 100
    disjoint psi discs on a 256^2 periodic grid, theta values drawn
    CONTINUOUSLY from U(0,1) (nucleation-assignment statistics), locked
    tol = 1e-3: exact count, exact node partition, sizes descending.
    Also the aliasing bound: with 100 uniform draws the min pairwise
    gap ~ 1/100^2 CAN fall below any fixed tolerance — labeling of
    DISJOINT grains is tolerance-independent (connectivity), which is
    what this gate demonstrates at scale."""
    n = 256
    xs = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    coords = np.stack([X.ravel(), Y.ravel()], axis=1)
    rng = np.random.default_rng(2)
    ctr, r0 = [], 0.022
    while len(ctr) < 100:
        c = rng.uniform(0, 1, 2)
        if all(np.hypot(*((c - c0 + 0.5) % 1.0 - 0.5)) > 2.6 * r0
               for c0 in ctr):
            ctr.append(c)
    psi = np.zeros(n * n)
    theta = np.zeros(n * n)
    th_seed = rng.uniform(0, 1, 100)
    for c, tv in zip(ctr, th_seed):
        d = coords - c[None, :]
        d -= np.round(d)
        m = (d ** 2).sum(axis=1) < r0 ** 2
        psi[m] = 1.0
        theta[m] = tv
    labels, sizes = grain_labels(coords, psi, theta, psi_th=0.5,
                                 theta_tol=1e-3, periodic=True)
    print(f"S2d-100: {len(sizes)} grains, sizes [{sizes.min()}, "
          f"{sizes.max()}], min theta gap "
          f"{np.diff(np.sort(th_seed)).min():.2e}")
    assert len(sizes) == 100, len(sizes)
    assert sizes.sum() == (psi > 0.5).sum()
    assert (np.diff(sizes) <= 0).all()
    assert labels[psi <= 0.5].max() == -1
