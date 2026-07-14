"""Solution-adaptive AMR march for Cahn--Hilliard (C3 Phase-3, stage A2).

The full dynamic-AMR cycle around the production :class:`CahnHilliardStepper`:

    estimate (|grad c| interface indicator)  ->  mark (refine / coarsen)
    ->  refine + coarsen  ->  2:1 balance  ->  rebuild mesh + constraints
    ->  conservatively transfer c, mu AND both BDF history levels
    ->  continue the march.

Refinement tracks the moving interface; coarsening reclaims the smoothed bulk.
Mass is conserved to solver tolerance across every remesh (the transfer core,
:mod:`diffsim.adaptivity.remesh`), and the free energy changes across a remesh
only by the (small, measured) transfer projection error -- never a jump.

Correctness, not speed: everything host-side except the CH assembly/solve.
"""
from __future__ import annotations

import time
import numpy as np

from ..octree.build import refine_elements, coarsen_elements
from ..octree.balance import balance2to1
from ..mesh.nodes import build_mesh
from ..mesh.constraints import build_constraints
from ..mesh.basis import basis_tables
from ..assembly.operators import DeviceMesh
from ..physics.cahn_hilliard import CahnHilliardStepper, np_rlog
from .remesh import conservative_transfer, field_mass


# --------------------------------------------------------------------------
# stepper construction on an arbitrary octree
# --------------------------------------------------------------------------

def build_ch_stepper(tree, M, kappa, dt, order, energy, device,
                     linsolver="splu", newton_tol=1e-10):
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=tree.dim), device)
    st = CahnHilliardStepper(dm, M, kappa, dt, order=order, energy=energy,
                             linsolver=linsolver, newton_tol=newton_tol)
    return st, mesh, cons


# --------------------------------------------------------------------------
# error/interface indicator + marking
# --------------------------------------------------------------------------

def interface_indicator(stepper, mesh):
    """Per-leaf interface indicator: the max |grad c| over the element's
    quadrature points (large on the diffuse interface, ~0 in the flat bulk)."""
    _, g = stepper._gp_scalar(stepper.x[0::2])
    dim = mesh.dim
    ind = np.zeros(len(mesh.tree))
    for pv, eids in mesh.bins.items():
        gp = g[pv].reshape(len(eids), -1, dim)          # [ne, nqp, dim]
        mag = np.sqrt((gp ** 2).sum(2)).max(1)          # [ne]
        ind[np.asarray(eids)] = mag
    return ind


def mark(tree, ind, refine_thr, coarse_thr, max_level, min_level):
    lev = tree.levels.astype(np.int64)
    refine_mask = (ind > refine_thr) & (lev < max_level)
    coarse_mask = (ind < coarse_thr) & (lev > min_level)
    return refine_mask, coarse_mask


def _leaf_ids(tree):
    return tree.keys.astype(np.uint64), tree.levels.astype(np.uint8)


def adapt_tree(tree, refine_mask, coarse_mask):
    """One refine+coarsen+balance pass. Refine wins over coarsen where both are
    marked (they are disjoint by construction: the indicator is either high or
    low). 2:1 balance may re-refine a just-coarsened cell near the interface --
    that only helps correctness."""
    # identify coarsen candidates by (key, level) so we can re-find them after
    # refinement re-sorts the leaf array
    cand = set(zip(tree.keys[coarse_mask].tolist(),
                   tree.levels[coarse_mask].tolist()))
    ref = refine_elements(tree, refine_mask & ~coarse_mask)
    if cand:
        cmask = np.array([(int(k), int(l)) in cand
                          for k, l in zip(ref.keys.tolist(),
                                          ref.levels.tolist())], bool)
        ref = coarsen_elements(ref, cmask)
    return balance2to1(ref)


# --------------------------------------------------------------------------
# free energy (for the continuity gate)
# --------------------------------------------------------------------------

def free_energy(mesh, cons, tables, c_free, kappa, energy, fh_A=1.0, fh_B=3.0):
    """F = INT [ f(c) + (kappa/2)|grad c|^2 ] dV by quadrature."""
    if not isinstance(tables, dict):
        tables = {mesh.p: tables}
    full = np.asarray(cons.T @ np.asarray(c_free))
    dim = mesh.dim
    h_all = mesh.tree.h()
    F = 0.0
    for pv, eids in mesh.bins.items():
        tb = tables[pv]
        conn = mesh.conn_of[pv]
        vals = full[conn]                                # [ne, nbf]
        c_gp = np.einsum("qa,ea->eq", tb.N, vals)        # [ne, nqp]
        h = h_all[eids]
        jac = (h / 2.0) ** dim
        wq = tb.w[None, :] * jac[:, None]                # [ne, nqp]
        if energy == "fh":
            f = fh_A * (c_gp * np_rlog(c_gp) + (1 - c_gp) * np_rlog(1 - c_gp)) \
                + fh_B * c_gp * (1 - c_gp)
        else:
            f = 0.25 * (c_gp ** 2 - 1.0) ** 2
        grad2 = np.zeros_like(c_gp)
        for d in range(dim):
            gd = (np.einsum("qa,ea->eq", tb.dN[:, :, d], vals)
                  * (2.0 / h)[:, None])                   # [ne, nqp]
            grad2 += gd ** 2
        F += float(((f + 0.5 * kappa * grad2) * wq).sum())
    return F


# --------------------------------------------------------------------------
# the remesh (state carried across)
# --------------------------------------------------------------------------

def remesh(stepper, mesh, cons, new_tree, M, kappa, energy, device,
           linsolver="splu", newton_tol=1e-10):
    """Rebuild the stepper on ``new_tree`` and conservatively carry the full
    state (c, mu, both BDF history levels, t, dt, dt_prev)."""
    tb = basis_tables(1, dim=mesh.dim)
    new_st, new_mesh, new_cons = build_ch_stepper(
        new_tree, M, kappa, stepper.dt, stepper.order, energy, device,
        linsolver=linsolver, newton_tol=newton_tol)
    c_old = stepper.x[0::2]
    mu_old = stepper.x[1::2]
    h0, h1 = stepper.hist[0], stepper.hist[1]
    c_n, mu_n, h0_n, h1_n = conservative_transfer(
        mesh, cons, tb, new_mesh, new_cons, tb, [c_old, mu_old, h0, h1])
    new_st.x = np.zeros(new_st.nfree * 2)
    new_st.x[0::2] = c_n
    new_st.x[1::2] = mu_n
    new_st.hist = [h0_n, h1_n]
    new_st.t = stepper.t
    new_st.dt = stepper.dt
    new_st.dt_prev = stepper.dt_prev
    return new_st, new_mesh, new_cons


# --------------------------------------------------------------------------
# driven cycle
# --------------------------------------------------------------------------

def amr_march(tree0, c0_fn, M, kappa, dt, t_end, device,
              order=1, energy="poly", remesh_every=5,
              refine_frac=0.15, coarse_frac=0.03,
              max_level=6, min_level=2, linsolver="splu",
              record=True, verbose=False):
    """Run the solution-adaptive AMR march. Returns a dict of trajectories and
    per-remesh diagnostics (mass drift, energy jump vs transfer error, refined-
    region/interface overlap, dof count) plus the final state + mesh."""
    tb = basis_tables(1, dim=tree0.dim)
    st, mesh, cons = build_ch_stepper(tree0, M, kappa, dt, order, energy,
                                      device, linsolver=linsolver)
    st.set_initial(c0_fn, mu_init="consistent")

    # Initial adaptation: resolve the interface of the INITIAL CONDITION before
    # the march (refine-only, re-seeding the exact analytic IC each round) so
    # step 0 sees a fully resolved interface. Without this, the first few steps
    # run on an under-resolved interface and inject error a uniform-fine run
    # never pays -- the AMR advantage is lost. Standard AMR practice.
    for _ in range(max_level + 1):
        ind = interface_indicator(st, mesh)
        mx = float(ind.max()) if ind.max() > 0 else 1.0
        refine_mask, _ = mark(mesh.tree, ind, refine_frac * mx, -1.0,
                              max_level, min_level)
        if not refine_mask.any():
            break
        new_tree = adapt_tree(mesh.tree, refine_mask,
                              np.zeros(len(mesh.tree), bool))
        st, mesh, cons = build_ch_stepper(new_tree, M, kappa, dt, order,
                                          energy, device, linsolver=linsolver)
        st.set_initial(c0_fn, mu_init="consistent")

    rec = dict(t=[], dofs=[], mass=[], energy=[],
               remesh_t=[], remesh_mass_before=[], remesh_mass_after=[],
               remesh_E_before=[], remesh_E_after=[], overlap=[], nleaf=[])

    def snap():
        rec["t"].append(st.t)
        rec["dofs"].append(st.nfree)
        rec["mass"].append(field_mass(mesh, cons, tb, st.x[0::2]))
        rec["energy"].append(free_energy(mesh, cons, tb, st.x[0::2],
                                          kappa, energy))
        rec["nleaf"].append(len(mesh.tree))

    if record:
        snap()
    step_i = 0
    t0 = time.perf_counter()
    while st.t < t_end - 1e-12:
        st.step()
        step_i += 1
        if step_i % remesh_every == 0 and st.t < t_end - 1e-12:
            ind = interface_indicator(st, mesh)
            mx = float(ind.max()) if ind.max() > 0 else 1.0
            r_thr, c_thr = refine_frac * mx, coarse_frac * mx
            refine_mask, coarse_mask = mark(mesh.tree, ind, r_thr, c_thr,
                                            max_level, min_level)
            new_tree = adapt_tree(mesh.tree, refine_mask, coarse_mask)
            m_before = field_mass(mesh, cons, tb, st.x[0::2])
            E_before = free_energy(mesh, cons, tb, st.x[0::2], kappa, energy)
            st, mesh, cons = remesh(st, mesh, cons, new_tree, M, kappa,
                                    energy, device, linsolver=linsolver)
            m_after = field_mass(mesh, cons, tb, st.x[0::2])
            E_after = free_energy(mesh, cons, tb, st.x[0::2], kappa, energy)
            # refined-region / interface overlap: fraction of the finest cells
            # that sit on the interface band (high indicator)
            ind2 = interface_indicator(st, mesh)
            fine = mesh.tree.levels >= mesh.tree.levels.max()
            band = ind2 > c_thr
            ov = float((fine & band).sum() / max(fine.sum(), 1))
            rec["remesh_t"].append(st.t)
            rec["remesh_mass_before"].append(m_before)
            rec["remesh_mass_after"].append(m_after)
            rec["remesh_E_before"].append(E_before)
            rec["remesh_E_after"].append(E_after)
            rec["overlap"].append(ov)
            if verbose:
                print(f"  remesh t={st.t:.4f} dofs={st.nfree} "
                      f"dmass={m_after - m_before:+.2e} "
                      f"dE={E_after - E_before:+.2e} overlap={ov:.2f}",
                      flush=True)
        if record:
            snap()
    wall = time.perf_counter() - t0
    return dict(rec=rec, stepper=st, mesh=mesh, cons=cons, wall=wall,
                final_c=st.x[0::2].copy())
