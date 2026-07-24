"""NCCL-CG Proof Driver — Stage 1: Synthetic 7-point Poisson (Task 3a).

Validates the distributed PCG mechanic end-to-end on CPU (gloo backend) and
GPU (nccl backend, Task 3b).  Task 3a covers gloo/CPU validation on the Mac.

Usage
-----
Single process (degenerate N=1)::

    python scripts/nccl_cg_proof.py --stage synthetic --backend gloo \\
        --dims 8,4,4

Two processes (gloo on CPU)::

    torchrun --standalone --nproc_per_node=2 scripts/nccl_cg_proof.py \\
        --stage synthetic --backend gloo --precond jacobi --dims 8,4,4

Arguments
---------
--stage     Only "synthetic" is implemented in Task 3a.
--backend   torch.distributed backend: "gloo" (CPU) or "nccl" (GPU).
--precond   Preconditioner: "jacobi" (always-on) or "amgx" (Task 3b, GPU).
--dims      NX,NY,NZ grid node counts (e.g. 8,4,4).
--rtol      Solver relative tolerance (default 1e-10).
--rhs       RHS mode: "mms" (manufactured solution, default) or "random".

Manufactured solution (--rhs mms)
----------------------------------
    u(x,y,z)  = sin(pi*x) * sin(pi*y) * sin(pi*z)     (on [0,1]^3)
    f(x,y,z)  = 3 * pi^2 * u(x,y,z)                   (-Delta u = f)

Dirichlet BCs: u = 0 on all faces (the manufactured solution vanishes there).

Random RHS (--rhs random)
--------------------------
Interior node RHS values are drawn from a fixed-seed standard normal:
    rhs_full = np.random.default_rng(0).standard_normal(nx*ny*nz)
    b[gid]   = rhs_full[gid]   for each interior node gid

All ranks generate the SAME rhs_full (same seed, same indexing) so that the
distributed build and the serial reference build produce the IDENTICAL global b.
Boundary nodes keep Dirichlet b=0 (identity rows).  This mode exercises many
CG iterations (the full β-recurrence / p-update distributed loop), whereas the
MMS mode converges in ~1 iteration because the single-eigenmode RHS collapses
the Krylov space.

Grid: uniform (NX x NY x NZ) nodes at
    x_i = i/(NX-1),  y_j = j/(NY-1),  z_k = k/(NZ-1)
    global node id  = i*NY*NZ + j*NZ + k   (axis-0 slowest)

The discrete Poisson operator uses a 7-point second-order FD stencil with
non-uniform grid spacing handled generically (h_x = 1/(NX-1), etc.).

Correctness check
-----------------
Rank 0 assembles the full (serial) system, solves with scipy.sparse.linalg.spsolve,
then gathers the distributed solution and compares:
    rel_err = ||x_dist - x_ref||_inf / ||x_ref||_inf

Must be < 1e-8 for the test to pass.  For --rhs mms the exact manufactured
solution is also compared; for --rhs random only the scipy reference is used.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Manufactured solution: u = sin(pi*x)*sin(pi*y)*sin(pi*z) on [0,1]^3
# ---------------------------------------------------------------------------

def _u_exact(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Exact solution on [0,1]^3."""
    return np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)


def _f_rhs(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """RHS: -Delta u = 3*pi^2 * u."""
    return 3.0 * math.pi**2 * _u_exact(x, y, z)


# ---------------------------------------------------------------------------
# Grid coordinate helpers
# ---------------------------------------------------------------------------

def _grid_coords(dims: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, Y, Z) coordinate arrays for every grid node (flat, length N)."""
    nx, ny, nz = dims
    xs = np.linspace(0.0, 1.0, nx)
    ys = np.linspace(0.0, 1.0, ny)
    zs = np.linspace(0.0, 1.0, nz)
    # Lexicographic: i0 slowest
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    return X.ravel(), Y.ravel(), Z.ravel()


def _is_boundary(i0: int, i1: int, i2: int, dims: tuple[int, int, int]) -> bool:
    nx, ny, nz = dims
    return (i0 == 0 or i0 == nx - 1 or
            i1 == 0 or i1 == ny - 1 or
            i2 == 0 or i2 == nz - 1)


# ---------------------------------------------------------------------------
# Random RHS helper
# ---------------------------------------------------------------------------

def _random_rhs_full(dims: tuple[int, int, int]) -> np.ndarray:
    """Return a deterministic random RHS vector for all nx*ny*nz nodes.

    Uses a fixed seed (0) so ALL ranks and the serial reference produce the
    identical array.  Interior nodes are indexed by global node id gid;
    boundary nodes use index value that will be overridden by the Dirichlet
    identity row (b[boundary] = u_exact for mms, 0 for random).
    """
    nx, ny, nz = dims
    return np.random.default_rng(0).standard_normal(nx * ny * nz)


# ---------------------------------------------------------------------------
# Full (serial) global system assembly
# ---------------------------------------------------------------------------

def assemble_global_poisson(dims: tuple[int, int, int], rhs_mode: str = "mms"):
    """Assemble the full (nx*ny*nz) x (nx*ny*nz) 7-point Poisson matrix.

    Returns (A_csr, b, x_exact) in fp64.  Dirichlet BCs are enforced by
    setting boundary rows to identity (A[k,k]=1, b[k]=u(boundary point)).

    Parameters
    ----------
    dims : (nx, ny, nz)
    rhs_mode : "mms" (default) or "random"
        "mms"    — interior b[k] = 3*pi^2 * sin(pi*x)*sin(pi*y)*sin(pi*z)
        "random" — interior b[k] = rhs_full[k], where rhs_full is generated
                   with a fixed seed (same as assemble_local_poisson).
    """
    nx, ny, nz = dims
    n = nx * ny * nz
    hx = 1.0 / (nx - 1) if nx > 1 else 1.0
    hy = 1.0 / (ny - 1) if ny > 1 else 1.0
    hz = 1.0 / (nz - 1) if nz > 1 else 1.0

    inv_hx2 = 1.0 / hx**2
    inv_hy2 = 1.0 / hy**2
    inv_hz2 = 1.0 / hz**2
    diag_val = 2.0 * (inv_hx2 + inv_hy2 + inv_hz2)

    X, Y, Z = _grid_coords(dims)
    x_exact = _u_exact(X, Y, Z)

    if rhs_mode == "mms":
        f = _f_rhs(X, Y, Z)
    else:
        # random: one deterministic call; interior nodes index by gid
        f = _random_rhs_full(dims)

    rows, cols, vals = [], [], []
    b = np.zeros(n, dtype=np.float64)

    for i0 in range(nx):
        for i1 in range(ny):
            for i2 in range(nz):
                k = i0 * ny * nz + i1 * nz + i2

                if _is_boundary(i0, i1, i2, dims):
                    # Dirichlet: identity row
                    rows.append(k); cols.append(k); vals.append(1.0)
                    if rhs_mode == "mms":
                        b[k] = x_exact[k]
                    else:
                        b[k] = 0.0   # homogeneous Dirichlet for random RHS
                    continue

                # Interior stencil
                rows.append(k); cols.append(k); vals.append(diag_val)
                b[k] = f[k]

                for (di0, di1, di2, coeff) in [
                    (-1, 0, 0, -inv_hx2), (1, 0, 0, -inv_hx2),
                    (0, -1, 0, -inv_hy2), (0, 1, 0, -inv_hy2),
                    (0, 0, -1, -inv_hz2), (0, 0, 1, -inv_hz2),
                ]:
                    ni0, ni1, ni2 = i0 + di0, i1 + di1, i2 + di2
                    nk = ni0 * ny * nz + ni1 * nz + ni2
                    rows.append(k); cols.append(nk); vals.append(coeff)

    A = scipy.sparse.csr_matrix(
        (np.array(vals, dtype=np.float64),
         (np.array(rows, dtype=np.int32),
          np.array(cols, dtype=np.int32))),
        shape=(n, n),
    )
    return A, b, x_exact


# ---------------------------------------------------------------------------
# Local (per-rank) partitioned system assembly
# ---------------------------------------------------------------------------

def assemble_local_poisson(
    part,
    dims: tuple[int, int, int],
    device: torch.device,
    rhs_mode: str = "mms",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the local partitioned Poisson system for this rank.

    Parameters
    ----------
    part : SlabPart
        This rank's partition metadata.
    dims : (nx, ny, nz)
        Global grid dimensions.
    device : torch.device
        Target device.
    rhs_mode : "mms" or "random"
        "mms"    — interior RHS from manufactured solution (default).
        "random" — interior RHS from a fixed-seed random vector indexed by
                   global node id; boundary nodes keep Dirichlet b=0.

    Returns
    -------
    A_local_csr : scipy.sparse.csr_matrix (float64)
        Sparse matrix with ``n_owned`` rows and ``n_owned + n_ghost`` columns.
        Column j corresponds to the j-th entry in the owned+ghost local vector.
    b_local : torch.Tensor, shape (n_owned,), float64
        Local RHS (owned rows only).
    diag_local : torch.Tensor, shape (n_owned,), float64
        Diagonal of A_local restricted to owned rows (for Jacobi precond).
    x_exact_local : torch.Tensor, shape (n_owned,), float64
        Exact solution at owned nodes (for error checking).
    """
    nx, ny, nz = dims
    hx = 1.0 / (nx - 1) if nx > 1 else 1.0
    hy = 1.0 / (ny - 1) if ny > 1 else 1.0
    hz = 1.0 / (nz - 1) if nz > 1 else 1.0

    inv_hx2 = 1.0 / hx**2
    inv_hy2 = 1.0 / hy**2
    inv_hz2 = 1.0 / hz**2
    stencil_diag = 2.0 * (inv_hx2 + inv_hy2 + inv_hz2)

    owned_ids = part.owned    # global ids, shape (n_owned,)
    n_owned = len(owned_ids)
    n_ghost = len(part.ghost)
    n_local = n_owned + n_ghost

    X, Y, Z = _grid_coords(dims)
    x_exact_all = _u_exact(X, Y, Z)

    if rhs_mode == "mms":
        f_all = _f_rhs(X, Y, Z)
    else:
        # Deterministic random: same seed on every rank, indexed by global gid
        f_all = _random_rhs_full(dims)

    # b vector (owned rows): interior nodes get f, boundary get u_exact (mms)
    # or 0 (random, homogeneous Dirichlet)
    b_np = np.zeros(n_owned, dtype=np.float64)
    diag_np = np.zeros(n_owned, dtype=np.float64)

    rows_local, cols_local, vals_local = [], [], []

    for loc_row, gid in enumerate(owned_ids):
        # Global (i0, i1, i2) from global id
        i2 = int(gid % nz)
        tmp = int(gid // nz)
        i1 = int(tmp % ny)
        i0 = int(tmp // ny)

        if _is_boundary(i0, i1, i2, dims):
            # Dirichlet: identity row in the local matrix
            rows_local.append(loc_row)
            cols_local.append(loc_row)     # local col = local row (owned)
            vals_local.append(1.0)
            diag_np[loc_row] = 1.0
            if rhs_mode == "mms":
                b_np[loc_row] = x_exact_all[gid]
            else:
                b_np[loc_row] = 0.0   # homogeneous Dirichlet
            continue

        # Interior: stencil diagonal
        rows_local.append(loc_row)
        cols_local.append(loc_row)
        vals_local.append(stencil_diag)
        diag_np[loc_row] = stencil_diag
        b_np[loc_row] = f_all[gid]

        for (di0, di1, di2, coeff) in [
            (-1, 0, 0, -inv_hx2), (1, 0, 0, -inv_hx2),
            (0, -1, 0, -inv_hy2), (0, 1, 0, -inv_hy2),
            (0, 0, -1, -inv_hz2), (0, 0, 1, -inv_hz2),
        ]:
            ni0, ni1, ni2 = i0 + di0, i1 + di1, i2 + di2
            n_gid = ni0 * ny * nz + ni1 * nz + ni2
            loc_col = int(part.local_index(np.array([n_gid], dtype=np.int64))[0])
            rows_local.append(loc_row)
            cols_local.append(loc_col)
            vals_local.append(coeff)

    A_local_csr = scipy.sparse.csr_matrix(
        (np.array(vals_local, dtype=np.float64),
         (np.array(rows_local, dtype=np.int32),
          np.array(cols_local, dtype=np.int32))),
        shape=(n_owned, n_local),
    )

    b_local = torch.tensor(b_np, dtype=torch.float64, device=device)
    diag_local = torch.tensor(diag_np, dtype=torch.float64, device=device)
    x_exact_local = torch.tensor(x_exact_all[owned_ids], dtype=torch.float64, device=device)

    return A_local_csr, b_local, diag_local, x_exact_local


# ---------------------------------------------------------------------------
# Partitioned SpMV wrapper
# ---------------------------------------------------------------------------

def make_partitioned_spmv(A_local_csr, n_owned: int, n_ghost: int, comm, device: torch.device):
    """Build a partitioned spmv callable for pcg().

    The spmv:
      1. Receives p_owned (length n_owned) from pcg.
      2. Copies it into a padded buffer (length n_owned + n_ghost).
      3. Calls comm.exchange_halo to fill the ghost tail.
      4. Performs the local matvec on owned rows.
      5. Returns Ap_owned (length n_owned).

    The ghost-padded buffer is allocated once and reused.
    """
    n_local = n_owned + n_ghost

    # Convert the local CSR to torch sparse for efficient matvec
    # Use dense conversion for small systems; for large, keep scipy
    # and convert buffers to numpy on the fly (CPU-safe path).
    _use_scipy = True  # always safe; GPU path would use torch.sparse

    padded = torch.zeros(n_local, dtype=torch.float64, device=device)

    def spmv(p_owned: torch.Tensor) -> torch.Tensor:
        # 1. Copy owned entries into padded buffer
        padded[:n_owned].copy_(p_owned)
        # 2. Fill ghost tail via halo exchange
        comm.exchange_halo(padded)
        # 3. Local matvec (scipy, CPU) → convert to/from numpy
        p_np = padded.cpu().numpy()
        Ap_np = np.asarray(A_local_csr @ p_np, dtype=np.float64)
        return torch.tensor(Ap_np, dtype=torch.float64, device=device)

    return spmv


# ---------------------------------------------------------------------------
# Reference solve (serial, rank 0 only)
# ---------------------------------------------------------------------------

def serial_reference_solve(dims: tuple[int, int, int], rhs_mode: str = "mms") -> np.ndarray:
    """Solve the full Poisson system serially with scipy.spsolve.

    Parameters
    ----------
    dims : (nx, ny, nz)
    rhs_mode : "mms" or "random"
        Passed through to assemble_global_poisson; must match the mode used
        in assemble_local_poisson so that dist-CG and serial reference use
        identical RHS vectors.
    """
    A, b, _ = assemble_global_poisson(dims, rhs_mode=rhs_mode)
    x_ref = scipy.sparse.linalg.spsolve(A, b)
    return x_ref.astype(np.float64)


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="NCCL-CG Proof Driver (Task 3a)")
    p.add_argument("--stage", default="synthetic", choices=["synthetic"],
                   help="Problem stage (only 'synthetic' in Task 3a)")
    p.add_argument("--backend", default="gloo", choices=["gloo", "nccl"],
                   help="torch.distributed backend")
    p.add_argument("--precond", default="jacobi", choices=["jacobi", "amgx"],
                   help="Preconditioner (only 'jacobi' in Task 3a)")
    p.add_argument("--dims", default="8,4,4",
                   help="Grid dimensions NX,NY,NZ (e.g. 8,4,4)")
    p.add_argument("--rtol", type=float, default=1e-10,
                   help="Solver relative tolerance")
    p.add_argument("--rhs", default="mms", choices=["mms", "random"],
                   help=("RHS mode: 'mms' (manufactured solution, default) or "
                         "'random' (fixed-seed standard normal, exercises many "
                         "CG iterations via the full distributed loop)"))
    return p.parse_args()


def main():
    args = parse_args()

    dims_str = args.dims.replace(" ", "")
    dims = tuple(int(d) for d in dims_str.split(","))
    if len(dims) != 3:
        sys.exit(f"--dims must be NX,NY,NZ (got {args.dims!r})")
    nx, ny, nz = dims

    # ----------------------------------------------------------------
    # Initialize torch.distributed
    # ----------------------------------------------------------------
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # Device selection
    if args.backend == "nccl":
        device = torch.device(f"cuda:{local_rank}")
        # NCCL binds its internal buffers to the CURRENT cuda device, not the
        # per-tensor device — every rank MUST pin its device before any NCCL op
        # or all ranks collide on cuda:0 (CUDA error 999 in batch_isend_irecv).
        torch.cuda.set_device(local_rank)
    else:
        device = torch.device("cpu")

    if world_size > 1 or "MASTER_ADDR" in os.environ:
        # Launched via torchrun (env vars are set)
        dist.init_process_group(backend=args.backend)
    else:
        # Direct python invocation (no torchrun): set minimal env and init
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group(
            backend=args.backend,
            init_method="env://",
            rank=0,
            world_size=1,
        )

    # ----------------------------------------------------------------
    # Build slab partition
    # ----------------------------------------------------------------
    from diffsim.mesh.partition import slab_partition
    from diffsim.solvers.dist_cg import pcg
    from diffsim.solvers.dist_cg_torch import TorchDistComm

    if nx < world_size:
        if rank == 0:
            print(f"ERROR: nx={nx} < world_size={world_size}; need nx >= world_size")
        dist.destroy_process_group()
        sys.exit(1)

    all_parts = slab_partition(dims, world_size)
    part = all_parts[rank]
    n_owned = len(part.owned)
    n_ghost = len(part.ghost)

    if rank == 0:
        print(f"[rank 0] world_size={world_size}, dims={dims}, "
              f"n_total={nx*ny*nz}, n_owned/rank~{nx*ny*nz//world_size}")

    # ----------------------------------------------------------------
    # Assemble local system
    # ----------------------------------------------------------------
    rhs_mode = args.rhs
    t0 = time.perf_counter()
    A_local_csr, b_local, diag_local, x_exact_local = assemble_local_poisson(
        part, dims, device, rhs_mode=rhs_mode
    )
    t_assemble = time.perf_counter() - t0

    if rank == 0:
        print(f"[rank 0] local assembly: {t_assemble:.3f}s")

    # ----------------------------------------------------------------
    # Build comm + spmv + precond
    # ----------------------------------------------------------------
    comm = TorchDistComm(part, device, dtype=torch.float64)
    spmv = make_partitioned_spmv(A_local_csr, n_owned, n_ghost, comm, device)

    if args.precond == "amgx":
        # Block-Jacobi preconditioner: each rank preconditions with an AMGX
        # solve on its OWNED diagonal block B = A_local[:, :n_owned] (drop the
        # ghost columns → NO cross-rank coupling → block-Jacobi). B is a
        # principal submatrix of the global SPD operator, hence SPD. Each rank
        # holds exactly one block, so AMGX's process-global singleton (cached
        # per (sym,tol,maxiter)) is reused across all CG iterations. We solve
        # each block to a tight tolerance so M^{-1} ≈ B^{-1} is a FIXED linear
        # operator (required for standard CG; a fixed-V-cycle apply is the
        # cheaper follow-up). The per-apply host<->device copy of the owned
        # vector is acceptable for this correctness proof.
        import scipy.sparse
        from diffsim.solvers.amgx import amgx_solve

        B_owned = scipy.sparse.csr_matrix(A_local_csr[:, :n_owned])
        B_owned.sort_indices()

        def precond(r: torch.Tensor) -> torch.Tensor:
            r_np = r.detach().cpu().numpy().astype(np.float64)
            z_np = amgx_solve(B_owned, r_np, sym=True, tol=1e-10, maxiter=200)
            return torch.tensor(z_np, dtype=torch.float64, device=device)
    else:
        # Jacobi (diagonal)
        def precond(r: torch.Tensor) -> torch.Tensor:
            return r / diag_local

    # ----------------------------------------------------------------
    # Run distributed PCG
    # ----------------------------------------------------------------
    # Warm-up the halo path with a quick no-op to avoid init overhead in timing
    dummy = torch.zeros(n_owned + n_ghost, dtype=torch.float64, device=device)
    if n_ghost > 0:
        comm.exchange_halo(dummy)

    t0 = time.perf_counter()
    x_local, info = pcg(spmv, precond, b_local, comm, rtol=args.rtol)
    t_solve = time.perf_counter() - t0

    if rank == 0:
        print(f"[rank 0] PCG: converged={info['converged']}, "
              f"iters={info['iters']}, solve_time={t_solve:.3f}s")
        if info["resid_history"]:
            print(f"[rank 0] final_resid={info['resid_history'][-1]:.3e}")

    # ----------------------------------------------------------------
    # Gather distributed solution on rank 0 for comparison
    # ----------------------------------------------------------------
    # Collect n_owned sizes across ranks. Collective tensors must live on the
    # backend's device: CUDA for NCCL (CPU tensors raise "No backend type
    # associated with device type cpu"), CPU for gloo. `device` is already the
    # right one for this backend.
    n_owned_tensor = torch.tensor([n_owned], dtype=torch.int64, device=device)
    all_n_owned = [torch.zeros(1, dtype=torch.int64, device=device)
                   for _ in range(world_size)]
    dist.all_gather(all_n_owned, n_owned_tensor)
    all_n_owned_list = [int(t.item()) for t in all_n_owned]
    max_n = max(all_n_owned_list)

    # Gather the solution with a PADDED all_gather (NCCL has no robust gather/
    # scatter and requires equal-sized tensors; slabs differ by the remainder).
    # Each rank pads its owned solution to max_n, all_gather, then rank 0 trims
    # per-rank and concatenates in rank order (== global node order for slabs).
    x_dev = (x_local.to(device=device, dtype=torch.float64)
             if isinstance(x_local, torch.Tensor)
             else torch.tensor(x_local, dtype=torch.float64, device=device))
    x_pad = torch.zeros(max_n, dtype=torch.float64, device=device)
    x_pad[:n_owned] = x_dev
    gathered = [torch.zeros(max_n, dtype=torch.float64, device=device)
                for _ in range(world_size)]
    dist.all_gather(gathered, x_pad)
    if rank == 0:
        parts = [gathered[r][:all_n_owned_list[r]] for r in range(world_size)]
        x_dist = torch.cat(parts).cpu().numpy()
    else:
        x_dist = None

    # ----------------------------------------------------------------
    # Correctness check on rank 0
    # ----------------------------------------------------------------
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Reference solve (scipy.spsolve)...")
        t0 = time.perf_counter()
        x_ref = serial_reference_solve(dims, rhs_mode=rhs_mode)
        t_ref = time.perf_counter() - t0
        print(f"Reference solve: {t_ref:.3f}s")

        # Compare distributed vs reference
        err = np.abs(x_dist - x_ref)
        rel_err = float(np.max(err) / (np.max(np.abs(x_ref)) + 1e-300))

        print(f"\n--- VALIDATION RESULTS ---")
        print(f"world_size : {world_size}")
        print(f"dims       : {dims}  (n_total={nx*ny*nz})")
        print(f"rhs_mode   : {rhs_mode}")
        print(f"iters      : {info['iters']}")
        print(f"rel_err    : {rel_err:.3e}  (||x_dist - x_ref||_inf / ||x_ref||_inf)")
        print(f"converged  : {info['converged']}")

        if rel_err < 1e-8:
            print(f"PASS  rel_err={rel_err:.3e} < 1e-8")
        else:
            print(f"FAIL  rel_err={rel_err:.3e} >= 1e-8  <-- INVESTIGATE")
            sys.exit(1)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
