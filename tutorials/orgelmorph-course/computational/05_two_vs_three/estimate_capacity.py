"""OrgElMorph course - Computational C5: memory-capacity estimator (CLI).

    python estimate_capacity.py --dim 3 --n 128 --fields 2 --solver blockch
    python estimate_capacity.py --dim 2 --level 8

Estimate the FULL device-memory footprint of a Cahn-Hilliard solve at a given
resolution -- not just the CSR values, but every buffer a real solve holds:
row pointers, column indices, duplicate device index mirrors, the Newton/BDF
solution+history vectors, local-to-global maps, quadrature/geometry tables,
solver workspace (direct fill-in OR preconditioner blocks), and the saved
output snapshot.  Prints a breakdown, the total, and whether the run clears
the int32 CSR ceiling and the card's memory.

The point (spec C5): "cheap in 2-D" is a statement about the GROWTH LAW, and
the memory that actually decides feasibility is far more than the matrix
values -- so estimate it completely before you launch.
"""
import argparse

INT32_MAX = 2 ** 31            # CSR slot-arithmetic ceiling (~2.1e9)


def stencil_nodes(dim, p):
    """Nodes coupled to a given node in a structured degree-p FE mesh: every
    node sharing an element.  For degree p that is a (2p+1)^dim block (P1 ->
    3^dim = 9 in 2-D, 27 in 3-D).  This -- element CONNECTIVITY, basis ORDER,
    and dimension -- is the real origin of nnz/dof, NOT a 3^dim-1 finite-
    difference stencil."""
    return (2 * p + 1) ** dim


def estimate(dim, n, p=1, fields=2, solver="blockch", idx_bytes=4,
             hist=2, snapshots=1, device_mem_gb=48.0):
    """Return a dict: per-component bytes, total, and feasibility flags.

    Model (per FE contract, not a hand-wave):
      nodes   = (n+1)^dim              (uniform grid, degree-p nodes folded in
                                        via the (p*n+1)^dim count below)
      dofs    = nodes * fields
      nnz     = dofs * fields * stencil_nodes   (each scalar coupling is a
                                                 fields x fields block)
    """
    nodes = (p * n + 1) ** dim
    dofs = nodes * fields
    stencil = stencil_nodes(dim, p)
    # each row (a dof) couples to `stencil` nodes, each contributing `fields`
    # column entries (the (c,mu) block is dense within a node pair)
    nnz = dofs * stencil * fields

    val_b = nnz * 8                              # float64 CSR values
    col_b = nnz * idx_bytes                      # CSR column indices
    rowptr_b = (dofs + 1) * idx_bytes            # CSR row pointers
    # a device-resident assembler keeps an int32 (indptr, indices) MIRROR
    dup_idx_b = (nnz + dofs + 1) * idx_bytes
    # Newton solution + rhs + increment + BDF history (`hist` old states)
    vec_b = (3 + hist) * dofs * 8
    # local-to-global maps: conn (nodes-per-elem * n_elem) int32
    npe = (p + 1) ** dim
    n_elem = n ** dim
    l2g_b = n_elem * npe * idx_bytes
    # quadrature + geometry tables (per element: element size, a few doubles)
    geo_b = n_elem * 8 * (dim + 2)
    # saved output snapshot(s): the conserved field on the grid
    out_b = snapshots * nodes * fields * 8

    # solver workspace -- the big, solver-dependent term
    if solver in ("splu", "cudss"):
        # direct factorization fill-in: bandwidth^ (dim-1) growth.  Rough
        # measured-class factor: 2-D ~ 6x nnz, 3-D ~ 30x nnz (the cuDSS wall).
        fill = 6.0 if dim == 2 else 30.0
        work_b = nnz * (8 + idx_bytes) * fill
        work_label = f"direct fill-in (~{fill:.0f}x nnz)"
    else:
        # blockch: a handful of extracted block copies (Amm/Acm/Amc/W1/W2)
        # on the shared node pattern + Krylov vectors; ~4x the block nnz.
        block_nnz = nodes * stencil            # one scalar block's nnz
        work_b = 5 * block_nnz * (8 + idx_bytes) + 30 * dofs * 8
        work_label = "blockch preconditioner blocks + Krylov vectors"

    comps = {
        "CSR values (float64)": val_b,
        "CSR column indices": col_b,
        "CSR row pointers": rowptr_b,
        "device index mirror (dup)": dup_idx_b,
        "solution + rhs + BDF history": vec_b,
        "local-to-global (conn) map": l2g_b,
        "quadrature / geometry": geo_b,
        f"solver workspace: {work_label}": work_b,
        "output snapshot(s)": out_b,
    }
    total = sum(comps.values())
    return dict(dim=dim, n=n, p=p, fields=fields, solver=solver,
                nodes=nodes, dofs=dofs, nnz=nnz, stencil=stencil,
                nnz_per_dof=nnz / dofs, components=comps, total_bytes=total,
                total_gb=total / 1e9,
                int32_overflow=nnz >= INT32_MAX,
                int32_headroom=nnz / INT32_MAX,
                fits_card=total < device_mem_gb * 1e9,
                device_mem_gb=device_mem_gb)


def _fmt_bytes(b):
    for unit in ("B", "KB", "MB", "GB"):
        if b < 1024 or unit == "GB":
            return f"{b:8.2f} {unit}"
        b /= 1024


def print_report(est):
    print(f"=== capacity estimate: {est['dim']}-D, n={est['n']}, "
          f"p={est['p']}, fields={est['fields']}, solver={est['solver']} ===")
    print(f"  nodes {est['nodes']:,}  dofs {est['dofs']:,}  "
          f"nnz {est['nnz']:,}  (stencil {est['stencil']} nodes, "
          f"{est['nnz_per_dof']:.0f} nnz/dof)")
    print("  memory breakdown:")
    for name, b in sorted(est["components"].items(), key=lambda kv: -kv[1]):
        frac = 100 * b / est["total_bytes"]
        print(f"    {name:42s} {_fmt_bytes(b)}  ({frac:4.1f}%)")
    print(f"    {'TOTAL':42s} {_fmt_bytes(est['total_bytes'])}")
    print(f"  int32 CSR ceiling: nnz/2^31 = {est['int32_headroom']:.3f} "
          f"({'OVERFLOW -> needs int64/block-mask' if est['int32_overflow'] else 'ok'})")
    print(f"  fits a {est['device_mem_gb']:.0f} GB card: "
          f"{'YES' if est['fits_card'] else 'NO -> matrix-free / multi-GPU'}")


def main():
    ap = argparse.ArgumentParser(description="CH memory-capacity estimator")
    ap.add_argument("--dim", type=int, default=3, choices=(2, 3))
    ap.add_argument("--n", type=int, help="cells per side")
    ap.add_argument("--level", type=int, help="2^level cells per side "
                    "(alternative to --n)")
    ap.add_argument("--p", type=int, default=1, help="basis degree")
    ap.add_argument("--fields", type=int, default=2, help="fields (c,mu -> 2)")
    ap.add_argument("--solver", default="blockch",
                    choices=("splu", "cudss", "blockch"))
    ap.add_argument("--idx-bytes", type=int, default=4, choices=(4, 8),
                    help="CSR index width (4=int32, 8=int64)")
    ap.add_argument("--device-mem-gb", type=float, default=48.0)
    args = ap.parse_args()
    n = args.n if args.n else (1 << args.level if args.level else 64)
    est = estimate(args.dim, n, p=args.p, fields=args.fields,
                   solver=args.solver, idx_bytes=args.idx_bytes,
                   device_mem_gb=args.device_mem_gb)
    print_report(est)


if __name__ == "__main__":
    main()
