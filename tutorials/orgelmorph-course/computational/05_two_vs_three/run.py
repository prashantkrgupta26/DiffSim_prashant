"""OrgElMorph course - Computational C5 driver (the file you run).

    python run.py                # live 2-D vs tiny-3-D scaling + cited ladder

Measures how dofs, nnz, memory, and step time grow with refinement in
2-D versus 3-D on real Cahn-Hilliard systems, then echoes the
device-scale ladder from the dev notes.  Compare with EXPECTED.md."""
import argparse

from scaling import (measure_scaling, CITED_LADDER, INT32_MAX,
                     int32_headroom, CSR_BYTES_PER_NNZ)


def _print_rows(recs, tag):
    print(f"  {tag}:")
    print(f"    {'level':>5} {'dofs':>8} {'nnz':>10} {'nnz/dof':>8} "
          f"{'mem MB':>8} {'step s':>7}")
    for r in recs:
        print(f"    {r['level']:>5} {r['dofs']:>8} {r['nnz']:>10} "
              f"{r['nnz_per_dof']:>8.1f} {r['mem_mb']:>8.1f} "
              f"{r['step_s']:>7.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    print("=== LIVE scaling: real Cahn-Hilliard systems ===\n")
    s = measure_scaling(device=args.device)
    _print_rows(s["two_d"], "2-D (uniform box)")
    _print_rows(s["three_d"], "3-D (uniform box)")
    print(f"\n  dofs per refinement level: "
          f"2-D x{s['dof_growth_2d']:.1f}, 3-D x{s['dof_growth_3d']:.1f}  "
          f"(theory: x4 vs x8)")
    print(f"  nnz per dof (stencil density): "
          f"2-D {s['nnz_per_dof_2d']:.0f}, 3-D {s['nnz_per_dof_3d']:.0f}  "
          f"(9-point vs 27-point)")
    # a same-ish-dof 2-D vs 3-D comparison
    a = s["two_d"][-1]      # 2-D L7
    b = s["three_d"][-1]    # 3-D L5
    print(f"  at comparable dofs ({a['dofs']} 2-D vs {b['dofs']} 3-D): "
          f"3-D has {b['nnz'] / a['nnz']:.1f}x the nnz and "
          f"{b['step_s'] / a['step_s']:.1f}x the step time")

    print("\n=== CITED device-scale ladder (dev notes; not re-run) ===")
    print(f"  {'case':>28} {'dofs':>11} {'nnz':>14} {'int32?':>7}")
    for label, dofs, nnz, note in CITED_LADDER:
        frac = int32_headroom(nnz)
        flag = "OVER" if frac > 1.0 else f"{frac * 100:.0f}%"
        print(f"  {label:>28} {dofs:>11,} {nnz:>14,} {flag:>7}   {note}")
    print(f"\n  int32 CSR ceiling: nnz < {INT32_MAX:,} (2^31)")
    print(f"  CSR storage ~ {CSR_BYTES_PER_NNZ} bytes/nnz "
          f"(float64 val + int32 col); the LU FILL of a 3-D direct solve "
          f"is far larger -> the cuDSS memory wall (C4)")

    ok = (s["dof_growth_3d"] > s["dof_growth_2d"]
          and s["nnz_per_dof_3d"] > s["nnz_per_dof_2d"])
    print(f"\n  CHECK (3-D grows faster in both dofs and nnz/dof): "
          f"{'PASS' if ok else 'FAIL'}")
    print("\nCompare with EXPECTED.md; render figures with gen_figures.py.")


if __name__ == "__main__":
    main()
