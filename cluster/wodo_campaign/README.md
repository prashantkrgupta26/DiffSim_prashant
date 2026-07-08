# Wodo CMS-2012 Nova Campaign (M4 track c)

Full-resolution replication of Wodo & Ganapathysubramanian, Comput.
Mater. Sci. 55 (2012) 113-126, figs 3-7, on Nova — plus ONE reduced-3D
stretch. Runner: `benchmarks/wodo_nova.py` (case table via `--list`).
All marches are DEVICE-BOUND (`use_device_assembly=True`: slot-map
scatter into a once-per-mesh CSR pattern + zero-copy torch-CSR cuDSS;
measured on the workstation at 96x48: 56x vs host-splu, 8x vs
host-cudss, 36 ms/step).

## Submit lines (from the repo root on Nova)

```bash
# fill FIXME_PARTITION in both scripts first:  sinfo -o "%P %G %N"
sbatch cluster/wodo_campaign/a100_wodo.sbatch   # 14 cases, full res, seq.
sbatch cluster/wodo_campaign/h200_wodo.sbatch   # ONE 3-D stretch case
```

## What runs

- **a100_wodo.sbatch** — sequential full-resolution campaign,
  14 cases, gate-critical first:
  fig6 Np={5,100,20}/Nf=5 Bi=0.4; fig7 chi={(1,.3,.6),(1,.6,.3),
  (1,.3,.3)} at Np=100/Nf=5 Bi=0.3; fig4 Bi={0.03,0.3,3};
  fig5 blends {1:1, 1:0.8} Bi=0.3; fig3 1-D Bi={0.1,1,10}.
  2-D full = 250x100 (the paper's own mesh; level-9 strip via the
  v1.1 generalized metric); fig3 = 4x256. Model config = the fig67
  gate-passing one exactly: CHC noise 1e-3, var_mob (D_ratio 1e-3),
  b_reg 1e-3, phi_s0 = 0.75, march to phi_s = 0.05 (`--to-phis`).
  Per-case wall cap 3 h; one `RESULT ...` line per case appended to
  `results.txt` (reason/steps/A_p/A_d/layers/top comps/mass
  drift/wall).
- **h200_wodo.sbatch** — the reduced-3D stretch: 128x128x48
  (~3.4M dofs; phi_s0=0.66, Lx=Ly=3.3, Bi=0.3). try/except ALLOC —
  EITHER outcome is the result (it locates the cuDSS wall for the
  coupled 4-dof CH factor on 141 GB). Full-res 3-D (230x230x70,
  ~15M dofs) needs the CH block preconditioner — recorded M4 item;
  do not attempt it with a direct solver.

## Mirror-folder workflow (how results come home)

Baskar runs the sbatch jobs; everything a case produces lands in
`cluster/results/wodo-<card>-<jobid>/`:

```
results.txt              # one RESULT line per case — the summary
<x>-<jobid>.out          # slurm stdout (per-case tails)
fig<N>_<case>.log        # full per-case log
f<N>_<case>_full_phip_h*.npy    # phi_p snapshots at h=0.9/0.7/0.6/0.5/0.4
f<N>_<case>_full_final.npz      # final phi_p, phi_f, h, t
f3d_stretch_final.npz           # 3-D nodal fields (if it ran)
```

Drop that directory into the repo root as a mirror folder, following
the band-study convention (`nova-band_sweep-mirror/`):

```bash
# on Nova
tar czf wodo-a100-<jobid>.tgz cluster/results/wodo-a100-<jobid>
# on the workstation, from the repo root
mkdir -p nova-wodo_campaign-mirror
tar xzf wodo-a100-<jobid>.tgz -C nova-wodo_campaign-mirror --strip-components=2
```

The analysis agent then reads `nova-wodo_campaign-mirror/` directly
(regime table, anisotropy gates, morphology plots vs the paper's
figures). `results.txt` alone is enough for the regime-level gates;
the `.npy/.npz` snapshots are needed for the figure-by-figure
morphology comparison.

## Expectations / gates (regime level, from the 96x48 sweep)

- fig6: A_p(final, N=100) >> A_p(final, N=5) (sweep: 374612 vs 2.3);
  N=5 percolated, N=100 multilayer.
- fig7: chi_fs=0.6 vs chi_ps=0.6 enrich OPPOSITE components in the
  top 10%; chips breaks the multilayer (their Sec. 7.4).
- fig4: total drying time ordered Bi=0.03 >> 0.3 >> 3; domain size
  decreases with Bi (morphology frozen at high Bi).
- fig5: 1:1 percolated vs 1:0.8 multiple stripes.
- fig3: t_final(Bi=10) << t_final(Bi=0.1); top-initiated vs
  homogeneous onset (committed workstation evidence: 0.33 vs 20.5).
- ALL cases: mass_drift < 1e-10 (measured class: 1e-15/16), simplex
  bounds respected.
