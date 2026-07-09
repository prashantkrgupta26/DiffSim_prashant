# heat-mass — coupled thermal / buoyant flow

Scalar transport (advection–diffusion with SUPG/VMS stabilization) coupled to the
momentum equation through Boussinesq buoyancy, plus consistent surface-flux
(Nusselt) functionals on immersed boundaries.

| Script | Case | Reference | Measured |
|---|---|---|---|
| `devahl_davis.py` | buoyancy-driven square cavity, Ra 10³–10⁶ | de Vahl Davis (1983) | Nu_mid 1.1174 / Nu_max 2.2426 — 0.05 % / 0.02 % |
| `coupled_nu.py` | heated immersed cylinder, coupled Nu | literature band | Nu_D 3.21 |
| `heated_cylinder.py` | consistent heat-flux functional | — | advective-face flux, exonerated-α study |

## Run

```bash
python benchmarks/heat-mass/devahl_davis.py            # the Nu convergence table
python benchmarks/heat-mass/coupled_nu.py --solver cudss
```

## Notes

The VMS scalar residual is complete through the `−κΔT` strong term (the lapN
tables), which is what lifts p2 to observed order 3.00 — the p1/band/p2
comparison lives in the M2 report. de Vahl Davis is the tightest thermal gate in
the suite (two digits on both Nusselt measures); its baseline is
`tests/baselines/m2_devahl_davis.json`.
