# Named film configs — the validated case library

Each YAML here is a complete, named, validated (or honestly-flagged)
case for `python -m diffsim.film <config.yaml>`. They are the SINGLE
SOURCE OF TRUTH for case parameters: the cluster kits
(`cluster/wodo_campaign/`) submit these files with `--set` overrides
for 3-D extensions instead of duplicating numbers.

To run a NEW material system, copy the closest case and edit four
numbers: `physics.chi`, `physics.N`, `physics.blend`, `evaporation.Bi`.
Everything else carries validated campaign defaults.

| config | system | what it shows |
|---|---|---|
| `wodo2012_fig3_bi{0.1,1,10}` | Wodo CMS-2012 Fig 3, 1-D drying | Bi ladder: top-initiated (Bi=10) vs homogeneous (Bi=0.1) onset; committed gate physics (`benchmarks/phase-field/wodo_fig3.py`) |
| `wodo2012_fig6_n{5,20,100}` | Fig 6, chain-length ladder (Nf=5 fixed) | percolated -> multilayered as Np grows |
| `wodo2012_fig7_{sym,chifs,chips}` | Fig 7, solvent selectivity | the less-soluble component wets the free surface; chips breaks the multilayer |
| `negi2018_{6000,3000,1500,500}rpm` | Negi 2018 spin-coated PDPP5T:PC71BM | drying-rate (rpm) ladder; domain size grows as drying slows |

Provenance of parameter values: the Wodo cases are read out of the
committed campaign tables (`benchmarks/phase-field/wodo_nova.py` FIG3 /
FIG2D and `wodo_fig67.py` — including the fig67 PARAMETER NOTE: Fig 6
varies Np with Nf = 5 FIXED). The Negi cases document their
weight->volume blend assumption and the eps^2 -> kappa
nondimensionalization inside `negi2018_6000rpm.yaml`.

Preflight policy: all Wodo cases are `preflight: strict`. The Negi
cases are `preflight: warn` because the paper's PHYSICAL eps^2 = 1e-10
J/m under-resolves the forming interfaces at the paper-scale 4 nm mesh
(xi_early ~ 1.5 elements < the validated 2.5-element floor) — a
recorded replication-fidelity deviation, monitored in flight; see the
note in `negi2018_6000rpm.yaml`.

Mesh note: the resolver picks the smallest octree level that fits the
requested cell counts (e.g. 250x100 -> level 8), which can differ from
the historical campaign's level (wodo_nova used level 9 for 250x100).
The PHYSICAL mesh is identical — same cell counts, same domain; the
generalized metric (wodo_film v1.1) absorbs the tree spacing.
