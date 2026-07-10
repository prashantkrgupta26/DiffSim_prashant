# M5 — OrgElMorph: Organic Electronics Morphology

Ratified by Baskar 2026-07-10 (rulings R1-R3 in
docs/theory/crystallization_formulation_p1.md — the formulation memo is
the technical contract; this spec is the milestone ledger).

## Mission

Model coupled phase separation + crystallization of organic blends —
thermal annealing first, then evaporation-induced structure formation —
with an (M, K)-generic multi-CH x multi-AC solver extensible from
ternary to quaternary (2 active + 2 solvents; 3 active + 1 solvent) and
beyond, with per-crystal orientation tracking, delivered through the
group-facing film front-end.

## Tracks

(a) DELIVERED (the enabling week, all pushed): the blockch
    preconditioner stack G1-G5 (full-res 3-D on one 48 GB card; five
    design laws), device-resident assembly (node-graph pattern),
    diffsim.film front-end (chi/N/blend/Bi first-class; 13 configs) +
    four-layer RunLog, Negi-2018 2-D replication on the SI-corrected
    mobility (both phenomena + the rpm-selected end-state observation),
    Wodo + Negi 3-D Nova kits.
(b) S0-S4 crystallization ladder per the P1 memo: S0 amorphous
    regression; S1 binary annealing (MMS + QUANTITATIVE kinetics vs
    2310.11844 + morphology classes, orientation included); S2 ternary
    annealing (2512.16390 validation, experiments as stretch); S3
    evaporation-induced (film frame); S4 quaternary both variants +
    per-solvent evaporation.
(c) Front-end + RunLog growth: species-list configs, crystallization
    preflight rules, grain identification/analysis outputs (theta-based
    crystal labeling, size distributions, crystallinity kinetics).
(d) Hero campaigns: Nova 3-D (Wodo + Negi kits ready; annealing +
    evaporative crystallization heroes join as S1-S3 land).
(e) Feeds: paper B (production results + the solver story), the M6
    candidate (learning rung 2 — learned free energies now including
    crystalline terms: the extended-FH basis is exactly the learnable
    surface).

## Standing constraints

House rules unchanged: measure-then-lock, findings-log honesty,
commit-per-green, Gramian-before-compute for any inverse work, the
parity-tolerance distribution lesson, mobility closures always NAMED
(wodo | negi | ...), no dyadic feature dims in gates.

## Open externals carried

GH200 coherent probe (M1d); Nova submissions (kits ready); papers
priority call (three outlines local).
