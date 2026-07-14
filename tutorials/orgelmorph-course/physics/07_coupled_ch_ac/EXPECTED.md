# P7 — expected results (self-check)

Running `python run.py` (defaults: 64×64, p1 bulk, χ_aa = 1.2,
χ_ca = 2.6, seeded crystal, fixed seed) should reproduce the following.

| quantity | value |
|---|---|
| (2,1) ON: crystallizer φ₀ inside → outside crystal | 0.852 → 0.436 |
| (2,1) ON demixing contrast | +0.416 |
| (2,0) OFF demixing contrast | +0.000 |
| amplification (ON/OFF, floored) | 21× |
| (3,1) contrast | +0.391 |
| (3,2) contrast | +0.170 |

**What must be true regardless of hardware:**

- **Crystallization purifies its species.** With the coupling ON, the
  crystallizing species concentrates strongly inside the crystal
  (inside ≫ outside; contrast ≈ +0.4). This is crystallization-driven
  demixing.
- **Pure Cahn–Hilliard does not demix here.** With K = 0 the same blend
  stays mixed (contrast ≈ 0), because χ_aa = 1.2 is below the demixing
  threshold — the composition pattern is *created* by crystallization,
  not merely amplified. (The "amplification" number is therefore a
  floored ratio; the honest statement is that pure CH gives ≈0.)
- **The mechanism scales up the (M,K) ladder.** (3,1) and (3,2) both show
  positive demixing contrast — adding species and crystallizable phases
  is a change of two integers, and the crystallization-driven demixing
  carries over.

Notes: the crystal stays modest in size (area ≈ 0.03) — the metric is the
*purity contrast*, not the crystal area. The (3,2) run (ten coupled
fields) is the heaviest; it is the slowest config to run. Third-
significant-figure differences from a different card are normal.
