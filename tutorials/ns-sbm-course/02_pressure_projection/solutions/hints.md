# 02 — hints & selected solutions (instructor-only)

## Hints for the exploratory questions

**Q (enclosed pin vs outflow Dirichlet).** Enclosed flow has no outflow, so
pressure is defined only up to a constant — a single DOF pin (free-node 0)
fixes the gauge. External flow *has* an outflow, and imposing $p'=0$ there is
both a gauge fix *and* the physically-correct do-nothing pressure boundary;
a node-0 pin in external flow would over-constrain and fight the outflow. The
two are disjoint by construction (`consistent_projection` item #3).

**Q (which consistency item cures the open-outflow blow-up?).** Item #1 (the
PSPG-consistent, *collocated* coarse divergence) most directly: the
full-flux by-parts PPE fabricates a spurious boundary term
$\sigma(u_h\cdot n, q)_\Gamma$ at the open outflow, which is what
destabilises the base split. Keeping the divergence collocated with $q$
removes it and makes the split's continuity match the monolithic PSPG row
term-for-term. Item #3 (disjoint outflow BCs) and #4 (rotational pressure)
ride along as secular-drift cures.

## Full solution — why the enclosed cavity shows no consistency effect

The base `LerayProjectionStepper` on the *enclosed* cavity uses the default
node-0 pressure pin. Because there is no open outflow, there is no spurious
boundary term to fabricate, so `consistent_projection` on/off gives the same
faithful result (`max|proj−mono| ≈ 0.049`). The correct pedagogical takeaway:
the consistency machinery is *targeted at the outflow*, and the way to
demonstrate its necessity is Chapter 03's open-channel square, where the
rung-A base split can blow up and the weak `consistent_projection` path is
the validated one.

## Full solution — the two `||div u||` values

Projection ≈ 2.0, monolithic ≈ 0.19. Both are pointwise weak divergences of
the equal-order VMS field; they differ because the two engines reach the
steady state by different operators (a split correction vs a coupled solve)
and the pointwise norm is not what either controls. Confirm the *shared*
gate by checking (in the projection) that the PPE-space identity
$\|\sigma B^\top\hat u - K_p\phi\|$ is machine-zero — that is the number
that agrees.
