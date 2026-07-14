"""Chapter 00 smoke test — the first run that exercises the whole workflow.

Runs a tiny binary Cahn-Hilliard march through the standard harness
(config -> provenance -> results.json -> tolerance check), so a new student
confirms that (a) the device toolchain works and (b) the course's
config/provenance/checker plumbing works, in one command:

    python run.py --config configs/smoke_cuda.yaml --output outputs/smoke

It writes the standard outputs/<run>/ layout and checks the results against
baseline.yaml. Pair it with ``doctor.py`` (env probe). See EXPECTED.md.
"""
from __future__ import annotations

import os
import sys

import numpy as np

# make the course common/ importable (course root is two levels up)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                os.pardir)))
from common import config as cfgmod          # noqa: E402
from common.run_base import build_parser, run_tutorial   # noqa: E402

SCHEMA = cfgmod.ConfigSchema(name="smoke", fields={
    "level": cfgmod.Field(int, default=4, min=2, max=8),
    "steps": cfgmod.Field(int, default=20, min=1),
    "dt": cfgmod.Field(float, default=0.02, min=0.0),
    "kappa": cfgmod.Field(float, default=5e-4, min=0.0),
    "seed": cfgmod.Field(int, default=0),
    "device": cfgmod.Field(str, default="cuda:0"),
    "precision": cfgmod.Field(str, default="fp64"),
})


def smoke_run(cfg, ctx):
    from diffsim.octree.build import build_uniform
    from diffsim.mesh.nodes import build_mesh
    from diffsim.mesh.constraints import build_constraints
    from diffsim.mesh.basis import basis_tables
    from diffsim.assembly.operators import DeviceMesh
    from diffsim.physics.cahn_hilliard import CahnHilliardStepper

    # CLI --device (ctx.device) is the source of truth; the config field is a
    # convenience default recorded in the resolved config.
    device = ctx.device or cfg.get("device")
    level, steps = cfg["level"], cfg["steps"]
    ctx.log(f"smoke: {2**level}x{2**level} CH, {steps} steps on {device}")
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)
    ctx.provenance.update(
        mesh={"level": level, "side": 2 ** level, "dim": 2, "p": 1},
        time_integrator="BDF1", linear_tol=1e-10)

    st = CahnHilliardStepper(dm, 1.0, cfg["kappa"], cfg["dt"], order=1,
                             energy="poly")
    rng = np.random.default_rng(cfg["seed"])
    st.set_initial(lambda x: 0.05 * rng.standard_normal(len(x)),
                   mu_init="consistent")
    cmins, cmaxs = [], []
    for _ in range(steps):
        c, _mu = st.step()
        c = np.asarray(c)
        cmins.append(float(c.min()))
        cmaxs.append(float(c.max()))
    ctx.history = {"c_min": np.array(cmins), "c_max": np.array(cmaxs)}
    c = np.asarray(st.hist[0])
    return {
        "n_steps": steps,
        "side": 2 ** level,
        "all_finite": bool(np.all(np.isfinite(c))),
        "c_min": min(cmins),
        "c_max": max(cmaxs),
    }


def main():
    args = build_parser("OrgElMorph 00 smoke test").parse_args()
    here = os.path.dirname(__file__)
    # small indefinite (c, mu) CH block -> scipy SuperLU pivoting is exact;
    # cuDSS (no pivoting) diverges here, so splu is the documented auto choice.
    run_tutorial(smoke_run, schema=SCHEMA, args=args,
                 default_output=os.path.join(here, "outputs", "smoke"),
                 baseline=os.path.join(here, "baseline.yaml"),
                 default_solver="splu")


if __name__ == "__main__":
    main()
