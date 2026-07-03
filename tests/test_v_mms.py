import json, pathlib
import numpy as np
import pytest
from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.assembly.dirichlet import DirichletPoisson
from diffsim.physics.poisson import l2_error

pytestmark = pytest.mark.tier3

U = lambda x: np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1]) * np.sin(np.pi * x[:, 2])
F = lambda x: 3.0 * np.pi**2 * U(x)
BASE = pathlib.Path(__file__).parent / "baselines" / "m0_baselines.json"

def _err(level, p, device):
    m = build_mesh(build_uniform(level), p=p)
    c = build_constraints(m)
    dm = DeviceMesh.from_mesh(m, c, basis_tables(p), device)
    u = DirichletPoisson(dm).solve(g_fn=U, f_fn=F, tol=1e-13)
    return l2_error(dm, u, U)

def test_V2_convergence_orders(device):
    results = {}
    for p, expected in ((1, 2.0), (2, 3.0)):
        levels = [2, 3, 4] if p == 1 else [1, 2, 3]
        errs = [_err(l, p, device) for l in levels]
        orders = [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
        assert abs(orders[-1] - expected) < 0.10, (p, orders)   # cuFEM ±0.10 criterion
        results[f"p{p}"] = errs
    # regression lock: create baseline on first run, compare thereafter
    if BASE.exists():
        ref = json.loads(BASE.read_text())
        for k, v in results.items():
            assert np.allclose(v, ref[k], rtol=1e-8), (k, v, ref[k])
    else:
        BASE.parent.mkdir(parents=True, exist_ok=True)
        BASE.write_text(json.dumps(results, indent=2))
        pytest.skip("baseline created; re-run to compare")
