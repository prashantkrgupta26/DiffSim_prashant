# A3 — Immersed geometry: the Shifted Boundary Method

LEARNING OUTCOME. You can solve a PDE on a domain that has NO body-fitted
mesh — only a signed-distance oracle — and you understand the three moving
parts: element classification (keep/discard), the surrogate staircase
boundary, and the Taylor shift that restores accuracy. You also meet the
INHOMOGENEOUS boundary data that A2 deferred: here g lives on the TRUE
circle and is mapped to the staircase by the shift.

BACKGROUND. THE PROBLEM. Solve  -div(grad u) = f  INSIDE a disk of radius 0.3 centered in
the unit square, with u = g on the circle — but WITHOUT a body-fitted mesh.
The mesh is a uniform Cartesian octree of the whole square; the circle is
known only through a signed-distance oracle psi(x) (negative inside).

THE METHOD (SBM, Main & Scovazzi 2018). Keep only elements sufficiently
inside the disk; their outer faces form a STAIRCASE approximation of the
circle (the surrogate boundary). Impose the Dirichlet condition there,
Nitsche-style, but TAYLOR-SHIFT everything to the true boundary using the
distance vector d(x) from each surrogate quadrature point to its closest
point on the circle:

    S u = u + grad(u).d          (+ half d^T H(u) d for quadratic elements)

We verify the whole pipeline with the method of manufactured solutions (MMS):
pick u*(x) = sin(pi x) cos(pi y), derive f = -lap(u*), impose g = u*, and
measure how fast the discrete solution converges to u* under refinement.
Expected: second order in L2 for linear elements. You will see it.

EXPECTED RESULTS: errors ~ 3.2e-3 / 6.6e-4 / 1.5e-4, orders 2.28 / 2.13
(slightly superconvergent early — common for SBM on smooth geometry).

Run:  python tutorials/A_foundations/A3_shifted_boundary.py

??? example "Full script — `tutorials/A_foundations/A3_shifted_boundary.py` (run it!)"

    ```python
    import numpy as np
    
    from diffsim.octree.build import build_uniform
    from diffsim.mesh.nodes import build_mesh
    from diffsim.mesh.constraints import build_constraints
    from diffsim.mesh.basis import basis_tables
    from diffsim.mesh.faces import face_tables
    from diffsim.assembly.operators import DeviceMesh
    from diffsim.geometry.csg import Sphere
    from diffsim.sbm.surrogate import classify_lambda, extract_surrogate, GeometryData
    from diffsim.sbm.poisson import SBMPoisson
    from diffsim.physics.poisson import l2_error_masked
    
    DEVICE = "cuda:0"
    
    # The manufactured truth: a smooth field with nonzero curvature everywhere.
    u_star = lambda x: np.sin(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1])
    f_star = lambda x: 2 * np.pi ** 2 * u_star(x)      # = -lap(u*)
    
    
    def solve_at_level(level: int, p: int = 1) -> float:
        """One full SBM solve; returns the L2 error against u* on the disk."""
        # (1) GEOMETRY. An oracle, not a mesh: psi < 0 inside the disk.
        oracle = Sphere((0.5, 0.5), 0.3)
    
        # (2) BACKGROUND OCTREE. Uniform level-`level` refinement of [0,1]^2
        #     (2^level elements per axis). Adaptivity is available but not needed
        #     for a convergence study.
        tree = build_uniform(level, dim=2)
    
        # (3) CLASSIFICATION (the lambda-criterion, production semantics).
        #     lambda = 0.0 keeps only elements FULLY inside the disk — the most
        #     conservative surrogate. Volume fractions are estimated by dense
        #     Gauss sampling of psi, with a Lipschitz narrow-band to skip
        #     far-away elements.
        retained, _ = classify_lambda(tree, oracle, lam=0.0, domain="inside")
    
        # (4) SURROGATE BOUNDARY: the exposed whole faces of the retained set —
        #     a staircase Gamma~ that approximates the circle from inside.
        surrogate = extract_surrogate(retained)
    
        # (5) MESH + CONSTRAINTS + DEVICE UPLOAD. build_constraints handles
        #     hanging nodes / p-transitions (none here); DeviceMesh ships
        #     connectivity + basis tables to the GPU.
        mesh = build_mesh(retained, p=p)
        cons = build_constraints(mesh)
        dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(p, dim=2), DEVICE)
    
        # (6) THE SHIFT DATA. For every quadrature point on every surrogate face:
        #     the closest true-boundary point (Newton projection on psi), the
        #     distance vector d, and the true normal. This is precisely the
        #     geometric information the Taylor shift needs — and, later in
        #     tutorial 02, the door through which SHAPE GRADIENTS flow.
        geo = GeometryData.evaluate(oracle, retained, surrogate,
                                    face_tables(p, 2), domain="inside")
    
        # (7) ASSEMBLE + SOLVE. SBMPoisson adds the volume stiffness and the
        #     three Nitsche face terms (consistency, adjoint-consistency on the
        #     SHIFTED test function, penalty alpha/h on shifted trial x test).
        problem = SBMPoisson(dm, geo, surrogate, g_fn=u_star, alpha=10.0)
        u = problem.solve(f_fn=f_star)
    
        # (8) MEASURE. L2 error at quadrature points, masked to the true disk.
        return l2_error_masked(dm, u, u_star,
                               lambda x: oracle.classify(x) < 0)
    
    
    if __name__ == "__main__":
        print(__doc__.split("Run:")[0])
        errors = {}
        for level in (4, 5, 6):
            errors[level] = solve_at_level(level)
            h = 1.0 / 2 ** level
            print(f"  level {level}  (h = {h:.4f}):   L2 error = {errors[level]:.3e}")
        print("\nobserved convergence orders (should approach 2.0):")
        lv = sorted(errors)
        for a, b in zip(lv, lv[1:]):
            print(f"  level {a} -> {b}:  order = "
                  f"{np.log2(errors[a] / errors[b]):.2f}")
        print("""
    EXPLORE
      (a) Set p=2 in solve_at_level and watch the order go to 3 — then read
          _shift_fn_for in src/diffsim/sbm/poisson.py to see the Hessian term
          that makes it possible (and why p=1 must NOT include it).
      (b) Change lam=0.0 to lam=1.0 (keep every intercepted element). The
          surrogate hugs the circle from OUTSIDE the retained set instead —
          the order is unchanged. Why does the shift not care about the sign
          of d?
      (c) Replace Sphere with diffsim.geometry.gridsdf.GridSDF.from_oracle(
          Sphere((0.5,0.5),0.3), n=128) — the same solve through a sampled
          voxel geometry. Compare the error floors.
    """)
    
    # PERFORMANCE CORNER: classification samples psi at Gauss points inside a
    # Lipschitz narrow band. Time classify_lambda vs level at 4..8 and fit the
    # exponent. It should track the number of INTERCEPTED elements O(2^level),
    # not the total O(4^level) — the narrow band is why.
    ```
