"""TriMesh (STL) geometry backend (spec S4.1 backend table, row 2): 3D only.

Forward queries use wp.Mesh (fp32 BVH) for CANDIDATE location only —
nearest triangle + inside/outside sign via winding number; the closest point
itself is recomputed in FP64 torch on the found triangle (clamped
barycentric regions; the region is selected from the FP64 computation, then
the smooth per-region formula keeps autograd exactness — piecewise-smooth,
spec S1.5). This mirrors production SBMCalc (nanoflann KD-tree + analytic
projection with edge/vertex fallback), with vertex positions as the
differentiable parameter via warp-free torch closest-point math.

distance_vector is overridden (no Newton projection): d = y - x;
n_grad = sign * (x - y)/|x - y| (the gradient direction of psi = sign*dist).
psi(x) is provided for classification and masks; its autograd path exists
(distance to the fixed candidate triangle) but oracle gradients should flow
through distance_torch/closest points, not psi Hessians — hence
near_eikonal = False and no Newton use.
"""
import numpy as np
import torch
import warp as wp

from .oracle import SDFOracle
from ..assembly.operators import _kernel_cache


def _make_query_kernel():
    key = ("trimesh_query",)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @wp.kernel(module="unique", enable_backward=False)
    def trimesh_query(mesh_id: wp.uint64,
                      pts: wp.array(dtype=wp.vec3),
                      sign: wp.array(dtype=wp.float64),
                      face: wp.array(dtype=wp.int32)):
        i = wp.tid()
        q = wp.mesh_query_point_sign_winding_number(mesh_id, pts[i], 1.0e6)
        sign[i] = wp.float64(q.sign)
        face[i] = q.face

    _kernel_cache[key] = trimesh_query
    return trimesh_query


def _closest_point_on_triangle(p, a, b, c):
    """FP64 torch closest point on triangle abc for query p [N,3] — the
    Ericson region-based algorithm; regions selected on detached values,
    per-region formula smooth in (p, a, b, c)."""
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = (ab * ap).sum(1)
    d2 = (ac * ap).sum(1)
    bp = p - b
    d3 = (ab * bp).sum(1)
    d4 = (ac * bp).sum(1)
    cp = p - c
    d5 = (ab * cp).sum(1)
    d6 = (ac * cp).sum(1)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    eps = 0.0
    # region masks (detached)
    with torch.no_grad():
        m_a = (d1 <= eps) & (d2 <= eps)
        m_b = (d3 >= -eps) & (d4 <= d3)
        m_c = (d6 >= -eps) & (d5 <= d6)
        m_ab = (~m_a) & (~m_b) & (vc <= eps) & (d1 >= -eps) & (d3 <= eps)
        m_ac = (~m_a) & (~m_c) & (vb <= eps) & (d2 >= -eps) & (d6 <= eps)
        m_bc = (~m_b) & (~m_c) & (va <= eps) & ((d4 - d3) >= -eps) & ((d5 - d6) >= -eps)
        m_in = ~(m_a | m_b | m_c | m_ab | m_ac | m_bc)
    y = torch.empty_like(p)
    y[m_a] = a[m_a]
    y[m_b] = b[m_b]
    y[m_c] = c[m_c]
    if m_ab.any():
        v = (d1 / (d1 - d3)).unsqueeze(1)
        y[m_ab] = (a + v * ab)[m_ab]
    if m_ac.any():
        w = (d2 / (d2 - d6)).unsqueeze(1)
        y[m_ac] = (a + w * ac)[m_ac]
    if m_bc.any():
        w = ((d4 - d3) / ((d4 - d3) + (d5 - d6))).unsqueeze(1)
        y[m_bc] = (b + w * (c - b))[m_bc]
    if m_in.any():
        denom = (va + vb + vc).unsqueeze(1)
        v = vb.unsqueeze(1) / denom
        w = vc.unsqueeze(1) / denom
        y[m_in] = (a + v * ab + w * ac)[m_in]
    return y


class TriMeshOracle(SDFOracle):
    near_eikonal = False

    def __init__(self, verts: np.ndarray, tris: np.ndarray, device="cuda:0"):
        self.verts = torch.tensor(np.asarray(verts, np.float64))
        self.tris = np.ascontiguousarray(tris, np.int32)
        self.dim = 3
        self._device = device
        self._wp_mesh = wp.Mesh(
            points=wp.array(np.asarray(verts, np.float32), dtype=wp.vec3,
                            device=device),
            indices=wp.array(self.tris.ravel(), dtype=wp.int32, device=device))

    @property
    def params(self):
        return [self.verts]

    def update_vertices(self, new_verts) -> None:
        """Geometry-update contract (evaluation finding 4, CONFIRMED):
        the fp32 warp BVH is built once at construction, while self.verts
        is exposed as a differentiable parameter — an optimizer mutating
        verts in place would leave the BVH candidates/signs STALE. All
        in-place vertex updates must go through here: refreshes the warp
        mesh points and refits the BVH in one epoch."""
        import torch as _t
        nv = (new_verts.detach() if isinstance(new_verts, _t.Tensor)
              else _t.tensor(np.asarray(new_verts, np.float64)))
        with _t.no_grad():
            self.verts.copy_(nv)
        self._wp_mesh.points.assign(
            wp.array(nv.numpy().astype(np.float32), dtype=wp.vec3,
                     device=self._device))
        self._wp_mesh.refit()

    def _query(self, pts: np.ndarray):
        """(sign[N], face[N]) from the fp32 BVH — candidates only."""
        d = self._device
        pd = wp.array(np.asarray(pts, np.float32), dtype=wp.vec3, device=d)
        sign = wp.zeros(len(pts), dtype=wp.float64, device=d)
        face = wp.zeros(len(pts), dtype=wp.int32, device=d)
        k = _make_query_kernel()
        wp.launch(k, dim=len(pts), inputs=[self._wp_mesh.id, pd, sign, face],
                  device=d)
        return sign.numpy(), face.numpy()

    def _closest_torch(self, pts: np.ndarray):
        """(y, sign) with y graph-connected to self.verts."""
        sign, face = self._query(pts)
        p = torch.tensor(np.asarray(pts, np.float64))
        tri = self.tris[face]                                  # [N, 3]
        a = self.verts[tri[:, 0]]
        b = self.verts[tri[:, 1]]
        c = self.verts[tri[:, 2]]
        y = _closest_point_on_triangle(p, a, b, c)
        return y, torch.tensor(sign, dtype=torch.float64), p

    def psi(self, x: torch.Tensor) -> torch.Tensor:
        y, sign, p = self._closest_torch(x.detach().numpy())
        return sign * (x - y).norm(dim=1)

    def classify(self, pts: np.ndarray) -> np.ndarray:
        y, sign, p = self._closest_torch(pts)
        with torch.no_grad():
            return (sign * (p - y).norm(dim=1)).numpy()

    def distance_vector(self, pts: np.ndarray):
        d_t, n_t, ok = self.distance_torch(pts)
        return d_t.detach().numpy(), n_t.detach().numpy(), ok

    def distance_torch(self, pts: np.ndarray):
        """(d, n_grad, ok) torch, graph-connected to verts. n_grad =
        sign (x - y)/|x - y| — the direction of increasing psi."""
        y, sign, p = self._closest_torch(pts)
        d = y - p
        dist = d.norm(dim=1, keepdim=True).clamp_min(1e-300)
        n = sign.unsqueeze(1) * (p - y) / dist
        ok = np.ones(len(pts), dtype=bool)
        return d, n, ok


def icosphere(n_sub: int, center, radius: float, device="cuda:0"):
    """Subdivided icosahedron on a sphere — procedural STL stand-in."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], np.float64)
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]], np.int64)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    for _ in range(n_sub):
        edge_mid = {}
        new_faces = []
        vlist = list(verts)

        def midpoint(i, j):
            key = (min(i, j), max(i, j))
            if key not in edge_mid:
                m = vlist[i] + vlist[j]
                m /= np.linalg.norm(m)
                edge_mid[key] = len(vlist)
                vlist.append(m)
            return edge_mid[key]

        for f in faces:
            a, b, c = int(f[0]), int(f[1]), int(f[2])
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        verts = np.array(vlist)
        faces = np.array(new_faces, np.int64)
    verts = np.asarray(center, np.float64) + radius * verts
    return TriMeshOracle(verts, faces, device=device)
