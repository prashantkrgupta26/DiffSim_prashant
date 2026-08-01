"""MergedTriMesh — ONE merged triangle soup for the TRUCK case (no SDF
formalism; task TU2 amendment 1).

The truck geometry is the UNION of many STL bodies.  Rather than a per-body
oracle + CSG-union routing, we CONCATENATE every body's (verts, tris) into one
triangle soup — applying each body's position offset and a single isotropic
scale into the octree unit cube — and build ONE ``wp.Mesh`` BVH over it.

In/out for the carve is ``wp.mesh_query_point_sign_winding_number`` on the
merged soup (winding number treats the whole soup as a single closed union,
including overlapping bodies, with no explicit boolean routing).  Surrogate
shift vectors are the EXACT closest-point-on-triangle from the same BVH query
(FP64 recompute on the winning triangle), mirroring
``diffsim.geometry.trimesh.TriMeshOracle``.

The exposed surface is oracle-compatible:
  * ``classify(pts) -> signed distance`` (negative INSIDE the solid), so it
    plugs straight into ``classify_lambda(tree, mesh, lam, domain="outside")``.
  * ``distance_vector(pts, y0=None) -> (d, n_grad, ok)`` for the volumetric
    ``GeometryData.evaluate`` surrogate cache.
  * ``distance_torch`` for autograd-connected variants.
  * ``query_batch(points) -> (dist, dvec, normal, sign)`` (task convenience).
"""
import numpy as np
import torch
import warp as wp

from .trimesh import _make_query_kernel, _closest_point_on_triangle
from ..device import default_device


def _read_stl(path: str):
    """Return (verts[Nv,3] float64, tris[Nt,3] int64) from an STL file.

    Uses meshio when available; otherwise falls back to a minimal binary-STL
    reader (80-byte header + uint32 triangle count + 50-byte records).  The
    binary reader de-duplicates vertices so wp.Mesh gets a compact index set.
    """
    try:
        import meshio
        m = meshio.read(path)
        verts = np.ascontiguousarray(m.points, np.float64)
        tris = None
        for cb in m.cells:
            if cb.type == "triangle":
                tris = np.ascontiguousarray(cb.data, np.int64)
                break
        if tris is None:
            raise ValueError(f"no triangle cells in {path}")
        return verts, tris
    except ImportError:
        return _read_binary_stl(path)


def _read_binary_stl(path: str):
    """Minimal binary-STL reader (no meshio dependency)."""
    with open(path, "rb") as fh:
        fh.read(80)                                   # header
        (ntri,) = np.frombuffer(fh.read(4), np.uint32)
        rec = np.frombuffer(fh.read(int(ntri) * 50), np.uint8).reshape(ntri, 50)
    # each 50-byte record: 12B normal + 3*12B verts + 2B attr
    coords = rec[:, 12:48].copy().view(np.float32).reshape(ntri, 3, 3)
    flat = coords.reshape(-1, 3).astype(np.float64)
    uniq, inv = np.unique(np.round(flat, 8), axis=0, return_inverse=True)
    tris = inv.reshape(ntri, 3).astype(np.int64)
    return np.ascontiguousarray(uniq), np.ascontiguousarray(tris)


class MergedTriMesh:
    """Merged triangle-soup oracle over the union of many STL bodies.

    Parameters
    ----------
    verts : (Nv, 3) float64
        Concatenated vertices (already offset + scaled into the working frame).
    tris : (Nt, 3) int64
        Triangles indexing ``verts``.
    device : optional
        Warp device; defaults to ``default_device()``.
    """
    near_eikonal = False

    def __init__(self, verts, tris, device=None):
        device = default_device() if device is None else device
        verts = np.ascontiguousarray(verts, np.float64)
        tris = np.ascontiguousarray(tris)
        if verts.ndim != 2 or verts.shape[1] != 3:
            raise ValueError(f"verts must be [Nv,3], got {verts.shape}")
        if tris.ndim != 2 or tris.shape[1] != 3 or len(tris) == 0:
            raise ValueError(f"tris must be non-empty [Nt,3], got {tris.shape}")
        if tris.min() < 0 or tris.max() >= len(verts):
            raise ValueError("triangle indices out of vertex range")
        self.verts = torch.tensor(verts)
        self.tris = np.ascontiguousarray(tris, np.int32)
        self.dim = 3
        self._device = device
        self._wp_mesh = wp.Mesh(
            points=wp.array(np.asarray(verts, np.float32), dtype=wp.vec3,
                            device=device),
            indices=wp.array(self.tris.ravel(), dtype=wp.int32, device=device))

    # ---- constructors ------------------------------------------------------

    @classmethod
    def from_bodies(cls, bodies, config_dir, scale=1.0, device=None,
                    stl_paths=None):
        """Build from a list of ``BodySpec`` (config loader).

        Each body's STL is read from ``config_dir/body.mesh_path`` (or the
        override in ``stl_paths[i]``), translated by ``body.position``, then the
        whole soup is multiplied by the isotropic ``scale`` (physical -> unit
        cube).  Vertex index bases are offset so the concatenation is one soup.
        """
        import os
        all_v, all_t, base = [], [], 0
        for i, b in enumerate(bodies):
            path = (stl_paths[i] if stl_paths is not None
                    else os.path.join(config_dir, b.mesh_path))
            v, t = _read_stl(path)
            v = (v + np.asarray(b.position, np.float64)) * float(scale)
            all_v.append(v)
            all_t.append(t + base)
            base += len(v)
        verts = np.concatenate(all_v, axis=0)
        tris = np.concatenate(all_t, axis=0)
        return cls(verts, tris, device=device)

    @property
    def params(self):
        return [self.verts]

    # ---- BVH candidate query ----------------------------------------------

    def _query(self, pts):
        """(sign[N], face[N]) from the fp32 BVH — candidates only."""
        d = self._device
        pts = np.ascontiguousarray(pts, np.float32)
        pd = wp.array(pts, dtype=wp.vec3, device=d)
        sign = wp.zeros(len(pts), dtype=wp.float64, device=d)
        face = wp.zeros(len(pts), dtype=wp.int32, device=d)
        k = _make_query_kernel()
        wp.launch(k, dim=len(pts), inputs=[self._wp_mesh.id, pd, sign, face],
                  device=d)
        return sign.numpy(), face.numpy()

    def _closest_torch(self, pts):
        """(y, sign, p) with y graph-connected to self.verts."""
        sign, face = self._query(pts)
        p = torch.tensor(np.ascontiguousarray(pts, np.float64))
        tri = self.tris[face]
        a = self.verts[tri[:, 0]]
        b = self.verts[tri[:, 1]]
        c = self.verts[tri[:, 2]]
        y = _closest_point_on_triangle(p, a, b, c)
        return y, torch.tensor(sign, dtype=torch.float64), p

    # ---- oracle surface ----------------------------------------------------

    def psi(self, x):
        """Signed distance (negative inside) as a torch tensor."""
        y, sign, p = self._closest_torch(x.detach().numpy())
        return sign * (x - y).norm(dim=1)

    def classify(self, pts):
        """Signed distance (negative INSIDE the merged solid), numpy [N].

        Winding-number sign is negative inside (matches TriMeshOracle), so
        domain='outside' in classify_lambda selects the fluid around the truck.
        """
        y, sign, p = self._closest_torch(pts)
        with torch.no_grad():
            return (sign * (p - y).norm(dim=1)).numpy()

    def distance_torch(self, pts):
        """(d, n_grad, ok) torch, graph-connected to verts.  n_grad =
        sign*(x - y)/|x - y| — the direction of increasing psi."""
        y, sign, p = self._closest_torch(pts)
        d = y - p
        dist = d.norm(dim=1, keepdim=True).clamp_min(1e-300)
        n = sign.unsqueeze(1) * (p - y) / dist
        ok = np.ones(len(pts), dtype=bool)
        return d, n, ok

    def distance_vector(self, pts, y0=None):
        """(d, n_grad, ok) numpy.  y0 ignored (BVH selects candidates)."""
        d_t, n_t, ok = self.distance_torch(pts)
        return d_t.detach().numpy(), n_t.detach().numpy(), ok

    def query_batch(self, points):
        """(dist, dvec, normal, sign) numpy for a batch of query points.

        dist   : [N]   signed distance (negative inside)
        dvec   : [N,3] shift vector query -> closest point on Gamma (y - x)
        normal : [N,3] unit boundary normal in the +psi direction
        sign   : [N]   winding sign (+1 outside, -1 inside)
        """
        y, sign, p = self._closest_torch(points)
        with torch.no_grad():
            dvec = (y - p)
            dist = sign * dvec.norm(dim=1)
            nrm = dvec.norm(dim=1, keepdim=True).clamp_min(1e-300)
            normal = sign.unsqueeze(1) * (p - y) / nrm
            return (dist.numpy(), dvec.numpy(), normal.numpy(), sign.numpy())
