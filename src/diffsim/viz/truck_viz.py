"""Truck-case per-frame visualization hook + Q-criterion computation.

Animation extracts (per viz_interval step):
  - Q-criterion isosurface .vtp (ROI-clipped, Q > Q_thresh)
  - Centerline (y-mid / z-mid) slice .vtu or .vtp
  - Truck-surface Cp .vtp (pressure coefficient on the merged STL triangles)

Full .vtu checkpoints at sparser checkpoint_interval.
Time-averaged (u, p) accumulators updated every viz step.

Q-criterion: Q = 0.5*(||Omega||^2 - ||S||^2) where:
  S_ij = 0.5*(du_i/dx_j + du_j/dx_i)   (symmetric rate-of-strain)
  Omega_ij = 0.5*(du_i/dx_j - du_j/dx_i) (antisymmetric vorticity)
  Q > 0 selects vortex cores (rotation-dominated).

Implementation: element-gradient averaging on the host (CPU numpy):
  1. For each element e and Gauss point q, compute grad u [dim x dim] from
     dN tables (dN[q,a,d] * 2/h_e = d phi_a / d x_d in physical coords).
  2. Sum to nodal accumulators: Q_node[i] += sum_q w_q * Q_gp / (n_elements at i).
  3. Output: nodal Q array [N_nodes].

Cp = (p - p_ref) / (0.5 * rho * U^2) with p_ref = 0 (pin), rho=1, U=1.
Surface pressure sampled at surrogate-face centroids, mapped to STL triangles
via MergedTriMesh.query_batch closest-point (approximate nearest-face binning).

ROI clip: box [x0,x1,y0,y1,z0,z1] in unit-cube coords.  Default = full domain
(no clip).  Clip is applied to centroid of each element/triangle.
"""
from __future__ import annotations

import pathlib
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Q-criterion on the host (element-gradient averaging)
# ---------------------------------------------------------------------------

def nodal_q_criterion(mesh, u_node: np.ndarray) -> np.ndarray:
    """Compute nodal Q-criterion from velocity node array.

    Parameters
    ----------
    mesh : Mesh
        DiffSim Mesh with .conn_of, .bins, .tree, .node_coords attributes.
    u_node : np.ndarray [N_nodes, 3]
        Velocity at all (full) nodes — NOT free DOFs.  Shape must be (Nn, 3).

    Returns
    -------
    Q_node : np.ndarray [N_nodes]
        Q-criterion at each node, averaged from surrounding elements.
        Q = 0.5 * (||Omega||^2 - ||S||^2) where:
          S_ij = 0.5*(grad_u_ij + grad_u_ji)   symmetric strain rate
          Omega_ij = 0.5*(grad_u_ij - grad_u_ji) antisymmetric vorticity
        Q > 0 in vortex cores (rotation-dominated).

    Notes
    -----
    Element-gradient route: for each element e with element size h_e,
    the physical gradient at Gauss point q is:
      grad_u[i,j] = sum_a dN[q,a,j] * (2/h_e) * u_node[conn[e,a], i]
    This matches the gp_grad_vec kernel convention (d*dim+c layout).
    Q is averaged to nodes by accumulating over all GPs in each element
    weighted by Gauss weight w_q, then distributing to the element's nodes.
    """
    from ..mesh.basis import basis_tables

    u_node = np.asarray(u_node, dtype=np.float64)
    dim = mesh.dim
    Nn = len(mesh.node_coords)
    Q_accum = np.zeros(Nn, dtype=np.float64)
    count = np.zeros(Nn, dtype=np.float64)

    h_all = mesh.tree.h()  # [Ne]

    for pv in sorted(mesh.bins.keys()):
        tb = basis_tables(int(pv), dim)  # Tables with .dN [nqp, nbf, dim], .w [nqp]
        eids = mesh.bins[pv]
        conn = mesh.conn_of[pv]  # [Ne_p, nbf]
        h_elem = h_all[eids]     # [Ne_p]
        # u at element nodes: [Ne_p, nbf, dim]
        u_elem = u_node[conn]    # [Ne_p, nbf, dim]

        Ne_p = len(eids)
        nqp = tb.nqp
        nbf = tb.nbf
        dN = tb.dN   # [nqp, nbf, dim]
        w = tb.w     # [nqp]

        # grad_u at each GP: [Ne_p, nqp, dim, dim]
        # grad_u[e,q,i,j] = sum_a dN[q,a,j] * (2/h_e) * u_elem[e,a,i]
        # Using einsum: "qad, ead -> eqd" then multiply by 2/h
        # Result layout: [Ne_p, nqp, dim_j, dim_i] via (q,a,d),(e,a,c)->eqdc
        # grad_u_ij = d u_j / d x_i  (consistent with gp_grad_vec d*dim+c)
        dscale = (2.0 / h_elem)  # [Ne_p]
        # grad_u[e,q,i,j] = dscale[e] * sum_a dN[q,a,i] * u_elem[e,a,j]
        # einsum: eqij = dscale[e] * sum_a dN[q,a,i] * u_elem[e,a,j]
        grad_u = np.einsum("qai,eaj->eqij", dN, u_elem) * dscale[:, None, None, None]
        # [Ne_p, nqp, dim, dim]

        # Symmetric (S) and antisymmetric (Omega) parts
        # S_ij = 0.5*(grad_u_ij + grad_u_ji)
        # Omega_ij = 0.5*(grad_u_ij - grad_u_ji)
        S = 0.5 * (grad_u + grad_u.transpose(0, 1, 3, 2))
        Omega = 0.5 * (grad_u - grad_u.transpose(0, 1, 3, 2))

        # ||S||^2 = sum_ij S_ij^2,  ||Omega||^2 = sum_ij Omega_ij^2
        S_sq = (S ** 2).sum(axis=(-2, -1))         # [Ne_p, nqp]
        Omega_sq = (Omega ** 2).sum(axis=(-2, -1)) # [Ne_p, nqp]

        # Q at each GP
        Q_gp = 0.5 * (Omega_sq - S_sq)  # [Ne_p, nqp]

        # Weight by Gauss weight, sum over qp to get element-average Q
        Q_elem = (Q_gp * w[None, :]).sum(axis=1) / w.sum()  # [Ne_p]

        # Scatter to nodes (simple scatter-add then normalize by count)
        # Each element contributes its Q to each of its nodes
        for a in range(nbf):
            np.add.at(Q_accum, conn[:, a], Q_elem)
            np.add.at(count, conn[:, a], 1.0)

    nz = count > 0
    Q_node = np.where(nz, Q_accum / count, 0.0)
    return Q_node


# ---------------------------------------------------------------------------
# Surface Cp sampler
# ---------------------------------------------------------------------------

def surface_cp(
    mesh,
    p_node: np.ndarray,
    merged,
    sf,
    *,
    p_ref: float = 0.0,
    rho: float = 1.0,
    U_inf: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute surface pressure coefficient on the merged STL triangles.

    Strategy:
      1. Compute surrogate-face centroids (midpoints of the surrogate face
         bounding boxes in unit-cube coords).
      2. Sample p at those centroids by nodal interpolation (nearest-node).
      3. Map sampled p to STL triangle centroids via MergedTriMesh.query_batch
         closest-point (approximate: the surrogate-face centroid closest to each
         STL triangle centroid).
      4. Cp = (p - p_ref) / (0.5 * rho * U_inf^2).

    Returns
    -------
    tri_centroids : [Nt, 3]  STL triangle centroids (unit-cube coords)
    Cp : [Nt]                pressure coefficient per STL triangle
    tri_cp_map : [Nt]        index of the nearest surrogate face centroid used
    """
    tree = mesh.tree
    from ..sbm.surrogate import face_gauss_points
    from ..mesh.faces import face_tables

    # Surrogate-face centroids: average the face GP positions at face order p=1
    ftab = face_tables(1, mesh.dim)
    xq_all = face_gauss_points(tree, sf, ftab)  # [Nf*nqf, dim]
    nqf = ftab.nqf
    nf = len(sf.elem)
    xq_reshaped = xq_all.reshape(nf, nqf, mesh.dim)
    sf_centroids = xq_reshaped.mean(axis=1)   # [Nf, dim]

    # Sample pressure at surrogate-face centroids by nearest-node lookup
    coords = np.asarray(mesh.node_coords, dtype=np.float64)
    p_node = np.asarray(p_node, dtype=np.float64)

    # For each surrogate-face centroid, find nearest mesh node
    # (at CI scales this is acceptable — O(Nf * Nn) brute-force for small meshes,
    # or use KD-tree for larger ones)
    from scipy.spatial import cKDTree
    kd = cKDTree(coords)
    _, nn_idx = kd.query(sf_centroids)
    p_sf = p_node[nn_idx]  # [Nf]

    # STL triangle centroids
    verts = merged.verts.numpy()                          # [Nv, 3]
    tris = np.asarray(merged.tris, dtype=np.int64)       # [Nt, 3]
    tri_centroids = verts[tris].mean(axis=1)              # [Nt, 3]

    # Map: for each STL triangle centroid, find nearest surrogate-face centroid
    kd_sf = cKDTree(sf_centroids)
    _, tri_sf_idx = kd_sf.query(tri_centroids)
    p_tri = p_sf[tri_sf_idx]   # [Nt]

    denom = 0.5 * rho * U_inf ** 2
    if denom == 0.0:
        raise ValueError(
            f"Cp denominator is zero (0.5*rho*U_inf^2=0); "
            f"got rho={rho!r}, U_inf={U_inf!r}.  "
            "Check that U_inf != 0 and rho != 0 before calling surface_cp."
        )
    Cp = (p_tri - p_ref) / denom

    return tri_centroids, Cp, tri_sf_idx


# ---------------------------------------------------------------------------
# ROI clip helpers
# ---------------------------------------------------------------------------

def _clip_mask_points(pts: np.ndarray, roi: Optional[Tuple]) -> np.ndarray:
    """Boolean mask for points inside roi box [x0,x1,y0,y1,z0,z1] or all-True."""
    if roi is None:
        return np.ones(len(pts), dtype=bool)
    x0, x1, y0, y1, z0, z1 = roi
    return (
        (pts[:, 0] >= x0) & (pts[:, 0] <= x1) &
        (pts[:, 1] >= y0) & (pts[:, 1] <= y1) &
        (pts[:, 2] >= z0) & (pts[:, 2] <= z1)
    )


# ---------------------------------------------------------------------------
# Per-frame extracts
# ---------------------------------------------------------------------------

def write_q_isosurface(
    mesh,
    u_node: np.ndarray,
    path: pathlib.Path,
    *,
    Q_node: Optional[np.ndarray] = None,
    umag_node: Optional[np.ndarray] = None,
    Q_thresh: float = 0.5,
    roi: Optional[Tuple] = None,
) -> Optional[pathlib.Path]:
    """Write Q-isosurface polydata as .vtp.

    Parameters
    ----------
    mesh : Mesh
    u_node : [Nn, 3] velocity at all nodes
    path : output .vtp path
    Q_node : precomputed Q (computed if None)
    umag_node : |u| at nodes (computed from u_node if None)
    Q_thresh : isosurface value (default 0.5; vortex cores)
    roi : optional [x0,x1,y0,y1,z0,z1] clip box

    Returns the written path or None if pyvista is not available / isosurface
    is empty.
    """
    try:
        import pyvista as pv
    except ImportError:
        print("[truck_viz] pyvista not available; skipping Q isosurface", flush=True)
        return None

    if Q_node is None:
        Q_node = nodal_q_criterion(mesh, u_node)
    if umag_node is None:
        umag_node = np.linalg.norm(u_node, axis=1)

    # Build a VTK unstructured grid from the octree mesh (hex8 elements only,
    # p=1 — the truck gate uses p=1 exclusively).
    coords = np.asarray(mesh.node_coords, dtype=np.float64)  # [Nn, 3]
    if coords.shape[1] == 2:
        coords = np.column_stack([coords, np.zeros(len(coords))])

    from ..viz.export import _lattice_to_vtk
    dim = mesh.dim

    all_cells = []
    cell_types = []
    for pv_key in sorted(mesh.bins.keys()):
        conn = mesh.conn_of[pv_key]   # [Ne, nbf]
        perm = _lattice_to_vtk(dim, int(pv_key))
        conn_vtk = conn[:, perm]      # [Ne, nbf]
        # pyvista hex connectivity
        nbf = conn_vtk.shape[1]
        if nbf == 8:
            vtktype = pv.CellType.HEXAHEDRON
        elif nbf == 27:
            vtktype = pv.CellType.TRIQUADRATIC_HEXAHEDRON
        else:
            continue
        for row in conn_vtk:
            all_cells.append([nbf] + list(row))
            cell_types.append(int(vtktype))

    if not all_cells:
        return None

    flat_cells = np.array([v for row in all_cells for v in row], dtype=np.int64)
    cell_types_arr = np.array(cell_types, dtype=np.uint8)
    grid = pv.UnstructuredGrid(flat_cells, cell_types_arr, coords)
    grid.point_data["Q"] = Q_node.astype(np.float32)
    grid.point_data["velocity_magnitude"] = umag_node.astype(np.float32)

    # ROI clip
    if roi is not None:
        x0, x1, y0, y1, z0, z1 = roi
        grid = grid.clip_box([x0, x1, y0, y1, z0, z1], invert=False)

    # Isosurface
    try:
        iso = grid.contour([Q_thresh], scalars="Q")
    except Exception:
        iso = None

    if iso is None or iso.n_points == 0:
        print(f"[truck_viz] Q isosurface at thresh={Q_thresh} is empty; "
              "writing empty polydata", flush=True)
        iso = pv.PolyData()

    path.parent.mkdir(parents=True, exist_ok=True)
    iso.save(str(path))
    return path


def write_centerline_slice(
    mesh,
    u_node: np.ndarray,
    p_node: np.ndarray,
    path: pathlib.Path,
    *,
    umag_node: Optional[np.ndarray] = None,
    Q_node: Optional[np.ndarray] = None,
    slice_axis: str = "z",
    roi: Optional[Tuple] = None,
) -> Optional[pathlib.Path]:
    """Write a centerline slice as .vtp (y-mid or z-mid cross-section).

    The slice is taken through the domain center along the specified axis.
    Uses PyVista's slice_orthogonal approach on the hex mesh.

    Returns written path or None.
    """
    try:
        import pyvista as pv
    except ImportError:
        print("[truck_viz] pyvista not available; skipping centerline slice",
              flush=True)
        return None

    if umag_node is None:
        umag_node = np.linalg.norm(u_node, axis=1)
    if Q_node is None:
        Q_node = nodal_q_criterion(mesh, u_node)

    coords = np.asarray(mesh.node_coords, dtype=np.float64)
    if coords.shape[1] == 2:
        coords = np.column_stack([coords, np.zeros(len(coords))])

    from ..viz.export import _lattice_to_vtk
    dim = mesh.dim

    all_cells = []
    cell_types = []
    for pv_key in sorted(mesh.bins.keys()):
        conn = mesh.conn_of[pv_key]
        perm = _lattice_to_vtk(dim, int(pv_key))
        conn_vtk = conn[:, perm]
        nbf = conn_vtk.shape[1]
        if nbf == 8:
            vtktype = pv.CellType.HEXAHEDRON
        elif nbf == 27:
            vtktype = pv.CellType.TRIQUADRATIC_HEXAHEDRON
        else:
            continue
        for row in conn_vtk:
            all_cells.append([nbf] + list(row))
            cell_types.append(int(vtktype))

    if not all_cells:
        return None

    flat_cells = np.array([v for row in all_cells for v in row], dtype=np.int64)
    cell_types_arr = np.array(cell_types, dtype=np.uint8)
    grid = pv.UnstructuredGrid(flat_cells, cell_types_arr, coords)
    grid.point_data["velocity_magnitude"] = umag_node.astype(np.float32)
    grid.point_data["pressure"] = np.asarray(p_node, dtype=np.float32)
    u_arr = np.asarray(u_node, dtype=np.float32)
    if u_arr.shape[1] == 3:
        grid.point_data["velocity"] = u_arr
    grid.point_data["Q"] = Q_node.astype(np.float32)

    # ROI clip
    if roi is not None:
        x0, x1, y0, y1, z0, z1 = roi
        grid = grid.clip_box([x0, x1, y0, y1, z0, z1], invert=False)

    # Determine slice position (center of domain)
    axis_map = {"x": 0, "y": 1, "z": 2}
    ax = axis_map.get(slice_axis, 2)
    center_val = 0.5 * (coords[:, ax].min() + coords[:, ax].max())

    origin = coords.mean(axis=0).copy()
    origin[ax] = center_val
    normal = np.zeros(3); normal[ax] = 1.0

    try:
        slc = grid.slice(normal=normal.tolist(), origin=origin.tolist())
    except Exception as exc:
        print(f"[truck_viz] slice failed: {exc}", flush=True)
        slc = pv.PolyData()

    path.parent.mkdir(parents=True, exist_ok=True)
    slc.save(str(path))
    return path


def write_truck_surface_cp(
    mesh,
    p_node: np.ndarray,
    merged,
    sf,
    path: pathlib.Path,
    *,
    p_ref: float = 0.0,
    rho: float = 1.0,
    U_inf: float = 1.0,
    roi: Optional[Tuple] = None,
) -> Optional[pathlib.Path]:
    """Write truck-surface Cp as .vtp (one value per STL triangle).

    Returns written path or None.
    """
    try:
        import pyvista as pv
    except ImportError:
        print("[truck_viz] pyvista not available; skipping surface Cp", flush=True)
        return None

    verts = merged.verts.numpy()
    tris = np.asarray(merged.tris, dtype=np.int64)

    tri_centroids, Cp, _ = surface_cp(
        mesh, p_node, merged, sf,
        p_ref=p_ref, rho=rho, U_inf=U_inf)

    # ROI clip by triangle centroid
    if roi is not None:
        mask = _clip_mask_points(tri_centroids, roi)
        tris = tris[mask]
        Cp = Cp[mask]
        tri_centroids = tri_centroids[mask]

    if len(tris) == 0:
        surf = pv.PolyData()
    else:
        # Build polydata
        if verts.shape[1] == 2:
            verts = np.column_stack([verts, np.zeros(len(verts))])
        faces = np.column_stack(
            [np.full(len(tris), 3, np.int64), tris]).ravel()
        surf = pv.PolyData(verts.astype(np.float32), faces)
        surf.cell_data["Cp"] = Cp.astype(np.float32)

    path.parent.mkdir(parents=True, exist_ok=True)
    surf.save(str(path))
    return path


# ---------------------------------------------------------------------------
# Full .vtu checkpoint
# ---------------------------------------------------------------------------

def write_vtu_checkpoint(
    mesh,
    u_node: np.ndarray,
    p_node: np.ndarray,
    path: pathlib.Path,
    *,
    Q_node: Optional[np.ndarray] = None,
) -> pathlib.Path:
    """Write full .vtu checkpoint with u, p, |u|, Q fields."""
    from ..viz.export import export_vtu

    if Q_node is None:
        Q_node = nodal_q_criterion(mesh, u_node)
    umag = np.linalg.norm(u_node, axis=1)

    fields = {
        "velocity": u_node,
        "pressure": p_node,
        "velocity_magnitude": umag,
        "Q": Q_node,
    }
    return export_vtu(mesh, path, fields=fields)


# ---------------------------------------------------------------------------
# VizHook — wired into the truck driver march loop
# ---------------------------------------------------------------------------

class TruckVizHook:
    """Per-step viz callback for the truck driver march loop.

    Parameters
    ----------
    mesh : Mesh
    cons : Constraints
    merged : MergedTriMesh
    sf : SurrogateFaces
    viz_dir : pathlib.Path
        Root directory for all extracts.
    viz_interval : int
        Write frame extracts every this many steps.
    checkpoint_interval : int
        Write full .vtu every this many steps (should be > viz_interval).
    Q_thresh : float
        Q isosurface threshold (default 0.5).
    roi : tuple or None
        [x0,x1,y0,y1,z0,z1] ROI clip in unit-cube coords.
    U_inf : float
        Free-stream velocity for Cp normalization (default 1.0).
    """

    def __init__(
        self,
        mesh,
        cons,
        merged,
        sf,
        viz_dir: pathlib.Path,
        *,
        viz_interval: int = 1,
        checkpoint_interval: int = 10,
        Q_thresh: float = 0.5,
        roi: Optional[Tuple] = None,
        U_inf: float = 1.0,
    ):
        self.mesh = mesh
        self.cons = cons
        self.merged = merged
        self.sf = sf
        self.viz_dir = pathlib.Path(viz_dir)
        self.viz_interval = viz_interval
        self.checkpoint_interval = checkpoint_interval
        self.Q_thresh = Q_thresh
        self.roi = roi
        self.U_inf = U_inf
        self.T = cons.T.tocsr()

        # Time-averaged accumulators
        dim = mesh.dim
        Nn = len(mesh.node_coords)
        self._u_avg = np.zeros((Nn, dim), dtype=np.float64)
        self._p_avg = np.zeros(Nn, dtype=np.float64)
        self._n_avg = 0

        # Written paths log
        self.written: list = []

        # Create output dirs
        for sub in ("frames/q_iso", "frames/centerline", "frames/surface_cp",
                    "checkpoints", "time_avg"):
            (self.viz_dir / sub).mkdir(parents=True, exist_ok=True)

    def _to_full(self, x_free: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert free DOF vector [nfree * ndof] -> (u_node [Nn,3], p_node [Nn])."""
        T = self.T
        dim = self.mesh.dim
        ndof = dim + 1
        # x_free is [nfree * ndof]; reshape to [nfree, ndof]
        x_f = x_free.reshape(-1, ndof)
        u_free = x_f[:, :dim]   # [nfree, dim]
        p_free = x_f[:, dim]    # [nfree]
        u_full = np.asarray(T @ u_free)  # [Nn, dim]
        p_full = np.asarray(T @ p_free)  # [Nn]
        return u_full, p_full

    def __call__(self, step: int, info: dict):
        """Called at the end of each march step.

        Parameters
        ----------
        step : int  (0-indexed)
        info : dict with at least 'x' key (free DOF vector [nfree*ndof])
        """
        x_free = info.get("x")
        if x_free is None:
            return

        u_full, p_full = self._to_full(x_free)

        # Time-average accumulator
        self._u_avg += u_full
        self._p_avg += p_full
        self._n_avg += 1

        do_frame = ((step + 1) % self.viz_interval == 0)
        do_checkpoint = ((step + 1) % self.checkpoint_interval == 0)

        if do_frame:
            frame_idx = (step + 1) // self.viz_interval
            fname = f"frame_{frame_idx:06d}"

            # Compute Q once, reuse across frame writes
            Q_node = nodal_q_criterion(self.mesh, u_full)

            # 1. Q isosurface
            q_path = self.viz_dir / "frames" / "q_iso" / f"{fname}.vtp"
            p = write_q_isosurface(
                self.mesh, u_full, q_path,
                Q_node=Q_node,
                Q_thresh=self.Q_thresh,
                roi=self.roi)
            if p:
                self.written.append(("q_iso", step, p))

            # 2. Centerline slice (z-mid: side view)
            sl_path = self.viz_dir / "frames" / "centerline" / f"{fname}.vtp"
            p = write_centerline_slice(
                self.mesh, u_full, p_full, sl_path,
                Q_node=Q_node,
                slice_axis="z",
                roi=self.roi)
            if p:
                self.written.append(("centerline", step, p))

            # 3. Surface Cp
            cp_path = self.viz_dir / "frames" / "surface_cp" / f"{fname}.vtp"
            p = write_truck_surface_cp(
                self.mesh, p_full, self.merged, self.sf, cp_path,
                U_inf=self.U_inf,
                roi=self.roi)
            if p:
                self.written.append(("surface_cp", step, p))

        if do_checkpoint:
            ckpt_path = (self.viz_dir / "checkpoints" /
                         f"step_{step+1:06d}.vtu")
            p = write_vtu_checkpoint(self.mesh, u_full, p_full, ckpt_path)
            self.written.append(("checkpoint", step, p))

    def write_time_average(self):
        """Write time-averaged (u, p) .vtu if any steps accumulated."""
        if self._n_avg == 0:
            return None
        u_avg = self._u_avg / self._n_avg
        p_avg = self._p_avg / self._n_avg
        out = self.viz_dir / "time_avg" / "u_p_avg.vtu"
        return write_vtu_checkpoint(self.mesh, u_avg, p_avg, out)
