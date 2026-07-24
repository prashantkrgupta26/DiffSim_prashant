"""Octree-mesh -> VTK Unstructured Grid exporter (Phase 1, spec S4).

The single data path for the viz module: everything else (renders, share,
plots) consumes the .vtu this writes. `meshio` is imported inside the function
bodies so `import diffsim.viz` never fails without the [viz] extra installed.

Node ordering note (load-bearing): `Mesh.conn_of[p]` lists each element's nodes
in the DiffSim x-fastest tensor lattice (`a = sum_i idx_i * (p+1)^i`, axis 0 =
x fastest; see mesh/nodes.py). VTK's Quad/Hexahedron/BiQuadraticQuad/
TriquadraticHexahedron expect their own canonical vertex order, so we apply an
explicit lattice->VTK permutation per (dim, p) before writing.
"""
import os
import pathlib
import warnings
from typing import Union

import numpy as np


# VTK cell types for (dim, p) pairs:
#   Quad (4), BiQuadQuad (9), Hex (8), TriquadHex (27)
_VTK_TYPE = {(2, 1): 9, (2, 2): 28, (3, 1): 12, (3, 2): 29}

# meshio cell-block names for the same (dim, p) pairs.
_MESHIO_NAME = {
    (2, 1): "quad",
    (2, 2): "quad9",
    (3, 1): "hexahedron",
    (3, 2): "hexahedron27",
}


def _lattice_to_vtk(dim: int, p: int) -> np.ndarray:
    """Permutation mapping DiffSim x-fastest lattice index -> VTK vertex index.

    Returns an int array `perm` of length (p+1)^dim such that
    `conn_of[p][:, perm]` is in VTK's canonical node order.  Built by inverting
    VTK's own ordering:  vtk_order[k] gives the lattice index of VTK vertex k,
    so `perm = vtk_order` (conn[:, vtk_order] picks lattice node for each VTK
    slot).  Supports the four (dim, p) combinations the octree emits.
    """
    npe = p + 1

    def lat(coord):
        # lattice index: a = sum_i coord_i * npe^i, axis 0 (x) fastest.
        a = 0
        for i, c in enumerate(coord):
            a += c * (npe ** i)
        return a

    if dim == 2 and p == 1:
        # VTK_QUAD: CCW corners.
        corners = [(0, 0), (1, 0), (1, 1), (0, 1)]
        return np.array([lat(c) for c in corners], np.int64)

    if dim == 2 and p == 2:
        # VTK_BIQUADRATIC_QUAD (28): 4 corners, 4 edge mids, 1 center.
        pts = [(0, 0), (2, 0), (2, 2), (0, 2),           # corners
               (1, 0), (2, 1), (1, 2), (0, 1),           # edge midpoints
               (1, 1)]                                   # center
        return np.array([lat(c) for c in pts], np.int64)

    if dim == 3 and p == 1:
        # VTK_HEXAHEDRON: bottom CCW then top CCW.
        corners = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                   (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
        return np.array([lat(c) for c in corners], np.int64)

    if dim == 3 and p == 2:
        # VTK_TRIQUADRATIC_HEXAHEDRON (29), 27 nodes. Coordinates on the
        # doubled 0/1/2 grid, in VTK's canonical order:
        #   8 corners, 12 edge midpoints, 6 face centers, 1 body center.
        corners = [(0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0),
                   (0, 0, 2), (2, 0, 2), (2, 2, 2), (0, 2, 2)]
        bottom_edges = [(1, 0, 0), (2, 1, 0), (1, 2, 0), (0, 1, 0)]
        top_edges = [(1, 0, 2), (2, 1, 2), (1, 2, 2), (0, 1, 2)]
        vert_edges = [(0, 0, 1), (2, 0, 1), (2, 2, 1), (0, 2, 1)]
        # VTK face-center order: -x, +x, -y, +y, -z, +z
        faces = [(0, 1, 1), (2, 1, 1), (1, 0, 1), (1, 2, 1),
                 (1, 1, 0), (1, 1, 2)]
        body = [(1, 1, 1)]
        pts = corners + bottom_edges + top_edges + vert_edges + faces + body
        return np.array([lat(c) for c in pts], np.int64)

    raise ValueError(f"unsupported (dim, p) = ({dim}, {p})")


def export_vtu(
    source,
    path: Union[str, os.PathLike],
    *,
    fields: dict = None,
    time_series: bool = False,
) -> pathlib.Path:
    """Convert a DiffSim field source to a VTK Unstructured Grid (.vtu).

    Source types:
      - Mesh object + fields dict: full octree cell topology (the good path,
        required for the mesh-slice figures).  Pass the Mesh as `source` and a
        dict of {name: ndarray(Nn,) or (Nn, 3)} as `fields`.
      - Path string/Path to a .npz file: degrades to a point-cloud VTU (no cell
        topology, so no element_size/level cell data).  A warning is emitted;
        fine for field contours, NOT for mesh-slice figures.

    Writes point data (nodal arrays) and cell data (element_size, level).
    Returns the written .vtu path.
    """
    try:
        import meshio
    except ImportError as e:
        raise ImportError(
            "export_vtu requires 'meshio'. Install with: pip install diffsim[viz]"
        ) from e

    path = pathlib.Path(path)

    # -- Resolve source --------------------------------------------------------
    if isinstance(source, (str, os.PathLike)):
        npz = np.load(str(source))
        coords = np.asarray(npz["coords"], dtype=np.float64)
        loaded_fields = {k: npz[k] for k in npz.files
                         if k not in ("coords", "h", "t")}
        warnings.warn(
            "export_vtu received an .npz path: writing a point-cloud VTU "
            "(no octree cell topology). Cell data (element_size, level) and "
            "mesh-slice figures require passing the Mesh object instead.",
            stacklevel=2,
        )
        _write_meshio_npz_only(coords, loaded_fields, path, meshio)
        return path

    # source is a Mesh object
    mesh = source
    dim = mesh.dim
    coords = np.asarray(mesh.node_coords, dtype=np.float64)   # [Nn, dim]
    if coords.shape[1] == 2:
        # meshio / VTK points are 3-D; pad 2-D coords with z = 0.
        coords = np.column_stack([coords, np.zeros(len(coords))])
    h_elem = mesh.tree.h()             # [Ne]
    levels = mesh.tree.levels          # uint8 [Ne]

    # -- Build per-p cell blocks (with lattice->VTK reordering) ---------------
    cells = []
    element_size_cells = []
    level_cells = []

    for pv in sorted(mesh.bins.keys()):
        eids = mesh.bins[pv]           # element indices for this p
        conn = mesh.conn_of[pv]        # int32 [nb, (p+1)^dim] x-fastest lattice
        perm = _lattice_to_vtk(dim, int(pv))
        conn_vtk = np.ascontiguousarray(conn[:, perm])
        cells.append((_MESHIO_NAME[(dim, int(pv))], conn_vtk))
        element_size_cells.append(h_elem[eids].astype(np.float64))
        level_cells.append(levels[eids].astype(np.int32))

    # -- Point data ------------------------------------------------------------
    point_data = {}
    if fields is not None:
        for k, v in fields.items():
            point_data[k] = np.asarray(v, dtype=np.float64)
        if "phi_p" in fields and "phi_f" in fields:
            point_data["phi_s"] = (
                1.0 - point_data["phi_p"] - point_data["phi_f"])

    # -- Cell data (meshio expects one array per cell-block, in a list) --------
    cell_data = {
        "element_size": element_size_cells,
        "level": level_cells,
    }

    m = meshio.Mesh(
        points=coords,
        cells=cells,
        point_data=point_data,
        cell_data=cell_data,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    meshio.write(str(path), m)

    if time_series:
        _write_pvd(path, [path])

    return path


def export_vtu_sbm(
    mesh,
    path: Union[str, os.PathLike],
    *,
    fields: dict = None,
    retained_tree,
    frac: np.ndarray,
    geom=None,
    sf=None,
) -> pathlib.Path:
    """`export_vtu` extended with SBM cell data (spec S4 carve-out story).

    Adds three cell-data arrays on top of the base VTU:
      element_type  int32 [Ne]: 0=interior (frac==1), 1=intercepted
                    (0<frac<1), 2=surrogate (element in ``sf.elem``; overrides
                    0/1 -- the SBM-active element wins the label).
      frac_in       float64 [Ne]: domain-inside fraction from classify_lambda.
      d_mean        float64 [Ne]: mean ||d|| over the surrogate GPs on that
                    element (from GeometryData.d); 0.0 on non-surrogate cells.

    The three arrays are indexed by the retained-tree element ordering, which
    is exactly `mesh.bins`' element ordering (mesh is built from retained_tree).
    """
    try:
        import meshio
    except ImportError as e:
        raise ImportError(
            "export_vtu_sbm requires 'meshio'. Install with: pip install diffsim[viz]"
        ) from e

    frac = np.asarray(frac, dtype=np.float64)
    ne = len(retained_tree)
    if len(frac) != ne:
        raise ValueError(
            f"frac has length {len(frac)} but retained_tree has {ne} elements")

    # -- element_type ---------------------------------------------------------
    elem_type = np.zeros(ne, dtype=np.int32)
    elem_type[frac < 1.0] = 1
    if sf is not None:
        elem_type[np.unique(sf.elem)] = 2

    # -- d_mean per element ---------------------------------------------------
    d_mean = np.zeros(ne, dtype=np.float64)
    if sf is not None and geom is not None:
        d = np.asarray(geom.d, dtype=np.float64)      # [Nf*nqf, dim]
        nf = len(sf.elem)
        if nf > 0:
            nqf = d.shape[0] // nf
            dnorm = np.linalg.norm(d, axis=1).reshape(nf, nqf)   # [Nf, nqf]
            per_face = dnorm.mean(axis=1)                        # [Nf]
            # average over all surrogate faces belonging to each element
            accum = np.zeros(ne, dtype=np.float64)
            count = np.zeros(ne, dtype=np.float64)
            np.add.at(accum, sf.elem, per_face)
            np.add.at(count, sf.elem, 1.0)
            nz = count > 0
            d_mean[nz] = accum[nz] / count[nz]

    # -- write the base VTU, then splice in the three cell arrays -------------
    export_vtu(mesh, path, fields=fields)
    m = meshio.read(str(path))

    # meshio cell_data is per-block; map the element-ordered arrays onto blocks
    # using the same bins ordering export_vtu used.
    et_blocks, fr_blocks, dm_blocks = [], [], []
    for pv in sorted(mesh.bins.keys()):
        eids = mesh.bins[pv]
        et_blocks.append(elem_type[eids])
        fr_blocks.append(frac[eids])
        dm_blocks.append(d_mean[eids])
    m.cell_data["element_type"] = et_blocks
    m.cell_data["frac_in"] = fr_blocks
    m.cell_data["d_mean"] = dm_blocks

    meshio.write(str(path), m)
    return pathlib.Path(path)


def export_body_vtp(
    vertices: np.ndarray,
    triangles: np.ndarray,
    path: Union[str, os.PathLike],
) -> pathlib.Path:
    """Write the embedded-body surface as a VTK PolyData (.vtp) file."""
    try:
        import meshio
    except ImportError as e:
        raise ImportError(
            "export_body_vtp requires 'meshio'. Install with: pip install diffsim[viz]"
        ) from e

    path = pathlib.Path(path)
    verts = np.asarray(vertices, dtype=np.float64)
    if verts.shape[1] == 2:
        verts = np.column_stack([verts, np.zeros(len(verts))])
    tris = np.asarray(triangles, dtype=np.int32)
    m = meshio.Mesh(points=verts, cells=[("triangle", tris)])
    path.parent.mkdir(parents=True, exist_ok=True)
    meshio.write(str(path), m)
    return path


def _write_meshio_npz_only(coords, loaded_fields, path, meshio):
    """Fallback: write a point-cloud VTU when no Mesh connectivity is given."""
    coords = np.asarray(coords, dtype=np.float64)
    if coords.shape[1] == 2:
        coords = np.column_stack([coords, np.zeros(len(coords))])
    verts = np.arange(len(coords)).reshape(-1, 1)
    pdata = {k: np.asarray(v, dtype=np.float64) for k, v in loaded_fields.items()}
    if "phi_p" in pdata and "phi_f" in pdata:
        pdata["phi_s"] = 1.0 - pdata["phi_p"] - pdata["phi_f"]
    m = meshio.Mesh(points=coords, cells=[("vertex", verts)], point_data=pdata)
    path.parent.mkdir(parents=True, exist_ok=True)
    meshio.write(str(path), m)


def _write_pvd(vtu_path: pathlib.Path, files: list) -> pathlib.Path:
    pvd = vtu_path.with_suffix(".pvd")
    lines = ['<?xml version="1.0"?>',
             '<VTKFile type="Collection" version="0.1">',
             '  <Collection>']
    for i, f in enumerate(files):
        lines.append(
            f'    <DataSet timestep="{i}" file="{pathlib.Path(f).name}"/>')
    lines += ['  </Collection>', '</VTKFile>']
    pvd.write_text("\n".join(lines))
    return pvd
