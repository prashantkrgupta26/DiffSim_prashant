"""Ahmed reference body (Ahmed et al. 1984) for the solver-escalation
campaign — clean-geometry rung between snapshot lab and the truck gate.

Standard proportions at unit body length L=1: W=389/1044, H=288/1044,
front-edge radius R=100/1044, slant length 222/1044 at ``slant_deg``.
DELIBERATE deviations, recorded per the spec: (1) NO support stilts
(sub-resolution features are the T5 lesson); (2) front rounding is applied
in the x-y profile only (2.5-D extrusion, flat sides) rather than 3-D
fillets — adequate for the solver-survival rung; noted for the production
accuracy phase.
"""
import numpy as np

_W = 389.0 / 1044.0
_H = 288.0 / 1044.0
_R = 100.0 / 1044.0
_SLANT_LEN = 222.0 / 1044.0


def ahmed_profile(slant_deg=25.0, front_radius=_R, n_arc=16):
    """Closed convex CCW (x, y) polygon of the side profile, unit length.

    x: 0 (front) -> 1 (rear); y: 0 (bottom) -> H (roof)."""
    r = float(front_radius)
    th = np.deg2rad(float(slant_deg))
    dx, dy = _SLANT_LEN * np.cos(th), _SLANT_LEN * np.sin(th)
    pts = []
    # front-bottom arc: (0, r) -> (r, 0), center (r, r)
    for a in np.linspace(np.pi, 1.5 * np.pi, n_arc + 1):
        pts.append((r + r * np.cos(a), r + r * np.sin(a)))
    pts.append((1.0, 0.0))            # bottom rear
    pts.append((1.0, _H - dy))        # rear face top (below slant)
    pts.append((1.0 - dx, _H))        # slant leading edge on the roof
    # front-top arc: (r, H) -> (0, H - r), center (r, H - r)
    for a in np.linspace(0.5 * np.pi, np.pi, n_arc + 1):
        pts.append((r + r * np.cos(a), _H - r + r * np.sin(a)))
    return np.asarray(pts, np.float64)


def ahmed_verts_tris(length=0.06, slant_deg=25.0, n_arc=16):
    """Watertight triangulated Ahmed body, body-local coords, min corner
    at the origin; returns (verts[N,3] float64, tris[M,3] int64)."""
    prof = ahmed_profile(slant_deg=slant_deg, n_arc=n_arc) * float(length)
    m = len(prof)
    w = _W * float(length)
    # z=0 verts: indices 0..m-1; z=w verts: indices m..2m-1
    verts = np.vstack([np.column_stack([prof, np.zeros(m)]),
                       np.column_stack([prof, np.full(m, w)])])
    tris = []
    # Side walls: profile is CCW in (x,y); for each CCW edge i->j the outward
    # normal points to the right of the direction of travel (away from interior).
    # Correct outward winding: (i, j, m+j), (i, m+j, m+i).
    for i in range(m):
        j = (i + 1) % m
        tris.append((i, j, m + j))
        tris.append((i, m + j, m + i))
    # z=0 cap: outward normal -z; viewed from outside (-z direction) CCW.
    # Profile CCW in xy => reverse fan for -z outward normal: (0, i+1, i).
    # z=w cap: outward normal +z; viewed from outside (+z direction) CCW.
    # Profile CCW in xy => fan (m, m+i, m+i+1).
    for i in range(1, m - 1):
        tris.append((0, i + 1, i))              # z=0 cap, normal -z
        tris.append((m, m + i, m + i + 1))      # z=w cap, normal +z
    verts = np.ascontiguousarray(verts, np.float64)
    tris = np.asarray(tris, np.int64)
    # Orientation guard: enclosed volume must be positive (outward normals).
    a, b_, c = (verts[tris[:, i]] for i in range(3))
    vol = float(np.einsum("ij,ij->i", a, np.cross(b_, c)).sum() / 6.0)
    if vol < 0:
        tris = tris[:, [0, 2, 1]]
    return verts, tris


def ahmed_merged(length=0.06, clearance=0.003, x_front=0.32,
                 band_level=11, slant_deg=25.0):
    """Ahmed placed in the unit channel [0,1] x [0,1/8] x [0,1/8].

    clearance is the ground gap; it must be resolvable (>= 4 cells at the
    band level) — the T5 sub-resolution lesson, enforced here."""
    h_band = 2.0 ** (-int(band_level))
    assert clearance >= h_band, (
        f"Ahmed clearance {clearance} < 1 cell at band level {band_level} "
        f"(h={h_band}) — unresolvable gap (T5 lesson)")
    from diffsim.geometry.merged_trimesh import MergedTriMesh
    verts, tris = ahmed_verts_tris(length=length, slant_deg=slant_deg)
    w = verts[:, 2].max()
    verts = verts + np.asarray(
        [float(x_front), float(clearance), 0.0625 - w / 2.0], np.float64)
    return MergedTriMesh(verts, tris)


def write_ahmed_stl(path, **kwargs):
    """Binary STL of the Ahmed body (for the render gate)."""
    import struct
    verts, tris = ahmed_verts_tris(**kwargs)
    with open(path, "wb") as f:
        f.write(b"\0" * 80)
        f.write(struct.pack("<I", len(tris)))
        for t in tris:
            a, b_, c = verts[t[0]], verts[t[1]], verts[t[2]]
            n = np.cross(b_ - a, c - a)
            nn = np.linalg.norm(n)
            n = n / nn if nn > 0 else n
            f.write(struct.pack("<3f", *n))
            for p in (a, b_, c):
                f.write(struct.pack("<3f", *p))
            f.write(struct.pack("<H", 0))
    return path
