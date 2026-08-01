"""STL-in-domain render smoke test (offscreen pyvista)."""
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))


def _write_box_stl(path):
    """Minimal binary STL: unit cube scaled to 0.05, 12 triangles."""
    v = np.array([[x, y, z] for x in (0, 1) for y in (0, 1)
                  for z in (0, 1)], np.float32) * 0.05
    faces = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5),
             (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6),
             (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    with open(path, "wb") as f:
        f.write(b"\0" * 80)
        f.write(struct.pack("<I", len(faces)))
        for a, b, c in faces:
            f.write(struct.pack("<3f", 0, 0, 0))
            for i in (a, b, c):
                f.write(struct.pack("<3f", *v[i]))
            f.write(struct.pack("<H", 0))


def test_render_stl_in_domain(tmp_path):
    from render_mesh import render_stl_in_domain
    stl = tmp_path / "box.stl"
    _write_box_stl(stl)
    prefix = str(tmp_path / "box_dom")
    render_stl_in_domain(str(stl), prefix, position=(0.3, 0.0, 0.04))
    for tag in ("iso", "side", "front"):
        p = tmp_path / f"box_dom_{tag}.png"
        assert p.exists() and p.stat().st_size > 1000
