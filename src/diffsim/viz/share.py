"""ParaView state-file (.pvsm) templates (Phase 1, spec §7).

`paraview_state(vtu_path, physics=...)` emits a minimal, valid ParaView state
XML that (a) declares an XMLUnstructuredGridReader pointing at the VTU by
relative filename, (b) sets the color array to a physics-appropriate default,
and (c) optionally adds a camera.  The template is a pure XML string — no
ParaView install is needed to generate it; a collaborator opens the .pvsm +
VTU together in desktop ParaView and lands on the intended scene.

Companion web path (Glance, spec §7) needs no code beyond the exporter; see
docs/dev/how-to-view-in-glance.md.
"""
import os
import pathlib
from typing import Union


# Per-physics default color field + intended filter (documented in the state
# header comment; Phase-1 templates stay minimal-but-valid).
_PHYSICS_DEFAULTS = {
    "film": {"color_by": "phi_p", "filter": "slice"},
    "xdd": {"color_by": "n", "filter": "none"},
    "ns": {"color_by": "velocity_magnitude", "filter": "lic"},
}


_PVSM_TEMPLATE = """<?xml version="1.0"?>
<ParaView version="5.10">
  <!-- DiffSim viz Phase-1 state template. physics="{physics}",
       intended filter="{filter}". Open this .pvsm together with the VTU
       (referenced below by relative filename) in desktop ParaView. -->
  <ServerManagerState>
    <Proxy group="sources" type="XMLUnstructuredGridReader" id="1001">
      <Property name="FileName" number_of_elements="1">
        <Element index="0" value="{vtu_name}"/>
      </Property>
    </Proxy>
    <Proxy group="representations" type="GeometryRepresentation" id="2001">
      <Property name="ColorArrayName" number_of_elements="2">
        <Element index="0" value="POINTS"/>
        <Element index="1" value="{color_by}"/>
      </Property>
    </Proxy>
{camera_section}  </ServerManagerState>
</ParaView>
"""


def _camera_xml(camera: dict) -> str:
    pos = camera.get("position", [0.0, 0.0, 1.0])
    focal = camera.get("focal_point", [0.0, 0.0, 0.0])
    up = camera.get("view_up", [0.0, 1.0, 0.0])

    def _vec(name, v):
        elems = "".join(
            f'        <Element index="{i}" value="{c}"/>\n'
            for i, c in enumerate(v))
        return (f'      <Property name="{name}" '
                f'number_of_elements="{len(v)}">\n{elems}'
                f'      </Property>\n')

    body = _vec("CameraPosition", pos) + _vec("CameraFocalPoint", focal) \
        + _vec("CameraViewUp", up)
    return ('    <Proxy group="views" type="RenderView" id="3001">\n'
            f'{body}    </Proxy>\n')


def paraview_state(
    vtu_path: Union[str, os.PathLike],
    *,
    physics: str = "film",
    color_by: str = None,
    camera: dict = None,
    output: Union[str, os.PathLike] = None,
) -> str:
    """Generate a ParaView state (.pvsm) template for the given VTU.

    Parameterized by physics type to pre-select the color field:
      film: color by phi_p           (intended slice on element_size)
      xdd:  color by n (density)      (JV overlay is a downstream annotation)
      ns:   color by velocity_magnitude (intended LIC surface + slice)

    The .pvsm references the VTU by relative filename (basename) so the
    collaborator can move both files together.  Returns the XML string; writes
    to `output` if given.
    """
    vtu_path = pathlib.Path(vtu_path)
    defaults = _PHYSICS_DEFAULTS.get(physics, _PHYSICS_DEFAULTS["film"])
    cb = color_by or defaults["color_by"]
    cam_xml = _camera_xml(camera) if camera else ""
    pvsm = _PVSM_TEMPLATE.format(
        vtu_name=vtu_path.name,
        color_by=cb,
        camera_section=cam_xml,
        physics=physics,
        filter=defaults["filter"],
    )
    if output is not None:
        pathlib.Path(output).write_text(pvsm)
    return pvsm
