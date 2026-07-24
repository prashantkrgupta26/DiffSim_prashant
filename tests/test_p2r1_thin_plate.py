"""P2-R1 in-CI gate: 2-D thin-plate blocked-channel two-sided shell surrogate.

Thin wrapper over the driver ``p2r1_thin_plate_blocked_channel.run_gate``: at a
small level it asserts the two-sided shell (a) BLOCKS the flow (zero downstream
through-flow), (b) develops a pressure JUMP across the plate, (c) recovers a net
plate force within tolerance of a carved-out thin-rectangle reference on the
same excluded band, and (d) the two-sided coupling is LOAD-BEARING (dropping the
loaded side collapses both the force and the jump). tier5 (assembles + splu-
solves a small 2-D monolithic system)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from p2r1_thin_plate_blocked_channel import run_gate

pytestmark = pytest.mark.tier5


def test_thin_plate_blocked_channel_two_sided_shell():
    res = run_gate(level=5, steps=45, verbose=False)
    v = res["verdict"]
    assert v["blocked"], res["shell"]              # zero downstream through-flow
    assert v["jump_ok"], res["shell"]              # a real pressure jump
    assert v["ref_ok"], (res["shell"], res["carved"])   # matches carved ref
    assert v["loadbearing"], (res["shell"], res["one_sided"])  # coupling bites
    assert v["overall"] == "PASS", res
