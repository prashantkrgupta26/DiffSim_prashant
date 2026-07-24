"""Tests for slab partition and halo index maps.

Tests cover:
- Completeness and disjointness of owned slabs
- Ghost correctness (adjacent planes from neighbors)
- send/recv symmetry between neighbor ranks
- local_index mapping (owned to [0, n_owned), ghost to [n_owned, n_owned+n_ghost))
- Boundary cases (n_ranks=1, unequal grid sizes)
"""
import pytest
import numpy as np
from diffsim.mesh.partition import slab_partition, SlabPart


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(params=[
    (4, 4, 4),
    (8, 4, 4),
    (5, 3, 3),
])
def dims_fixture(request):
    """Parametrized grid dimensions."""
    return request.param


@pytest.fixture(params=[1, 2, 4])
def n_ranks_fixture(request):
    """Parametrized rank counts."""
    return request.param


@pytest.fixture
def partition_fixture(dims_fixture, n_ranks_fixture):
    """Parametrized partitions (dims, n_ranks, partition_list)."""
    if n_ranks_fixture > dims_fixture[0]:
        pytest.skip(f"n_ranks={n_ranks_fixture} > nx={dims_fixture[0]}")
    return dims_fixture, n_ranks_fixture, slab_partition(dims_fixture, n_ranks_fixture)


# ============================================================================
# Tests
# ============================================================================

def test_completeness_and_disjointness(partition_fixture):
    """Owned slabs partition [0, nx*ny*nz) completely and disjointly."""
    dims, n_ranks, parts = partition_fixture
    nx, ny, nz = dims
    total_nodes = nx * ny * nz

    # Collect all owned ids across all ranks
    all_owned = []
    for part in parts:
        all_owned.extend(part.owned)

    all_owned = np.array(all_owned, dtype=np.int64)

    # Check completeness: union covers all nodes
    assert set(all_owned) == set(range(total_nodes)), \
        f"Owned ids do not cover [0, {total_nodes})"

    # Check disjointness: no duplicates
    assert len(all_owned) == len(set(all_owned)), \
        "Owned ids have duplicates"

    # Each rank's owned ids are contiguous (for axis-0 slab structure)
    for r, part in enumerate(parts):
        owned_set = set(part.owned)
        # No id should appear in multiple ranks
        for other_r in range(n_ranks):
            if other_r != r:
                overlap = owned_set & set(parts[other_r].owned)
                assert len(overlap) == 0, \
                    f"Rank {r} and rank {other_r} have overlapping owned ids"


def test_ghost_correctness(partition_fixture):
    """Ghost nodes are exactly the adjacent axis-0 planes of neighbors."""
    dims, n_ranks, parts = partition_fixture
    nx, ny, nz = dims

    def to_global_id(i0: int, i1: int, i2: int) -> int:
        """Compute global node id (axis 0 slowest)."""
        return i0 * ny * nz + i1 * nz + i2

    def plane_ids(i0: int) -> set:
        """Get all global ids for a plane at axis-0 index i0."""
        ids = set()
        for i1 in range(ny):
            for i2 in range(nz):
                ids.add(to_global_id(i0, i1, i2))
        return ids

    # Compute axis-0 ranges for each rank (same logic as slab_partition)
    rows_per_rank = nx // n_ranks
    i0_ranges = []
    for r in range(n_ranks):
        i0_start = r * rows_per_rank
        if r == n_ranks - 1:
            i0_stop = nx
        else:
            i0_stop = (r + 1) * rows_per_rank
        i0_ranges.append((i0_start, i0_stop))

    # For each rank, verify ghost correctness
    for r, part in enumerate(parts):
        i0_start, i0_stop = i0_ranges[r]
        expected_ghost = set()

        # Lower neighbor's last plane
        if r > 0:
            prev_i0_start, prev_i0_stop = i0_ranges[r - 1]
            expected_ghost.update(plane_ids(prev_i0_stop - 1))

        # Upper neighbor's first plane
        if r < n_ranks - 1:
            next_i0_start, next_i0_stop = i0_ranges[r + 1]
            expected_ghost.update(plane_ids(next_i0_start))

        actual_ghost = set(part.ghost)
        assert actual_ghost == expected_ghost, \
            f"Rank {r} ghost mismatch: expected {expected_ghost}, got {actual_ghost}"


def test_send_recv_symmetry(partition_fixture):
    """send/recv are symmetric: rank r sends to nbr iff nbr receives from r."""
    dims, n_ranks, parts = partition_fixture

    for r in range(n_ranks):
        part = parts[r]

        # Check each neighbor in send dict
        for nbr, ids_to_send in part.send.items():
            # Neighbor nbr should have r in its recv dict
            assert r in parts[nbr].recv, \
                f"Rank {r} sends to {nbr}, but {nbr} does not recv from {r}"

            recv_ids = parts[nbr].recv[r]
            assert set(ids_to_send) == set(recv_ids), \
                f"Rank {r} send to {nbr} != rank {nbr} recv from {r}"

        # Check each recv has a corresponding send in the neighbor
        for nbr, ids_to_recv in part.recv.items():
            # Neighbor nbr should have r in its send dict
            assert r in parts[nbr].send, \
                f"Rank {r} recvs from {nbr}, but {nbr} does not send to {r}"

            send_ids = parts[nbr].send[r]
            assert set(ids_to_recv) == set(send_ids), \
                f"Rank {r} recv from {nbr} != rank {nbr} send to {r}"


def test_local_index_owned(partition_fixture):
    """local_index maps owned ids to [0, n_owned)."""
    dims, n_ranks, parts = partition_fixture

    for part in parts:
        n_owned = len(part.owned)
        local_indices = part.local_index(part.owned)

        # Should be in [0, n_owned)
        assert np.all(local_indices >= 0), "local_index result < 0"
        assert np.all(local_indices < n_owned), "local_index result >= n_owned"

        # Should be contiguous and match order
        expected = np.arange(n_owned, dtype=np.int64)
        assert np.array_equal(local_indices, expected), \
            f"local_index(owned) != [0, {n_owned})"


def test_local_index_ghost(partition_fixture):
    """local_index maps ghost ids to [n_owned, n_owned + n_ghost)."""
    dims, n_ranks, parts = partition_fixture

    for part in parts:
        n_owned = len(part.owned)
        n_ghost = len(part.ghost)

        if n_ghost == 0:
            continue  # Skip if no ghosts (e.g., n_ranks=1)

        local_indices = part.local_index(part.ghost)

        # Should be in [n_owned, n_owned + n_ghost)
        assert np.all(local_indices >= n_owned), "local_index(ghost) < n_owned"
        assert np.all(local_indices < n_owned + n_ghost), \
            "local_index(ghost) >= n_owned + n_ghost"

        # Should be contiguous and start at n_owned
        expected = np.arange(n_owned, n_owned + n_ghost, dtype=np.int64)
        assert np.array_equal(local_indices, expected), \
            f"local_index(ghost) != [{n_owned}, {n_owned + n_ghost})"


def test_local_index_round_trip(partition_fixture):
    """local_index round-trip: owned+ghost maps to contiguous local range."""
    dims, n_ranks, parts = partition_fixture

    for part in parts:
        n_owned = len(part.owned)
        n_ghost = len(part.ghost)
        n_local = n_owned + n_ghost

        # Combine owned and ghost
        combined = np.concatenate([part.owned, part.ghost])
        local_indices = part.local_index(combined)

        # Should be [0, 1, 2, ..., n_local-1]
        expected = np.arange(n_local, dtype=np.int64)
        assert np.array_equal(local_indices, expected), \
            f"local_index(owned+ghost) != [0, {n_local})"


def test_local_index_out_of_range(partition_fixture):
    """local_index raises ValueError for ids outside owned ∪ ghost."""
    dims, n_ranks, parts = partition_fixture
    nx, ny, nz = dims
    total_nodes = nx * ny * nz

    for part in parts:
        # Find a node not in owned ∪ ghost
        owned_set = set(part.owned)
        ghost_set = set(part.ghost)
        for candidate in range(total_nodes):
            if candidate not in owned_set and candidate not in ghost_set:
                # Found an out-of-range id
                with pytest.raises(ValueError):
                    part.local_index(np.array([candidate], dtype=np.int64))
                break


def test_n_ranks_1_degenerate(dims_fixture):
    """Special case: n_ranks=1 has all owned, no ghosts, empty send/recv."""
    dims = dims_fixture
    nx, ny, nz = dims

    parts = slab_partition(dims, n_ranks=1)
    assert len(parts) == 1, "n_ranks=1 should produce 1 partition"

    part = parts[0]
    total_nodes = nx * ny * nz

    # Owned should be all nodes
    assert len(part.owned) == total_nodes, \
        f"n_ranks=1 owned should have {total_nodes} nodes"
    assert set(part.owned) == set(range(total_nodes)), \
        "n_ranks=1 owned should cover all nodes"

    # Ghost should be empty
    assert len(part.ghost) == 0, "n_ranks=1 ghost should be empty"

    # send/recv should be empty
    assert len(part.send) == 0, "n_ranks=1 send should be empty"
    assert len(part.recv) == 0, "n_ranks=1 recv should be empty"


def test_boundary_ranks_half_width_halo(dims_fixture):
    """Boundary ranks (first/last) have one-sided halos."""
    # Use n_ranks=2 to test first and last rank
    dims = dims_fixture
    nx, ny, nz = dims

    if nx < 2:
        pytest.skip("Need nx >= 2 for n_ranks=2")

    parts = slab_partition(dims, n_ranks=2)

    # First rank: only has upper neighbor (rank 1)
    first = parts[0]
    assert 0 not in first.send, "First rank should not send to rank -1"
    assert 0 not in first.recv, "First rank should not recv from rank -1"
    assert 1 in first.send or len(first.ghost) == 0, \
        "First rank should send to rank 1 (if has ghosts)"

    # Last rank: only has lower neighbor (rank 0)
    last = parts[1]
    assert 2 not in last.send, "Last rank should not send to rank 2"
    assert 2 not in last.recv, "Last rank should not recv from rank 2"
    assert 0 in last.send or len(last.ghost) == 0, \
        "Last rank should send to rank 0 (if has ghosts)"


def test_interior_rank_two_sided_halo():
    """Interior ranks (with n_ranks >= 3) have two-sided halos."""
    dims = (6, 3, 3)  # 6 rows, enough for 3 ranks
    parts = slab_partition(dims, n_ranks=3)

    # Middle rank (rank 1): should have both neighbors
    middle = parts[1]
    assert 0 in middle.send and 0 in middle.recv, \
        "Middle rank should send/recv with lower neighbor"
    assert 2 in middle.send and 2 in middle.recv, \
        "Middle rank should send/recv with upper neighbor"


def test_ghost_ordering_lower_then_upper():
    """Ghost ids are ordered: lower-rank plane first, then upper-rank plane."""
    dims = (6, 3, 3)
    parts = slab_partition(dims, n_ranks=3)

    middle = parts[1]
    nx, ny, nz = dims
    rows_per_rank = nx // 3

    # Middle rank has i0 range [2, 4)
    # Lower neighbor's last plane is at i0=1
    # Upper neighbor's first plane is at i0=4

    def to_global_id(i0: int, i1: int, i2: int) -> int:
        return i0 * ny * nz + i1 * nz + i2

    # Expected ghost: plane at i0=1, then plane at i0=4
    expected_ghost_lower = set()
    for i1 in range(ny):
        for i2 in range(nz):
            expected_ghost_lower.add(to_global_id(1, i1, i2))

    expected_ghost_upper = set()
    for i1 in range(ny):
        for i2 in range(nz):
            expected_ghost_upper.add(to_global_id(4, i1, i2))

    # Split middle.ghost into lower and upper based on size
    n_plane = ny * nz
    ghost_lower = set(middle.ghost[:n_plane])
    ghost_upper = set(middle.ghost[n_plane:])

    assert ghost_lower == expected_ghost_lower, \
        "Lower ghost plane mismatch"
    assert ghost_upper == expected_ghost_upper, \
        "Upper ghost plane mismatch"


def test_assertion_n_ranks_exceeds_nx():
    """slab_partition raises AssertionError if n_ranks > nx."""
    dims = (3, 4, 4)
    with pytest.raises(AssertionError):
        slab_partition(dims, n_ranks=5)


# ============================================================================
# Summary/integration
# ============================================================================

def test_small_synthetic_2rank():
    """Explicit test on small (4,4,4) grid with 2 ranks."""
    dims = (4, 4, 4)
    parts = slab_partition(dims, n_ranks=2)

    assert len(parts) == 2
    ny, nz = 4, 4
    n_plane = ny * nz

    # Rank 0: rows 0-1 (2 planes * 16 nodes/plane = 32 nodes)
    r0 = parts[0]
    assert len(r0.owned) == 2 * n_plane, "Rank 0 owned size"
    assert len(r0.ghost) == n_plane, "Rank 0 ghost size (1 plane from rank 1)"

    # Rank 1: rows 2-3 (2 planes * 16 nodes/plane = 32 nodes)
    r1 = parts[1]
    assert len(r1.owned) == 2 * n_plane, "Rank 1 owned size"
    assert len(r1.ghost) == n_plane, "Rank 1 ghost size (1 plane from rank 0)"

    # Check send/recv symmetry
    assert set(r0.send[1]) == set(r1.recv[0])
    assert set(r1.send[0]) == set(r0.recv[1])

    # Owned should partition the full set
    full_set = set(r0.owned) | set(r1.owned)
    nx, ny, nz = 4, 4, 4
    assert full_set == set(range(nx * ny * nz)), \
        "Full set should be all 64 nodes"
    assert set(r0.owned) & set(r1.owned) == set(), \
        "Owned sets should be disjoint"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
