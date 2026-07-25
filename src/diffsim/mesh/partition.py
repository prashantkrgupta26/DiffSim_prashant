"""Slab partition + halo index maps for distributed solvers.

Partition a uniform grid with axis-0-slowest lexicographic numbering into
contiguous row-block slabs for distributed solvers (NCCL, AMGX per rank).

Global node id = i0*ny*nz + i1*nz + i2, where axis 0 is slowest-varying.
Each rank owns a contiguous slab of axis-0 rows; ghosts are 1-plane halos
from neighbors (7-point stencil).
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=False)
class SlabPart:
    """Partition metadata for one rank's subdomain slab.

    Attributes
    ----------
    rank : int
        Rank index (0 <= rank < n_ranks).
    owned : np.ndarray
        1-D int64 array of global node ids this rank owns (contiguous axis-0 slab).
        For rank r with axis-0 range [i0_start, i0_stop), contains all ny*nz nodes
        for each i0 in that range, in lexicographic order (slowest first).
    ghost : np.ndarray
        1-D int64 array of global ids of the ghost planes (halo nodes).
        For interior ranks: union of the last plane of rank r-1 and first plane of rank r+1.
        For boundary ranks: only the interior-side plane.
        For n_ranks=1: empty array.
    send : dict[int, np.ndarray]
        For each neighbor rank, the owned global ids that neighbor needs (its ghost).
        Key = neighbor rank integer.
    recv : dict[int, np.ndarray]
        For each neighbor rank, the ghost global ids this rank receives from it.
        Key = neighbor rank integer.
    """
    rank: int
    owned: np.ndarray
    ghost: np.ndarray
    send: dict[int, np.ndarray]
    recv: dict[int, np.ndarray]

    def local_index(self, global_ids: np.ndarray) -> np.ndarray:
        """Map global node ids to local indices within (owned + ghost).

        Owned nodes map to [0, n_owned).
        Ghost nodes map to [n_owned, n_owned + n_ghost).

        Parameters
        ----------
        global_ids : np.ndarray
            1-D int64 array of global node ids to map.

        Returns
        -------
        np.ndarray
            1-D int64 array of local indices.

        Raises
        ------
        ValueError
            If any global id is not in owned ∪ ghost.
        """
        n_owned = len(self.owned)
        n_ghost = len(self.ghost)
        local = np.empty(len(global_ids), dtype=np.int64)

        # Create inverse maps: global_id -> local index
        owned_map = {gid: i for i, gid in enumerate(self.owned)}
        ghost_map = {gid: n_owned + i for i, gid in enumerate(self.ghost)}

        # Map each global id
        for i, gid in enumerate(global_ids):
            if gid in owned_map:
                local[i] = owned_map[gid]
            elif gid in ghost_map:
                local[i] = ghost_map[gid]
            else:
                raise ValueError(
                    f"Global id {gid} not in owned ∪ ghost for rank {self.rank}"
                )

        return local


def slab_partition(dims: tuple[int, int, int], n_ranks: int) -> list[SlabPart]:
    """Partition a uniform grid into contiguous axis-0 slabs.

    Partition a uniform grid with dims=(nx, ny, nz) nodes (where axis 0 is
    slowest-varying in lexicographic order) into n_ranks contiguous slabs
    along axis 0. Each rank owns floor(nx/n_ranks) or floor(nx/n_ranks)+1
    complete axis-0 rows; the last rank takes the remainder.

    Parameters
    ----------
    dims : tuple[int, int, int]
        Grid dimensions (nx, ny, nz). Total nodes = nx*ny*nz.
    n_ranks : int
        Number of ranks. Must satisfy 1 <= n_ranks <= nx.

    Returns
    -------
    list[SlabPart]
        One SlabPart per rank, with owned/ghost/send/recv indexed.

    Raises
    ------
    AssertionError
        If n_ranks > nx.
    """
    nx, ny, nz = dims
    assert n_ranks <= nx, f"n_ranks={n_ranks} > nx={nx}"

    # Compute per-rank slab boundaries (axis-0 rows)
    rows_per_rank = nx // n_ranks
    partition_list = []

    # Compute i0 ranges for each rank
    i0_ranges = []
    for r in range(n_ranks):
        i0_start = r * rows_per_rank
        if r == n_ranks - 1:
            i0_stop = nx  # Last rank gets remainder
        else:
            i0_stop = (r + 1) * rows_per_rank
        i0_ranges.append((i0_start, i0_stop))

    # Helper: convert (i0, i1, i2) to global node id (axis 0 slowest)
    def to_global_id(i0: int, i1: int, i2: int) -> int:
        return i0 * ny * nz + i1 * nz + i2

    # Helper: get all global ids for a plane at axis-0 index i0
    def plane_ids(i0: int) -> np.ndarray:
        ids = np.zeros(ny * nz, dtype=np.int64)
        for i1 in range(ny):
            for i2 in range(nz):
                ids[i1 * nz + i2] = to_global_id(i0, i1, i2)
        return ids

    # Build partition for each rank
    for r in range(n_ranks):
        i0_start, i0_stop = i0_ranges[r]

        # Owned: all nodes in this rank's axis-0 range
        owned_ids = []
        for i0 in range(i0_start, i0_stop):
            owned_ids.extend(plane_ids(i0))
        owned = np.array(owned_ids, dtype=np.int64)

        # Ghost: adjacent planes from neighbors
        ghost_ids = []
        send_dict = {}
        recv_dict = {}

        # Lower neighbor (rank r-1): take their last plane
        if r > 0:
            prev_i0_start, prev_i0_stop = i0_ranges[r - 1]
            ghost_plane = plane_ids(prev_i0_stop - 1)
            ghost_ids.extend(ghost_plane)
            # recv from rank r-1: their last plane
            recv_dict[r - 1] = ghost_plane
            # send to rank r-1: our first plane
            send_dict[r - 1] = plane_ids(i0_start)

        # Upper neighbor (rank r+1): take their first plane
        if r < n_ranks - 1:
            next_i0_start, next_i0_stop = i0_ranges[r + 1]
            ghost_plane = plane_ids(next_i0_start)
            ghost_ids.extend(ghost_plane)
            # recv from rank r+1: their first plane
            recv_dict[r + 1] = ghost_plane
            # send to rank r+1: our last plane
            send_dict[r + 1] = plane_ids(i0_stop - 1)

        ghost = np.array(ghost_ids, dtype=np.int64)

        partition_list.append(
            SlabPart(
                rank=r,
                owned=owned,
                ghost=ghost,
                send=send_dict,
                recv=recv_dict,
            )
        )

    return partition_list


def extract_local_block(K_global_csr, part: SlabPart):
    """Extract a rank's local operator block from a global CSR matrix.

    Given the global (n x n) CSR matrix and a SlabPart, return the local
    operator with ``n_owned`` rows and ``n_owned + n_ghost`` columns.  Each
    column global id appearing in an owned row is remapped to its local index
    via ``part.local_index``.

    The 1-plane halo (``part.ghost``) covers the full p1 hex 27-point stencil
    because: interior node ``(i0, i1, i2)`` couples to all
    ``(i0+di0, i1+di1, i2+di2)`` with ``di0 in {-1, 0, 1}`` — axis-0 changes
    by at most 1, so the ±1 axis-0 planes are exactly what the halo provides.
    Axis-1 and axis-2 changes stay within the slab's owned nodes.

    Parameters
    ----------
    K_global_csr : scipy.sparse.csr_matrix
        Square (n x n) global stiffness matrix in CSR format.
    part : SlabPart
        Partition metadata for one rank.

    Returns
    -------
    scipy.sparse.csr_matrix
        Local matrix of shape ``(n_owned, n_owned + n_ghost)`` in fp64 CSR.
        Column j corresponds to ``(owned ++ ghost)[j]`` (local flat index).

    Raises
    ------
    AssertionError
        If any column global id appearing in an owned row is NOT in
        ``owned ∪ ghost`` (i.e., the halo is insufficient).
    """
    import scipy.sparse

    owned_ids = part.owned          # global ids, shape (n_owned,)
    ghost_ids = part.ghost          # global ids, shape (n_ghost,)
    n_owned = len(owned_ids)
    n_ghost = len(ghost_ids)
    n_local = n_owned + n_ghost

    # Build inverse map: global id -> local column index
    local_col_map = {}
    for loc, gid in enumerate(owned_ids):
        local_col_map[int(gid)] = loc
    for loc, gid in enumerate(ghost_ids):
        local_col_map[int(gid)] = n_owned + loc

    # Slice owned rows from the global CSR
    # K_global_csr[owned_ids, :] extracts rows in one shot
    owned_row_block = K_global_csr[owned_ids, :]  # shape (n_owned, n_global)

    # Build the local CSR by remapping column indices
    src = owned_row_block.tocsr()
    src.sort_indices()

    rows_out, cols_out, vals_out = [], [], []
    for loc_row in range(n_owned):
        row_start = src.indptr[loc_row]
        row_end   = src.indptr[loc_row + 1]
        for ptr in range(row_start, row_end):
            gcol = int(src.indices[ptr])
            val  = float(src.data[ptr])
            assert gcol in local_col_map, (
                f"rank {part.rank}: owned row {loc_row} (gid={owned_ids[loc_row]}) "
                f"has column gid={gcol} not in owned∪ghost — halo too narrow"
            )
            rows_out.append(loc_row)
            cols_out.append(local_col_map[gcol])
            vals_out.append(val)

    A_local = scipy.sparse.csr_matrix(
        (np.array(vals_out, dtype=np.float64),
         (np.array(rows_out, dtype=np.int32),
          np.array(cols_out, dtype=np.int32))),
        shape=(n_owned, n_local),
    )
    return A_local
