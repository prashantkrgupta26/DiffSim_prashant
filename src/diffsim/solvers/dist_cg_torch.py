"""Torch-distributed communicator for the NCCL-CG proof (Task 3a/3b).

Provides ``TorchDistComm`` — a drop-in ``Comm`` implementation backed by
``torch.distributed``.  Backend-agnostic: works under **gloo** (CPU) for
Task 3a validation and **nccl** (GPU) for Task 3b without any code change.

Also provides ``build_halo_index_tensors`` to translate a ``SlabPart``'s
global send/recv maps into contiguous local-index tensors for the halo
exchange, and ``PartitionedSpmv`` for building the partitioned 7-point
Poisson SpMV from a ``SlabPart``.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.distributed as dist

from diffsim.mesh.partition import SlabPart
from diffsim.solvers.dist_cg import Comm


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------

def build_halo_index_tensors(
    part: SlabPart,
    device: torch.device,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Build local-index tensors for halo send/recv from a SlabPart.

    Translates the global-id-based ``part.send`` / ``part.recv`` dicts
    into local-index tensors (into the rank's owned+ghost flat buffer).

    Parameters
    ----------
    part : SlabPart
        Partition metadata for this rank.
    device : torch.device
        Target device for the index tensors.

    Returns
    -------
    send_idx : dict[int, torch.Tensor]
        For each neighbor rank, a 1-D int64 tensor of LOCAL indices in the
        owned+ghost buffer that this rank sends to that neighbor.
    recv_idx : dict[int, torch.Tensor]
        For each neighbor rank, a 1-D int64 tensor of LOCAL indices in the
        owned+ghost buffer where ghost data from that neighbor is written.
    """
    send_idx: dict[int, torch.Tensor] = {}
    recv_idx: dict[int, torch.Tensor] = {}

    for nbr, global_ids in part.send.items():
        local = part.local_index(global_ids)
        send_idx[nbr] = torch.tensor(local, dtype=torch.int64, device=device)

    for nbr, global_ids in part.recv.items():
        local = part.local_index(global_ids)
        recv_idx[nbr] = torch.tensor(local, dtype=torch.int64, device=device)

    return send_idx, recv_idx


# ---------------------------------------------------------------------------
# TorchDistComm
# ---------------------------------------------------------------------------

class TorchDistComm:
    """Distributed communicator using ``torch.distributed``.

    Implements the ``Comm`` protocol used by ``pcg()``.  Works with any
    backend supported by ``torch.distributed`` — use **gloo** for CPU
    validation (Task 3a) and **nccl** for GPU runs (Task 3b).

    Parameters
    ----------
    part : SlabPart
        This rank's partition metadata (owned + ghost + send/recv maps).
    device : torch.device
        Device for all tensors.  Must be consistent across ranks.
    dtype : torch.dtype, optional
        Floating-point dtype; defaults to ``torch.float64`` (fp64).

    Notes on ``exchange_halo``
    --------------------------
    The caller passes a *padded* flat vector of length
    ``n_owned + n_ghost``; the owned entries (``[:n_owned]``) must be
    populated; ghost entries (``[n_owned:]``) are filled by this call.

    Internally, each rank posts non-blocking isend/irecv pairs
    (``torch.distributed.batch_isend_irecv``) for every neighbor, then
    waits on all handles before returning.  The send buffer holds
    ``local_vec[send_idx[nbr]]`` and the recv result is scattered into
    ``local_vec[recv_idx[nbr]]``.

    Notes on ``allreduce_sum``
    --------------------------
    Wraps the scalar in a 1-element tensor, calls ``dist.all_reduce`` with
    ``ReduceOp.SUM``, and returns the result as a Python float.
    """

    def __init__(
        self,
        part: SlabPart,
        device: torch.device,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self._part = part
        self._device = device
        self._dtype = dtype

        # Pre-compute local index tensors for send/recv
        self._send_idx, self._recv_idx = build_halo_index_tensors(part, device)

        # Pre-allocate send buffers (reused each halo exchange)
        self._send_bufs: dict[int, torch.Tensor] = {
            nbr: torch.empty(idx.numel(), dtype=dtype, device=device)
            for nbr, idx in self._send_idx.items()
        }

    # ------------------------------------------------------------------
    # Comm protocol implementation
    # ------------------------------------------------------------------

    def allreduce_sum(self, x: float) -> float:
        """All-reduce a Python float by SUM across all ranks."""
        t = torch.tensor([float(x)], dtype=self._dtype, device=self._device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return float(t.item())

    def exchange_halo(self, local_vec: torch.Tensor) -> None:
        """Fill ghost entries of *local_vec* from owning ranks in-place.

        Parameters
        ----------
        local_vec : torch.Tensor, shape (n_owned + n_ghost,)
            Flat owned+ghost buffer.  Owned entries ``[:n_owned]`` must be
            current; ghost entries ``[n_owned:]`` are written by this call.
        """
        if not self._send_idx:
            # No neighbors (world_size==1 or boundary with no ghosts)
            return

        ops = []

        # Post sends first (gather values into pre-allocated send bufs)
        for nbr, send_idx in self._send_idx.items():
            buf = self._send_bufs[nbr]
            buf.copy_(local_vec[send_idx])
            ops.append(dist.P2POp(dist.isend, buf, nbr))

        # Post recvs (we write directly into local_vec at ghost positions)
        # Allocate temporary recv buffers to avoid aliasing issues
        recv_bufs: dict[int, torch.Tensor] = {}
        for nbr, recv_idx in self._recv_idx.items():
            rbuf = torch.empty(recv_idx.numel(), dtype=self._dtype, device=self._device)
            recv_bufs[nbr] = rbuf
            ops.append(dist.P2POp(dist.irecv, rbuf, nbr))

        # Launch all sends and recvs together
        handles = dist.batch_isend_irecv(ops)

        # Wait for completion
        for h in handles:
            h.wait()

        # Scatter received data into ghost slots
        for nbr, recv_idx in self._recv_idx.items():
            local_vec[recv_idx] = recv_bufs[nbr]
