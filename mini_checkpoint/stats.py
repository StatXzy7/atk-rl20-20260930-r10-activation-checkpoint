"""Memory proxy: count *real saved tensors*, distinguishing storage from refs.

These numbers are a deterministic **proxy** for activation memory.  They are
NOT the device peak memory reported by the caching allocator: they ignore
fragmentation, allocator rounding/reserved pools, autograd metadata, workspace
buffers and everything that is not an explicitly saved tensor.  Real peak
memory, when measurable, must be reported separately (see ``benchmark.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


@dataclass(frozen=True)
class TensorMemStats:
    num_refs: int          # number of tensor objects counted
    num_storages: int      # number of distinct underlying storages
    unique_bytes: int      # bytes across distinct storages (each counted once)
    ref_bytes: int         # numel * element_size summed over every reference
    num_views: int         # refs sharing storage with another counted ref

    def as_dict(self) -> dict:
        return {
            "num_refs": self.num_refs,
            "num_storages": self.num_storages,
            "unique_bytes": self.unique_bytes,
            "ref_bytes": self.ref_bytes,
            "num_views": self.num_views,
        }


def tensor_mem_stats(tensors: Iterable[torch.Tensor]) -> TensorMemStats:
    """Count references vs. distinct storages.

    Two tensor references that are views / share-slice the same
    ``untyped_storage()`` contribute one storage, but each reference is still
    listed separately.  Two distinct tensors that merely contain equal values
    have distinct storages.  This is exactly the distinction the task asks for:
    repeated references and shared storage must not be double-counted as
    "saved memory", while naive element counting must be shown separately.
    """
    tensor_list = list(tensors)
    storage_first_ref: dict[int, int] = {}
    num_refs = 0
    ref_bytes = 0
    unique_bytes = 0
    num_views = 0
    for t in tensor_list:
        num_refs += 1
        ref_bytes += t.nelement() * t.element_size()
        storage = t.untyped_storage()
        key = (storage.device, storage.data_ptr(), storage.nbytes())
        if key in storage_first_ref:
            num_views += 1
        else:
            storage_first_ref[key] = num_refs - 1
            unique_bytes += storage.nbytes()
    return TensorMemStats(
        num_refs=num_refs,
        num_storages=len(storage_first_ref),
        unique_bytes=unique_bytes,
        ref_bytes=ref_bytes,
        num_views=num_views,
    )


class EagerActivationCollector:
    """Collect every module output in a plain eager sequential forward.

    This is what the autograd graph keeps alive for an un-checkpointed network:
    all intermediate activations (the inputs/outputs a backward may need).
    """

    def __init__(self, modules: Iterable[nn.Module]):
        self.modules = list(modules)
        self.activations: list[torch.Tensor] = []
        self._handles = []

    def __enter__(self):
        def hook(_module, _inputs, output):
            if isinstance(output, torch.Tensor):
                self.activations.append(output)
            elif isinstance(output, (tuple, list)):
                self.activations.extend(o for o in output
                                        if isinstance(o, torch.Tensor))

        for m in self.modules:
            self._handles.append(m.register_forward_hook(hook))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
