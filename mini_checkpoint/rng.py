"""Capture / restore global RNG state for the devices we may draw from.

Checkpoint recomputation must reproduce the *exact* dropout mask used in the
original forward.  We therefore snapshot the CPU generator state and the state
of every already-initialised CUDA device before running a segment, and restore
the snapshot immediately before the recomputation.

After recomputation we restore the state that was current at the *start of the
backward pass*, so the caller's RNG stream advances exactly as it would in a
plain (non-checkpointed) forward/backward: one stream position per random draw,
once.  See README.md "RNG lifecycle" for the full argument.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RNGState:
    cpu: torch.Tensor
    cuda: dict  # device_index -> ByteTensor


def get_rng_state() -> RNGState:
    cuda_states: dict[int, torch.Tensor] = {}
    if torch.cuda.is_available():
        # Only touch devices that have been initialised: querying
        # device_count() / get_device_state is harmless and avoids creating a
        # CUDA context the caller did not have.
        for idx in range(torch.cuda.device_count()):
            if torch.cuda.is_initialized():
                cuda_states[idx] = torch.cuda.get_rng_state(idx)
    return RNGState(cpu=torch.random.get_rng_state(), cuda=cuda_states)


def set_rng_state(state: RNGState) -> None:
    torch.random.set_rng_state(state.cpu)
    if torch.cuda.is_available():
        for idx, tensor in state.cuda.items():
            torch.cuda.set_rng_state(tensor, idx)


@contextmanager
def rng_state(state: RNGState):
    saved = get_rng_state()
    set_rng_state(state)
    try:
        yield
    finally:
        set_rng_state(saved)
