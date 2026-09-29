"""Helpers to build identical eager references for comparison."""

from __future__ import annotations

import torch
from torch import nn

from .network import CheckpointedSequential


def clone_network(ckpt_model: CheckpointedSequential) -> nn.Sequential:
    """Plain ``nn.Sequential`` with the same module instances' *values*.

    Modules that are shared (the same instance appearing twice) are cloned as
    separate instances here, which would change semantics; callers that need
    shared parameters should construct both networks from the same factory.
    ``build_pair`` handles that correctly.
    """
    return nn.Sequential(*[m for m in ckpt_model.module_list])


def build_pair(factory, segments, *, preserve_rng_state=True, share_modules=False):
    """Build ``(eager_model, ckpt_model)`` with equal initial parameters.

    ``factory(shared: bool)`` returns a list of modules.  When ``share_modules``
    is True the *same* module instances back both networks (so a reused module
    is genuinely shared and identical in both paths).
    """
    modules = factory(shared=share_modules)
    if share_modules:
        eager = nn.Sequential(*modules)
        ckpt = CheckpointedSequential(
            *modules, segments=segments, preserve_rng_state=preserve_rng_state
        )
        return eager, ckpt

    eager_modules = [m for m in modules]
    ckpt_modules = []
    with torch.no_grad():
        for m in eager_modules:
            copy = _clone_module(m)
            ckpt_modules.append(copy)
    eager = nn.Sequential(*eager_modules)
    ckpt = CheckpointedSequential(
        *ckpt_modules, segments=segments, preserve_rng_state=preserve_rng_state
    )
    return eager, ckpt


def _clone_module(module: nn.Module) -> nn.Module:
    import copy

    return copy.deepcopy(module)
