"""A small sequential network with user-defined checkpoint segment boundaries.

The network is an ordinary ``nn.Sequential``-style stack of ``Linear`` layers,
non-linear activations and ``Dropout``.  ``segments`` selects where the
checkpoint boundaries fall.  Boundaries may be placed *between* any two
modules; cutting through a module is not meaningful here.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Union

import torch
from torch import nn

from .checkpoint import checkpoint_segment


SegmentsSpec = Union[int, Sequence[int]]


def normalize_segments(num_modules: int, segments: SegmentsSpec) -> list[tuple[int, int]]:
    """Turn the user spec into half-open module index ranges ``[start, end)``.

    Accepted forms
    --------------
    * ``int k`` (1 <= k <= num_modules): the stack is split into ``k``
      contiguous, as-even-as-possible segments (the first segments carry the
      remainder when num_modules is not divisible by k).
    * sequence of interior cut points ``c`` (1 <= c < num_modules, strictly
      increasing): a segment boundary is inserted before module ``c``.
    """
    if num_modules <= 0:
        raise ValueError("cannot segment an empty module list")

    if isinstance(segments, int) and not isinstance(segments, bool):
        k = segments
        if k < 1 or k > num_modules:
            raise ValueError(
                f"segment count must be in [1, {num_modules}], got {k}"
            )
        base, rem = divmod(num_modules, k)
        # leading segments take the remainder
        cuts = [base * i + min(i, rem) for i in range(1, k)]
    elif isinstance(segments, Iterable):
        cuts = list(segments)
        if not cuts:
            raise ValueError("segment cut-point list must be non-empty "
                             "(use segments=1 for a single segment)")
        if any(not isinstance(c, int) or isinstance(c, bool) for c in cuts):
            raise TypeError("segment cut points must be ints")
        if any(c < 1 or c >= num_modules for c in cuts):
            raise ValueError(
                f"cut points must lie in [1, {num_modules - 1}], got {cuts}"
            )
        if cuts != sorted(set(cuts)):
            raise ValueError(f"cut points must be strictly increasing: {cuts}")
    else:
        raise TypeError(
            "segments must be an int or a sequence of int cut points, got "
            f"{type(segments)!r}"
        )

    bounds = [0, *cuts, num_modules]
    return list(zip(bounds[:-1], bounds[1:]))


class CheckpointedSequential(nn.Module):
    """Sequential stack whose forward is split into recomputing segments.

    Parameters
    ----------
    *modules:
        ``nn.Module`` children (Linear / activations / Dropout / any pure
        module).  Modules may be repeated to share parameters; the same module
        instance can appear in several segments.
    segments:
        See :func:`normalize_segments`.
    preserve_rng_state:
        Forwarded to every segment; leave True when dropout is present.
    """

    def __init__(
        self,
        *modules: nn.Module,
        segments: SegmentsSpec = 1,
        preserve_rng_state: bool = True,
    ):
        super().__init__()
        if not modules:
            raise ValueError("at least one module is required")
        self.layers = nn.ModuleList(modules)
        self.segment_ranges = normalize_segments(len(modules), segments)
        self.preserve_rng_state = preserve_rng_state

    @property
    def module_list(self) -> list[nn.Module]:
        return list(self.layers)

    def _make_runner(self, start: int, end: int):
        mods = self.module_list[start:end]

        def runner(x, _params):
            for module in mods:
                x = module(x)
            return x

        return runner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mods = self.module_list
        # Version of every boundary at the moment it was produced. A later
        # segment must not mutate an earlier boundary in place.
        boundary_versions: list[tuple[torch.Tensor, int]] = []
        for seg_index, (start, end) in enumerate(self.segment_ranges):
            # Register the incoming boundary before running this segment, then
            # verify no earlier/current boundary was mutated in place.
            boundary_versions.append((x, x._version))
            self._check_boundary_versions(boundary_versions)
            runner = self._make_runner(start, end)
            params = [p for m in mods[start:end] for p in m.parameters()]
            x = checkpoint_segment(
                runner,
                x,
                params,
                preserve_rng_state=self.preserve_rng_state,
            )
            self._check_boundary_versions(boundary_versions)
        return x

    @staticmethod
    def _check_boundary_versions(boundary_versions):
        for tensor, version in boundary_versions:
            if tensor._version != version:
                raise RuntimeError(
                    "a saved boundary activation was modified in place after "
                    "being produced; checkpointing cannot recompute it. Move "
                    "the in-place op or avoid mutating activations shared "
                    "across segment boundaries."
                )

    def num_saved_boundaries(self) -> int:
        """Number of boundary activations retained for one forward call."""
        return len(self.segment_ranges)
