"""Segment-level activation checkpointing, implemented from scratch.

The core recomputation machinery is a custom ``torch.autograd.Function``
(``_CheckpointSegment``).  It does **not** use ``torch.utils.checkpoint``.

Design
------
For each segment (a slice of the network between two user-specified boundaries)
the forward pass runs the segment normally but keeps *only*:

  * the segment input (boundary tensor), and
  * the segment output (boundary tensor),

dropping all per-layer intermediates.  The backward pass:

  1. restores the per-device RNG snapshot taken before that segment's forward
     (so dropout draws the same mask),
  2. re-runs the segment under ``enable_grad`` to rebuild its local graph,
  3. calls ``torch.autograd.grad`` on that local graph,
  4. restores the RNG state that existed at the start of the backward pass.

Inter-segment boundaries are ``detach()``-ed while *retaining*
``requires_grad``.  Each segment therefore rebuilds an independent local graph;
the gradient reaching a detached boundary leaf is routed by the autograd engine
to the previous segment's Function node (whose output that boundary is), so
chains and shared-parameter reuse both work exactly as in eager mode.  The
external input of the first segment is NOT detached, so its ``.grad`` reaches
the caller's leaf.

Weight tensors are passed as ordinary Function inputs (deduplicated per
segment), so their gradients are accumulated through the normal autograd
accumulator; a parameter shared by several segments receives the sum of the
per-segment gradients.

Explicitly unsupported (see README.md):
  * higher-order derivatives (double backward) -> rejected at runtime;
  * layers with side effects other than global-RNG use (they run twice);
  * in-place mutation of segment inputs or shared inputs -> rejected at runtime.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

from .rng import RNGState, get_rng_state, set_rng_state


def _dedup(tensors):
    """Order-preserving deduplication by tensor identity."""
    seen = set()
    out = []
    for t in tensors:
        if t is None or id(t) in seen:
            continue
        seen.add(id(t))
        out.append(t)
    return out


class _CheckpointSegment(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        runner: Callable,
        preserve_rng_state: bool,
        rng_before: Optional[RNGState],
        boundary_in: torch.Tensor,
        *params: torch.Tensor,
    ):
        if not isinstance(boundary_in, torch.Tensor):
            raise TypeError(
                "segment input must be a Tensor; got %r" % type(boundary_in)
            )

        ctx.preserve_rng_state = preserve_rng_state
        ctx.rng_before = rng_before

        boundary_in_vc = boundary_in._version
        with torch.no_grad():
            out = runner(boundary_in, params)
        if boundary_in._version != boundary_in_vc:
            raise RuntimeError(
                "segment runner modified its input tensor in place; in-place "
                "mutation of segment inputs or shared inputs is not supported."
            )
        # A segment may harmlessly pass through a view of its input (Dropout in
        # eval mode, an identity activation): storage is shared without
        # mutation and recomputes fine. The dangerous in-place overwrite case
        # is caught above via the version counter.

        ctx.runner = runner
        ctx.save_for_backward(boundary_in, *params)
        ctx.boundary_in_version = boundary_in._version
        ctx.num_params = len(params)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        if torch.is_grad_enabled():
            raise NotImplementedError(
                "higher-order derivatives through a recomputing checkpoint "
                "segment are not supported: the local graph rebuilt during "
                "backward is discarded. Detected create_graph=True / double "
                "backward."
            )
        boundary_in = ctx.saved_tensors[0]
        params = list(ctx.saved_tensors[1 : 1 + ctx.num_params])

        if boundary_in._version != ctx.boundary_in_version:
            raise RuntimeError(
                "a segment boundary tensor was modified in place between the "
                "forward and backward passes; checkpointing cannot recompute "
                "correctly. Do not mutate saved activations in place."
            )

        needs_in = boundary_in.requires_grad
        needs_params = [p.requires_grad for p in params]

        grad_in: Optional[torch.Tensor] = None
        param_grads: list[Optional[torch.Tensor]] = [None] * len(params)

        if grad_out is not None:
            backward_entry_rng = get_rng_state()
            if ctx.preserve_rng_state and ctx.rng_before is not None:
                set_rng_state(ctx.rng_before)
            try:
                with torch.enable_grad():
                    # Rebuild the local graph from a detached leaf: gradients
                    # w.r.t. this leaf are routed to the ORIGINAL boundary input
                    # (kept in saved_tensors) by the engine, exactly as reentrant
                    # checkpoint does with detach_variable. This keeps each
                    # segment's recomputed graph independent (so cross-segment
                    # parameter reuse cannot traverse a freed graph).
                    detached_in = boundary_in.detach()
                    if needs_in:
                        detached_in.requires_grad_(True)
                    local_targets = [detached_in] if needs_in else []
                    local_targets += [
                        p for p, need in zip(params, needs_params) if need
                    ]
                    recomputed = ctx.runner(detached_in, params)
                grads = torch.autograd.grad(
                    recomputed,
                    local_targets,
                    grad_outputs=grad_out,
                    allow_unused=True,
                )
            finally:
                # Restore the stream to where it was at backward entry, undoing
                # whatever the recomputation consumed.
                set_rng_state(backward_entry_rng)

            cursor = 0
            if needs_in:
                grad_in = grads[cursor]
                cursor += 1
            for index, need in enumerate(needs_params):
                if need:
                    param_grads[index] = grads[cursor]
                    cursor += 1

        # Slots: runner, preserve_rng_state, rng_before, boundary_in, then one
        # per parameter. Non-Tensor metadata slots receive None.
        return (None, None, None, grad_in, *param_grads)


def checkpoint_segment(
    runner: Callable,
    boundary_in: torch.Tensor,
    params,
    *,
    preserve_rng_state: bool = True,
) -> torch.Tensor:
    """Run one segment in the forward, scheduling it for recomputation.

    Parameters
    ----------
    runner:
        Callable ``runner(x, params) -> y``. Must be deterministic given ``x``
        and the global RNG state. Modules with arbitrary side effects run a
        second time during the backward and are not supported transparently.
    boundary_in:
        Segment input tensor. Saved for recomputation.
    params:
        Weight tensors the segment reads (e.g. the modules' ``parameters()``),
        deduplicated per segment by identity.
    preserve_rng_state:
        When True (default), RNG state is snapshotted for exact dropout
        reproduction and the caller's stream is left in eager-equivalent
        position. Set False only for fully deterministic segments.
    """
    params = _dedup(params)
    rng_before = get_rng_state() if preserve_rng_state else None
    return _CheckpointSegment.apply(
        runner, preserve_rng_state, rng_before, boundary_in, *params
    )
