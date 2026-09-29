"""Correctness tests for the from-scratch segmented checkpointing."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from mini_checkpoint import (
    CheckpointedSequential,
    EagerActivationCollector,
    checkpoint_segment,
    normalize_segments,
    tensor_mem_stats,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def make_modules(width=6, depth=3, p=0.5, dtype=torch.float64, shared=False):
    """Linear / activation / dropout stack, optionally reusing one Linear."""
    mods = []
    shared_linear = nn.Linear(width, width, dtype=dtype)
    for i in range(depth):
        if shared and i % 2 == 1:
            mods.append(shared_linear)  # same instance, shared parameters
        else:
            mods.append(nn.Linear(width, width, dtype=dtype))
        mods.append(nn.Tanh())
        mods.append(nn.Dropout(p))
    mods.append(nn.Linear(width, 2, dtype=dtype))
    return mods


def build_pair(segments, *, shared=False, preserve=True, seed=0,
               width=6, depth=3, p=0.5, dtype=torch.float64):
    mods = make_modules(width=width, depth=depth, p=p, dtype=dtype,
                        shared=shared)
    eager = nn.Sequential(*mods)
    ckpt = CheckpointedSequential(
        *mods, segments=segments, preserve_rng_state=preserve
    )
    return eager, ckpt


def clone_grads(model):
    # model.parameters() yields each distinct parameter once, in
    # first-occurrence order; shared instances are de-duplicated by nn.Module.
    return [
        p.grad.detach().clone() if p.grad is not None else None
        for p in model.parameters()
    ]


def assert_equal_grads(grad_a, grad_b, exact=True):
    assert len(grad_a) == len(grad_b)
    for idx, (ga, gb) in enumerate(zip(grad_a, grad_b)):
        if ga is None:
            assert gb is None
            continue
        assert gb is not None, f"missing grad for parameter #{idx}"
        if exact:
            assert torch.equal(ga, gb), f"grad mismatch for parameter #{idx}"
        else:
            torch.testing.assert_close(ga, gb, rtol=1e-5, atol=1e-6)


def run_path(model, x):
    model.zero_grad(set_to_none=True)
    y = model(x)
    y.sum().backward()
    xgrad = None if x.grad is None else x.grad.detach().clone()
    return y.detach().clone(), xgrad, clone_grads(model)


# ---------------------------------------------------------------------------
# Segment spec
# ---------------------------------------------------------------------------

def test_normalize_segments_uniform():
    assert normalize_segments(6, 1) == [(0, 6)]
    assert normalize_segments(6, 2) == [(0, 3), (3, 6)]
    assert normalize_segments(6, 3) == [(0, 2), (2, 4), (4, 6)]
    # remainder carried by leading segments
    assert normalize_segments(7, 3) == [(0, 3), (3, 5), (5, 7)]


def test_normalize_segments_cuts_uneven():
    assert normalize_segments(10, [1]) == [(0, 1), (1, 10)]
    assert normalize_segments(10, [2, 3, 9]) == [(0, 2), (2, 3), (3, 9), (9, 10)]


@pytest.mark.parametrize("spec", [0, 11, [], [0], [10], [2, 2], [3, 1], 1.5])
def test_normalize_segments_rejects_bad(spec):
    with pytest.raises((ValueError, TypeError)):
        normalize_segments(10, spec)


# ---------------------------------------------------------------------------
# Output / gradient parity, single / multi / uneven segments
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("segments", [1, 2, 4, [2], [1, 3, 8]])
def test_matches_eager_double(segments):
    torch.manual_seed(123)
    x = torch.randn(5, 6, dtype=torch.float64)

    torch.manual_seed(42)
    eager, ckpt = build_pair(segments, seed=42)
    eager.train(); ckpt.train()

    # identical seeds on both paths -> identical dropout draws in forward
    xe = x.detach().clone().requires_grad_(True)
    xc = x.detach().clone().requires_grad_(True)

    torch.manual_seed(777)
    ye, gxe, ge = run_path(eager, xe)
    torch.manual_seed(777)
    yc, gxc, gc = run_path(ckpt, xc)

    assert torch.equal(ye, yc)          # identical masks + numerics
    assert torch.equal(gxe, gxc)
    assert_equal_grads(ge, gc, exact=True)


def test_matches_eager_float32_tight():
    torch.manual_seed(123)
    x = torch.randn(8, 8, dtype=torch.float32)
    eager = nn.Sequential(
        nn.Linear(8, 8), nn.Tanh(), nn.Dropout(0.4),
        nn.Linear(8, 8), nn.Tanh(), nn.Dropout(0.4),
        nn.Linear(8, 2),
    )
    ckpt = CheckpointedSequential(*list(eager), segments=3)
    eager.train(); ckpt.train()
    xe = x.detach().clone().requires_grad_(True)
    xc = x.detach().clone().requires_grad_(True)
    torch.manual_seed(5); ye, gxe, ge = run_path(eager, xe)
    torch.manual_seed(5); yc, gxc, gc = run_path(ckpt, xc)
    torch.testing.assert_close(ye, yc, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(gxe, gxc, rtol=1e-5, atol=1e-6)
    assert_equal_grads(ge, gc, exact=False)


# ---------------------------------------------------------------------------
# Shared module parameters across segments
# ---------------------------------------------------------------------------

def test_shared_module_parameters():
    torch.manual_seed(0)
    x = torch.randn(4, 6, dtype=torch.float64)
    eager, ckpt = build_pair(2, shared=True)
    eager.train(); ckpt.train()
    xe = x.detach().clone().requires_grad_(True)
    xc = x.detach().clone().requires_grad_(True)
    torch.manual_seed(9); ye, gxe, ge = run_path(eager, xe)
    torch.manual_seed(9); yc, gxc, gc = run_path(ckpt, xc)
    assert torch.equal(ye, yc)
    assert torch.equal(gxe, gxc)
    assert_equal_grads(ge, gc, exact=True)

    # The shared linear is the same object in both models, so its .grad is
    # exactly the eager gradient (summed contributions over repeated uses).
    shared = [m for m in ckpt.module_list if isinstance(m, nn.Linear)][1]
    eager_shared = [m for m in eager if isinstance(m, nn.Linear)][1]
    assert shared is eager_shared
    assert shared.weight.grad is not None


# ---------------------------------------------------------------------------
# Input requires_grad on/off, frozen parameters
# ---------------------------------------------------------------------------

def test_input_no_grad():
    torch.manual_seed(0)
    x = torch.randn(4, 6, dtype=torch.float64, requires_grad=False)
    eager, ckpt = build_pair(2)
    eager.train(); ckpt.train()
    torch.manual_seed(1); ye, _, ge = run_path(eager, x)
    torch.manual_seed(1); yc, _, gc = run_path(ckpt, x)
    assert torch.equal(ye, yc)
    assert_equal_grads(ge, gc, exact=True)
    assert x.grad is None


def test_frozen_parameters():
    torch.manual_seed(0)
    x = torch.randn(4, 6, dtype=torch.float64)
    eager, ckpt = build_pair(2)
    # freeze the first linear in both (same module instances)
    first = [m for m in eager if isinstance(m, nn.Linear)][0]
    for p in first.parameters():
        p.requires_grad_(False)
    eager.train(); ckpt.train()
    xe = x.detach().clone().requires_grad_(True)
    xc = x.detach().clone().requires_grad_(True)
    torch.manual_seed(2); ye, gxe, ge = run_path(eager, xe)
    torch.manual_seed(2); yc, gxc, gc = run_path(ckpt, xc)
    assert torch.equal(ye, yc)
    assert torch.equal(gxe, gxc)
    assert_equal_grads(ge, gc, exact=True)
    assert first.weight.grad is None


# ---------------------------------------------------------------------------
# RNG stream: after forward+backward the stream must equal eager's, and draws
# continuing afterwards must match bit-for-bit.
# ---------------------------------------------------------------------------

def _post_path_rng(model, x, seed):
    torch.manual_seed(seed)
    y = model(x)
    y.sum().backward()
    return torch.random.get_rng_state().clone()


@pytest.mark.parametrize("segments", [1, 2, 5, [1, 2, 7]])
def test_rng_stream_position_matches_eager(segments):
    x = torch.randn(5, 6, dtype=torch.float64)
    eager, ckpt = build_pair(segments)
    eager.train(); ckpt.train()

    state_e = _post_path_rng(eager, x, 2024)
    state_c = _post_path_rng(ckpt, x, 2024)
    assert torch.equal(state_e, state_c), (
        "post forward+backward RNG stream position differs from eager"
    )


def test_subsequent_draws_identical_after_backward():
    x = torch.randn(5, 6, dtype=torch.float64)
    eager, ckpt = build_pair(3)
    eager.train(); ckpt.train()

    def draws_after(model, x, seed):
        torch.manual_seed(seed)
        y = model(x); y.sum().backward()
        # caller keeps sampling after the whole forward+backward
        return torch.rand(1000)

    a = draws_after(eager, x, 31)
    b = draws_after(ckpt, x, 31)
    assert torch.equal(a, b)


def test_interleaved_backward_draws_are_consumed_like_eager():
    # A random draw *between* the forward and the backward must land at the
    # same stream offset on both paths; then post-backward states still match.
    x = torch.randn(5, 6, dtype=torch.float64)
    eager, ckpt = build_pair(2)
    eager.train(); ckpt.train()

    def run(model):
        torch.manual_seed(7)
        y = model(x)
        mid = torch.rand(10)                 # draw before backward
        y.sum().backward()
        after = torch.rand(10)
        return mid, after

    me, ae = run(eager)
    mc, ac = run(ckpt)
    assert torch.equal(me, mc)
    assert torch.equal(ae, ac)


def test_recompute_uses_forward_mask_not_new_mask():
    # If recompute drew a fresh mask, gradients would (almost surely) differ
    # from eager. This is the direct numerical check, double precision exact.
    x = torch.randn(6, 6, dtype=torch.float64)
    eager, ckpt = build_pair(2, p=0.9)
    eager.train(); ckpt.train()
    xe = x.clone().requires_grad_(True); xc = x.clone().requires_grad_(True)
    torch.manual_seed(11); ye, gxe, ge = run_path(eager, xe)
    torch.manual_seed(11); yc, gxc, gc = run_path(ckpt, xc)
    assert torch.equal(ye, yc)
    assert torch.equal(gxe, gxc)
    assert_equal_grads(ge, gc, exact=True)


# ---------------------------------------------------------------------------
# Double precision gradcheck on the RNG-free subproblem
# ---------------------------------------------------------------------------

def _deterministic_runner(width):
    l1 = nn.Linear(width, width, dtype=torch.float64)
    l2 = nn.Linear(width, width, dtype=torch.float64)
    params = [l1.weight, l1.bias, l2.weight, l2.bias]

    def runner(x, ps):
        w1, b1, w2, b2 = ps
        return torch.tanh(torch.tanh(x @ w1.T + b1) @ w2.T + b2)

    return runner, params


def test_gradcheck_single_segment():
    torch.manual_seed(0)
    width = 4
    runner, params = _deterministic_runner(width)
    x = torch.randn(3, width, dtype=torch.float64, requires_grad=True)
    inputs = (x, *[p.detach().clone().requires_grad_(True) for p in params])

    def fn(x, w1, b1, w2, b2):
        return checkpoint_segment(runner, x, [w1, b1, w2, b2],
                                  preserve_rng_state=False)

    assert torch.autograd.gradcheck(
        fn, inputs, eps=1e-6, atol=1e-6, rtol=1e-4, raise_exception=True
    )


def test_gradcheck_two_segments_chained():
    torch.manual_seed(0)
    width = 4
    runner_a, params_a = _deterministic_runner(width)
    runner_b, params_b = _deterministic_runner(width)
    x = torch.randn(2, width, dtype=torch.float64, requires_grad=True)
    pa = [p.detach().clone().requires_grad_(True) for p in params_a]
    pb = [p.detach().clone().requires_grad_(True) for p in params_b]
    inputs = (x, *pa, *pb)

    def fn(x, *allp):
        a = allp[:4]; b = allp[4:]
        h = checkpoint_segment(runner_a, x, list(a), preserve_rng_state=False)
        return checkpoint_segment(runner_b, h, list(b),
                                  preserve_rng_state=False)

    assert torch.autograd.gradcheck(
        fn, inputs, eps=1e-6, atol=1e-6, rtol=1e-4, raise_exception=True
    )


# ---------------------------------------------------------------------------
# Unsupported cases are rejected / clearly bounded
# ---------------------------------------------------------------------------

class _Counting(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
    def forward(self, x):
        self.calls += 1
        return x + 1.0


def test_side_effect_layer_runs_twice_documented_boundary():
    # Not silently "once like eager": checkpointing must re-execute it.
    counting = _Counting()
    model = CheckpointedSequential(nn.Linear(4, 4), counting, nn.Linear(4, 4),
                                   segments=1)
    x = torch.randn(2, 4)
    model(x).sum().backward()
    assert counting.calls == 2  # documented, not claimed transparent


class _Inplace(nn.Module):
    def forward(self, x):
        return x.sigmoid_()  # mutates the segment input in place


def test_inplace_segment_input_rejected():
    # Boundary inside: first segment ends before the in-place op, so the
    # in-place op mutates the saved boundary of the next segment.
    # cut after module 0: segment 1 = [_Inplace, Linear] mutates its own
    # saved boundary input in place -> rejected during that segment's forward
    model = CheckpointedSequential(nn.Linear(4, 4), _Inplace(), nn.Linear(4, 4),
                                   segments=[1])
    x = torch.randn(2, 4)
    with pytest.raises(RuntimeError, match="in place|in-place"):
        model(x)


class _InplaceInternal(nn.Module):
    def __init__(self, dtype=torch.float64):
        super().__init__()
        self.lin = nn.Linear(4, 4, dtype=dtype)
    def forward(self, x):
        h = self.lin(x)
        return h.relu_()  # mutates a fresh internal tensor: safe to recompute


def test_inplace_internal_activation_is_safe():
    model = CheckpointedSequential(
        _InplaceInternal(), nn.Linear(4, 2, dtype=torch.float64), segments=2)
    eager = nn.Sequential(
        _InplaceInternal(), nn.Linear(4, 2, dtype=torch.float64))
    # share identical weights
    with torch.no_grad():
        for a, b in zip(model.parameters(), eager.parameters()):
            b.copy_(a)
    x = torch.randn(3, 4, dtype=torch.float64)
    xe = x.clone().requires_grad_(True); xc = x.clone().requires_grad_(True)
    torch.manual_seed(1); ye, gxe, ge = run_path(eager, xe)
    torch.manual_seed(1); yc, gxc, gc = run_path(model, xc)
    assert torch.equal(ye, yc)
    assert torch.equal(gxe, gxc)
    assert_equal_grads(ge, gc, exact=True)


def test_higher_order_rejected():
    width = 4
    runner, params = _deterministic_runner(width)
    x = torch.randn(2, width, dtype=torch.float64, requires_grad=True)
    ps = [p.detach().clone().requires_grad_(True) for p in params]
    y = checkpoint_segment(runner, x, ps, preserve_rng_state=False)
    # The first backward already runs under grad mode when create_graph=True,
    # which is exactly the higher-order request we refuse.
    with pytest.raises(NotImplementedError, match="higher-order"):
        torch.autograd.grad(y.sum(), x, create_graph=True)


# ---------------------------------------------------------------------------
# Memory proxy: real tensors, refs vs shared storage
# ---------------------------------------------------------------------------

def test_tensor_mem_stats_distinguishes_storage_and_refs():
    a = torch.arange(12, dtype=torch.float64)        # 12*8 = 96 bytes
    v = a.view(3, 4)                                 # same storage
    b = torch.arange(12, dtype=torch.float64)        # distinct storage
    stats = tensor_mem_stats([a, v, b])
    assert stats.num_refs == 3
    assert stats.num_storages == 2
    assert stats.unique_bytes == 192
    assert stats.ref_bytes == 288
    assert stats.num_views == 1


def test_proxy_fewer_boundaries_with_fewer_segments():
    torch.manual_seed(0)
    x = torch.randn(8, 6, dtype=torch.float64)
    mods = make_modules(width=6, depth=3, dtype=torch.float64)

    eager = nn.Sequential(*mods).eval()
    with EagerActivationCollector(mods) as coll:
        with torch.no_grad():
            eager(x)
    eager_bytes = tensor_mem_stats(coll.activations).unique_bytes

    def ckpt_bytes(k):
        from mini_checkpoint.benchmark import saved_boundary_tensors
        mods_k = make_modules(width=6, depth=3, dtype=torch.float64)
        model = CheckpointedSequential(*mods_k, segments=k).eval()
        bounds = saved_boundary_tensors(model, x)
        return tensor_mem_stats(bounds).unique_bytes

    one = ckpt_bytes(1)
    multi = ckpt_bytes(4)
    # k boundaries retained; more segments -> more retained activation bytes,
    # and every checkpointed variant saves far less than full eager.
    assert one < multi < eager_bytes


def test_module_invocation_counts_extra_compute():
    from mini_checkpoint.benchmark import CallCounter
    mods = make_modules(width=5, depth=2, dtype=torch.float64)
    model = CheckpointedSequential(*mods, segments=2)
    counter = CallCounter(mods)
    x = torch.randn(3, 5, dtype=torch.float64)
    model(x).sum().backward()
    n_modules = len(mods)
    # forward n + recompute n (every segment recomputed once) = 2n
    assert counter.count == 2 * n_modules
    counter.close()


def test_shared_module_across_distinct_segments():
    # Same Linear instance is used in segment 0 AND segment 1, with dropout in
    # between. Gradient on the shared weights must sum both contributions and
    # match an eager network using the same instance.
    dtype = torch.float64
    shared = nn.Linear(5, 5, dtype=dtype)
    head = nn.Linear(5, 5, dtype=dtype)
    tail = nn.Linear(5, 2, dtype=dtype)
    # modules: shared, Tanh, Dropout, head, Tanh, Dropout, shared, tail
    mods = [shared, nn.Tanh(), nn.Dropout(0.5), head, nn.Tanh(),
            nn.Dropout(0.5), shared, tail]
    eager = nn.Sequential(*mods)
    ckpt = CheckpointedSequential(*mods, segments=2)  # cut after module 4
    assert ckpt.segment_ranges == [(0, 4), (4, 8)]
    eager.train(); ckpt.train()

    x = torch.randn(4, 5, dtype=dtype)
    xe = x.clone().requires_grad_(True)
    xc = x.clone().requires_grad_(True)
    torch.manual_seed(4242); ye, gxe, ge = run_path(eager, xe)
    torch.manual_seed(4242); yc, gxc, gc = run_path(ckpt, xc)
    assert torch.equal(ye, yc)
    assert torch.equal(gxe, gxc)
    assert_equal_grads(ge, gc, exact=True)
    # parameter list is de-duplicated by nn.Module despite two uses
    assert sum(p is shared.weight for p in ckpt.parameters()) == 1


def test_early_frozen_segment_grad_still_routes():
    # First segment frozen, later segment trainable: input gradient must still
    # flow back through both Function nodes, and frozen params stay grad-free.
    mods = [nn.Linear(4, 4, dtype=torch.float64), nn.ReLU(),
            nn.Linear(4, 2, dtype=torch.float64)]
    model = CheckpointedSequential(*mods, segments=2)
    for p in mods[0].parameters():
        p.requires_grad_(False)
    x = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    model(x).sum().backward()
    assert x.grad is not None
    assert mods[0].weight.grad is None
    assert mods[2].weight.grad is not None
