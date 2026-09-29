"""Real measurements for segmented checkpointing.

Run:  python -m mini_checkpoint.benchmark [--d D] [--depth L] [--batch B]

Reported for each segment count k:
  * saved-activation proxy (real tensor bytes; refs vs distinct storages);
  * number of module invocations during one forward + one backward
    (eager = 1x per module; checkpointed = 2x -> +1 full forward of work);
  * wall-clock time of forward+backward (CPU here; never assumed to improve);
  * real device peak memory when a CUDA device is available, via
    torch.cuda.max_memory_allocated, reset around each trial.

On a CPU-only build there is no equivalent of max_memory_allocated; we report
process max RSS (ru_maxrss) purely as an OS-level, non-comparable number and
say so explicitly. No models or datasets are downloaded: everything is
synthetic.
"""

from __future__ import annotations

import argparse
import resource
import time

import torch
from torch import nn

from .network import CheckpointedSequential, normalize_segments
from .stats import EagerActivationCollector, tensor_mem_stats


def build_modules(depth: int, width: int, p: float, dtype) -> list[nn.Module]:
    mods: list[nn.Module] = []
    for i in range(depth):
        mods.append(nn.Linear(width, width, dtype=dtype))
        mods.append(nn.Tanh())
        mods.append(nn.Dropout(p))
    mods.append(nn.Linear(width, width, dtype=dtype))
    return mods


class CallCounter:
    """Count forward invocations of modules (including recomputations)."""

    def __init__(self, modules):
        self.handles = [
            m.register_forward_hook(self._hook)
            for m in modules
            if not isinstance(m, nn.modules.container.ModuleList)
            and len(list(m.children())) == 0
        ]
        self.count = 0

    def _hook(self, *args):
        self.count += 1

    def close(self):
        for h in self.handles:
            h.remove()


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def saved_boundary_tensors(model: CheckpointedSequential, x: torch.Tensor):
    """Return the exact list of segment-input tensors kept for recomputation."""
    captured: list[torch.Tensor] = []
    orig = model._make_runner

    def spy_runner(start, end):
        inner = orig(start, end)

        def runner(tensor, params):
            captured.append(tensor)
            return inner(tensor, params)

        return runner

    model._make_runner = spy_runner
    try:
        with torch.no_grad():
            model(x)
    finally:
        model._make_runner = orig
    return captured


def measure_proxy(modules_factory, segments, x):
    mods = modules_factory()
    model = CheckpointedSequential(*mods, segments=segments).eval()
    # Use training=True so dropout modules exist, but proxy only needs shapes;
    # dropout does not change tensor shape so eval is fine.
    boundaries = saved_boundary_tensors(model, x)
    return tensor_mem_stats(boundaries)


def run_forward_backward(model, x):
    y = model(x)
    y.sum().backward()


def reset_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_bytes():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated()
    return None


def time_forward_backward(model, x, repeats):
    """Return (best wall time, best device peak bytes or None on CPU)."""
    model.train()
    best = float("inf")
    best_peak = None
    for _ in range(repeats):
        for p in model.parameters():
            p.grad = None
        reset_peak()
        _sync()
        t0 = time.perf_counter()
        run_forward_backward(model, x)
        _sync()
        dt = time.perf_counter() - t0
        best = min(best, dt)
        peak = peak_bytes()
        if peak is not None:
            best_peak = peak if best_peak is None else min(best_peak, peak)
    return best, best_peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    torch.manual_seed(0)
    n_modules = args.depth * 3 + 1
    x = torch.randn(args.batch, args.d, dtype=dtype)

    def factory():
        torch.manual_seed(123)
        return build_modules(args.depth, args.d, 0.5, dtype)

    # Eager reference proxy: all module outputs are retained.
    eager_mods = factory()
    eager = nn.Sequential(*eager_mods)
    eager.eval()
    with EagerActivationCollector(eager_mods) as coll:
        with torch.no_grad():
            eager(x)
    eager_stats = tensor_mem_stats(coll.activations)

    print(f"device: {'cuda' if torch.cuda.is_available() else 'cpu-only'}")
    print(f"network: {args.depth} x (Linear/Tanh/Dropout) + Linear, "
          f"width={args.d}, batch={args.batch}")
    print(f"modules={n_modules}, input={tuple(x.shape)}, dtype={args.dtype}")
    print()
    print("k = segment count; saved = distinct storage bytes actually retained")
    print("-" * 88)
    print(f"{'k':>3} {'saved_refs':>10} {'saved_stor':>10} "
          f"{'saved_MiB':>10} {'calls fwd+bwd':>13} {'best_s':>9} "
          f"{'peak_MiB':>9}")
    print("-" * 88)

    for k in [1, 2, 3, 4, 6, n_modules]:
        segments = k  # int -> as-even-as-possible split
        stats = measure_proxy(factory, segments, x)

        mods_t = factory()
        model = CheckpointedSequential(*mods_t, segments=segments)
        model.train()
        counter = CallCounter(mods_t)
        run_forward_backward(model, x)
        total_calls = counter.count
        counter.close()
        elapsed, peak = time_forward_backward(model, x, args.repeats)
        peak_txt = f"{peak / 2**20:8.2f}" if peak is not None else "     N/A"

        print(f"{k:>3} {stats.num_refs:>10} {stats.num_storages:>10} "
              f"{stats.unique_bytes / 2**20:>10.3f} {total_calls:>13} "
              f"{elapsed:>9.4f} {peak_txt:>9}")

    print("-" * 88)
    # Eager timing / calls.
    mods_e = factory()
    eager_train = nn.Sequential(*mods_e)
    eager_train.train()
    counter_e = CallCounter(mods_e)
    run_forward_backward(eager_train, x)
    eager_calls = counter_e.count
    counter_e.close()
    eager_time, eager_peak = time_forward_backward(eager_train, x, args.repeats)
    eager_peak_txt = (f"{eager_peak / 2**20:8.2f}"
                      if eager_peak is not None else "     N/A")
    print(f"{'eager':>3} {eager_stats.num_refs:>10} {eager_stats.num_storages:>10} "
          f"{eager_stats.unique_bytes / 2**20:>10.3f} {eager_calls:>13} "
          f"{eager_time:>9.4f} {eager_peak_txt:>9}")
    print()
    print("Notes:")
    print(" * saved_MiB is a retained-tensor proxy, NOT device peak memory.")
    print(" * calls: each module runs once per forward; checkpointed modules")
    print("   run one extra time during recompute (2x). Extra compute is")
    print("   ~+1 full network forward regardless of k or cut placement.")
    print(" * timing is wall-clock best-of-N; on CPU recompute usually costs")
    print("   real time; speedup is not guaranteed and not assumed.")
    if torch.cuda.is_available():
        print(" * CUDA present: true max_memory_allocated can be enabled here.")
    else:
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(f" * CPU-only build: no device peak allocator. Process max RSS "
              f"~{rss_kb / 1024:.1f} MiB (OS-level, includes everything, "
              f"not a saved-activation number).")


if __name__ == "__main__":
    main()
