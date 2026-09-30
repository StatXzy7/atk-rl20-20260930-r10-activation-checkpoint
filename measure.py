from __future__ import annotations

import copy
import time

import torch
from torch import nn

from checkpointing import (
    CheckpointedSequential,
    TensorMemoryStats,
    checkpoint_memory_stats,
    count_saved_tensor_payloads,
)


def build_layers() -> list[nn.Module]:
    layers = []
    for _ in range(8):
        layers.extend((nn.Linear(256, 256), nn.Tanh(), nn.Dropout(0.1)))
    layers.append(nn.Linear(256, 32))
    return layers


def make_pair(boundaries):
    layers = build_layers()
    reference = nn.Sequential(*copy.deepcopy(layers))
    checkpoint = CheckpointedSequential(*copy.deepcopy(layers), boundaries=boundaries)
    with torch.no_grad():
        for reference_param, checkpoint_param in zip(
            reference.parameters(), checkpoint.parameters(), strict=True
        ):
            checkpoint_param.copy_(reference_param)
    reference.train()
    checkpoint.train()
    return reference, checkpoint


def saved_payload_collector():
    tensors = []

    def pack(tensor):
        tensors.append(tensor)
        return tensor

    def unpack(tensor):
        return tensor

    return tensors, pack, unpack


def synchronize_if_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_run(model, x, seed):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for parameter in model.parameters():
        parameter.grad = None
    torch.manual_seed(seed)
    synchronize_if_cuda()
    start = time.perf_counter()
    y = model(x)
    y.sum().backward()
    synchronize_if_cuda()
    elapsed = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
    return y.detach(), elapsed, peak


def format_bytes(value):
    if value is None:
        return "n/a (CPU)"
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if amount < 1024 or unit == "GiB":
            return f"{amount:.1f} {unit}"
        amount /= 1024


def checkpoint_proxy_bytes(stats: TensorMemoryStats) -> tuple[int, int, int, int]:
    payload_logical = stats.activation.logical_bytes + stats.rng_state.logical_bytes
    payload_unique = (
        stats.activation.unique_storage_bytes + stats.rng_state.unique_storage_bytes
    )
    payload_repeated = (
        stats.activation.repeated_reference_bytes + stats.rng_state.repeated_reference_bytes
    )
    parameter_references = stats.parameter.referenced_storage_bytes
    return payload_logical, payload_unique, payload_repeated, parameter_references


def main():
    torch.set_num_threads(1)
    x = torch.randn(128, 256)
    configs = [
        ("reference", None),
        ("1 segment", 1),
        ("2 segments", 2),
        ("4 segments", 4),
        ("8 segments", 8),
        ("uneven cuts=(9,15,21,24)", (9, 15, 21, 24)),
    ]

    print("# Activation checkpoint measurements")
    print()
    print(f"torch={torch.__version__}, cuda_available={torch.cuda.is_available()}")
    print("input=(128,256), 8 hidden Linear/Tanh/Dropout blocks + output Linear")
    print("time is one forward plus one backward; best of five timed runs after one warmup")
    print()
    print(
        "| model | payload logical B | unique storage B | repeated backing B | "
        "parameter refs B | wall s | real device peak |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | --- |")

    for label, boundaries in configs:
        reference, checkpoint = make_pair(boundaries if boundaries is not None else 1)
        model = reference if boundaries is None else checkpoint

        if boundaries is not None:
            checkpoint(x)
            cp_stats = checkpoint_memory_stats(checkpoint)
            logical, unique, repeated, parameter_refs = checkpoint_proxy_bytes(cp_stats)

            torch.manual_seed(1234)
            expected = reference(x)
            torch.manual_seed(1234)
            actual = checkpoint(x)
            assert torch.equal(expected, actual)
        else:
            payloads, pack, unpack = saved_payload_collector()
            with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
                timed_run(model, x, 1234)
            ref_stats = count_saved_tensor_payloads(payloads)
            logical = ref_stats.logical_bytes
            unique = ref_stats.unique_storage_bytes
            repeated = ref_stats.repeated_reference_bytes
            parameter_refs = 0
            del payloads, pack, unpack

        elapsed_times = []
        peak = None
        for iteration in range(6):
            _, elapsed, peak = timed_run(model, x, 2000 + iteration)
            if iteration > 0:
                elapsed_times.append(elapsed)

        print(
            f"| {label} | {logical} | {unique} | {repeated} | "
            f"{parameter_refs} | {min(elapsed_times):.6f} | {format_bytes(peak)} |"
        )

    print()
    print("Payload bytes are a saved-tensor proxy, not allocator peak memory.")
    print("Checkpoint payload contains segment boundaries and RNG snapshots; it does not cache dropout outputs.")
    print("Parameter refs count repeated Python/autograd references to model-owned storage, not new parameter bytes.")
    print("CPU wall time is environment dependent; checkpointing trades saved activations for recomputation.")


if __name__ == "__main__":
    main()
