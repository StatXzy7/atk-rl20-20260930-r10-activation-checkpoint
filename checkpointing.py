from __future__ import annotations

from dataclasses import dataclass
import threading
import weakref

import torch
import torch.nn.functional as F
from torch import nn


ALLOWED_ACTIVATIONS = (
    nn.Identity,
    nn.ReLU,
    nn.LeakyReLU,
    nn.PReLU,
    nn.ELU,
    nn.GELU,
    nn.Sigmoid,
    nn.Tanh,
    nn.Softmax,
    nn.LogSoftmax,
    nn.Hardtanh,
    nn.Hardsigmoid,
    nn.Hardswish,
    nn.SiLU,
    nn.Mish,
    nn.Tanhshrink,
    nn.Softplus,
    nn.Softshrink,
)


def _validate_layers(layers: list[nn.Module]) -> None:
    if not layers:
        raise ValueError("at least one layer is required")
    for index, layer in enumerate(layers):
        if isinstance(layer, (nn.Linear, nn.Dropout, *ALLOWED_ACTIVATIONS)):
            if getattr(layer, "inplace", False):
                raise ValueError(f"layer {index} uses in-place mutation, which is not supported")
            continue
        raise ValueError(
            f"layer {index} ({type(layer).__name__}) is not a supported stateless "
            "Linear/activation/Dropout layer; arbitrary side-effect layers are rejected"
        )


def normalize_boundaries(
    num_layers: int,
    boundaries: int | list[int] | tuple[int, ...],
) -> tuple[int, ...]:
    if not isinstance(num_layers, int) or num_layers <= 0:
        raise ValueError("num_layers must be a positive integer")
    if isinstance(boundaries, int):
        num_segments = boundaries
        if num_segments <= 0:
            raise ValueError("num_segments must be positive")
        if num_segments > num_layers:
            raise ValueError("num_segments cannot exceed the number of layers")
        base, remainder = divmod(num_layers, num_segments)
        cuts = []
        cursor = 0
        for segment in range(num_segments):
            size = base + (1 if segment < remainder else 0)
            cursor += size
            if segment + 1 < num_segments:
                cuts.append(cursor)
        return tuple(cuts)

    if not isinstance(boundaries, (list, tuple)):
        raise TypeError("boundaries must be a positive segment count or a tuple/list of cuts")
    if not all(type(cut) is int for cut in boundaries):
        raise TypeError("boundary cuts must be ints")
    cuts = tuple(boundaries)
    if any(cut <= 0 or cut >= num_layers for cut in cuts):
        raise ValueError("boundary cuts must satisfy 0 < cut < num_layers")
    if list(cuts) != sorted(cuts) or len(set(cuts)) != len(cuts):
        raise ValueError("boundary cuts must be strictly increasing")
    return cuts


def _segment_slices(boundaries: tuple[int, ...]) -> list[slice]:
    points = (0, *boundaries)
    return [slice(start, end) for start, end in zip(points, (*points[1:], None))]


def _unique_params(modules: list[nn.Module]) -> list[nn.Parameter]:
    params = []
    seen = set()
    for module in modules:
        for param in module.parameters(recurse=False):
            key = (param.device.type, param.device.index, param.data_ptr())
            if key not in seen:
                seen.add(key)
                params.append(param)
    return params


def _rng_devices(*groups: list[torch.Tensor]) -> list[tuple[str, int | None]]:
    devices: dict[tuple[str, int | None], None] = {}
    has_cpu = False
    for tensors in groups:
        for tensor in tensors:
            device = tensor.device
            if device.type == "cpu":
                has_cpu = True
            else:
                devices[(device.type, device.index)] = None
    ordered: list[tuple[str, int | None]] = []
    if has_cpu:
        ordered.append(("cpu", None))
    ordered.extend(devices)
    return ordered


def _capture_rng(devices: list[tuple[str, int | None]]) -> list[torch.Tensor]:
    states = []
    for kind, index in devices:
        if kind == "cpu":
            states.append(torch.random.get_rng_state())
        elif kind == "cuda":
            states.append(torch.cuda.get_rng_state(index if index is not None else 0))
        else:
            raise ValueError(f"unsupported RNG device type: {kind}")
    return states


def _set_rng(devices: list[tuple[str, int | None]], states: list[torch.Tensor]) -> None:
    for (kind, index), state in zip(devices, states):
        if kind == "cpu":
            torch.random.set_rng_state(state)
        elif kind == "cuda":
            torch.cuda.set_rng_state(state, index if index is not None else 0)
        else:
            raise ValueError(f"unsupported RNG device type: {kind}")


def _storage_key(tensor: torch.Tensor) -> tuple[str, int | None, int]:
    return (tensor.device.type, tensor.device.index, tensor.untyped_storage().data_ptr())


@dataclass(frozen=True)
class TensorGroupStats:
    references: int
    logical_bytes: int
    referenced_storage_bytes: int
    unique_storage_bytes: int

    @property
    def repeated_reference_bytes(self) -> int:
        return max(0, self.referenced_storage_bytes - self.unique_storage_bytes)


@dataclass(frozen=True)
class TensorMemoryStats:
    activation: TensorGroupStats
    parameter: TensorGroupStats
    rng_state: TensorGroupStats


def _stats_for_tensors(tensors: list[torch.Tensor]) -> TensorGroupStats:
    storage_tensors = {}
    logical = 0
    referenced_storage = 0
    for tensor in tensors:
        logical += tensor.numel() * tensor.element_size()
        referenced_storage += tensor.untyped_storage().nbytes()
        key = _storage_key(tensor)
        storage_tensors[key] = tensor
    return TensorGroupStats(
        references=len(tensors),
        logical_bytes=logical,
        referenced_storage_bytes=referenced_storage,
        unique_storage_bytes=sum(
            tensor.untyped_storage().nbytes() for tensor in storage_tensors.values()
        ),
)


def checkpoint_contexts(model: nn.Module) -> list[object]:
    contexts = []
    seen = set()
    for context in getattr(model, "_checkpoint_contexts", []):
        if context is not None and id(context) not in seen:
            seen.add(id(context))
            contexts.append(context)
    return contexts


def checkpoint_memory_stats(model: nn.Module) -> TensorMemoryStats:
    activations = []
    parameters = []
    rng_states = []
    for context in checkpoint_contexts(model):
        if not getattr(context, "retained_for_backward", False):
            continue
        activations.append(context.saved_tensors[0])
        parameters.extend(context.params)
        rng_states.extend(context.rng_states)
    return TensorMemoryStats(
        activation=_stats_for_tensors(activations),
        parameter=_stats_for_tensors(parameters),
        rng_state=_stats_for_tensors(rng_states),
    )


def count_saved_tensor_payloads(tensors: list[torch.Tensor]) -> TensorGroupStats:
    return _stats_for_tensors(list(tensors))


class _SegmentCheckpoint(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        layers: tuple[nn.Module, ...],
        owner_ref,
        rng_devices: list[tuple[str, int | None]],
        x: torch.Tensor,
        *params: torch.Tensor,
    ) -> torch.Tensor:
        ctx.layers = layers
        ctx.num_params = len(params)
        ctx.params = tuple(params)
        ctx.rng_devices = rng_devices
        ctx.retained_for_backward = True
        ctx.owner_ref = owner_ref

        rng_states = _capture_rng(rng_devices)
        ctx.rng_states = rng_states
        boundary_input = x.detach()
        if x.requires_grad:
            boundary_input.requires_grad_(True)
        with torch.no_grad():
            output = boundary_input
            for layer in layers:
                output = layer(output)

        ctx.save_for_backward(boundary_input)
        owner = owner_ref() if owner_ref is not None else None
        if owner is not None:
            owner._checkpoint_contexts.append(ctx)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if torch.is_grad_enabled():
            raise RuntimeError(
                "activation checkpointing supports first-order gradients only; "
                "create_graph=True/higher-order differentiation is rejected"
            )

        boundary = ctx.saved_tensors[0]
        params = list(ctx.params)
        rng_devices = ctx.rng_devices
        caller_rng = _capture_rng(rng_devices)

        try:
            _set_rng(rng_devices, ctx.rng_states)
            with torch.enable_grad():
                recomputed = boundary
                outputs = [recomputed]
                for layer in ctx.layers:
                    recomputed = layer(recomputed)
                    outputs.append(recomputed)

                differentiable_params = [
                    param for param in params if param.requires_grad
                ]
                differentiated_inputs = differentiable_params[:]
                if boundary.requires_grad:
                    differentiated_inputs.insert(0, boundary)

                gradients = torch.autograd.grad(
                    outputs[-1],
                    differentiated_inputs,
                    grad_output,
                    allow_unused=True,
                    only_inputs=True,
                )

                if boundary.requires_grad:
                    grad_boundary = gradients[0]
                    parameter_gradients = gradients[1:]
                else:
                    grad_boundary = None
                    parameter_gradients = gradients
                if boundary.requires_grad and grad_boundary is None:
                    grad_boundary = torch.zeros_like(boundary)
                grad_params = []
                gradient_iter = iter(parameter_gradients)
                for param in params:
                    if param.requires_grad:
                        grad_params.append(next(gradient_iter))
                    else:
                        grad_params.append(None)
        finally:
            _set_rng(rng_devices, caller_rng)
            ctx.retained_for_backward = False
            ctx.rng_states = ()
            owner = ctx.owner_ref() if ctx.owner_ref is not None else None
            if owner is not None:
                owner._checkpoint_contexts = [
                    active for active in owner._checkpoint_contexts if active is not ctx
                ]

        return (None, None, None, grad_boundary, *grad_params)


class CheckpointedSequential(nn.Module):
    """Sequential Linear/activation/Dropout network with explicit activation checkpoints."""

    def __init__(
        self,
        *layers: nn.Module,
        boundaries: int | list[int] | tuple[int, ...],
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(list(layers))
        self.boundaries = normalize_boundaries(len(self.layers), boundaries)
        _validate_layers(list(self.layers))
        self._segment_layers = [
            list(self.layers[segment_slice]) for segment_slice in _segment_slices(self.boundaries)
        ]
        self._segment_params = [_unique_params(part) for part in self._segment_layers]
        devices = {
            (param.device.type, param.device.index)
            for param in self.parameters()
        }
        if len(devices) > 1:
            raise ValueError("mixed-device networks are not supported")
        self._context_lock = threading.Lock()
        self._checkpoint_contexts: list[object] = []

    @property
    def num_segments(self) -> int:
        return len(self._segment_layers)

    def clear_context_records(self) -> None:
        self._checkpoint_contexts.clear()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError("input must be a tensor")

        needs_grad = torch.is_grad_enabled() and (
            x.requires_grad
            or any(param.requires_grad for param in self.parameters())
        )
        if not needs_grad:
            with torch.no_grad():
                for layer in self.layers:
                    x = layer(x)
            return x

        with self._context_lock:
            self._checkpoint_contexts.clear()
            for part, params in zip(self._segment_layers, self._segment_params):
                rng_devices = _rng_devices([x], params)
                x = _SegmentCheckpoint.apply(
                    tuple(part),
                    weakref.ref(self),
                    rng_devices,
                    x,
                    *params,
                )
        return x


def checkpoint_segment_linear_tanh_linear(
    x: torch.Tensor,
    weight1: torch.Tensor,
    bias1: torch.Tensor,
    weight2: torch.Tensor,
    bias2: torch.Tensor,
) -> torch.Tensor:
    """Deterministic low-level checkpoint used by the double-precision gradcheck test."""

    layers = (
        lambda x: F.linear(x, weight1, bias1),
        torch.tanh,
        lambda x: F.linear(x, weight2, bias2),
    )
    return _SegmentCheckpoint.apply(
        tuple(layers),
        None,
        [("cpu", None)],
        x,
        weight1,
        bias1,
        weight2,
        bias2,
    )
