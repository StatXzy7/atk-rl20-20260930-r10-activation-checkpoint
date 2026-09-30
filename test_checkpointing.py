import copy

import pytest
import torch
from torch import nn

from checkpointing import (
    CheckpointedSequential,
    checkpoint_memory_stats,
    checkpoint_segment_linear_tanh_linear,
    count_saved_tensor_payloads,
    normalize_boundaries,
)


def build_reference():
    return nn.Sequential(
        nn.Linear(4, 6),
        nn.Tanh(),
        nn.Dropout(0.4),
        nn.Linear(6, 5),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(5, 3),
    )


def build_checkpoint(boundaries):
    reference = build_reference()
    checkpoint = CheckpointedSequential(
        *[copy.deepcopy(layer) for layer in reference],
        boundaries=boundaries,
    )
    with torch.no_grad():
        for reference_param, checkpoint_param in zip(
            reference.parameters(), checkpoint.parameters(), strict=True
        ):
            checkpoint_param.copy_(reference_param)
    return reference, checkpoint


def grads_by_name(model):
    return {
        name: param.grad.detach().clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }


def assert_parameter_grads_equal(reference, checkpoint):
    reference_grads = [param.grad for param in reference.parameters()]
    checkpoint_grads = [param.grad for param in checkpoint.parameters()]
    assert len(reference_grads) == len(checkpoint_grads)
    for index, (reference_grad, checkpoint_grad) in enumerate(
        zip(reference_grads, checkpoint_grads, strict=True)
    ):
        assert torch.equal(reference_grad, checkpoint_grad), index


@pytest.mark.parametrize(
    ("boundaries", "seed"),
    [
        (1, 11),
        (3, 12),
        ((1, 3, 5), 13),
        ((2, 6), 14),
        ((1, 2, 3, 4, 5, 6), 15),
    ],
)
def test_output_and_gradients_match_reference(boundaries, seed):
    torch.manual_seed(seed)
    reference, checkpoint = build_checkpoint(boundaries)
    reference.train()
    checkpoint.train()
    x_ref = torch.randn(8, 4)
    x_cp = x_ref.detach().clone()

    torch.manual_seed(1000)
    y_ref = reference(x_ref)
    torch.manual_seed(1000)
    y_cp = checkpoint(x_cp)

    assert torch.equal(y_ref, y_cp)

    y_ref.sum().backward()
    y_cp.sum().backward()

    assert len(grads_by_name(reference)) == len(grads_by_name(checkpoint))
    assert_parameter_grads_equal(reference, checkpoint)


def test_input_not_requiring_grad_still_updates_parameters():
    torch.manual_seed(0)
    reference, checkpoint = build_checkpoint(2)
    reference.train()
    checkpoint.train()
    x_ref = torch.randn(8, 4)
    x_cp = x_ref.detach().clone()

    torch.manual_seed(20)
    y_ref = reference(x_ref)
    torch.manual_seed(20)
    y_cp = checkpoint(x_cp)
    y_ref.sum().backward()
    y_cp.sum().backward()

    assert x_ref.grad is None
    assert x_cp.grad is None
    assert_parameter_grads_equal(reference, checkpoint)


def test_repeated_module_parameters_accumulate_like_reference():
    shared = nn.Linear(4, 4)
    final_layer = nn.Linear(4, 2)
    reference = nn.Sequential(
        shared,
        nn.Tanh(),
        nn.Dropout(0.5),
        shared,
        nn.ReLU(),
        final_layer,
    )
    checkpoint = CheckpointedSequential(
        shared,
        nn.Tanh(),
        nn.Dropout(0.5),
        shared,
        nn.ReLU(),
        final_layer,
        boundaries=(2, 4),
    )
    reference.train()
    checkpoint.train()

    x_ref = torch.randn(5, 4)
    x_cp = x_ref.detach().clone()
    torch.manual_seed(44)
    y_ref = reference(x_ref)
    torch.manual_seed(44)
    y_cp = checkpoint(x_cp)

    assert torch.equal(y_ref, y_cp)
    y_ref.sum().backward()
    y_cp.sum().backward()

    assert len(list(checkpoint.parameters())) == len(list(reference.parameters()))
    assert torch.equal(shared.weight.grad, checkpoint.layers[0].weight.grad)
    assert torch.equal(shared.bias.grad, checkpoint.layers[0].bias.grad)
    assert torch.equal(final_layer.weight.grad, checkpoint.layers[5].weight.grad)
    assert torch.equal(final_layer.bias.grad, checkpoint.layers[5].bias.grad)


def test_rng_sequence_is_unchanged_after_forward_and_backward():
    reference, checkpoint = build_checkpoint((2, 5))
    reference.train()
    checkpoint.train()
    x_ref = torch.randn(7, 4)
    x_cp = x_ref.detach().clone()

    torch.manual_seed(700)
    y_ref = reference(x_ref)
    state_after_reference_forward = torch.random.get_rng_state().clone()
    y_ref.sum().backward()
    sample_after_reference_backward = torch.randn(3)

    torch.manual_seed(700)
    y_cp = checkpoint(x_cp)
    state_after_checkpoint_forward = torch.random.get_rng_state().clone()
    y_cp.sum().backward()
    sample_after_checkpoint_backward = torch.randn(3)

    assert torch.equal(state_after_reference_forward, state_after_checkpoint_forward)
    assert torch.equal(sample_after_reference_backward, sample_after_checkpoint_backward)


def test_checkpoint_retains_only_boundaries_and_rng_states():
    model = CheckpointedSequential(
        nn.Linear(4, 5),
        nn.Tanh(),
        nn.Dropout(0.2),
        nn.Linear(5, 3),
        boundaries=(2,),
    )
    x = torch.randn(6, 4, requires_grad=True)
    model(x)
    stats = checkpoint_memory_stats(model)

    assert stats.activation.references == 2
    assert stats.activation.logical_bytes == 6 * 4 * 4 + 6 * 5 * 4
    assert stats.activation.unique_storage_bytes == stats.activation.logical_bytes
    assert stats.rng_state.references == 2


def test_payload_counter_distinguishes_shared_storage_from_repeated_references():
    storage = torch.arange(4, dtype=torch.float32)
    view_a = storage[:2]
    view_b = storage[1:]
    stats = count_saved_tensor_payloads([view_a, view_b])

    assert stats.references == 2
    assert stats.logical_bytes == 20
    assert stats.referenced_storage_bytes == 32
    assert stats.unique_storage_bytes == 16
    assert stats.repeated_reference_bytes == 16


def test_higher_order_gradients_are_rejected():
    model = CheckpointedSequential(nn.Linear(3, 3), nn.Tanh(), boundaries=1)
    x = torch.randn(2, 3, requires_grad=True)
    with pytest.raises(RuntimeError, match="first-order gradients only"):
        torch.autograd.grad(model(x).sum(), x, create_graph=True)


def test_side_effect_and_stateful_layers_are_rejected():
    class MutatesInput(nn.Module):
        def forward(self, x):
            x.sigmoid_()
            return x

    with pytest.raises(ValueError, match="side-effect layers"):
        CheckpointedSequential(nn.Linear(3, 3), MutatesInput(), boundaries=1)
    with pytest.raises(ValueError, match="side-effect layers"):
        CheckpointedSequential(nn.Linear(3, 3), nn.BatchNorm1d(3), boundaries=1)


def test_inplace_activation_is_rejected():
    with pytest.raises(ValueError, match="in-place"):
        CheckpointedSequential(nn.Linear(3, 3), nn.ReLU(inplace=True), boundaries=1)


def test_boundary_validation():
    assert normalize_boundaries(5, 2) == (3,)
    assert normalize_boundaries(6, 3) == (2, 4)
    with pytest.raises(ValueError):
        normalize_boundaries(3, 4)
    with pytest.raises(ValueError):
        normalize_boundaries(4, (0, 2))
    with pytest.raises(ValueError):
        normalize_boundaries(4, (2, 2))


def test_double_precision_gradcheck_deterministic_subproblem():
    torch.manual_seed(0)
    inputs = (
        torch.randn(3, 4, dtype=torch.float64, requires_grad=True),
        torch.randn(5, 4, dtype=torch.float64, requires_grad=True),
        torch.randn(5, dtype=torch.float64, requires_grad=True),
        torch.randn(2, 5, dtype=torch.float64, requires_grad=True),
        torch.randn(2, dtype=torch.float64, requires_grad=True),
    )
    assert torch.autograd.gradcheck(
        checkpoint_segment_linear_tanh_linear,
        inputs,
        eps=1e-6,
        atol=1e-8,
    )
