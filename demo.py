"""Tiny end-to-end demonstration (synthetic tensors only).

Shows: eager vs checkpointed give identical outputs/gradients (float64), the
post-backward RNG stream is identical, and prints the memory proxy.
"""

import torch
from torch import nn

from mini_checkpoint import (
    CheckpointedSequential,
    EagerActivationCollector,
    tensor_mem_stats,
)
from mini_checkpoint.benchmark import saved_boundary_tensors


def main():
    torch.manual_seed(0)
    dtype = torch.float64
    mods = [
        nn.Linear(8, 8, dtype=dtype), nn.Tanh(), nn.Dropout(0.5),
        nn.Linear(8, 8, dtype=dtype), nn.Tanh(), nn.Dropout(0.5),
        nn.Linear(8, 2, dtype=dtype),
    ]
    eager = nn.Sequential(*[m for m in mods])
    ckpt = CheckpointedSequential(*mods, segments=2)
    eager.train(); ckpt.train()
    x = torch.randn(4, 8, dtype=dtype)

    xe = x.clone().requires_grad_(True)
    xc = x.clone().requires_grad_(True)
    torch.manual_seed(123)
    ye = eager(xe); ye.sum().backward()
    torch.manual_seed(123)
    yc = ckpt(xc); yc.sum().backward()

    print("outputs identical (float64):", torch.equal(ye, yc))
    print("input grads identical      :", torch.equal(xe.grad, xc.grad))
    same = all(
        (a.grad is None and b.grad is None)
        or (a.grad is not None and b.grad is not None
            and torch.equal(a.grad, b.grad))
        for a, b in zip(eager.parameters(), ckpt.parameters())
    )
    print("all param grads identical  :", same)

    def state_after(model, inp):
        torch.manual_seed(123)
        model(inp).sum().backward()
        return torch.random.get_rng_state().clone(), torch.rand(1000).clone()

    se, draw_e = state_after(eager, xe)
    sc, draw_c = state_after(ckpt, xc)
    print("post-fwd+bwd RNG equal     :", torch.equal(se, sc))
    print("draws after backward equal :", torch.equal(draw_e, draw_c))

    ckpt.eval()
    with torch.no_grad():
        proxy = tensor_mem_stats(saved_boundary_tensors(ckpt, x))
    print("checkpoint saved proxy     :", proxy.as_dict())
    eager_eval = nn.Sequential(*mods).eval()
    with EagerActivationCollector(mods) as coll, torch.no_grad():
        eager_eval(x)
    print("eager activations proxy    :",
          tensor_mem_stats(coll.activations).as_dict())


if __name__ == "__main__":
    main()
