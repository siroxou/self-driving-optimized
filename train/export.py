"""Export a trained PPO model to .npz weights for sim/policy.py (and the browser).

    python -m train.export runs/generalist/model.zip web/models/generalist.npz
"""
import argparse

import numpy as np
from torch import nn


def export(model, path):
    """Write the actor network (hidden layers + action head) as w0, b0, w1, b1, ..."""
    policy = model.policy
    assert isinstance(policy.activation_fn(), nn.Tanh), "sim/policy.py assumes tanh hidden layers"
    linears = [m for m in policy.mlp_extractor.policy_net if isinstance(m, nn.Linear)] + [policy.action_net]
    arrays = {}
    for i, layer in enumerate(linears):
        arrays[f"w{i}"] = layer.weight.detach().cpu().numpy().astype(np.float32)
        arrays[f"b{i}"] = layer.bias.detach().cpu().numpy().astype(np.float32)
    np.savez_compressed(path, **arrays)


if __name__ == "__main__":
    from stable_baselines3 import PPO

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model")
    p.add_argument("out")
    a = p.parse_args()
    export(PPO.load(a.model, device="cpu"), a.out)
    print("wrote", a.out)
