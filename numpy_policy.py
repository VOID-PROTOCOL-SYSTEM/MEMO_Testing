"""Torch-free, WASM/Pyodide-safe re-implementation of the pieces of
train_hybrid_aco_policies.py that are needed at *inference* time inside
the marimo notebook.

Why this exists
----------------
`torch` has no wheel that runs inside Pyodide/marimo-WASM, so a GitHub
Pages WASM export can never `import torch`. The trained policy itself is
tiny (two 3-layer MLPs, ~64 hidden units), so instead of shipping torch
we ship its *weights* as plain .npz arrays (see convert_checkpoints.py)
and do the forward pass with numpy. This file is a drop-in replacement
for the two things the notebook actually calls from
`train_hybrid_aco_policies` at runtime:

    trainmod.Actor / trainmod.Critic   -> loaded from .npz instead of .pt
    trainmod.construct_plan(...)       -> identical logic, numpy instead
                                           of torch for the forward pass
                                           and Python's `random` instead
                                           of torch's RNG for sampling.

Everything else in train_hybrid_aco_policies.py (training loop, workers,
argparse, multiprocessing) is offline-training-only code and is not
imported here or needed in the deployed notebook.

Note on reproducibility: the original construct_plan sampled with
torch.multinomial, seeded by torch.manual_seed(). This module samples
with Python's `random` module instead, seeded the same way the rest of
the notebook seeds it (`random.seed(...)`). The *policy* is identical
(same trained weights, same softmax probabilities) but individual sampled
rollouts will differ bit-for-bit from a torch run with the same seed,
since the underlying RNG stream is different. Statistically this makes
no difference to solution quality.
"""

import os
import random as _random

import numpy as np


# ------------------------------------------------------------------
# 1. Numpy MLP standing in for the torch Actor / Critic
# ------------------------------------------------------------------

class NumpyMLP:
    """Linear(in,64) -> Tanh -> Linear(64,64) -> Tanh -> Linear(64,1),
    matching train_hybrid_aco_policies.Actor / .Critic exactly."""

    def __init__(self, w0, b0, w2, b2, w4, b4):
        self.w0, self.b0 = w0.astype(np.float32), b0.astype(np.float32)
        self.w2, self.b2 = w2.astype(np.float32), b2.astype(np.float32)
        self.w4, self.b4 = w4.astype(np.float32), b4.astype(np.float32)

    def __call__(self, feats):
        x = np.asarray(feats, dtype=np.float32)
        squeeze_back = x.ndim == 1
        if squeeze_back:
            x = x[None, :]
        h = np.tanh(x @ self.w0.T + self.b0)
        h = np.tanh(h @ self.w2.T + self.b2)
        out = (h @ self.w4.T + self.b4)[:, 0]
        return out[0] if squeeze_back else out

    def eval(self):
        return self  # kept only so call sites that do actor.eval() still work


def _mlp_from_npz(npz, prefix):
    return NumpyMLP(
        npz[f"{prefix}.net.0.weight"], npz[f"{prefix}.net.0.bias"],
        npz[f"{prefix}.net.2.weight"], npz[f"{prefix}.net.2.bias"],
        npz[f"{prefix}.net.4.weight"], npz[f"{prefix}.net.4.bias"],
    )


def load_policy(path):
    """Equivalent of the old torch.load(...).load_state_dict(...) pair,
    but reading the .npz produced by convert_checkpoints.py."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find policy checkpoint:\n{path}")
    npz = np.load(path)
    actor = _mlp_from_npz(npz, "actor")
    critic = _mlp_from_npz(npz, "critic")
    return actor, critic


# ------------------------------------------------------------------
# 2. Features - identical to train_hybrid_aco_policies.py (torch-free
#    already in the original, copied verbatim)
# ------------------------------------------------------------------

def candidate_features(here, cand, load, cap, spent, budget, dist, mass_of, value_of,
                        pheromone, n_remaining, n_total, dist_scale):
    if cand == "STOP":
        return [0.0, 0.0, 0.0, load / cap, spent / max(budget, 1e-9), 1.0,
                n_remaining / max(n_total, 1)]
    d = max(dist(here, cand), 1e-9)
    vpm = value_of[cand] / mass_of[cand]
    return [
        vpm / 5.0,
        d / max(dist_scale, 1e-9),
        mass_of[cand] / cap,
        load / cap,
        spent / max(budget, 1e-9),
        pheromone.get((here, cand), 1.0),
        n_remaining / max(n_total, 1),
    ]


def state_features(spent, budget, load, cap, remaining, value_of, mass_of, n_total):
    mean_vpm = (sum(value_of[u] / mass_of[u] for u in remaining) / len(remaining)
                if remaining else 0.0)
    return [spent / max(budget, 1e-9), load / cap, len(remaining) / max(n_total, 1),
            mean_vpm / 5.0]


# ------------------------------------------------------------------
# 3. construct_plan - same algorithm as train_hybrid_aco_policies.py,
#    numpy softmax/argmax instead of torch, Python `random` for sampling
# ------------------------------------------------------------------

def _softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def construct_plan(inst, actor, critic, pheromone, dist_scale, mode="sample"):
    """mode: 'greedy' -> deterministic argmax, no trajectory recorded.
             anything else -> stochastic sample (matches the original,
             where the notebook actually passes a bool through `mode`,
             so this always falls into the sampling branch - preserved
             as-is rather than "fixed", to keep behaviour identical to
             the pre-WASM notebook)."""
    SUPPLIES, CAP, BUDGET = inst['SUPPLIES'], inst['CAP'], inst['BUDGET']
    dist, trip_cost, best_order_fn = inst['dist'], inst['trip_cost'], inst['best_order']
    SHAFT, MASS, VALUE, EXIT_LEG = inst['SHAFT'], inst['MASS'], inst['VALUE'], inst['EXIT_LEG']

    remaining = set(SUPPLIES)
    n_total = len(SUPPLIES)
    plan, spent = [], 0.0
    trajectory = []

    while remaining:
        trip, here, load = [], SHAFT, 0
        while True:
            candidates = [u for u in remaining if u not in trip and load + MASS[u] <= CAP]
            options = list(candidates) + (["STOP"] if trip else [])
            if not options:
                break

            feats = np.array(
                [candidate_features(here, c, load, CAP, spent, BUDGET, dist, MASS, VALUE,
                                     pheromone, len(remaining), n_total, dist_scale)
                 for c in options],
                dtype=np.float32)
            probs = _softmax(actor(feats))

            if mode == "greedy":
                idx = int(np.argmax(probs))
            else:
                idx = _random.choices(range(len(options)), weights=probs.tolist(), k=1)[0]
                s_feat = state_features(spent, BUDGET, load, CAP, remaining, VALUE, MASS, n_total)
                trajectory.append((s_feat, feats.tolist(), idx))

            pick = options[idx]
            if pick == "STOP":
                break
            trip.append(pick)
            load += MASS[pick]
            here = pick

        if not trip:
            break
        trip = best_order_fn(trip)
        c = trip_cost(trip)
        if spent + c + EXIT_LEG > BUDGET:
            break
        spent += c
        plan.append(trip)
        for u in trip:
            remaining.discard(u)

    return plan, trajectory
