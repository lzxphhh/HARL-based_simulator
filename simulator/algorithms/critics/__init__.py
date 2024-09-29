"""Critic registry."""
from simulator.algorithms.critics.v_critic import VCritic
CRITIC_REGISTRY = {
    "happo": VCritic,
    "mappo": VCritic,
}
