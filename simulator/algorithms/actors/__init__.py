"""Algorithm registry."""
from simulator.algorithms.actors.happo import HAPPO
from simulator.algorithms.actors.mappo import MAPPO

ALGO_REGISTRY = {
    "happo": HAPPO,
    "mappo": MAPPO,
}
