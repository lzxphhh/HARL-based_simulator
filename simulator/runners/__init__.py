"""Runner registry."""
from simulator.runners.on_policy_ha_runner import OnPolicyHARunner
from simulator.runners.on_policy_ma_runner import OnPolicyMARunner
# from simulator.runners.on_policy_prediction_runner import OnPolicyPredictionRunner

MARL_RUNNER_REGISTRY = {
    "happo": OnPolicyHARunner,
    "mappo": OnPolicyMARunner,
}

# PREDICTION_RUNNER_REGISTRY = {
#     "GAT": OnPolicyPredictionRunner,
# }
