"""Runner for on-policy prediction algorithms."""
import numpy as np
import torch
from simulator.runners.predict_base_runner import PredictBaseRunner

class OnPolicyPredictionRunner(PredictBaseRunner):
    """Runner for on-policy prediction algorithms."""

    def train(self):
        """Training procedure for predictor."""
        local_predictor_train_infos = []

        local_predictor_train_info = self.local_predictor[0].train(self.local_predictor_buffer, self.num_agents)
        for _ in torch.randperm(self.num_agents):
            local_predictor_train_infos.append(local_predictor_train_info)

        global_predictor_train_infos = self.global_predictor.train(self.global_predictor_buffer, self.value_normalizer)

        return local_predictor_train_infos, global_predictor_train_infos
