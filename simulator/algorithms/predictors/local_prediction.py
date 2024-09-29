import numpy as np
import torch
import torch.nn as nn
from simulator.utils.envs_tools import check
from simulator.utils.models_tools import get_grad_norm
from simulator.algorithms.predictors.prediction_base import PredictionBase


class LOC_PREDICTION(PredictionBase):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        """Initialize lcoal world model.
        Args:
            args: (dict) arguments. # yaml里model和algo的config打包作为args进入OnPolicyBase
            obs_space: (gym.spaces or list) observation space. # 单个智能体的观测空间 eg: Box (18,)
            act_space: (gym.spaces) action space. # 单个智能体的动作空间 eg: Discrete(5,)
            device: (torch.device) device to use for tensor operations.
        """
        super(LOC_PREDICTION, self).__init__(args, obs_space, act_space, device)

        self.clip_param = args["clip_param"]  # PPO的clip参数
        self.ppo_epoch = args["ppo_epoch"]  # Number of epoch when optimizing the surrogate loss
        self.prediction_num_mini_batch = args["actor_num_mini_batch"]
        self.entropy_coef = args["entropy_coef"]  # Entropy coefficient for the loss calculation
        self.use_max_grad_norm = args["use_max_grad_norm"]  # TODO PPO相关
        self.max_grad_norm = args["max_grad_norm"]  # maximum value for the gradient clipping

    def update(self, sample):
        """Update actor network.
        Args:
            sample: (Tuple) contains data batch with which to update networks.
        Returns:
            policy_loss: (torch.Tensor) actor(policy) loss value.
            dist_entropy: (torch.Tensor) action entropies.
            actor_grad_norm: (torch.Tensor) gradient norm from actor update.
            imp_weights: (torch.Tensor) importance sampling weights.
        """
        """
            obs_batch: 【n_rollout_threads * episode_length * num_agents, *obs_shape】
            rnn_states_batch: [mini_batch_size, 1, rnn_hidden_dim]
            actions_batch: 【n_rollout_threads * episode_length, *act_shape=1】
            masks_batch: 【n_rollout_threads * episode_length, *mask=1】
            active_masks_batch: 【n_rollout_threads * episode_length, *mask=1】
            old_action_log_probs_batch: 【n_rollout_threads * episode_length, *act_shape=1】
            adv_targ: 【n_rollout_threads * episode_length, 1】
            available_actions_batch: 【n_rollout_threads * episode_length, action_space】
        """
        (
            obs_batch,
            rnn_states_actor_batch,
            actions_batch,
            prediction_errors_batch,
            rnn_states_local_batch,
            masks_batch,
            active_masks_batch,
        ) = sample

        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        # TODO: 需要设计loss function
        tensor_prediction_errors = torch.tensor(prediction_errors_batch, device=self.device, requires_grad=True)
        prediction_loss = torch.mean(tensor_prediction_errors)

        self.local_predictor_optimizer.zero_grad()

        prediction_loss.backward()

        self.local_predictor_optimizer.step()

        return prediction_loss

    def train(self, local_predictor_buffer, num_agents):
        train_info = {}
        train_info["prediction_loss"] = 0

        for _ in range(self.ppo_epoch):
            data_generators = []
            for agent_id in range(num_agents):
                data_generator = local_predictor_buffer[agent_id].recurrent_generator_prediction(
                    self.prediction_num_mini_batch, self.data_chunk_length
                )
                data_generators.append(data_generator)
            for _ in range(self.prediction_num_mini_batch):
                batches = [[] for _ in range(7)]
                for generator in data_generators:
                    sample = next(generator)
                    for i in range(7):
                        batches[i].append(sample[i])
                for i in range(7):
                    batches[i] = np.concatenate(batches[i], axis=0)
                prediction_loss = self.update(
                    tuple(batches)
                )

                train_info["prediction_loss"] += prediction_loss.item()

            num_updates = self.ppo_epoch * self.prediction_num_mini_batch

            for k in train_info.keys():
                train_info[k] /= num_updates

            return train_info