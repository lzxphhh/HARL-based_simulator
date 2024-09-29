"""On-policy buffer for global prediction that uses Environment-Provided (EP) state."""
import torch
import numpy as np
from simulator.utils.envs_tools import get_shape_from_obs_space
from simulator.utils.trans_tools import _flatten, _sa_cast, _ma_cast


class OnPolicyGlobalPredictionBuffer:
    """On-policy buffer for global prediction data storage."""

    def __init__(self, args, share_obs_space, num_agents):
        """Initialize on-policy global prediction buffer.
        Args:
            args: (dict) arguments
            share_obs_space: (gym.Space or list) share observation space
        """
        self.episode_length = args["episode_length"]  # 每个环境的episode长度
        self.n_rollout_threads = args["n_rollout_threads"] # 多进程环境数量
        self.hidden_sizes = args["hidden_sizes"] # 网络的隐藏层大小
        self.rnn_hidden_size = self.hidden_sizes[-1] # rnn隐藏层大小
        self.recurrent_n = args["recurrent_n"] # rnn的层数

        self.gamma = args["gamma"]  # 折扣因子
        self.gae_lambda = args["gae_lambda"]  # GAE的参数
        self.use_gae = args["use_gae"]  # 是否使用GAE

        self.use_proper_time_limits = args["use_proper_time_limits"]  # 是否考虑episode的提前结束

        share_obs_shape = get_shape_from_obs_space(share_obs_space)  # 获取单个智能体共享状态空间的形状，tuple of integer. eg: （54，）

        if isinstance(share_obs_shape[-1], list):
            share_obs_shape = share_obs_shape[:1]

        """
        Global Prediction Buffer里储存了： ALL (np.ndarray) NOTE： 在EP中所有agent的全局状态是一样的
        1. self.share_obs: 全局状态 [episode_length + 1, 进程数量, share_obs_shape]
        2. self.rnn_states: prediction的rnn状态 [episode_length + 1, 进程数量, recurrent_n, rnn_hidden_size]
        3. self.masks: 每一步的mask,环境是否done [episode_length + 1, 进程数量, 1]
        4. self.bad_masks: 每一步的bad_mask,是否提前结束truncation [episode_length + 1, 进程数量, 1] 和6一起看
        5. self.prediction_errors: 每一步的预测误差 [episode_length, 进程数量, 1]
        """

        # Buffer for share observations

        self.share_obs = np.zeros(
            (self.episode_length + 1,
             self.n_rollout_threads,
             *share_obs_shape),
            dtype=np.float32,
        )

        # Buffer for rnn states of prediction
        self.rnn_states_global = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                # num_agents,
                self.recurrent_n,
                self.rnn_hidden_size,
            ),
            dtype=np.float32,
        )

        self.actions = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, 1),
            dtype=np.float32,
        )

        # Buffer for prediction errors
        self.global_prediction_errors = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, 1), dtype=np.float32
        )
        # Buffer for value predictions made by this predictor
        self.value_preds = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32
        )

        # Buffer for masks indicating whether an episode is done at each timestep
        self.masks = np.ones(
            (self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32
        )

        # Buffer for bad masks indicating truncation and termination.
        # If 0, trunction; if 1 and masks is 0, termination; else, not done yet.
        self.bad_masks = np.ones_like(self.masks)

        self.step = 0

    def insert(
            self,
            share_obs,
            actions,
            global_prediction_errors,
            rnn_states_global,
            value_preds,
            masks,
    ):
        """Insert data into buffer."""
        self.share_obs[self.step + 1] = share_obs.copy()
        self.actions[self.step] = actions.copy()
        self.global_prediction_errors[self.step] = global_prediction_errors.copy()
        self.rnn_states_global[self.step + 1] = rnn_states_global.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.masks[self.step + 1] = masks.copy()

        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        """After an update, copy the data at the last step to the first position of the buffer."""
        self.share_obs[0] = self.share_obs[-1].copy()
        self.rnn_states_global[0] = self.rnn_states_global[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()

    def get_mean_prediction_errors(self):
        """Get mean rewards for logging."""
        flattened_prediction_errors = self.global_prediction_errors.flatten()
        return np.mean(flattened_prediction_errors)

    def recurrent_generator_prediction(self, prediction_num_mini_batch, data_chunk_length):
        """Training data generator for global prediction that uses RNN network.
        This generator splits the trajectories into chunks of length data_chunk_length,
        and therefore maybe more efficient than the naive_recurrent_generator_actor in training.
        Args:
            prediction_num_mini_batch: (int) Number of mini batches for prediction.
            data_chunk_length: (int) Length of data chunks.
        """

        # get episode_length, n_rollout_threads, and mini_batch_size
        episode_length, n_rollout_threads = self.global_prediction_errors.shape[0:2]
        batch_size = n_rollout_threads * episode_length
        data_chunks = batch_size // data_chunk_length
        mini_batch_size = data_chunks // prediction_num_mini_batch

        assert (
            episode_length % data_chunk_length == 0
        ), f"episode length ({episode_length}) must be a multiple of data chunk length ({data_chunk_length})."
        assert data_chunks >= 2, "need larger batch size"

        # shuffle indices
        rand = torch.randperm(data_chunks).numpy()
        sampler = [
            rand[i * mini_batch_size : (i + 1) * mini_batch_size]
            for i in range(prediction_num_mini_batch)
        ]

        # The following data operations first transpose the first two dimensions of the data (episode_length, n_rollout_threads)
        # to (n_rollout_threads, episode_length), then reshape the data to (n_rollout_threads * episode_length, *dim).
        # Take share_obs shape as an example:
        # (episode_length + 1, n_rollout_threads, *share_obs_shape) --> (episode_length, n_rollout_threads, *share_obs_shape)
        # --> (n_rollout_threads, episode_length, *share_obs_shape) --> (n_rollout_threads * episode_length, *share_obs_shape)
        if len(self.share_obs.shape) > 3:
            share_obs = (
                self.share_obs[:-1]
                .transpose(1, 0, 2, 3, 4)
                .reshape(-1, *self.share_obs.shape[2:])
            )
        else:
            share_obs = _sa_cast(self.share_obs[:-1])
        actions = _flatten(episode_length, n_rollout_threads, self.actions)
        value_preds = _sa_cast(self.value_preds[:-1])
        global_prediction_errors = _flatten(episode_length, n_rollout_threads, self.global_prediction_errors)
        masks = _sa_cast(self.masks[:-1])
        rnn_states_global = (
            self.rnn_states_global[:-1]
            .transpose(1, 0, 2, 3)
            .reshape(-1, *self.rnn_states_global.shape[2:])
        )

        # generate mini-batches
        for indices in sampler:
            share_obs_batch = []
            rnn_states_global_batch = []
            actions_batch = []
            global_prediction_errors_batch = []
            value_preds_batch = []
            masks_batch = []

            for index in indices:
                ind = index * data_chunk_length
                share_obs_batch.append(share_obs[ind : ind + data_chunk_length])
                actions_batch.append(actions[ind : ind + data_chunk_length])
                global_prediction_errors_batch.append(
                    global_prediction_errors[ind : ind + data_chunk_length]
                )
                masks_batch.append(masks[ind : ind + data_chunk_length])
                rnn_states_global_batch.append(
                    rnn_states_global[ind]
                )  # only the beginning rnn states are needed
                value_preds_batch.append(value_preds[ind: ind + data_chunk_length])

            L, N = data_chunk_length, mini_batch_size
            # These are all ndarrays of size (data_chunk_length, mini_batch_size, *dim)
            share_obs_batch = np.stack(share_obs_batch, axis=1)
            actions_batch = np.stack(actions_batch, axis=1)
            global_prediction_errors_batch = np.stack(global_prediction_errors_batch, axis=1)
            masks_batch = np.stack(masks_batch, axis=1)
            # rnn_states_batch is a (mini_batch_size, *dim) ndarray
            rnn_states_global_batch = np.stack(rnn_states_global_batch).reshape(
                N, *self.rnn_states_global.shape[2:]
            )
            value_preds_batch = np.stack(value_preds_batch, axis=1)

            # Flatten the (data_chunk_length, mini_batch_size, *dim) ndarrays to (data_chunk_length * mini_batch_size, *dim)
            share_obs_batch = _flatten(L, N, share_obs_batch)
            actions_batch = _flatten(L, N, actions_batch)
            global_prediction_errors_batch = _flatten(L, N, global_prediction_errors_batch)
            masks_batch = _flatten(L, N, masks_batch)
            value_preds_batch = _flatten(L, N, value_preds_batch)

            yield share_obs_batch, actions_batch, global_prediction_errors_batch, rnn_states_global_batch,  value_preds_batch, masks_batch
