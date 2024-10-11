"""On-policy buffer for actor."""

import torch
import numpy as np
from simulator.utils.trans_tools import _flatten, _sa_cast
from simulator.utils.envs_tools import get_shape_from_obs_space, get_shape_from_act_space

class OnPolicyLocalPredictionBuffer:
    """On-policy buffer for local prediction data storage."""

    def __init__(self, args, obs_space, act_space):
        """Initialize on-policy actor buffer.
        Args:
            args: (dict) arguments # yaml里model和algo的config打包作为args进入OnPolicyActorBuffer
            obs_space: (gym.Space or list) observation space # 单个智能体的观测空间 eg: Box (18,)
            act_space: (gym.Space) action space # 单个智能体的动作空间 eg: Discrete(5,)
        """
        self.episode_length = args["episode_length"]  # 每个环境的episode长度
        self.n_rollout_threads = args["n_rollout_threads"]  # 多进程环境数量
        self.hidden_sizes = args["hidden_sizes"]  # actor网络的隐藏层大小
        self.rnn_hidden_size = self.hidden_sizes[-1]  # rnn隐藏层大小
        self.recurrent_n = args["recurrent_n"]  # rnn的层数

        obs_shape = get_shape_from_obs_space(obs_space)  # 获取单个智能体观测空间的形状，tuple of integer. eg: （18，）

        if isinstance(obs_shape[-1], list):
            obs_shape = obs_shape[:1]

        """
        Actor Buffer里储存了： ALL (np.ndarray)
        1. self.obs: local agent inputs to the actor. # 当前智能体的输入 [episode_length+1, 进程数量, obs_shape]
        2. self.rnn_states: rnn states of the actor. # 当前智能体的rnn状态 [episode_length+1, 进程数量, rnn层数, rnn层大小]
        3. self.available_actions: available actions of the actor. # 当前智能体的可用动作（仅离散） [episode_length+1, 进程数量, 动作空间大小]
        4. self.actions: actions of the actor. # 当前智能体的动作 [episode_length, 进程数量, 1（单个离散）]
        5. self.action_log_probs: action log probs of the actor. # 当前智能体选取的动作的log概率 [episode_length, 进程数量, 1（单个离散）]
        6. self.masks: 这个agent每一步的mask,是否done (rnn需要reset)  [episode_length+1, 进程数量, 1]
        7. self.active_masks:这个agent每一步的是否存活[episode_length+1, 进程数量, 1]
        """
        # Buffer for observations of this actor.
        self.obs = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, *obs_shape),
            dtype=np.float32,
        )

        # Buffer for rnn states of this actor.
        self.rnn_states_actor = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                self.recurrent_n,
                self.rnn_hidden_size,
            ),
            dtype=np.float32,
        )

        # Buffer for available actions of this actor.
        if act_space.__class__.__name__ == "Discrete":
            self.available_actions = np.ones(
                (self.episode_length + 1, self.n_rollout_threads, act_space.n),
                dtype=np.float32,
            )
        else:
            self.available_actions = None

        # 获取动作空间的维度，integer. eg: 1-》单个离散，
        act_shape = get_shape_from_act_space(act_space)

        # Buffer for actions of this actor.
        self.actions = np.zeros(
            (self.episode_length, self.n_rollout_threads, act_shape), dtype=np.float32
        )

        self.time_steps = [1, 2, 3, 4, 5]
        self.categories = ['all', 'CAV', 'HDV']
        self.local_prediction_errors = {}
        # Buffer for prediction errors of this actor.
        for t in self.time_steps:
            for category in self.categories:
                key = f'{t}s_{category}'
                self.local_prediction_errors[key] = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1), dtype=np.float32
            )

        # Buffer for rnn states of this predictor.
        self.rnn_states_local = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                self.recurrent_n,
                self.rnn_hidden_size,
            ),
            dtype=np.float32,
        )

        # Buffer for masks of this actor. Masks denotes at which point should the rnn states be reset.
        # 当前这个agent在不同并行环境的不同时间点是否done，如果done，那么就需要reset rnn
        self.masks = np.ones((self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32)

        # Buffer for active masks of this actor. Active masks denotes whether the agent is alive.
        # 当前这个agent在不同并行环境的不同时间点是否存活，如果不存活，那么就不需要计算loss，不需要更新参数
        self.active_masks = np.ones_like(self.masks)

        # happo的参数 # TODO: 这个参数是干嘛的？
        self.factor = None

        # 当前所有并行环境的当前步数
        self.step = 0

    def update_factor(self, factor):
        """Save factor for this actor.
        只有on_policy_ha_runner调用了这个函数"""
        self.factor = factor.copy()

    def insert(
        self,
        obs,
        rnn_states_actor,
        actions,
        localPre_errs,
        rnn_states_local,
        masks,
        active_masks=None,
    ):
        """Insert data into actor buffer."""
        self.obs[self.step + 1] = obs.copy()
        self.rnn_states_actor[self.step + 1] = rnn_states_actor.copy()
        self.actions[self.step] = actions.copy()
        for t in self.time_steps:
            for category in self.categories:
                key = f'{t}s_{category}'
                self.local_prediction_errors[key][self.step] = localPre_errs[key].copy()
        self.rnn_states_local[self.step + 1] = rnn_states_local.copy()
        self.masks[self.step + 1] = masks.copy()
        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks.copy()

        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        """After an update, copy the data at the last step to the first position of the buffer."""
        self.obs[0] = self.obs[-1].copy()
        self.rnn_states_actor[0] = self.rnn_states_actor[-1].copy()
        self.rnn_states_local[0] = self.rnn_states_local[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()

    def recurrent_generator_prediction(self, prediction_num_mini_batch, data_chunk_length):
        """Training data generator for actor that uses RNN network.
        当actor是RNN时，使用这个生成器。
        把轨迹分成长度为data_chunk_length的块，因此比naive_recurrent_generator_actor在训练时更有效率。
        This generator splits the trajectories into chunks of length data_chunk_length,
        and therefore maybe more efficient than the naive_recurrent_generator_actor in training.
        """

        # get episode_length, n_rollout_threads, and mini_batch_size
        # trajectory长度，进程数
        episode_length, n_rollout_threads = self.actions.shape[0:2]
        # batch_size = 进程数 * trajectory长度 (收集一次数据)
        batch_size = n_rollout_threads * episode_length
        # 把所有时间步根据data_chunk_length分成多个组时间步
        data_chunks = batch_size // data_chunk_length
        # 把data_chunks分成actor_num_mini_batch份
        mini_batch_size = data_chunks // prediction_num_mini_batch

        assert episode_length % data_chunk_length == 0, (
            f"episode length ({episode_length}) must be a multiple of data chunk length ({data_chunk_length})."
        )
        assert data_chunks >= 2, "need larger batch size"

        # shuffle indices
        rand = torch.randperm(data_chunks).numpy()
        sampler = [
            rand[i * mini_batch_size : (i + 1) * mini_batch_size]
            for i in range(prediction_num_mini_batch)
        ]

        # The following data operations first transpose the first two dimensions of the data (episode_length, n_rollout_threads)
        # to (n_rollout_threads, episode_length), then reshape the data to (n_rollout_threads * episode_length, *dim).
        # Take obs shape as an example:
        # (episode_length + 1, n_rollout_threads, *obs_shape) --> (episode_length, n_rollout_threads, *obs_shape)
        # --> (n_rollout_threads, episode_length, *obs_shape) --> (n_rollout_threads * episode_length, *obs_shape)
        if len(self.obs.shape) > 3:
            obs = self.obs[:-1].transpose(1, 0, 2, 3, 4).reshape(-1, *self.obs.shape[2:])
        else:
            obs = _sa_cast(self.obs[:-1])
        actions = _sa_cast(self.actions)
        local_prediction_errors = {}
        for t in self.time_steps:
            for category in self.categories:
                key = f'{t}s_{category}'
                local_prediction_errors[key] = _sa_cast(self.local_prediction_errors[key])
        # local_prediction_errors = _sa_cast(self.local_prediction_errors)
        masks = _sa_cast(self.masks[:-1])
        active_masks = _sa_cast(self.active_masks[:-1])
        rnn_states_actor = (
            self.rnn_states_actor[:-1].transpose(1, 0, 2, 3).reshape(-1, *self.rnn_states_actor.shape[2:])
        )
        rnn_states_local = (
            self.rnn_states_local[:-1].transpose(1, 0, 2, 3).reshape(-1, *self.rnn_states_local.shape[2:])
        )

        # generate mini-batches
        for indices in sampler:
            obs_batch = []
            rnn_states_actor_batch = []
            actions_batch = []
            local_prediction_errors_batch = {}
            for t in self.time_steps:
                for category in self.categories:
                    key = f'{t}s_{category}'
                    local_prediction_errors_batch[key] = []
            rnn_states_local_batch = []
            masks_batch = []
            active_masks_batch = []

            for index in indices:
                ind = index * data_chunk_length
                obs_batch.append(obs[ind : ind + data_chunk_length])
                rnn_states_actor_batch.append(rnn_states_actor[ind])
                actions_batch.append(actions[ind : ind + data_chunk_length])
                for t in self.time_steps:
                    for category in self.categories:
                        key = f'{t}s_{category}'
                        local_prediction_errors_batch[key].append(local_prediction_errors[key][ind : ind + data_chunk_length])
                # local_prediction_errors_batch.append(local_prediction_errors[ind : ind + data_chunk_length])
                rnn_states_local_batch.append(rnn_states_local[ind])  # TODO:only the beginning rnn states are needed?
                masks_batch.append(masks[ind : ind + data_chunk_length])
                active_masks_batch.append(active_masks[ind : ind + data_chunk_length])

            L, N = data_chunk_length, mini_batch_size
            # These are all ndarrays of size (data_chunk_length, mini_batch_size, *dim)
            obs_batch = np.stack(obs_batch, axis=1)
            rnn_states_actor_batch = np.stack(rnn_states_actor_batch).reshape(N, *self.rnn_states_actor.shape[2:])
            actions_batch = np.stack(actions_batch, axis=1)
            for t in self.time_steps:
                for category in self.categories:
                    key = f'{t}s_{category}'
                    local_prediction_errors_batch[key] = np.stack(local_prediction_errors_batch[key], axis=1)
            # local_prediction_errors_batch = np.stack(local_prediction_errors_batch, axis=1)
            masks_batch = np.stack(masks_batch, axis=1)
            active_masks_batch = np.stack(active_masks_batch, axis=1)
            # rnn_states_batch is a (mini_batch_size, *dim) ndarray
            rnn_states_local_batch = np.stack(rnn_states_local_batch).reshape(N, *self.rnn_states_local.shape[2:])

            # flatten the (data_chunk_length, mini_batch_size, *dim) ndarrays to (data_chunk_length * mini_batch_size, *dim)
            obs_batch = _flatten(L, N, obs_batch)
            actions_batch = _flatten(L, N, actions_batch)
            prediction_errors_batch = {}
            for t in self.time_steps:
                for category in self.categories:
                    key = f'{t}s_{category}'
                    prediction_errors_batch[key] = _flatten(L, N, local_prediction_errors_batch[key])
            # prediction_errors_batch = _flatten(L, N, local_prediction_errors_batch)
            masks_batch = _flatten(L, N, masks_batch)
            active_masks_batch = _flatten(L, N, active_masks_batch)
            yield obs_batch, rnn_states_actor_batch, actions_batch, prediction_errors_batch, rnn_states_local_batch, masks_batch, active_masks_batch
