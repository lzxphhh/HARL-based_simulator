"""MAPPO algorithm."""
import numpy as np
import torch
import torch.nn as nn
from simulator.utils.envs_tools import check
from simulator.utils.models_tools import get_grad_norm
from simulator.algorithms.actors.on_policy_base import OnPolicyBase


class MAPPO(OnPolicyBase):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        """Initialize MAPPO algorithm.
        Args:
            args: (dict) arguments. # yaml里model和algo的config打包作为args进入OnPolicyBase
            obs_space: (gym.spaces or list) observation space. # 单个智能体的观测空间 eg: Box (18,)
            act_space: (gym.spaces) action space. # 单个智能体的动作空间 eg: Discrete(5,)
            device: (torch.device) device to use for tensor operations.
        """
        super(MAPPO, self).__init__(args, obs_space, act_space, device)

        # 可以去看https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
        self.clip_param = args["clip_param"]  # PPO的clip参数
        self.ppo_epoch = args["ppo_epoch"]  # Number of epoch when optimizing the surrogate loss
        self.actor_num_mini_batch = args["actor_num_mini_batch"]
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
            rnn_states_batch,
            actions_batch,
            action_losss_batch,
            masks_batch,
            active_masks_batch,
            old_action_log_probs_batch,
            adv_targ,
            available_actions_batch,
        ) = sample

        # 检查数据类型和设备
        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)

        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # 计算当前新策略的log_prob以及entropy
        # reshape to do in a single forward pass for all steps
        # 在on policy base里面
        action_log_probs, dist_entropy, _ = self.evaluate_actions(
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
        )
        # update actor
        # 计算新旧策略比值
        imp_weights = getattr(torch, self.action_aggregation)(
            torch.exp(action_log_probs - old_action_log_probs_batch),
            dim=-1,
            keepdim=True,
        )
        # 计算surr1和surr2
        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self.use_policy_active_masks:
            # 死掉的agent的loss不计算
            policy_action_loss = (
                -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True) * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        # 计算actor的loss (为什么没有entropy loss)
        policy_loss = policy_action_loss

        self.actor_optimizer.zero_grad()

        # policy entropy -- original code
        # (policy_loss - dist_entropy * self.entropy_coef).backward()
        # policy_loss.backward()

        # add prediction loss
        entropy_loss = policy_loss - dist_entropy * self.entropy_coef
        action_loss = np.mean(action_losss_batch)
        # dynamic weight of each loss
        weight_action_loss = 0
        n_action_loss = self.args["action_loss_subzone"]
        w_action_loss = self.args["action_loss_weight"]
        for i in range(len(n_action_loss)):
            if action_loss > n_action_loss[i]:
                weight_action_loss = w_action_loss[i]
        combined_loss = weight_action_loss * action_loss + entropy_loss
        combined_loss.backward()

        if self.use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_grad_norm(self.actor.parameters())

        self.actor_optimizer.step()

        return policy_loss, entropy_loss, action_loss, combined_loss, dist_entropy, actor_grad_norm, imp_weights

    def train(self, actor_buffer, advantages, state_type):
        """Perform a training update for non-parameter-sharing MAPPO using minibatch GD.
        Args:
            actor_buffer: (OnPolicyActorBuffer) buffer containing training data related to actor.
            advantages: (np.ndarray) advantages. [num_steps, 环境数量, 1]
            state_type: (str) type of state.
        Returns:
            train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        train_info = {}
        train_info["policy_loss"] = 0
        train_info["entropy_loss"] = 0
        train_info["action_loss"] = 0
        train_info["combined_loss"] = 0
        train_info["dist_entropy"] = 0
        train_info["actor_grad_norm"] = 0
        train_info["ratio"] = 0

        # 如果这段trajectory全是done，那么不需要训练
        if np.all(actor_buffer.active_masks[:-1] == 0.0):
            return train_info

        if state_type == "EP":
            # 计算advantages的均值和标准差
            advantages_copy = advantages.copy()
            # 如果这个traj哪一步是done，那么这一步的advantage就是nan
            advantages_copy[actor_buffer.active_masks[:-1] == 0.0] = np.nan
            # 计算advantages的均值和标准差
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            # 对advantages进行归一化 trick
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        # 通常对一段通过on-policy采样的trajectory优化多个epoch
        for _ in range(self.ppo_epoch):
            if self.use_recurrent_policy:
                data_generator = actor_buffer.recurrent_generator_actor(
                    advantages, self.actor_num_mini_batch, self.data_chunk_length
                )
            elif self.use_naive_recurrent_policy:
                data_generator = actor_buffer.naive_recurrent_generator_actor(
                    advantages, self.actor_num_mini_batch
                )
            else:
                data_generator = actor_buffer.feed_forward_generator_actor(
                    advantages, self.actor_num_mini_batch
                )

            # sample出每一个batch的数据
            for sample in data_generator:
                # 计算actor的loss并且更新
                policy_loss, entropy_loss, action_loss, combined_loss, dist_entropy, actor_grad_norm, imp_weights  = self.update(sample)

                train_info["policy_loss"] += policy_loss.item()
                train_info["entropy_loss"] += entropy_loss.item()
                train_info["action_loss"] += action_loss.item()
                train_info["combined_loss"] += combined_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["actor_grad_norm"] += actor_grad_norm
                train_info["ratio"] += imp_weights.mean()

        # 总共更新了多少次
        num_updates = self.ppo_epoch * self.actor_num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    def share_param_train(self, actor_buffer, advantages, num_agents, state_type):
        """Perform a training update for parameter-sharing MAPPO using minibatch GD.
        共享参数版本的MAPPO的训练&更新 -- minibatch GD
        Args:
            actor_buffer: (list[OnPolicyActorBuffer]) buffer containing training data related to actor.
            advantages: (np.ndarray) advantages. 【episode_length, 进程数量, 1】
            num_agents: (int) number of agents.
            state_type: (str) type of state.
        Returns:
            train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        train_info = {}
        train_info["policy_loss"] = 0
        train_info["entropy_loss"] = 0
        train_info["action_loss"] = 0
        train_info["combined_loss"] = 0
        train_info["dist_entropy"] = 0
        train_info["actor_grad_norm"] = 0
        train_info["ratio"] = 0

        if state_type == "EP":
            advantages_ori_list = []
            advantages_copy_list = []
            # 对于每个agent，计算advantages的均值和标准差（每个agent的advantages都是一样的）
            for agent_id in range(num_agents):
                advantages_ori = advantages.copy()
                advantages_ori_list.append(advantages_ori)
                advantages_copy = advantages.copy()
                advantages_copy[actor_buffer[agent_id].active_masks[:-1] == 0.0] = np.nan
                advantages_copy_list.append(advantages_copy)
            advantages_ori_tensor = np.array(advantages_ori_list)
            advantages_copy_tensor = np.array(advantages_copy_list)
            mean_advantages = np.nanmean(advantages_copy_tensor)
            std_advantages = np.nanstd(advantages_copy_tensor)
            normalized_advantages = (advantages_ori_tensor - mean_advantages) / (std_advantages + 1e-5)
            advantages_list = []
            for agent_id in range(num_agents):
                advantages_list.append(normalized_advantages[agent_id])
        elif state_type == "FP":
            advantages_list = []
            for agent_id in range(num_agents):
                advantages_list.append(advantages[:, :, agent_id])

        for _ in range(self.ppo_epoch):
            data_generators = []
            for agent_id in range(num_agents):
                if self.use_recurrent_policy:
                    data_generator = actor_buffer[agent_id].recurrent_generator_actor(
                        advantages_list[agent_id],
                        self.actor_num_mini_batch,
                        self.data_chunk_length,
                    )
                elif self.use_naive_recurrent_policy:
                    data_generator = actor_buffer[agent_id].naive_recurrent_generator_actor(
                        advantages_list[agent_id], self.actor_num_mini_batch
                    )
                else:
                    data_generator = actor_buffer[agent_id].feed_forward_generator_actor(
                        advantages_list[agent_id], self.actor_num_mini_batch
                    )
                data_generators.append(data_generator)

            for _ in range(self.actor_num_mini_batch):
                batches = [[] for _ in range(9)] # 注意如果self.factor not none 需要更改个数
                for generator in data_generators:
                    sample = next(generator)
                    for i in range(9):
                        batches[i].append(sample[i])
                for i in range(8):
                    batches[i] = np.concatenate(batches[i], axis=0)
                if batches[8][0] is None:  # 判断有没有factor
                    batches[8] = None
                else:
                    batches[8] = np.concatenate(batches[8], axis=0)
                policy_loss, entropy_loss, action_loss, combined_loss, dist_entropy, actor_grad_norm, imp_weights = self.update(
                    tuple(batches)
                )

                train_info["policy_loss"] += policy_loss.item()
                train_info["entropy_loss"] += entropy_loss.item()
                train_info["action_loss"] += action_loss.item()
                train_info["combined_loss"] += combined_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["actor_grad_norm"] += actor_grad_norm
                train_info["ratio"] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.actor_num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info
