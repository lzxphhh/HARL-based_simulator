"""V Critic."""
import torch
import torch.nn as nn
from simulator.utils.models_tools import (
    get_grad_norm,
    huber_loss,
    mse_loss,
    update_linear_schedule,
)
from simulator.utils.envs_tools import check
from simulator.models.value_function_models.v_net import VNet


class VCritic:
    """V Critic.
    Critic that learns a V-function.
    """

    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        self.args = args  # yaml里model和algo的config打包作为args进入VCritic
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)  # dtype和device

        self.clip_param = args["clip_param"]  # PPO的clip参数

        # TODO: PPO相关 在下面的update函数中用到对应
        self.critic_epoch = args["critic_epoch"]
        self.critic_num_mini_batch = args["critic_num_mini_batch"]
        self.data_chunk_length = args["data_chunk_length"]
        self.value_loss_coef = args["value_loss_coef"]
        self.max_grad_norm = args["max_grad_norm"]  # The maximum value for the gradient clipping
        self.huber_delta = args["huber_delta"]

        self.use_recurrent_policy = args["use_recurrent_policy"]
        self.use_naive_recurrent_policy = args["use_naive_recurrent_policy"]
        self.use_max_grad_norm = args["use_max_grad_norm"]
        self.use_clipped_value_loss = args["use_clipped_value_loss"]
        self.use_huber_loss = args["use_huber_loss"]
        self.use_policy_active_masks = args["use_policy_active_masks"]

        self.critic_lr = args["critic_lr"]  # critic的学习率
        self.opti_eps = args["opti_eps"]  # critic Adam优化器的eps
        self.weight_decay = args["weight_decay"]  # critic Adam优化器的weight_decay

        self.share_obs_space = cent_obs_space  # 共享观测空间/全局状态空间 eg. Box(-inf, inf, (54,), float32)

        # 初始化critic网络，输入为单个agent的共享观测空间，输出为1维分数R+rnn_state
        self.critic = VNet(args, self.share_obs_space, self.device)

        # 初始化critic网络的优化器
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )

    def lr_decay(self, episode, episodes):
        """Decay the actor and critic learning rates.
        episode是当前episode的index，episodes是总共需要跑多少个episode
        Args:
            episode: (int) current training episode.
            episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_values(self, cent_obs, rnn_states_critic, masks):
        """Get value function predictions.
        Args:
            cent_obs: (np.ndarray) centralized input to the critic.
            rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
            masks: (np.ndarray) denotes points at which RNN states should be reset.
        Returns:
            values: (torch.Tensor) value function predictions. (并行环境数量, 1)
            rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """
        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)
        return values, rnn_states_critic

    def cal_value_loss(
            self, values, value_preds_batch, return_batch, value_normalizer=None
    ):
        """Calculate value function loss.
        Args:
            values: (torch.Tensor) value function predictions.
            value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
            return_batch: (torch.Tensor) reward to go returns.
            value_normalizer: (ValueNorm) normalize the rewards, denormalize critic outputs.
        Returns:
            value_loss: (torch.Tensor) value function loss.
        """
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
            -self.clip_param, self.clip_param
        )
        if value_normalizer is not None:
            value_normalizer.update(return_batch)
            error_clipped = (
                    value_normalizer.normalize(return_batch) - value_pred_clipped
            )
            error_original = value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self.use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self.use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        value_loss = value_loss.mean()

        return value_loss

    def update(self, sample, value_normalizer=None):
        """Update critic network.
        Args:
            sample: (Tuple) contains data batch with which to update networks.
            value_normalizer: (ValueNorm) normalize the rewards, denormalize critic outputs.
        Returns:
            value_loss: (torch.Tensor) value function loss.
            critic_grad_norm: (torch.Tensor) gradient norm from critic update.
        """
        """
        share_obs_batch: (torch.Tensor) agent的共享观测，shape为【n_rollout_threads * episode_length, *share_obs_shape】
        rnn_states_critic_batch: (torch.Tensor) critic的RNN状态，shape为[mini_batch_size, 1, rnn_hidden_dim]
        value_preds_batch: (torch.Tensor) critic的预测值，shape为【n_rollout_threads * episode_length, 1】
        return_batch: (torch.Tensor) agent的reward to go，shape为【n_rollout_threads * episode_length, 1】
        masks_batch: (torch.Tensor) agent的masks，shape为【n_rollout_threads * episode_length, 1】
        """
        (
            share_obs_batch,
            rnn_states_critic_batch,
            value_preds_batch,
            return_batch,
            masks_batch,
        ) = sample

        # 检查数据类型
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)

        values, _ = self.get_values(
            share_obs_batch, rnn_states_critic_batch, masks_batch
        )

        # 计算critic的loss
        value_loss = self.cal_value_loss(
            values, value_preds_batch, return_batch, value_normalizer=value_normalizer
        )

        self.critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self.use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.max_grad_norm
            )
        else:
            critic_grad_norm = get_grad_norm(self.critic.parameters())

        self.critic_optimizer.step()

        return value_loss, critic_grad_norm

    def train(self, critic_buffer, value_normalizer=None):
        """Perform a training update using minibatch GD.
        Args:
            critic_buffer: (OnPolicyCriticBufferEP or OnPolicyCriticBufferFP) buffer containing training data related to critic.
            value_normalizer: (ValueNorm) normalize the rewards, denormalize critic outputs.
        Returns:
            train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """

        train_info = {}

        train_info["value_loss"] = 0
        train_info["critic_grad_norm"] = 0

        # critic_epoch是critic更新的次数
        for _ in range(self.critic_epoch):
            if self.use_recurrent_policy:
                data_generator = critic_buffer.recurrent_generator_critic(
                    self.critic_num_mini_batch, self.data_chunk_length
                )
            elif self.use_naive_recurrent_policy:
                data_generator = critic_buffer.naive_recurrent_generator_critic(
                    self.critic_num_mini_batch
                )
            else:
                data_generator = critic_buffer.feed_forward_generator_critic(
                    self.critic_num_mini_batch
                )
            # sample出
            for sample in data_generator:
                # 计算critic的loss并且更新
                value_loss, critic_grad_norm = self.update(
                    sample, value_normalizer=value_normalizer
                )

                train_info["value_loss"] += value_loss.item()
                train_info["critic_grad_norm"] += critic_grad_norm

        num_updates = self.critic_epoch * self.critic_num_mini_batch

        for k, _ in train_info.items():
            train_info[k] /= num_updates

        return train_info

    def prep_training(self):
        """Prepare for training."""
        self.critic.train()

    def prep_rollout(self):
        """Prepare for rollout."""
        self.critic.eval()
