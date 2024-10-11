import torch
import torch.nn as nn
from simulator.utils.models_tools import (
    get_grad_norm,
    huber_loss,
    mse_loss,
    update_linear_schedule,
)
from simulator.utils.envs_tools import check
from simulator.models.predict_models.global_prediction_net import GlobalPrediction

class GLO_PREDICTION:
    def __init__(self, args, cent_obs_space, act_space, device=torch.device("cpu")):
        """Initialize global world model.
        """
        self.args = args  # yaml里model和algo的config打包作为args进入VCritic
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)  # dtype和device

        self.prediction_epoch = args["critic_epoch"]
        self.prediction_num_mini_batch = args["critic_num_mini_batch"]
        self.data_chunk_length = args["data_chunk_length"]
        self.value_loss_coef = args["value_loss_coef"]
        self.max_grad_norm = args["max_grad_norm"]  # The maximum value for the gradient clipping
        self.huber_delta = args["huber_delta"]

        self.use_recurrent_policy = args["use_recurrent_policy"]
        self.use_max_grad_norm = args["use_max_grad_norm"]
        self.use_clipped_value_loss = args["use_clipped_value_loss"]
        self.use_huber_loss = args["use_huber_loss"]
        self.use_policy_active_masks = args["use_policy_active_masks"]

        self.prediction_lr = args["critic_lr"]  # critic的学习率
        self.opti_eps = args["opti_eps"]  # critic Adam优化器的eps
        self.weight_decay = args["weight_decay"]  # critic Adam优化器的weight_decay

        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        self.global_predictor = GlobalPrediction(args, self.share_obs_space, self.act_space, self.device)
        self.global_predictor_optimizer = torch.optim.Adam(
            self.global_predictor.parameters(),
            lr=self.prediction_lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )
        # self.global_predictor_optimizer = torch.optim.AdamW(
        #     self.global_predictor.parameters(),
        #     lr=self.prediction_lr,
        #     eps=self.opti_eps,
        #     weight_decay=self.weight_decay,
        # )
        # self.global_predictor_optimizer = torch.optim.SGD(
        #     self.global_predictor.parameters(),
        #     lr=self.prediction_lr,
        #     eps=self.opti_eps,
        #     weight_decay=self.weight_decay,
        # )

    def lr_decay(self, episode, episodes):
        """Decay the actor and critic learning rates.
        episode是当前episode的index，episodes是总共需要跑多少个episode
        Args:
            episode: (int) current training episode.
            episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.global_predictor_optimizer, episode, episodes, self.prediction_lr)

    def get_predictions(self, cent_obs, rnn_states, masks, step):
        """Get value function predictions.
        Args:
            cent_obs: (np.ndarray) centralized input to the critic.
            rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
            masks: (np.ndarray) denotes points at which RNN states should be reset.
        Returns:
            values: (torch.Tensor) value function predictions. (并行环境数量, 1)
            rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """
        prediction_errors = self.global_predictor(cent_obs, step)
        rnn_states = torch.from_numpy(rnn_states)
        rnn_states = rnn_states.to(prediction_errors.device)
        return rnn_states, prediction_errors

    def act(self, cent_obs, rnn_states, masks, step):
        """Get value function predictions.
        Args:
            cent_obs: (np.ndarray) centralized input to the critic.
            rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
            masks: (np.ndarray) denotes points at which RNN states should be reset.
        Returns:
            values: (torch.Tensor) value function predictions. (并行环境数量, 1)
            rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """
        prediction_errors = self.global_predictor(cent_obs, step)
        rnn_states = torch.from_numpy(rnn_states)
        rnn_states = rnn_states.to(prediction_errors.device)
        return rnn_states, prediction_errors

    def cal_prediction_loss(self, prediction_errors):
        """Calculate the loss for the global world model.
        Args:
            prediction_errors: (torch.Tensor) prediction errors for the given inputs.
        Returns:
            loss: (torch.Tensor) loss for the global world model.
        """
        flattened_tensor = prediction_errors.view(-1)
        flattened_tensor = flattened_tensor.float()
        flattened_tensor.requires_grad_(True)
        loss = flattened_tensor.mean()
        return loss

    def update(self, sample, value_normalizer=None):
        (
            share_obs_batch,
            actions_batch,
            global_prediction_errors_batch,
            rnn_states_global_batch,
            value_preds_batch,
            masks_batch,
        ) = sample

        # 计算prediction的loss # TODO: 需要设计loss function
        tensor_prediction_errors = torch.tensor(
            global_prediction_errors_batch, device=self.device, requires_grad=True
        )
        # 获取张量并过滤掉 NaN 和零值元素
        valid_elements = tensor_prediction_errors[
            ~torch.isnan(tensor_prediction_errors) & (tensor_prediction_errors != 0)]

        # 计算非 NaN 且非零元素的平均值
        prediction_loss = valid_elements.mean() if valid_elements.numel() > 0 else torch.tensor(0.0)

        self.global_predictor_optimizer.zero_grad()

        prediction_loss.backward()

        self.global_predictor_optimizer.step()

        return prediction_loss

    def train(self, global_predictor_buffer, value_normalizer=None):
        """Perform a training update using minibatch GD.
        Args:
            critic_buffer: (OnPolicyCriticBufferEP or OnPolicyCriticBufferFP) buffer containing training data related to critic.
            value_normalizer: (ValueNorm) normalize the rewards, denormalize critic outputs.
        Returns:
            train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """

        train_info = {}

        train_info["global_prediction_loss"] = 0

        # prediction_epoch是更新的次数
        # for _ in range(self.prediction_epoch):
        data_generator = global_predictor_buffer.recurrent_generator_prediction(
                self.prediction_num_mini_batch, self.data_chunk_length
            )
        # sample出
        for sample in data_generator:
            # 计算prediction的loss并且更新
            prediction_loss = self.update(
                sample, value_normalizer=value_normalizer
            )

            train_info["global_prediction_loss"] += prediction_loss.item()

        # num_updates = self.prediction_epoch * self.prediction_num_mini_batch

        # for k, _ in train_info.items():
        #     train_info[k] /= num_updates

        return train_info

    def prep_training(self):
        """Prepare for training."""
        self.global_predictor.train()

    def prep_rollout(self):
        """Prepare for rollout."""
        self.global_predictor.eval()