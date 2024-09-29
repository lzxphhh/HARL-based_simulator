"""Base class for on-policy algorithms."""

import torch
from simulator.models.predict_models.local_prediction_net import LocalPrediction
from simulator.utils.models_tools import update_linear_schedule


class PredictionBase:
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        """Initialize Base class.
        Args:
            args: (dict) arguments.  # yaml里面的model和algo的config打包作为args进入OnPolicyBase
            obs_space: (gym.spaces or list) observation space. # 单个智能体的观测空间 eg: Box (18,)
            act_space: (gym.spaces) action space. # 单个智能体的动作空间 eg: Discrete(5,)
            device: (torch.device) device to use for tensor operations.
        """
        # save arguments
        # "model" and "algo" sections in $Algorithm config file
        self.args = args
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)   # dtype和device

        self.data_chunk_length = args["data_chunk_length"]
        self.use_recurrent_policy = args["use_recurrent_policy"]
        self.use_policy_active_masks = args["use_policy_active_masks"]

        self.lr = args["lr"]  # actor学习率
        self.opti_eps = args["opti_eps"]  # optimizer的epsilon
        self.weight_decay = args["weight_decay"] # optimizer的权重衰减
        # save observation and action spaces
        self.obs_space = obs_space
        self.act_space = act_space
        self.local_predictor = LocalPrediction(args, self.obs_space, self.act_space, self.device)
        self.local_predictor_optimizer = torch.optim.Adam(
            self.local_predictor.parameters(),
            lr=self.lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )


    def lr_decay(self, episode, episodes):
        """Decay the learning rates.
        episode是当前episode的index，episodes是总共需要跑多少个episode
        Args:
            episode: (int) current training episode.
            episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.local_predictor_optimizer, episode, episodes, self.lr)

    def get_predictions(
            self, obs, rnn_states_local, actions, masks
    ):
        """Compute prediction errors for the given inputs.
        输入:
            obs: (np.ndarray) local agent inputs to the actor. 所有环境下当前时刻某个agent的obs 【thread_num, obs_dim】
            rnn_states: (np.ndarray) if actor has RNN layer, RNN states for prediction.
                                上一时刻的rnn_state 【thread_num, rnn层数，rnn_state_dim】
            masks: (np.ndarray) denotes points at which RNN states should be reset. 【thread_num, 1】
        输出:
            prediction_errors: (torch.Tensor) prediction_errors for the given inputs. 【thread_num, 1】
            rnn_states: (torch.Tensor) updated RNN states for prediction. 【thread_num, rnn层数，rnn_state_dim】
        """
        rnn_states_local, local_prediction_error = self.local_predictor(
            obs, rnn_states_local, actions, masks
        )
        return rnn_states_local, local_prediction_error

    def act(
            self, obs, rnn_states_local, actions, masks
    ):
        rnn_states_local, local_prediction_error = self.local_predictor(
            obs, rnn_states_local, actions, masks
        )
        return rnn_states_local, local_prediction_error

    def update(self, sample):
        """Update actor network.
        Args:
            sample: (Tuple) contains data batch with which to update networks.
        """
        pass

    def train(self, local_predictor_buffer):
        pass

    def prep_training(self):
        """Prepare for training."""
        self.local_predictor.train()

    def prep_rollout(self):
        """Prepare for rollout."""
        # 测试world model网络结构
        self.local_predictor.eval()
