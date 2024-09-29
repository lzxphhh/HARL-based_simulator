import torch
import torch.nn as nn
from simulator.utils.envs_tools import check
from simulator.models.base.cnn import CNNBase
from simulator.models.base.mlp import MLPBase
from simulator.models.improve_predictor.GAT_aware import GATAware
from simulator.models.base.rnn import RNNLayer
from simulator.models.base.act import ACTLayer
from simulator.utils.envs_tools import get_shape_from_obs_space
import yaml
import copy
import time

class GlobalPrediction(nn.Module):
    """Stochastic policy model. Outputs actions given observations."""

    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        """Initialize StochasticPolicy model.
        Args:
            args: (dict) arguments containing relevant model information. #yaml里面的model和algo的config打包
            obs_space: (gym.Space) observation space.  # 单个智能体的观测空间 eg: Box (18,)
            action_space: (gym.Space) action space. # 单个智能体的动作空间 eg: Discrete(5,)
            device: (torch.device) specifies the device to run on (cpu/gpu).
        """
        super(GlobalPrediction, self).__init__()
        self.strategy = args['strategy']
        self.hidden_sizes = args["hidden_sizes"]  # MLP隐藏层神经元数量
        self.args = args  # yaml里面的model和algo的config打包
        self.gain = args["gain"]  # 激活函数的斜率或增益，增益较大的激活函数会更敏感地响应输入的小变化，而增益较小的激活函数则会对输入的小变化不那么敏感
        self.initialization_method = args["initialization_method"]  # 网络权重初始化方法
        self.use_policy_active_masks = args["use_policy_active_masks"]  # TODO：这是什么

        self.use_recurrent_policy = args["use_recurrent_policy"]
        # number of recurrent layers
        self.recurrent_n = args["recurrent_n"]  # RNN层数
        self.tpdv = dict(dtype=torch.float32, device=device)  # dtype和device

        obs_shape = get_shape_from_obs_space(obs_space)  # 获取观测空间的形状，tuple of integer. eg: （18，）

        # 根据观测空间的形状，选择CNN或者MLP作为基础网络，用于base提取特征，输入大小obs_shape，输出大小hidden_sizes[-1]
        base = CNNBase if len(obs_shape) == 3 else MLPBase
        self.base = base(args, obs_shape)
        self.prediction = GATAware(obs_shape[0], action_space.n, self.hidden_sizes[-1], 'Discrete', args)
        self.num_CAVs = args['num_CAVs']
        self.num_HDVs = args['num_HDVs']
        self.max_num_HDVs = args['max_num_HDVs']
        self.max_num_CAVs = args['max_num_CAVs']
        self.CAV_ids = [f'CAV_{i}' for i in range(self.num_CAVs)]
        self.HDV_ids = [f'HDV_{i}' for i in range(self.num_HDVs)]
        self.veh_ids = self.CAV_ids + self.HDV_ids
        self.prediction_output = {}
        for i in range(3):
            self.prediction_output[f'hist_{i + 1}'] = {veh_id: {pre_id: [] for pre_id in self.veh_ids} for veh_id in self.CAV_ids}
        # 如果使用RNN，初始化RNN层
        if self.use_recurrent_policy:
            self.rnn = RNNLayer(
                self.hidden_sizes[-1],
                self.hidden_sizes[-1],
                self.recurrent_n,
                self.initialization_method,
            )

        # 初始化ACT层, 用于输出动作(动作概率)，输入大小hidden_sizes[-1]，输出大小action_space.n
        self.act = ACTLayer(
            action_space,
            self.hidden_sizes[-1],
            self.initialization_method,
            self.gain,
            args,  # yaml里面的model和algo的config打包
        )

        self.to(device)
    def forward(
            self, obs, rnn_states, actions, masks, available_actions=None, deterministic=False
    ):
        """Compute actions from the given inputs.
        Args:
            obs: (np.ndarray / torch.Tensor) observation inputs into network.
            rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
            masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
            available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
            deterministic: (bool) whether to sample from action distribution or return the mode.
        Returns:
            actions: (torch.Tensor) actions to take.
            action_log_probs: (torch.Tensor) log probabilities of taken actions.
            rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        # 检查输入的dtype和device是否正确，变形到在cuda上的tensor以方便进入网络
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        # 用base提取特征-输入大小obs_shape，输出大小hidden_sizes[-1], eg: TensorShape([20, 120]) 并行环境数量 x hidden_sizes[-1]
        env_num = obs.size(0)
        prediction_features = self.base(obs)

        # 如果使用RNN，将特征和RNN状态输入RNN层，得到新的特征和RNN状态
        if self.use_recurrent_policy:
            prediction_features, rnn_states = self.rnn(prediction_features, rnn_states, masks)
        # 将特征和可用动作输入ACT层，得到动作，动作概率
        prediction_errors, _ = self.act(
            prediction_features, available_actions, deterministic
        )
        prediction_errors = prediction_errors.unsqueeze(1).repeat(1, self.num_CAVs, 1)
        return rnn_states, prediction_errors