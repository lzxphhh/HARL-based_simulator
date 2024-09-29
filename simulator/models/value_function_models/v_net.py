import torch
import torch.nn as nn
from simulator.models.base.cnn import CNNBase
from simulator.models.base.mlp import MLPBase
from simulator.models.base.rnn import RNNLayer
from simulator.utils.envs_tools import check, get_shape_from_obs_space
from simulator.utils.models_tools import init, get_init_method
from simulator.models.improve_critic.cross_aware_net import Cross_aware_net
from simulator.models.improve_critic.interaction_cross_net import Iteraction_cross_net

class VNet(nn.Module):
    """V Network. Outputs value function predictions given global states."""

    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        """Initialize VNet model.
        Args:
            args: (dict) arguments containing relevant model information.
            cent_obs_space: (gym.Space) centralized observation space.
            device: (torch.device) specifies the device to run on (cpu/gpu).
        """
        super(VNet, self).__init__()
        self.hidden_sizes = args["hidden_sizes"] # MLP隐藏层神经元数量
        self.initialization_method = args["initialization_method"]  # 网络权重初始化方法
        self.use_naive_recurrent_policy = args["use_naive_recurrent_policy"]
        self.use_recurrent_policy = args["use_recurrent_policy"]
        self.recurrent_n = args["recurrent_n"]  # RNN的层数
        self.tpdv = dict(dtype=torch.float32, device=device) # dtype和device

        # 获取网络权重初始化方法函数
        init_method = get_init_method(self.initialization_method)

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space) # 获取观测空间的形状，tuple of integer. eg: （54，）

        # 根据观测空间的形状，选择CNN或者MLP作为基础网络，用于base提取特征，输入大小cent_obs_shape，输出大小hidden_sizes[-1]
        base = CNNBase if len(cent_obs_shape) == 3 else MLPBase
        self.base = base(args, cent_obs_shape)
        self.cross_aware_old = Cross_aware_net(args, cent_obs_shape)
        self.cross_aware_new = Iteraction_cross_net(args, cent_obs_shape)
        # self.base_net = MLPBase(args, [16 * 3 + 18 * 6 + 2])

        # 如果使用RNN，初始化RNN层 #TODO: 暂时还没看
        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            self.rnn = RNNLayer(
                self.hidden_sizes[-1],
                self.hidden_sizes[-1],
                self.recurrent_n,
                self.initialization_method,
            )

        # 定义了一个初始化神经网络权重的函数 （特别是 nn.Linear 层的权重）
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        # 初始化了一个 nn.Linear 层 (self.linear)，该层将输入的self.hidden_sizes[-1]个特征映射到一个值，即state value
        self.v_out = init_(nn.Linear(self.hidden_sizes[-1], 1))

        self.to(device)

        self.example_extend_info = {
            # 'bottle_neck_position': torch.zeros(2),
            'hdv_stats': torch.zeros(24, 6),  # 0
            'cav_stats': torch.zeros(16, 6),  # 1
            'all_lane_stats': torch.zeros(18, 6),  # 2
            'bottle_neck_position': torch.zeros(2),  # 3
            'road_structure': torch.zeros(10),  # 4
            'road_end': torch.zeros(2),  # 5
            'target': torch.zeros(2),  # 6
            'self_stats': torch.zeros(1, 13),  # 7
            'distance_bott': torch.zeros(2),  # 8
            'distance_end': torch.zeros(2),  # 9
            'executed_action': torch.zeros(1, 2),  # 10
            'generation_action': torch.zeros(1, 1),  # 11
            'surround_hdv_stats': torch.zeros(6, 6),  # 12
            'surround_cav_stats': torch.zeros(6, 6),  # 13
            'surround_lane_stats': torch.zeros(3, 6),  # 14
        }

    def reconstruct_info(self, obs):
        reconstructed = self.reconstruct_obs_batch(obs, self.example_extend_info)
        return reconstructed['hdv_stats'], reconstructed['cav_stats'], reconstructed['all_lane_stats'], \
            reconstructed['bottle_neck_position'], reconstructed['road_structure'], reconstructed['road_end'], \
            reconstructed['target'], reconstructed['self_stats'], \
            reconstructed['distance_bott'], reconstructed['distance_end'], \
            reconstructed['executed_action'], reconstructed['generation_action'], \
            reconstructed['surround_hdv_stats'], reconstructed['surround_cav_stats'], \
            reconstructed['surround_lane_stats']

    def forward(self, cent_obs, rnn_states, masks):
        """Compute actions from the given inputs.
        Args:
            cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
            rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
            masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.
        Returns:
            values: (torch.Tensor) value function predictions.
            rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        # 检查输入的dtype和device是否正确，变形到在cuda上的tensor以方便进入网络
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        # 用base提取特征-输入大小obs_shape，输出大小hidden_sizes[-1], eg: TensorShape([20, 120]) 并行环境数量 x hidden_sizes[-1]
        # critic_features = self.base(cent_obs)
        # critic_features = self.hierarchical_net(cent_obs)
        # critic_features = self.cross_aware(cent_obs)
        critic_features = self.cross_aware_new(cent_obs)

        # 如果使用RNN，将特征和RNN状态输入RNN层，得到新的特征和RNN状态
        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)

        # 将特征输入v_out层，得到值函数预测值
        values = self.v_out(critic_features)

        return values, rnn_states

    def hierarchical_net(self, obs, batch_size=20):
        batch_size = obs.size(0)
        info_current = self.reconstruct_info(obs)
        cav_stats_current = info_current[1][:, :, :3]
        cav_stats_current = cav_stats_current.reshape(batch_size, -1)
        all_lane_stats = info_current[2]
        all_lane_stats = all_lane_stats.reshape(batch_size, -1)
        bottle_neck_position = info_current[3]

        combined_obs = torch.cat((cav_stats_current, all_lane_stats, bottle_neck_position), dim=1)
        embedding = self.base_net(combined_obs)
        return embedding

    def reconstruct_obs_batch(self, obs_batch, template_structure):
        device = obs_batch.device  # Get the device of obs_batch

        # Initialize the reconstructed_batch with the same structure as template_structure
        reconstructed_batch = {
            key: torch.empty((obs_batch.size(0),) + tensor.shape, device=device)
            for key, tensor in template_structure.items()
        }

        # Compute the cumulative sizes of each tensor in the template structure
        sizes = [tensor.numel() for tensor in template_structure.values()]
        cumulative_sizes = torch.cumsum(torch.tensor(sizes), dim=0)
        indices = [0] + cumulative_sizes.tolist()[:-1]

        # Split obs_batch into chunks based on the cumulative sizes
        split_tensors = torch.split(obs_batch, sizes, dim=1)

        # Assign the split tensors to the appropriate keys in the reconstructed_batch
        for key, split_tensor in zip(template_structure.keys(), split_tensors):
            reconstructed_batch[key] = split_tensor.view((-1,) + template_structure[key].shape)

        return reconstructed_batch
