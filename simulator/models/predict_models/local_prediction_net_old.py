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

class LocalPrediction(nn.Module):
    """Stochastic policy model. Outputs actions given observations."""

    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        """Initialize StochasticPolicy model.
        Args:
            args: (dict) arguments containing relevant model information. #yaml里面的model和algo的config打包
            obs_space: (gym.Space) observation space.  # 单个智能体的观测空间 eg: Box (18,)
            action_space: (gym.Space) action space. # 单个智能体的动作空间 eg: Discrete(5,)
            device: (torch.device) specifies the device to run on (cpu/gpu).
        """
        super(LocalPrediction, self).__init__()
        self.strategy = args['strategy']
        self.hidden_sizes = args["hidden_sizes"]  # MLP隐藏层神经元数量
        self.args = args  # yaml里面的model和algo的config打包
        self.gain = args["gain"]  # 激活函数的斜率或增益，增益较大的激活函数会更敏感地响应输入的小变化，而增益较小的激活函数则会对输入的小变化不那么敏感
        self.initialization_method = args["initialization_method"]  # 网络权重初始化方法
        self.use_policy_active_masks = args["use_policy_active_masks"]

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

        self.example_extend_info = {
            'road_structure': torch.zeros(10),  # 0
            'bottle_neck_position': torch.zeros(2),  # 1
            'road_end': torch.zeros(2),  # 2
            'target': torch.zeros(2),  # 3
            'self_stats': torch.zeros(1, 13),  # 4
            'distance_bott': torch.zeros(2),  # 5
            'distance_end': torch.zeros(2),  # 6
            'actor_action': torch.zeros(1, 3),  # 7
            'actual_action': torch.zeros(1, 3),  # 8
            # 'ego_cav_motion': torch.zeros(1, 15),            # 9
            # 'ego_hdv_motion': torch.zeros(1, 15),            # 10
            'ego_hist_motion': torch.zeros(1, 5 * 7),  # 9
            'surround_stats': torch.zeros(6, 36),  # 10
            'expand_surround_stats': torch.zeros(10, 20),  # 11
            'surround_relation_graph': torch.zeros(10, 10),  # 12
            'surround_IDs': torch.zeros(10),  # 13
            'surround_lane_stats': torch.zeros(3, 6),  # 14
            'hdv_stats': torch.zeros(self.max_num_HDVs, 7),  # 15
            'cav_stats': torch.zeros(self.max_num_CAVs, 7),  # 16
        }

    def reconstruct_data(self, obs):
        reconstructed = self.reconstruct_obs_batch(obs, self.example_extend_info)
        return (reconstructed['road_structure'], reconstructed['bottle_neck_position'], reconstructed['road_end'], \
                reconstructed['target'], reconstructed['self_stats'], \
                reconstructed['distance_bott'], reconstructed['distance_end'], \
                reconstructed['actor_action'], reconstructed['actual_action'], \
                reconstructed['ego_hist_motion'], reconstructed['surround_stats'], \
                reconstructed['expand_surround_stats'], reconstructed['surround_relation_graph'], \
                reconstructed['surround_IDs'], reconstructed['surround_lane_stats'], \
                reconstructed['hdv_stats'], reconstructed['cav_stats'])

    def forward(
            self, obs, rnn_states, actions, masks, available_actions=None
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
        reconstruct_info = self.reconstruct_data(obs)

        # 如果使用RNN，将特征和RNN状态输入RNN层，得到新的特征和RNN状态
        if self.use_recurrent_policy:
            prediction_features, rnn_states = self.rnn(prediction_features, rnn_states, masks)

        # prediction_loss
        actions = torch.from_numpy(actions).to(**self.tpdv)
        future_states, ego_id, prediction_groundtruth = self.prediction(reconstruct_info, actions, batch_size=obs.size(0))
        action_is_zero = (reconstruct_info[7] == 0)
        action_all_zero = torch.all(action_is_zero)
        ego2cav_is_zero = (reconstruct_info[9] == 0)
        ego2cav_all_zero = torch.all(ego2cav_is_zero)
        ego2hdv_is_zero = (reconstruct_info[10] == 0)
        ego2hdv_all_zero = torch.all(ego2hdv_is_zero)
        all_zero = action_all_zero and ego2cav_all_zero and ego2hdv_all_zero
        if all_zero:
            self.prediction_output['hist_3'][ego_id] = {pre_id: [] for pre_id in self.veh_ids}
            self.prediction_output['hist_2'][ego_id] = {pre_id: [] for pre_id in self.veh_ids}
            self.prediction_output['hist_1'][ego_id] = {pre_id: [] for pre_id in self.veh_ids}
        prediction_error_output = torch.zeros(env_num, 1, device=self.tpdv['device'])
        prediction_error = {}
        prediction_mae_error = {}
        if ego_id:
            for i in range(3):
                prediction_error[f'hist_{i + 1}'] = {pre_id: [] for pre_id in self.veh_ids}
                prediction_mae_error[f'hist_{i + 1}'] = {key: torch.zeros(env_num, 1, device=self.tpdv['device']) for key in self.veh_ids}
            if len(ego_id) > 10:
                print('ego_id:', ego_id)
            for veh_id in self.veh_ids:
                if self.prediction_output['hist_1'][ego_id][veh_id] != [] and prediction_groundtruth[veh_id] != []:
                    if prediction_groundtruth[veh_id].size() != self.prediction_output['hist_1'][ego_id][veh_id][:, 0, :].size():
                        continue
                    else:
                        prediction_error['hist_1'][veh_id] = self.prediction_output['hist_1'][ego_id][veh_id][:, 0, :] - \
                                                             prediction_groundtruth[veh_id][:, :] if self.prediction_output['hist_1'][ego_id][veh_id][:, 0, :] != [0, 0, 0] else [0, 0, 0]
                        # prediction_mse_error[veh_id] = torch.mean(torch.pow(self.prediction_error['hist_1'][ego_id][veh_id], 2), dim=1, keepdim=True)
                        prediction_mae_error['hist_1'][veh_id] = torch.mean(torch.abs(prediction_error['hist_1'][veh_id]), dim=1, keepdim=True)
                else:
                    prediction_error['hist_1'][veh_id] = []
                if self.prediction_output['hist_2'][ego_id][veh_id] != [] and prediction_groundtruth[veh_id] != []:
                    if prediction_groundtruth[veh_id].size() != self.prediction_output['hist_2'][ego_id][veh_id][:, 1, :].size():
                        continue
                    else:
                        prediction_error['hist_2'][veh_id] = self.prediction_output['hist_2'][ego_id][veh_id][:, 1, :] - \
                                                             prediction_groundtruth[veh_id][:, :] if self.prediction_output['hist_2'][ego_id][veh_id][:, 1, :] != [0, 0, 0] else [0, 0, 0]
                        # prediction_mse_error[veh_id] = torch.mean(torch.pow(self.prediction_error['hist_2'][ego_id][veh_id], 2), dim=1, keepdim=True)
                        prediction_mae_error['hist_2'][veh_id] = torch.mean(torch.abs(prediction_error['hist_2'][veh_id]), dim=1, keepdim=True)
                else:
                    prediction_error['hist_2'][veh_id] = []
                if self.prediction_output['hist_3'][ego_id][veh_id] != [] and prediction_groundtruth[veh_id] != []:
                    if prediction_groundtruth[veh_id].size() != self.prediction_output['hist_3'][ego_id][veh_id][:, 2, :].size():
                        continue
                    else:
                        prediction_error['hist_3'][veh_id] = self.prediction_output['hist_3'][ego_id][veh_id][:, 2, :] - \
                                                             prediction_groundtruth[veh_id][:, :] if self.prediction_output['hist_3'][ego_id][veh_id][:, 2, :] != [0, 0, 0] else [0, 0, 0]
                        # prediction_mse_error[veh_id] = torch.mean(torch.pow(self.prediction_error['hist_3'][ego_id][veh_id], 2), dim=1, keepdim=True)
                        prediction_mae_error['hist_3'][veh_id] = torch.mean(torch.abs(prediction_error['hist_3'][veh_id]), dim=1, keepdim=True)
                else:
                    prediction_error['hist_3'][veh_id] = []

            for i in range(env_num):
                error_cumulative = 0
                veh_count = 0
                for veh_id in self.veh_ids:
                    if prediction_mae_error['hist_1'][veh_id][i, 0] != 0 and veh_id != ego_id:
                        error_cumulative += prediction_mae_error['hist_1'][veh_id][i, 0]  # prediction_mse_error
                        veh_count += 1
                    if prediction_mae_error['hist_2'][veh_id][i, 0] != 0 and veh_id != ego_id:
                        error_cumulative += prediction_mae_error['hist_2'][veh_id][i, 0]  # prediction_mse_error
                        veh_count += 1
                    if prediction_mae_error['hist_3'][veh_id][i, 0] != 0 and veh_id != ego_id:
                        error_cumulative += prediction_mae_error['hist_3'][veh_id][i, 0]  # prediction_mse_error
                        veh_count += 1
                if veh_count != 0:
                    prediction_error_output[i, 0] = error_cumulative / veh_count
                else:
                    prediction_error_output[i, 0] = 0
            self.prediction_output['hist_3'][ego_id] = self.prediction_output['hist_2'][ego_id]
            self.prediction_output['hist_2'][ego_id] = self.prediction_output['hist_1'][ego_id]
            self.prediction_output['hist_1'][ego_id] = future_states

        return rnn_states, prediction_error_output

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
