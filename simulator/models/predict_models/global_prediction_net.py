import torch
import torch.nn as nn
from simulator.utils.envs_tools import check
from simulator.models.base.cnn import CNNBase
from simulator.models.base.mlp import MLPBase
from simulator.models.improve_predictor.flow_transformer import Flow_transformer_net
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
        self.num_CAVs = args['num_CAVs']
        self.num_HDVs = args['num_HDVs']
        self.max_num_HDVs = args['max_num_HDVs']
        self.max_num_CAVs = args['max_num_CAVs']
        self.prediction = Flow_transformer_net(args, obs_shape[0])
        self.num_lanes = 18
        self.lane_ids = [f'lane_{i}' for i in range(self.num_lanes)]
        self.prediction_output = {}
        self.prediction_log = {}
        self.groundtruth_log = {}
        self.prediction_error_log = {}
        self.n_rollout_threads = args['n_rollout_threads']
        self.timestamp = torch.zeros(self.n_rollout_threads)
        self.start_time = torch.zeros(self.n_rollout_threads)
        self.mse_loss_fn = torch.nn.MSELoss(reduction='none')
        self.zero_tensor = torch.zeros(1, device=device)
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
            # 'ego_cav_motion': torch.zeros(1, 15),                                          # 9
            # 'ego_hdv_motion': torch.zeros(1, 15),                                          # 10
            'ego_hist_motion': torch.zeros(1, 5 * 7),  # 9
            'surround_stats': torch.zeros(6, 36),  # 10
            'surround_relation_graph_simple': torch.zeros(7, 7),  # 11
            'expand_surround_stats': torch.zeros(10, 20),  # 12
            'surround_relation_graph': torch.zeros(10, 10),  # 13
            'surround_IDs': torch.zeros(10),  # 14
            'surround_lane_stats': torch.zeros(3, 6),  # 15
            'hdv_stats': torch.zeros(self.max_num_HDVs, 7),  # 16
            'cav_stats': torch.zeros(self.max_num_CAVs, 7),  # 17
            'vehicle_relation_graph': torch.zeros(self.max_num_CAVs + self.max_num_HDVs,
                                                  self.max_num_CAVs + self.max_num_HDVs),  # 18
            'all_lane_stats': torch.zeros(18, 6),  # 19
            'all_lane_evolution': torch.zeros(18, 26 * 5),  # 20
            'hdv_hist': torch.zeros(self.max_num_HDVs, 5 * 7),  # 21
            'cav_hist': torch.zeros(self.max_num_CAVs, 5 * 7),  # 22
        }

    def reconstruct_info(self, obs):
        reconstructed = self.reconstruct_obs_batch(obs, self.example_extend_info)
        return reconstructed['road_structure'], reconstructed['bottle_neck_position'], reconstructed['road_end'], \
            reconstructed['target'], reconstructed['self_stats'], \
            reconstructed['distance_bott'], reconstructed['distance_end'], \
            reconstructed['actor_action'], reconstructed['actual_action'], \
            reconstructed['ego_hist_motion'], reconstructed['surround_stats'], reconstructed[
            'surround_relation_graph_simple'], \
            reconstructed['expand_surround_stats'], reconstructed['surround_relation_graph'], \
            reconstructed['surround_IDs'], reconstructed['surround_lane_stats'], \
            reconstructed['hdv_stats'], reconstructed['cav_stats'], reconstructed['vehicle_relation_graph'], \
            reconstructed['all_lane_stats'], reconstructed['all_lane_evolution'], \
            reconstructed['hdv_hist'], reconstructed['cav_hist']

    def forward(
            self, obs, step=None
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
        # 用base提取特征-输入大小obs_shape，输出大小hidden_sizes[-1], eg: TensorShape([20, 120]) 并行环境数量 x hidden_sizes[-1]
        env_num = obs.size(0)

        device = obs.device
        state_divisors = torch.tensor([700, 3.2, 15, 2 * torch.pi], device=obs.device)
        if step == 0:
            self.prediction_log = {}
            self.groundtruth_log = {}
            self.prediction_error_log = {}

        reconstruct_info = self.reconstruct_info(obs)
        if (reconstruct_info[4] == 0).all():
            # Initialize prediction_error dictionary with shape (batch_size, 1)
            prediction_error = torch.zeros(env_num, 1, device=device)
        else:
            # Initialize prediction_error dictionary with shape (batch_size, 1)
            prediction_error = torch.zeros(env_num, 1, device=device)
            # Update prediction and ground truth logs
            if step not in self.prediction_log:
                self.prediction_log[step] = {lane_id: {} for lane_id in self.lane_ids}
            if step not in self.groundtruth_log:
                self.groundtruth_log[step] = {lane_id: {} for lane_id in self.lane_ids}
                current_gt_x = reconstruct_info[16][:, :, 0] * state_divisors[0]
                num_env = current_gt_x.size(0)

                # 确保 self.start_time 的大小与 num_env 一致
                # 如果 self.start_time 不存在，或者大小与 num_env 不一致，则重新初始化
                if not hasattr(self, 'start_time') or self.start_time.size(0) != num_env:
                    self.start_time = torch.zeros(num_env, dtype=torch.float32, device=current_gt_x.device)

                # 比较每个元素是否小于等于 250，生成布尔张量
                less_than_250 = current_gt_x <= 250  # 形状与 current_gt_x 相同，元素为 True 或 False

                # 对每一行检查所有元素是否都为 True（即所有值都小于等于 250）
                Mark = less_than_250.all(dim=1)  # 形状为 (num_env,)，元素为 True 或 False

                # 当 Mark 为 True 时，将对应位置的 self.start_time 赋值为 step
                self.start_time[Mark] = step
                self.groundtruth_log[step].update(
                    {f'lane_{i}': reconstruct_info[19][:, i, :] for i in range(self.num_lanes)}
                )
            if step not in self.prediction_error_log:
                self.prediction_error_log[step] = {lane_id: {} for lane_id in self.lane_ids}
            future_prediction = self.prediction(reconstruct_info, env_num)
            self.prediction_log[step].update(
                    {f'lane_{i}': future_prediction[:, i, :, :] for i in range(self.num_lanes)}
                )
            # Batch compute errors
            if step > 5:
                for t_step in range(5):
                    past_time = step - t_step - 1
                    if past_time in self.prediction_log:
                        pred_log = self.prediction_log[past_time]
                        gt_log = self.groundtruth_log[step]
                        # Collect predictions and ground truths
                        preds = []
                        gts = []
                        valid_masks = []

                        for lane_id, pred_tensor in pred_log.items():
                            pred = pred_tensor[:, t_step, :]  # Shape: [env_num, 4]
                            gt = gt_log[lane_id]

                            # 创建一个布尔掩码，标记 self.start_time 为 0 的环境
                            invalid_envs = (step - self.start_time < 5)  # 形状: [num_lane]

                            # 将无效环境的 gt 数据置为零
                            gt[invalid_envs, :] = 0
                            # Shape: [env_num, 4]

                            # 创建预测值的有效性掩码
                            valid_pred = pred.abs().sum(dim=1) > 0  # 形状: [env_num]

                            # 创建真实值的有效性掩码
                            valid_gt = gt.abs().sum(dim=1) > 0  # 形状: [env_num]

                            # 组合两个有效性掩码，只有当预测值和真实值都有效时，valid 才为 True
                            valid = (valid_pred & valid_gt).float()  # 形状: [env_num]

                            preds.append(pred)
                            gts.append(gt)
                            valid_masks.append(valid)
                    if preds:
                        preds_tensor = torch.stack(preds, dim=1)  # Shape: [env_num, num_lane, 6]
                        gts_tensor = torch.stack(gts, dim=1)  # Shape: [env_num, num_lane, 6]
                        valid_masks_tensor = torch.stack(valid_masks, dim=1)  # Shape: [env_num, num_lane]

                        # Compute per-element squared errors
                        squared_errors = self.mse_loss_fn(preds_tensor, gts_tensor)  # Shape: [env_num, num_lane, 6]

                        # Sum over the state dimension (dim=2)
                        sum_squared_errors = squared_errors.sum(dim=2)  # Shape: [env_num, num_lane]

                        # Compute root mean squared error per vehicle and environment
                        errors = torch.sqrt(sum_squared_errors) * valid_masks_tensor  # Shape: [env_num, num_lane]

                        # 创建布尔掩码并转换为浮点数
                        mask = (errors != 0).float()

                        # 计算非零元素的数量，保持维度
                        count_nonzero = mask.sum(dim=1, keepdim=True)

                        # 计算非零元素的和
                        sum_errors = errors.sum(dim=1, keepdim=True)

                        # 计算平均值，处理除以零的情况
                        mean_error_all = torch.where(count_nonzero != 0, sum_errors / count_nonzero,
                                                     torch.zeros_like(sum_errors))

                        prediction_error = mean_error_all
        prediction_errors = prediction_error.unsqueeze(1).expand(-1, self.num_CAVs, -1)

        return prediction_errors

    def reconstruct_obs_batch(self, obs_batch, template_structure):
        device = obs_batch.device

        # Initialize the reconstructed_batch with the same structure as template_structure
        reconstructed_batch = {
            key: torch.empty((obs_batch.size(0),) + tensor.shape, device=device)
            for key, tensor in template_structure.items()
        }

        # Compute the sizes of each tensor in the template structure
        sizes = [tensor.numel() for tensor in template_structure.values()]
        indices = torch.cumsum(torch.tensor([0] + sizes), dim=0)

        # Split obs_batch into chunks based on the sizes
        split_tensors = [obs_batch[:, indices[i]:indices[i+1]] for i in range(len(indices)-1)]

        # Assign the split tensors to the appropriate keys in the reconstructed_batch
        for key, split_tensor, template_tensor in zip(template_structure.keys(), split_tensors, template_structure.values()):
            reconstructed_batch[key] = split_tensor.view((-1,) + template_tensor.shape)

        return reconstructed_batch