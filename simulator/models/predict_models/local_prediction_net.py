import torch
import torch.nn as nn
from simulator.utils.envs_tools import check
from simulator.models.base.mlp import MLPBase
from simulator.models.improve_predictor.GAT_aware import GATAware
from simulator.utils.envs_tools import get_shape_from_obs_space
from collections import defaultdict
import copy

class LocalPrediction(nn.Module):
    """Stochastic policy model. Outputs actions given observations."""

    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(LocalPrediction, self).__init__()
        self.strategy = args['strategy']
        self.hidden_sizes = args["hidden_sizes"]
        self.args = args
        self.gain = args["gain"]
        self.initialization_method = args["initialization_method"]
        self.use_policy_active_masks = args["use_policy_active_masks"]
        self.use_recurrent_policy = args["use_recurrent_policy"]
        self.recurrent_n = args["recurrent_n"]
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        base = MLPBase
        self.base = base(args, obs_shape)
        self.prediction = GATAware(obs_shape[0], action_space.n, self.hidden_sizes[-1], 'Discrete', args)
        self.num_CAVs = args['num_CAVs']
        self.num_HDVs = args['num_HDVs']
        self.max_num_HDVs = args['max_num_HDVs']
        self.max_num_CAVs = args['max_num_CAVs']
        self.CAV_ids = [f'CAV_{i}' for i in range(self.num_CAVs)]
        self.HDV_ids = [f'HDV_{i}' for i in range(self.num_HDVs)]
        self.max_CAV_ids = [f'CAV_{i}' for i in range(self.max_num_CAVs)]
        self.max_HDV_ids = [f'HDV_{i}' for i in range(self.max_num_HDVs)]
        self.veh_ids = self.CAV_ids + self.HDV_ids
        self.max_veh_ids = self.max_CAV_ids + self.max_HDV_ids
        self.prediction_log = {}
        self.groundtruth_log = {}
        self.prediction_error_log = {}
        self.n_rollout_threads = args['n_rollout_threads']
        self.timestamp = torch.zeros(self.n_rollout_threads)
        self.start_time = torch.zeros(self.n_rollout_threads)
        self.time_log = {
            '1s_CAV': 1, '2s_CAV': 2, '3s_CAV': 3, '4s_CAV': 4, '5s_CAV': 5,
            '1s_HDV': 11, '2s_HDV': 12, '3s_HDV': 13, '4s_HDV': 14, '5s_HDV': 15,
            '1s_all': 21, '2s_all': 22, '3s_all': 23, '4s_all': 24, '5s_all': 25
        }
        self.mse_loss_fn = torch.nn.MSELoss(reduction='none')
        self.zero_tensor = torch.zeros(1, device=device)
        self.first_episode = True

        self.to(device)

        self.example_extend_info = {
            'road_structure': torch.zeros(10),                       # 00
            'bottle_neck_position': torch.zeros(2),                  # 01
            'road_end': torch.zeros(2),                              # 02
            'target': torch.zeros(2),                                # 03
            'self_stats': torch.zeros(1, 13),                        # 04
            'distance_bott': torch.zeros(2),                         # 05
            'distance_end': torch.zeros(2),                          # 06
            'actor_action': torch.zeros(1, 3),                       # 07
            'actual_action': torch.zeros(1, 3),                      # 08
            'ego_hist_motion': torch.zeros(1, 5 * 7),                # 09
            'surround_stats': torch.zeros(6, 36),                    # 10
            'surround_relation_graph_simple': torch.zeros(7, 7),     # 11
            'expand_surround_stats': torch.zeros(10, 20),            # 12
            'surround_relation_graph': torch.zeros(10, 10),          # 13
            'surround_IDs': torch.zeros(10),                         # 14
            'surround_lane_stats': torch.zeros(3, 6),                # 15
            'hdv_stats': torch.zeros(self.max_num_HDVs, 7),          # 16
            'cav_stats': torch.zeros(self.max_num_CAVs, 7),          # 17
        }

    def reconstruct_data(self, obs):
        reconstructed = self.reconstruct_obs_batch(obs, self.example_extend_info)
        return (reconstructed['road_structure'], reconstructed['bottle_neck_position'], reconstructed['road_end'],
                reconstructed['target'], reconstructed['self_stats'],
                reconstructed['distance_bott'], reconstructed['distance_end'],
                reconstructed['actor_action'], reconstructed['actual_action'],
                reconstructed['ego_hist_motion'], reconstructed['surround_stats'],
                reconstructed['surround_relation_graph_simple'],
                reconstructed['expand_surround_stats'], reconstructed['surround_relation_graph'],
                reconstructed['surround_IDs'], reconstructed['surround_lane_stats'],
                reconstructed['hdv_stats'], reconstructed['cav_stats'])

    def forward(self, obs, actions, masks, agent_id=None, step=None):
        obs = check(obs).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv)
        env_num = obs.size(0)
        ego_id = f'CAV_{agent_id}'
        state_divisors = torch.tensor([700, 3.2, 15, 2 * torch.pi], device=obs.device)

        if step == 0 and agent_id == 0:
            if self.first_episode == True:
                self.prediction_log = {}
                self.groundtruth_log = {}
                self.prediction_error_log = {}
            elif env_num != self.n_rollout_threads:
                self.prediction_log = {}
                self.groundtruth_log = {}
                self.prediction_error_log = {}
                self.first_episode = True
            elif self.groundtruth_log and self.groundtruth_log[0][ego_id].size(0) == self.n_rollout_threads:
                self.first_episode = False


        reconstruct_info = self.reconstruct_data(obs)
        if (reconstruct_info[4] == 0).all():
            # Initialize prediction_error dictionary with shape (batch_size, 1)
            prediction_error = {key: torch.zeros(env_num, 1, device=obs.device) for key in self.time_log.keys()}
        else:
            # Initialize prediction_error dictionary with shape (batch_size, 1)
            prediction_error = {key: torch.zeros(env_num, 1, device=obs.device) for key in self.time_log.keys()}

            # Update prediction and ground truth logs
            if step not in self.prediction_log:
                self.prediction_log[step] = {veh_id: {} for veh_id in self.CAV_ids}
            if step not in self.groundtruth_log:
                self.groundtruth_log[step] = {veh_id: [] for veh_id in self.veh_ids}
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
                    {f'HDV_{i}': reconstruct_info[16][:, i, :2] * state_divisors[:2] for i in range(self.num_HDVs)}
                )
                self.groundtruth_log[step].update(
                    {f'CAV_{i}': reconstruct_info[17][:, i, :2] * state_divisors[:2] for i in range(self.num_CAVs)}
                )
            if step not in self.prediction_error_log:
                self.prediction_error_log[step] = {veh_id: {} for veh_id in self.CAV_ids}

            # Prediction operation
            prediction_output = self.prediction(reconstruct_info, actions, state_divisors, batch_size=obs.size(0))
            self.prediction_log[step][ego_id] = prediction_output


            # Batch compute errors
            if not self.first_episode or step > 5:
                for t_step in range(5):
                    step_key_all = f'{t_step + 1}s_all'
                    step_key_cav = f'{t_step + 1}s_CAV'
                    step_key_hdv = f'{t_step + 1}s_HDV'

                    # Collect past predictions and ground truths
                    if not self.first_episode:
                        past_time = (step - t_step - 1) if (step - t_step - 1 >= 0) else (step - t_step - 1 + 100)
                    else:
                        past_time = step - t_step - 1
                    if past_time in self.prediction_log:
                        pred_log = {key: value.clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
                                    for key, value in self.prediction_log[past_time][ego_id].items()}
                        gt_log = {key: value.clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
                                    for key, value in self.groundtruth_log[step].items()}

                        # Collect predictions and ground truths
                        preds = []
                        gts = []
                        valid_masks = []
                        veh_types = []

                        for veh_id, pred_tensor in pred_log.items():
                            if veh_id != ego_id and veh_id in gt_log:
                                pred = pred_tensor[:, t_step, :]  # Shape: [env_num, 4]
                                gt = gt_log[veh_id]

                                # 创建一个布尔掩码，标记 self.start_time 为 0 的环境
                                invalid_envs = (step - self.start_time < 5)  # 形状: [num_env]

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
                                veh_types.append(veh_id.startswith('CAV'))

                        if preds:
                            preds_tensor = torch.stack(preds, dim=1)  # Shape: [env_num, num_veh, 4]
                            gts_tensor = torch.stack(gts, dim=1)  # Shape: [env_num, num_veh, 4]
                            valid_masks_tensor = torch.stack(valid_masks, dim=1)  # Shape: [env_num, num_veh]
                            veh_types_tensor = torch.tensor(veh_types, device=obs.device)  # [num_veh]

                            # Compute per-element squared errors
                            squared_errors = self.mse_loss_fn(preds_tensor, gts_tensor)  # Shape: [env_num, num_veh, 4]

                            # Sum over the state dimension (dim=2)
                            sum_squared_errors = squared_errors.sum(dim=2)  # Shape: [env_num, num_veh]

                            # Compute root mean squared error per vehicle and environment
                            errors = torch.sqrt(sum_squared_errors) * valid_masks_tensor  # Shape: [env_num, num_veh]

                            # 创建布尔掩码并转换为浮点数
                            mask = (errors != 0).float()

                            # 计算非零元素的数量，保持维度
                            count_nonzero = mask.sum(dim=1, keepdim=True)

                            # 计算非零元素的和
                            sum_errors = errors.sum(dim=1, keepdim=True)

                            # 计算平均值，处理除以零的情况
                            mean_error_all = torch.where(count_nonzero != 0, sum_errors / count_nonzero,
                                                         torch.zeros_like(sum_errors))

                            prediction_error[step_key_all] = mean_error_all

                            # 对于 CAVs
                            if veh_types_tensor.any():
                                cav_mask = veh_types_tensor  # Shape: [num_veh]
                                errors_cav = errors[:, cav_mask]  # Shape: [env_num, num_cav_veh]

                                # 创建 CAV 的非零误差掩码
                                mask_cav = (errors_cav != 0).float()

                                # 计算非零元素的数量，保持维度
                                count_nonzero_cav = mask_cav.sum(dim=1, keepdim=True)

                                # 计算非零元素的和
                                sum_errors_cav = errors_cav.sum(dim=1, keepdim=True)

                                # 计算平均值，处理除以零的情况
                                mean_error_cav = torch.where(count_nonzero_cav != 0, sum_errors_cav / count_nonzero_cav,
                                                             torch.zeros_like(sum_errors_cav))

                                prediction_error[step_key_cav] = mean_error_cav

                            # 对于 HDVs
                            if (~veh_types_tensor).any():
                                hdv_mask = ~veh_types_tensor  # Shape: [num_veh]
                                errors_hdv = errors[:, hdv_mask]  # Shape: [env_num, num_hdv_veh]

                                # 创建 HDV 的非零误差掩码
                                mask_hdv = (errors_hdv != 0).float()

                                # 计算非零元素的数量，保持维度
                                count_nonzero_hdv = mask_hdv.sum(dim=1, keepdim=True)

                                # 计算非零元素的和
                                sum_errors_hdv = errors_hdv.sum(dim=1, keepdim=True)

                                # 计算平均值，处理除以零的情况
                                mean_error_hdv = torch.where(count_nonzero_hdv != 0, sum_errors_hdv / count_nonzero_hdv,
                                                             torch.zeros_like(sum_errors_hdv))

                                prediction_error[step_key_hdv] = mean_error_hdv

                def contains_value_greater_than(data, threshold):
                    import torch
                    import numpy as np

                    if isinstance(data, dict):
                        return any(contains_value_greater_than(value, threshold) for value in data.values())
                    elif isinstance(data, (list, tuple, set)):
                        return any(contains_value_greater_than(item, threshold) for item in data)
                    elif torch.is_tensor(data):
                        return (data > threshold).any().item()
                    elif isinstance(data, np.ndarray):
                        return np.any(data > threshold)
                    elif isinstance(data, (int, float)):
                        return data > threshold
                    else:
                        return False

                # 使用更新后的函数进行判断
                if contains_value_greater_than(prediction_error, 400):
                    print("存在值大于400的情况")
        self.prediction_error_log[step][ego_id] = prediction_error

        return prediction_error

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
