import yaml
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from simulator.models.base.mlp import MLPBase
from simulator.models.base.simple_layers import CrossAttention, GAT, MLP_improve, MultiVeh_GAT, TrajectoryDecoder
from einops import rearrange, repeat
import time

class TIE_rep(nn.Module):
    def __init__(self, obs_dim, action_dim, n_embd, action_type='Discrete', args=None):
        super(TIE_rep, self).__init__()
        self.max_num_HDVs = args['max_num_HDVs']
        self.max_num_CAVs = args['max_num_CAVs']
        self.num_HDVs = args['num_HDVs']
        self.num_CAVs = args['num_CAVs']
        self.hist_length = args['hist_length']
        # 单个智能体obs_dim
        self.obs_dim = obs_dim
        # 单个智能体action_dim
        self.action_dim = action_dim
        self.n_embd = n_embd
        self.action_type = action_type
        self.para_env = args['n_rollout_threads']

        self.gat_CAV_1a = GAT(nfeat=6, nhid=16, nclass=64, dropout=0.1, alpha=0.2, nheads=1)
        self.gat_HDV_5s = GAT(nfeat=3*5, nhid=32, nclass=64, dropout=0.1, alpha=0.2, nheads=1)

        self.mlp_surround_lane = MLPBase(args, [3 * 6])
        self.mlp_local_combined = MLPBase(args, [10+64*3])

        self.example_extend_info = {
            'road_structure': torch.zeros(10),               # 0
            'bottle_neck_position': torch.zeros(2),          # 1
            'road_end': torch.zeros(2),                      # 2
            'target': torch.zeros(2),                        # 3
            'self_stats': torch.zeros(1, 13),                # 4
            'distance_bott': torch.zeros(2),                 # 5
            'distance_end': torch.zeros(2),                  # 6
            'actor_action': torch.zeros(1, 3),               # 7
            'actual_action': torch.zeros(1, 3),              # 8
            # 'ego_cav_motion': torch.zeros(1, 15),            # 9
            # 'ego_hdv_motion': torch.zeros(1, 15),            # 10
            'ego_hist_motion': torch.zeros(1, 5 * 7),        # 9
            'surround_stats': torch.zeros(6, 36),            # 10
            'surround_relation_graph_simple': torch.zeros(7, 7),    # 11
            'expand_surround_stats': torch.zeros(10, 20),    # 12
            'surround_relation_graph': torch.zeros(10, 10),  # 13
            'surround_IDs': torch.zeros(10),                 # 14
            'surround_lane_stats': torch.zeros(3, 6),        # 15
            'hdv_stats': torch.zeros(self.max_num_HDVs, 7),  # 16
            'cav_stats': torch.zeros(self.max_num_CAVs, 7),  # 17
        }

    def reconstruct_info(self, obs):
        reconstructed = self.reconstruct_obs_batch(obs, self.example_extend_info)
        return (reconstructed['road_structure'], reconstructed['bottle_neck_position'],  reconstructed['road_end'], \
                reconstructed['target'], reconstructed['self_stats'], \
                reconstructed['distance_bott'], reconstructed['distance_end'], \
                reconstructed['actor_action'], reconstructed['actual_action'], \
                reconstructed['ego_hist_motion'], reconstructed['surround_stats'], reconstructed['surround_relation_graph_simple'], \
                reconstructed['expand_surround_stats'], reconstructed['surround_relation_graph'], \
                reconstructed['surround_IDs'], reconstructed['surround_lane_stats'], \
                reconstructed['hdv_stats'], reconstructed['cav_stats'])

    def forward(self, obs, batch_size=20):
        # obs: (n_rollout_thread, obs_dim)
        info_current = self.reconstruct_info(obs)
        # 初始化张量
        self.adj_surround_cav = torch.ones(batch_size, 6+1, 6+1, device='cuda')  # 7 = 6 + 1
        self.adj_surround_hdv = torch.ones(batch_size, 6+1, 6+1, device='cuda')
        # 将中间部分置零
        self.adj_surround_cav[:, 1:, 1:] = 0
        self.adj_surround_hdv[:, 1:, 1:] = 0
        # 获取 type_mask 信息
        type_masks = info_current[10][:, :, 0]  # shape: [batch_size, 6]
        # 创建掩码矩阵
        cav_mask = (type_masks == 0).float()  # 将布尔类型转换为 Float 类型
        hdv_mask = (type_masks == 1).float()
        # 创建索引
        batch_indices = torch.arange(batch_size).unsqueeze(1)  # shape: [batch_size, 1]
        indices = torch.arange(1, 7).unsqueeze(0)  # shape: [1, 6]
        # 对 cav 矩阵进行修改
        self.adj_surround_cav[batch_indices, 0, indices] = hdv_mask
        self.adj_surround_cav[batch_indices, indices, 0] = hdv_mask
        # 对 hdv 矩阵进行修改
        self.adj_surround_hdv[batch_indices, 0, indices] = cav_mask
        self.adj_surround_hdv[batch_indices, indices, 0] = cav_mask

        ################################################## trajectory-aware interaction encoder ######################################################
        # bottle_neck_0, distance_bott_0, road_end_0, distance_end_0, target_0
        # 01-'bottle_neck_position': torch.zeros(2),
        # 05-'distance_bott': torch.zeros(2),
        # 02-'road_end': torch.zeros(2),
        # 06-'distance_end': torch.zeros(2),
        # 03-'target': torch.zeros(2),
        local_road_info = torch.cat((info_current[5], info_current[2][:, :1], info_current[6], info_current[3]), dim=1)
        local_surround_cav_motion = torch.cat((info_current[10][:, :, 29:32], info_current[10][:, :, 33:]), dim=2)
        local_ego_motion = torch.cat((info_current[9][:, :, 28:31], info_current[9][:, :, 32:]), dim=2)
        hdv_hist = torch.cat((info_current[10][:, :, 1:4], info_current[10][:, :, 8:11], info_current[10][:, :, 15:18], info_current[10][:, :, 22:25], info_current[10][:, :, 29:32]), dim=2)
        ego_hist = torch.cat((info_current[9][:, :, :3], info_current[9][:, :, 7:10], info_current[9][:, :, 14:17], info_current[9][:, :, 21:24], info_current[9][:, :, 28:31]), dim=2)
        local_surround_lane_stats = info_current[15]

        combined_ego2hdv_hist = torch.cat((ego_hist, hdv_hist), dim=1)
        ego2hdv_relation_hist = self.gat_HDV_5s(combined_ego2hdv_hist, self.adj_surround_hdv.to(combined_ego2hdv_hist.device))
        combined_ego2cav_current_a = torch.cat((local_ego_motion, local_surround_cav_motion), dim=1)
        ego2cav_relation_current_a = self.gat_CAV_1a(combined_ego2cav_current_a, self.adj_surround_cav.to(combined_ego2cav_current_a.device))
        local_cav_relation = ego2cav_relation_current_a
        local_hdv_relation = ego2hdv_relation_hist
        local_lane_embedding = self.mlp_surround_lane(local_surround_lane_stats.view(local_surround_lane_stats.size(0), -1))
        exe_action = info_current[8].view(info_current[8].size(0), -1)
        local_combined_embedding = torch.cat((local_road_info, exe_action, local_cav_relation, local_hdv_relation, local_lane_embedding), dim=1)
        local_combined_embedding = self.mlp_local_combined(local_combined_embedding)
        return local_combined_embedding, info_current

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