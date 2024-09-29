import yaml
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from simulator.models.base.mlp import MLPBase
from simulator.models.base.simple_layers import CrossAttention, GAT, MLP_improve
from einops import rearrange, repeat

class Cross_aware_rep(nn.Module):
    def __init__(self, obs_dim, action_dim, n_embd, action_type='Discrete', args=None):
        super(Cross_aware_rep, self).__init__()
        self.max_num_HDVs = args['max_num_HDVs']
        self.max_num_CAVs = args['max_num_CAVs']
        self.num_HDVs = args['num_HDVs']
        self.num_CAVs = args['num_CAVs']
        self.hist_length = args['hist_length']
        # 单个智能体obs_dim
        self.obs_dim = obs_dim
        self.one_step_obs_dim = int(obs_dim / (self.hist_length + 1))
        # 单个智能体action_dim
        self.action_dim = action_dim
        self.n_embd = n_embd
        self.action_type = action_type

        ###### global encoder ######
        # self.mlp_road = MLPBase(args, [10])
        # self.mlp_all_cav_a = MLPBase(args, [self.max_num_CAVs*6])
        # self.mlp_all_cav = MLPBase(args, [self.max_num_CAVs*3])
        # self.mlp_all_hdv = MLPBase(args, [self.max_num_HDVs*3])
        # self.mlp_all_lane = MLPBase(args, [18 * 6])
        # self.mlp_global_combined = MLPBase(args, [10+64*3])

        ###### local encoder ######
        # self.mlp_target = MLPBase(args, [2])

        # self.gat_CAV_1s = GAT(nfeat=3, nhid=16, nclass=64, dropout=0.1, alpha=0.2, nheads=1)
        self.gat_CAV_1a = GAT(nfeat=6, nhid=16, nclass=64, dropout=0.1, alpha=0.2, nheads=1)
        # self.gat_HDV_1s = GAT(nfeat=3, nhid=16, nclass=64, dropout=0.1, alpha=0.2, nheads=1)
        self.gat_HDV_5s = GAT(nfeat=3*5, nhid=32, nclass=64, dropout=0.1, alpha=0.2, nheads=1)

        self.mlp_surround_lane = MLPBase(args, [3 * 6])
        # self.mlp_lane = MLP_improve(input_dim=3*6, output_dim=64, hidden_dim=64, num_layers=2, dropout=0.2)
        self.mlp_local_combined = MLPBase(args, [10+64*3])

        # cross-aware encoder
        # self.cross_attention = CrossAttention(64*3+10, 8, 64, 0.1)
        # self.mlp_combined = MLPBase(args, [64*3+10])

        self.example_extend_history = {
            'history_5': torch.zeros(self.one_step_obs_dim),
            'history_4': torch.zeros(self.one_step_obs_dim),
            'history_3': torch.zeros(self.one_step_obs_dim),
            'history_2': torch.zeros(self.one_step_obs_dim),
            'history_1': torch.zeros(self.one_step_obs_dim),
            'current': torch.zeros(self.one_step_obs_dim)
        }
        self.example_extend_info = {
            # 'bottle_neck_position': torch.zeros(2),
            'hdv_stats': torch.zeros(self.max_num_HDVs, 6),  # 0
            'cav_stats': torch.zeros(self.max_num_CAVs, 6),  # 1
            'all_lane_stats': torch.zeros(18, 6),            # 2
            'bottle_neck_position': torch.zeros(2),          # 3
            'road_structure': torch.zeros(10),               # 4
            'road_end': torch.zeros(2),                      # 5
            'target': torch.zeros(2),                        # 6
            'self_stats': torch.zeros(1, 13),                # 7
            'distance_bott': torch.zeros(2),                 # 8
            'distance_end': torch.zeros(2),                  # 9
            'executed_action': torch.zeros(1, 2),            # 10
            'generation_action': torch.zeros(1, 1),          # 11
            'surround_hdv_stats': torch.zeros(6, 6),         # 12
            'surround_cav_stats': torch.zeros(6, 6),         # 13
            'surround_lane_stats': torch.zeros(3, 6),        # 14
        }

    def reconstruct_history(self, obs):
        reconstructed = self.reconstruct_obs_batch(obs, self.example_extend_history)
        return reconstructed['history_5'], reconstructed['history_4'], reconstructed['history_3'], \
            reconstructed['history_2'], reconstructed['history_1'], \
            reconstructed['current']

    def reconstruct_info(self, obs):
        reconstructed = self.reconstruct_obs_batch(obs, self.example_extend_info)
        return reconstructed['hdv_stats'], reconstructed['cav_stats'], reconstructed['all_lane_stats'], \
            reconstructed['bottle_neck_position'], reconstructed['road_structure'], reconstructed['road_end'], \
            reconstructed['target'], reconstructed['self_stats'], \
            reconstructed['distance_bott'], reconstructed['distance_end'], \
            reconstructed['executed_action'], reconstructed['generation_action'], \
            reconstructed['surround_hdv_stats'], reconstructed['surround_cav_stats'], \
            reconstructed['surround_lane_stats']

    def forward(self, obs, batch_size=20):
        # obs: (n_rollout_thread, obs_dim)
        history_5, history_4, history_3, history_2, history_1, current = self.reconstruct_history(obs)

        # info_hist_5 = self.reconstruct_info(history_5)
        info_hist_4 = self.reconstruct_info(history_4)
        info_hist_3 = self.reconstruct_info(history_3)
        info_hist_2 = self.reconstruct_info(history_2)
        info_hist_1 = self.reconstruct_info(history_1)
        info_current = self.reconstruct_info(current)

        # self.adj_all_cav = torch.ones(batch_size, self.max_num_CAVs + 1, self.max_num_CAVs + 1)
        # self.adj_all_cav[:, 1:, 1:] = 0
        # self.adj_all_hdv = torch.ones(batch_size, self.max_num_HDVs + 1, self.max_num_HDVs + 1)
        # self.adj_all_hdv[:, 1:, 1:] = 0
        self.adj_surround_cav = torch.ones(batch_size, 6+1, 6+1)
        self.adj_surround_cav[:, 1:, 1:] = 0
        self.adj_surround_hdv = torch.ones(batch_size, 6+1, 6+1)
        self.adj_surround_hdv[:, 1:, 1:] = 0

        ################################################## global embedding ######################################################
        # global_road_info = info_current[4]
        # global_cav_stats = info_current[1]
        # ego_stats_current = info_current[7][:, :, :3]
        # cav_stats_current = info_current[1][:, :, :3]
        # hdv_stats_current = info_current[0][:, :, :3]
        # global_lane_stats = info_current[2]

        # global_cav_embedding = self.mlp_all_cav_a(global_cav_stats.view(global_cav_stats.size(0), -1))
        # global_cav_embedding = self.mlp_all_cav(cav_stats_current.reshape(cav_stats_current.size(0), -1))
        # global_hdv_embedding = self.mlp_all_hdv(hdv_stats_current.reshape(hdv_stats_current.size(0), -1))
        # global_lane_embedding = self.mlp_all_lane(global_lane_stats.view(global_lane_stats.size(0), -1))

        # global_ego2hdv_current = torch.cat((ego_stats_current, hdv_stats_current), dim=1)
        # global_ego2hdv_relation = self.gat_HDV_1s(global_ego2hdv_current, self.adj_all_hdv.to(global_ego2hdv_current.device))
        # global_ego2cav_current = torch.cat((ego_stats_current, cav_stats_current), dim=1)
        # global_ego2cav_relation = self.gat_CAV_1s(global_ego2cav_current, self.adj_all_cav.to(global_ego2cav_current.device))

        # global_combined_embedding = torch.cat((global_road_info, global_cav_embedding, global_hdv_embedding, global_lane_embedding), dim=1)
        # global_combined_embedding = self.mlp_global_combined(global_combined_embedding)

        ################################################## local embedding ######################################################
        # bottle_neck_0, distance_bott_0, road_end_0, distance_end_0, target_0
        # 03-'bottle_neck_position': torch.zeros(2),
        # 08-'distance_bott': torch.zeros(2),
        # 05-'road_end': torch.zeros(2),
        # 09-'distance_end': torch.zeros(2),
        # 06-'target': torch.zeros(2),
        local_road_info = torch.cat((info_current[8], info_current[5][:, :1], info_current[9], info_current[6]), dim=1)
        # local_road_info = torch.cat((info_current[4], info_current[3], info_current[8], info_current[5], info_current[9], info_current[6]), dim=1)
        # local_surround_cav_stats = info_current[13][:, :, :3]
        local_surround_cav_motion = info_current[13]
        # local_surround_hdv_stats = info_current[12][:, :, :3]
        # local_ego_stats = info_current[7][:, :, :3]
        local_ego_motion = torch.cat((info_current[7][:, :, :3], info_current[11], info_current[10]), dim=2)
        local_surround_lane_stats = info_current[14]

        ego_hist = torch.cat((info_hist_4[7][:, :, :3], info_hist_3[7][:, :, :3], info_hist_2[7][:, :, :3],
                              info_hist_1[7][:, :, :3], info_current[7][:, :, :3]), dim=2)
        hdv_hist = torch.cat((info_hist_4[12][:, :, :3], info_hist_3[12][:, :, :3], info_hist_2[12][:, :, :3],
                              info_hist_1[12][:, :, :3], info_current[12][:, :, :3]), dim=2)

        # combined_ego2hdv_current = torch.cat((local_ego_stats, local_surround_hdv_stats), dim=1)
        # ego2hdv_relation_current = self.gat_HDV_1s(combined_ego2hdv_current, self.adj_surround_hdv.to(combined_ego2hdv_current.device))
        combined_ego2hdv_hist = torch.cat((ego_hist, hdv_hist), dim=1)
        ego2hdv_relation_hist = self.gat_HDV_5s(combined_ego2hdv_hist, self.adj_surround_hdv.to(combined_ego2hdv_hist.device))

        # combined_ego2cav_current = torch.cat((local_ego_stats, local_surround_cav_stats), dim=1)
        # ego2cav_relation_current = self.gat_CAV_1s(combined_ego2cav_current, self.adj_surround_cav.to(combined_ego2cav_current.device))
        combined_ego2cav_current_a = torch.cat((local_ego_motion, local_surround_cav_motion), dim=1)
        ego2cav_relation_current_a = self.gat_CAV_1a(combined_ego2cav_current_a, self.adj_surround_cav.to(combined_ego2cav_current_a.device))

        local_cav_relation = ego2cav_relation_current_a
        local_hdv_relation = ego2hdv_relation_hist
        local_lane_embedding = self.mlp_surround_lane(local_surround_lane_stats.view(local_surround_lane_stats.size(0), -1))
        # local_lane_embedding = self.mlp_lane(local_surround_lane_stats.view(local_surround_lane_stats.size(0), -1))
        # local_lane_embedding = local_surround_lane_stats.view(local_surround_lane_stats.size(0), -1)
        exe_action = info_current[10].view(info_current[10].size(0), -1)
        gen_action = info_current[11].view(info_current[11].size(0), -1)
        local_combined_embedding = torch.cat((local_road_info, gen_action, exe_action, local_cav_relation, local_hdv_relation, local_lane_embedding), dim=1)
        local_combined_embedding = self.mlp_local_combined(local_combined_embedding)

        # cross-aware representation
        # input1 = global_combined_embedding.unsqueeze(1)
        # input2 = local_combined_embedding.unsqueeze(1)
        # cross_embedding = self.cross_attention(input1, input2)
        # cross_embedding = cross_embedding.squeeze(1)
        # cross_embedding = self.mlp_combined(cross_embedding)

        return local_combined_embedding

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