import yaml
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from simulator.models.base.mlp import MLPBase
from simulator.models.base.simple_layers import CrossAttention, GAT
from einops import rearrange, repeat

class Cross_aware_net(nn.Module):
    def __init__(self, args, obs_shape):
        super(Cross_aware_net, self).__init__()
        self.max_num_HDVs = args['max_num_HDVs']
        self.max_num_CAVs = args['max_num_CAVs']
        self.num_HDVs = args['num_HDVs']
        self.num_CAVs = args['num_CAVs']
        # 单个智能体shared_obs_dim
        self.shared_obs_dim = obs_shape[0]

        ###### global encoder ######
        self.mlp_all_cav = MLPBase(args, [self.max_num_CAVs * 3])
        self.mlp_all_hdv = MLPBase(args, [self.max_num_HDVs * 3])
        self.mlp_all_lane = MLPBase(args, [18 * 6])
        self.mlp_global_combined = MLPBase(args, [10 + 64 * 3])

        self.gat_all_vehs = GAT(nfeat=3, nhid=16, nclass=64, dropout=0.1, alpha=0.2, nheads=1)

        # cross-aware encoder
        self.cross_attention = CrossAttention(64, 8, 64, 0.1)

        self.example_extend_info = {
            # 'bottle_neck_position': torch.zeros(2),
            'hdv_stats': torch.zeros(self.max_num_HDVs, 6),  # 0
            'cav_stats': torch.zeros(self.max_num_CAVs, 6),  # 1
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
            'vehicle_relation_graph': torch.zeros(self.max_num_CAVs+self.max_num_HDVs, self.max_num_CAVs+self.max_num_HDVs),  # 15
        }

    def reconstruct_info(self, obs):
        reconstructed = self.reconstruct_obs_batch(obs, self.example_extend_info)
        return reconstructed['hdv_stats'], reconstructed['cav_stats'], reconstructed['all_lane_stats'], \
            reconstructed['bottle_neck_position'], reconstructed['road_structure'], reconstructed['road_end'], \
            reconstructed['target'], reconstructed['self_stats'], \
            reconstructed['distance_bott'], reconstructed['distance_end'], \
            reconstructed['executed_action'], reconstructed['generation_action'], \
            reconstructed['surround_hdv_stats'], reconstructed['surround_cav_stats'], \
            reconstructed['surround_lane_stats'], reconstructed['vehicle_relation_graph']

    def forward(self, obs, batch_size=20):
        batch_size = obs.size(0)
        info_current = self.reconstruct_info(obs)
        # self.adj_all_vehs = torch.ones(batch_size, self.max_num_CAVs + self.max_num_HDVs, self.max_num_CAVs + self.max_num_HDVs)
        # self.adj_all_vehs[:, 1:, 1:] = 0

        self.adj_all_vehs = info_current[15]

        ################################################## global embedding ######################################################
        global_road_info = info_current[4]
        cav_stats_current = info_current[1][:, :, :3]
        hdv_stats_current = info_current[0][:, :, :3]
        global_lane_stats = info_current[2]

        global_lane_embedding = self.mlp_all_lane(global_lane_stats.view(global_lane_stats.size(0), -1))
        global_cav_embedding = self.mlp_all_cav(cav_stats_current.reshape(cav_stats_current.size(0), -1))
        global_hdv_embedding = self.mlp_all_hdv(hdv_stats_current.reshape(hdv_stats_current.size(0), -1))
        global_combined_embedding = torch.cat(
            (global_road_info, global_cav_embedding, global_hdv_embedding, global_lane_embedding), dim=1)
        global_combined_embedding = self.mlp_global_combined(global_combined_embedding)

        ################################################## interaction embedding ######################################################
        all_vehicle_embedding = torch.cat((cav_stats_current, hdv_stats_current), dim=1)
        all_veh_relation = self.gat_all_vehs(all_vehicle_embedding, self.adj_all_vehs.to(all_vehicle_embedding.device))

        # cross-aware representation
        input1 = global_combined_embedding.unsqueeze(1)
        input2 = all_veh_relation.unsqueeze(1)
        cross_embedding = self.cross_attention(input1, input2)
        cross_embedding = cross_embedding.squeeze(1)

        return cross_embedding

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
