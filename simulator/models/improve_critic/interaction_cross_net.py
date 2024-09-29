import yaml
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from simulator.models.base.mlp import MLPBase
from simulator.models.base.simple_layers import CrossAttention, GAT, MultiVeh_GAT
from einops import rearrange, repeat

class Iteraction_cross_net(nn.Module):
    def __init__(self, args, obs_shape):
        super(Iteraction_cross_net, self).__init__()
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

        self.gat_all_vehs = MultiVeh_GAT(nfeat=3, nhid=16, nclass=64, dropout=0.2, alpha=0.2, nheads=1)
        # self.gat_all_vehs = GAT(nfeat=3, nhid=16, nclass=64, dropout=0.1, alpha=0.2, nheads=1)

        # cross-aware encoder
        self.cross_attention = CrossAttention(64, 8, 64, 0.1)

        self.example_extend_info = {
            'road_structure': torch.zeros(10),                                             # 0
            'bottle_neck_position': torch.zeros(2),                                        # 1
            'road_end': torch.zeros(2),                                                    # 2
            'target': torch.zeros(2),                                                      # 3
            'self_stats': torch.zeros(1, 13),                                              # 4
            'distance_bott': torch.zeros(2),                                               # 5
            'distance_end': torch.zeros(2),                                                # 6
            'actor_action': torch.zeros(1, 3),                                             # 7
            'actual_action': torch.zeros(1, 3),                                            # 8
            'ego_cav_motion': torch.zeros(1, 15),                                          # 9
            'ego_hdv_motion': torch.zeros(1, 15),                                          # 10
            'surround_stats': torch.zeros(6, 16),                                          # 11
            'expand_surround_stats': torch.zeros(10, 20),                                  # 12
            'surround_relation_graph': torch.zeros(10, 10),                                # 13
            'surround_IDs': torch.zeros(10),                                               # 14
            'surround_lane_stats': torch.zeros(3, 6),                                      # 15
            'hdv_stats': torch.zeros(self.max_num_HDVs, 7),                                # 16
            'cav_stats': torch.zeros(self.max_num_CAVs, 7),                                # 17
            'vehicle_relation_graph': torch.zeros(self.max_num_CAVs + self.max_num_HDVs,
                                                  self.max_num_CAVs + self.max_num_HDVs),  # 18
            'all_lane_stats': torch.zeros(18, 6),                                          # 19
        }

    def reconstruct_info(self, obs):
        reconstructed = self.reconstruct_obs_batch(obs, self.example_extend_info)
        return reconstructed['road_structure'], reconstructed['bottle_neck_position'], reconstructed['road_end'], \
            reconstructed['target'], reconstructed['self_stats'], \
            reconstructed['distance_bott'], reconstructed['distance_end'], \
            reconstructed['actor_action'], reconstructed['actual_action'], \
            reconstructed['ego_cav_motion'], reconstructed['ego_hdv_motion'], reconstructed['surround_stats'], \
            reconstructed['expand_surround_stats'], reconstructed['surround_relation_graph'], \
            reconstructed['surround_IDs'], reconstructed['surround_lane_stats'], \
            reconstructed['hdv_stats'], reconstructed['cav_stats'], reconstructed['vehicle_relation_graph'], \
            reconstructed['all_lane_stats']

    def forward(self, obs, batch_size=20):
        batch_size = obs.size(0)
        info_current = self.reconstruct_info(obs)
        # self.adj_all_vehs = torch.ones(batch_size, self.max_num_CAVs + self.max_num_HDVs, self.max_num_CAVs + self.max_num_HDVs)
        # self.adj_all_vehs[:, 1:, 1:] = 0

        self.adj_all_vehs = info_current[18]

        ################################################## global embedding ######################################################
        global_road_info = info_current[0]
        cav_stats_current = info_current[17][:, :, :3]
        hdv_stats_current = info_current[16][:, :, :3]
        global_lane_stats = info_current[19]

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
        input2 = all_veh_relation
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
