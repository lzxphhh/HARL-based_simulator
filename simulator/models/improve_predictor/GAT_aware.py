import yaml
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from simulator.models.base.mlp import MLPBase
from simulator.models.base.simple_layers import CrossAttention, GAT, MLP_improve, MultiVeh_GAT, TrajectoryDecoder
from einops import rearrange, repeat
import time

class GATAware(nn.Module):
    def __init__(self, obs_dim, action_dim, n_embd, action_type='Discrete', args=None):
        super(GATAware, self).__init__()
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

        # prediction encoder & decoder
        self.mlp_surround_lane = MLPBase(args, [3 * 6])
        self.gat_all_vehs = MultiVeh_GAT(nfeat=20, nhid=32, nclass=64, dropout=0.1, alpha=0.2, nheads=1)
        self.mlp_enc_combined = MLPBase(args, [10+64+64+1])
        self.Pre_decoder = TrajectoryDecoder(input_dim=64, hidden_dim=64, output_dim=3)
        self.CAV_ids = [f'CAV_{i}' for i in range(self.num_CAVs)]
        self.HDV_ids = [f'HDV_{i}' for i in range(self.num_HDVs)]
        self.veh_ids = self.CAV_ids + self.HDV_ids

    def forward(self, info_current, new_action, batch_size=20):
        # bottle_neck_0, distance_bott_0, road_end_0, distance_end_0, target_0
        # 01-'bottle_neck_position': torch.zeros(2),
        # 05-'distance_bott': torch.zeros(2),
        # 02-'road_end': torch.zeros(2),
        # 06-'distance_end': torch.zeros(2),
        # 03-'target': torch.zeros(2),
        local_road_info = torch.cat((info_current[5], info_current[2][:, :1], info_current[6], info_current[3]), dim=1)
        exe_action = info_current[8].view(info_current[11].size(0), -1)
        local_surround_lane_stats = info_current[15]
        local_lane_embedding = self.mlp_surround_lane(local_surround_lane_stats.view(local_surround_lane_stats.size(0), -1))

        ################################################## prediction encoder & decoder ######################################################

        self.adj_surround_vehs = info_current[13]
        enc_veh_hist = info_current[12]
        enc_veh_relation = self.gat_all_vehs(enc_veh_hist, self.adj_surround_vehs)
        other_info = torch.cat((local_road_info, exe_action, local_lane_embedding), dim=1)
        other_info_expanded = other_info.unsqueeze(1).repeat(1, 10, 1)
        action_expanded = new_action.unsqueeze(1).repeat(1, 10, 1)
        enc_combined_embedding = torch.cat((other_info_expanded, enc_veh_relation, action_expanded), dim=2)
        enc_combined_embedding = self.mlp_enc_combined(enc_combined_embedding)

        # Predict future states
        future_states = self.Pre_decoder(enc_combined_embedding.view(batch_size * 10, 64), future_steps=3)

        # Reshape to [batch_size, num_veh, future_steps, state_dim]
        future_states = future_states.view(batch_size, 10, 3, 3)

        # Change the relative state to absolute state
        ego_position_x = info_current[4][:, :, 0] * 700
        ego_position_y = info_current[4][:, :, 1]
        ego_speed = info_current[4][:, :, 2] * 15

        # Broadcasting and vectorized operations
        absolute_future_states = torch.zeros(batch_size, 10, 3, 3, device='cuda')
        absolute_future_states[:, :, :, 0] = future_states[:, :, :, 0] + ego_position_x[:, 0, None, None]
        absolute_future_states[:, :, :, 1] = future_states[:, :, :, 1] + ego_position_y[:, 0, None, None]
        absolute_future_states[:, :, :, 2] = ego_speed[:, 0, None, None] - future_states[:, :, :, 2]

        # Initialize prediction_output and prediction_groundtruth
        prediction_output = {veh_id: torch.zeros(batch_size, 3, 3, device='cuda') for veh_id in self.veh_ids}
        prediction_groundtruth = {veh_id: torch.zeros(batch_size, 3, device='cuda') for veh_id in self.veh_ids}

        # Process ego_id
        ego_id = self.CAV_ids[int(info_current[14][0, 0] - 200)] if info_current[14][0, 0] >= 200 else 0

        # Extract the type indicator (10 or 20)
        vehicle_ids = info_current[14]
        type_indicator = vehicle_ids // 100
        # Extract the specific ID number (x in HDV_x or CAV_x)
        specific_id = (vehicle_ids % 100).int()
        # Iterate over the tensor and map each ID
        for i in range(batch_size):
            for j in range(10):
                if type_indicator[i, j] == 1:
                    prediction_output[f'HDV_{specific_id[i, j].item()}'][i, :, :] = absolute_future_states[i, j, :, :]
                elif type_indicator[i, j] == 2:
                    prediction_output[f'CAV_{specific_id[i, j].item()}'][i, :, :] = absolute_future_states[i, j, :, :]

        # Process prediction_groundtruth using vectorized operations
        for veh_id in self.veh_ids:
            idx = int(veh_id[4:])
            if veh_id.startswith('CAV'):
                current_state = info_current[17][:, idx, :3]
            else:
                current_state = info_current[16][:, idx, :3]

            current_state[:, 0] = current_state[:, 0] * 700
            current_state[:, 2] = current_state[:, 2] * 15
            prediction_groundtruth[veh_id] = current_state
        return prediction_output, ego_id, prediction_groundtruth
