import torch
import torch.nn as nn
from simulator.models.base.mlp import MLPBase
from simulator.models.base.simple_layers import MultiVeh_GAT, TrajectoryDecoder, CrossAttention
from simulator.models.base.transformers import TransformerModule

class GATAware(nn.Module):
    def __init__(self, obs_dim, action_dim, n_embd, action_type='Discrete', args=None):
        super(GATAware, self).__init__()
        self.max_num_HDVs = args['max_num_HDVs']
        self.max_num_CAVs = args['max_num_CAVs']
        self.num_HDVs = args['num_HDVs']
        self.num_CAVs = args['num_CAVs']
        self.hist_length = args['hist_length']
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_embd = n_embd
        self.action_type = action_type
        self.para_env = args['n_rollout_threads']

        # Prediction encoder & decoder
        self.mlp_surround_lane = MLPBase(args, [3 * 6])
        hist_hidden_size = 64
        self.hdv_linear = nn.Linear(4, 32)
        self.hdv_gru_layer = nn.GRU(input_size=32, hidden_size=64, batch_first=True)
        self.cav_linear = nn.Linear(7, 32)
        self.cav_transformer = TransformerModule(32, 2, 1, 32)
        self.cav_state_linear = nn.Linear(4, 32)
        self.cav_action_linear = nn.Linear(1, 32)
        self.cav_cross_attn = CrossAttention(32, 8, 32, 0.1)
        self.cav_gru_layer = nn.GRU(input_size=32, hidden_size=64, num_layers=1, batch_first=True)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        self.gat_all_vehs = MultiVeh_GAT(nfeat=64, nhid=128, nclass=64, dropout=0.2, alpha=0.2, nheads=1)
        self.mlp_enc_combined = MLPBase(args, [8 + 32 + 64 + 64 + 64])
        self.Pre_decoder = TrajectoryDecoder(input_dim=8 + 32 + 64 + 64 + 64, hidden_dim=256, output_dim=2)
        self.CAV_ids = [f'CAV_{i}' for i in range(self.num_CAVs)]
        self.HDV_ids = [f'HDV_{i}' for i in range(self.num_HDVs)]
        self.veh_ids = self.CAV_ids + self.HDV_ids

    def forward(self, info_current, actions, state_divisors, batch_size=20):
        # Extract vehicle IDs and types for surrounding vehicles
        vehicle_ids = info_current[14][:, 1:7]  # Shape: [batch_size, num_vehicles]
        type_indicator = vehicle_ids // 100  # Vehicle type: 1 for HDV, 2 for CAV
        specific_id = (vehicle_ids % 100).long()  # Specific vehicle ID

        # Create a mask for valid vehicle IDs
        valid_mask = (vehicle_ids != 0)

        if not valid_mask.any():
            # All values in valid_mask are False--Skip the subsequent process
            prediction_output = {}
        else:
            # Concatenate local road information
            local_road_info = torch.cat(
                (info_current[1], info_current[5], info_current[2], info_current[6]), dim=1)
            local_surround_lane_stats = info_current[15]
            local_lane_embedding = self.mlp_surround_lane(
                local_surround_lane_stats.reshape(local_surround_lane_stats.size(0), -1))

            # Retrieve vehicle history and adjacency information
            ego_veh_hist = info_current[9]  # ego_hist_motion
            adj_surround_vehs = info_current[11]  # surround_relation_graph
            surround_veh_hist = info_current[10]
            num_env, num_veh, num_state = surround_veh_hist.size()

            # Prepare tensors for storing features
            device = ego_veh_hist.device
            ego_features = torch.zeros(num_env, 1, 64, device=device)
            dyn_features = torch.zeros(num_env, num_veh, 64, device=device)

            # Process ego vehicle dynamic features
            state_tensor = ego_veh_hist[:, 0, :].reshape(num_env, 5, 7)
            state_tensor = self.cav_linear(state_tensor)
            transformer_out = self.cav_transformer(state_tensor)
            # state_tensor_state = state_tensor[:, :, :4]
            # state_tensor_action = state_tensor[:, :, 4:5]
            # state_tensor_state = self.leaky_relu(self.cav_state_linear(state_tensor_state))
            # state_tensor_action = self.leaky_relu(self.cav_action_linear(state_tensor_action))
            # cross_attn_out = self.cav_cross_attn(state_tensor_state, state_tensor_action)
            _, gru_hidden_1 = self.cav_gru_layer(transformer_out) # transformer_out cross_attn_out
            ego_features = self.leaky_relu(gru_hidden_1.squeeze(0)).unsqueeze(1)  # [num_env, 1, 64]

            # Surrounding vehicles - batch processing for types
            vehicle_types = surround_veh_hist[:, :, 0].long()
            state_tensors = surround_veh_hist[:, :, 1:].reshape(num_env, num_veh, 5, 7)

            # Process type 0 vehicles (HDVs)
            mask_type_0 = (vehicle_types == 0)
            if mask_type_0.any():
                states_type_0 = state_tensors[mask_type_0].reshape(-1, 5, 7)[:, :, :4]
                hdv_linear_out = self.leaky_relu(self.hdv_linear(states_type_0))
                _, gru_hidden_0 = self.hdv_gru_layer(hdv_linear_out)
                dyn_features[mask_type_0] = self.leaky_relu(gru_hidden_0.squeeze(0))

            # Process type 1 vehicles (CAVs)
            mask_type_1 = (vehicle_types == 1)
            if mask_type_1.any():
                states_type_1 = state_tensors[mask_type_1].reshape(-1, 5, 7)
                # cav_linear_out = self.cav_linear(states_type_1)
                # transformer_out_1 = self.cav_transformer(cav_linear_out)
                state_tensor_state = states_type_1[:, :, :4]
                state_tensor_action = states_type_1[:, :, 4:5]
                state_tensor_state = self.cav_state_linear(state_tensor_state)
                state_tensor_action = self.cav_action_linear(state_tensor_action)
                cross_attn_out_1 = self.cav_cross_attn(state_tensor_state, state_tensor_action)
                _, gru_hidden_1 = self.cav_gru_layer(cross_attn_out_1) # transformer_out_1
                dyn_features[mask_type_1] = self.leaky_relu(gru_hidden_1.squeeze(0))

            # Concatenate ego and surrounding vehicle features
            veh_dynamics = torch.cat((ego_features, dyn_features), dim=1)
            enc_veh_relation = self.gat_all_vehs(veh_dynamics, adj_surround_vehs)

            # Process other information and expand for all vehicles
            actions_embedding = self.cav_action_linear(actions)
            other_info = torch.cat((local_road_info, local_lane_embedding, actions_embedding), dim=1)
            other_info_expanded = other_info.unsqueeze(1).expand(-1, 7, -1)

            # Combine embeddings and predict future states
            enc_combined_embedding = torch.cat((other_info_expanded, veh_dynamics, enc_veh_relation), dim=2)
            # enc_combined_embedding = self.mlp_enc_combined(enc_combined_embedding)
            future_states = self.Pre_decoder(enc_combined_embedding.reshape(batch_size * 7, 8 + 32 + 64 + 64 + 64), future_steps=5, deviation='none')
            future_states = future_states.reshape(batch_size, 7, 5, 2)

            # Compute absolute future states for surrounding vehicles (vehicles 1 to 6)
            num_vehicles = 6  # Excluding ego vehicle
            ego_position_x = info_current[4][:, 0, 0]  # [batch_size]
            ego_position_y = info_current[4][:, 0, 1]
            # ego_speed = info_current[4][:, 0, 2]
            # ego_heading = info_current[4][:, 0, 3]

            # Expand ego vehicle states to match the dimensions
            ego_position_x = ego_position_x[:, None, None]  # [batch_size, 1, 1]
            ego_position_y = ego_position_y[:, None, None]
            # ego_speed = ego_speed[:, None, None]
            # ego_heading = ego_heading[:, None, None]

            # Compute absolute future states using vectorized operations
            absolute_future_states = torch.empty(batch_size, num_vehicles, 5, 2, device=future_states.device)
            absolute_future_states[:, :, :, 0] = (future_states[:, 1:7, :, 0] + ego_position_x) * state_divisors[0]
            absolute_future_states[:, :, :, 1] = (future_states[:, 1:7, :, 1] + ego_position_y) * state_divisors[1]
            # absolute_future_states[:, :, :, 2] = ego_speed - future_states[:, 1:7, :, 2]
            # absolute_future_states[:, :, :, 3] = future_states[:, 1:7, :, 3] + ego_heading

            # Flatten tensors to merge batch and vehicle dimensions
            batch_indices = torch.arange(batch_size, device=device)[:, None].expand(-1, num_vehicles)
            valid_batch_indices = batch_indices[valid_mask]          # [num_valid_entries]
            valid_vehicle_ids = specific_id[valid_mask]              # [num_valid_entries]
            valid_vehicle_types = type_indicator[valid_mask]         # [num_valid_entries]
            valid_future_states = absolute_future_states[valid_mask] # [num_valid_entries, 5, 4]

            # Combine vehicle types and IDs to create unique keys
            unique_keys, inverse_indices = torch.unique(
                valid_vehicle_types * 100 + valid_vehicle_ids, return_inverse=True)
            num_unique_keys = unique_keys.size(0)
            predictions = torch.zeros(batch_size, num_unique_keys, 5, 2, device=device)

            # Assign predictions using advanced indexing
            predictions[valid_batch_indices, inverse_indices] = valid_future_states

            # Generate keys for the prediction_output dictionary
            unique_keys_np = unique_keys.cpu().numpy()
            vehicle_types = ['HDV' if (key // 100) == 1 else 'CAV' for key in unique_keys_np]
            vehicle_ids_list = (unique_keys_np % 100).astype(int)
            keys = [f'{vt}_{vid}' for vt, vid in zip(vehicle_types, vehicle_ids_list)]

            # Create the prediction_output dictionary
            prediction_output = {key: predictions[:, idx] for idx, key in enumerate(keys)}

        return prediction_output
