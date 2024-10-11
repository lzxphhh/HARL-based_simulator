import torch
import torch.nn as nn
import numpy as np
from simulator.models.base.mlp import MLPBase
from simulator.models.base.simple_layers import MultiVeh_GAT, TrajectoryDecoder, CrossAttention, CrossTransformer
from simulator.models.base.transformers import TransformerModule

class Flow_transformer_net(nn.Module):
    def __init__(self, args, obs_shape):
        super(Flow_transformer_net, self).__init__()
        self.max_num_HDVs = args['max_num_HDVs']
        self.max_num_CAVs = args['max_num_CAVs']
        self.num_HDVs = args['num_HDVs']
        self.num_CAVs = args['num_CAVs']
        # 单个智能体shared_obs_dim
        self.shared_obs_dim = obs_shape

        ###### global encoder ######
        # self.mlp_statis = MLPBase(args, [6])
        # self.mlp_veh_distribution = MLPBase(args, [20])
        self.statis_linear = nn.Linear(6, 32)
        self.distribution_linear = nn.Linear(20, 32)
        self.gru_layer = nn.GRU(input_size=32, hidden_size=64, batch_first=True)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

        self.gat_all_lanes = MultiVeh_GAT(nfeat=64, nhid=128, nclass=64, dropout=0.2, alpha=0.2, nheads=1)
        self.mlp_combined = MLPBase(args, [64 + 64])
        self.Pre_decoder = TrajectoryDecoder(input_dim=64, hidden_dim=32, output_dim=6)

        # cross-aware encoder
        self.cross_attention = CrossAttention(64, 8, 64, 0.1)
        self.cross_transformer = CrossTransformer(dim=32, depth=2, heads=8, dim_head=64, mlp_dim=64, dropout=0.2)

    def generate_connection_matrices(self, num_env, device):
        group_sizes = [4, 4, 4, 2, 4]
        group_indices = []
        start = 0
        for size in group_sizes:
            end = start + size
            group_indices.append((start, end))
            start = end

        # 定义 A1, A2, A3, A4 矩阵
        A1 = torch.tensor([
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [0, 1, 1, 1],
            [0, 0, 1, 1]
        ], dtype=torch.float32, device=device)

        A2 = torch.eye(4, dtype=torch.float32, device=device)

        A3 = torch.tensor([
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ], dtype=torch.float32, device=device)

        A4 = torch.tensor([
            [1, 1],
            [1, 1]
        ], dtype=torch.float32, device=device)

        # 初始化基础连接矩阵
        base_matrix = torch.zeros((18, 18), dtype=torch.float32, device=device)

        # 填充矩阵块
        idx1 = group_indices[0]  # (0, 4)
        idx2 = group_indices[1]  # (4, 8)
        idx3 = group_indices[2]  # (8, 12)
        idx4 = group_indices[3]  # (12, 14)
        idx5 = group_indices[4]  # (14, 18)

        # Block(1,1): A1
        base_matrix[idx1[0]:idx1[1], idx1[0]:idx1[1]] = A1

        # Block(2,1): A2
        base_matrix[idx2[0]:idx2[1], idx1[0]:idx1[1]] = A2
        # Block(2,2): A1
        base_matrix[idx2[0]:idx2[1], idx2[0]:idx2[1]] = A1

        # Block(3,2): A2
        base_matrix[idx3[0]:idx3[1], idx2[0]:idx2[1]] = A2
        # Block(3,3): A1
        base_matrix[idx3[0]:idx3[1], idx3[0]:idx3[1]] = A1

        # Block(4,3): A2
        base_matrix[idx4[0]:idx4[1], idx5[0]:idx5[1]] = A3
        # Block(4,4): A1
        base_matrix[idx4[0]:idx4[1], idx4[0]:idx4[1]] = A4

        # Block(5,4): A3
        base_matrix[idx5[0]:idx5[1], idx3[0]:idx3[1]] = A2
        # Block(5,5): A4
        base_matrix[idx5[0]:idx5[1], idx5[0]:idx5[1]] = A1

        # 为每个环境复制基础矩阵
        conn_matrices = base_matrix.unsqueeze(0).repeat(num_env, 1, 1)

        return conn_matrices

    def forward(self, info_current, batch_size):
        flow_hist = info_current[20]
        num_lane = 18
        device = flow_hist.device
        dyn_features = torch.zeros(batch_size, num_lane, 64, device=device)
        for i in range(num_lane):
            lane_hist = flow_hist[:, i, :]
            state_tensor = lane_hist.reshape(batch_size, 5, 26)
            statis_tensor = state_tensor[:, :, :6]
            veh_distribution = state_tensor[:, :, 6:]
            statis_embedding = self.statis_linear(statis_tensor)
            distribution_embedding = self.distribution_linear(veh_distribution)
            ct_embedding = self.cross_transformer(statis_embedding, distribution_embedding)
            _, gru_hidden_1 = self.gru_layer(ct_embedding)
            dyn_features[:, i, :] = self.leaky_relu(gru_hidden_1.squeeze(0))
        connection_matrices = self.generate_connection_matrices(batch_size, device)

        lane_relation = self.gat_all_lanes(dyn_features, connection_matrices)
        combined_embedding = torch.cat((dyn_features, lane_relation), dim=2)
        combined_embedding = self.mlp_combined(combined_embedding)
        future_states = self.Pre_decoder(combined_embedding.reshape(batch_size * 18, 64), future_steps=5, deviation='relu')
        future_states = future_states.reshape(batch_size, 18, 5, 6)

        return future_states




