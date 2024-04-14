# %%
# -*- coding:utf-8 -*-
"""
@Time: 2023/02/07 10:33
@Author: shengtudai
@File: model.py
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

# %%
class GNN(nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(64, 32)
        self.conv2 = GCNConv(32, 16)

        self.norm = nn.LayerNorm(1024)
        self.adapt = nn.AdaptiveMaxPool1d(1024)
        self.w0 = nn.Linear(1024, 512)
        self.w1 = nn.Linear(512, 128)
        self.w2 = nn.Linear(128, 64)

    def forward(self, x, edge_index):
        x = self.adapt(x)
        x = self.norm(x)
        x = F.elu(self.w0(x))
        x = F.elu(self.w1(x))
        x = F.elu(self.w2(x))

        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# 二维编码，使用卷积
class EdgeEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z, edge_label_index):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        x = torch.einsum('ij,ik->ijk', src, dst).view(-1, 1, 16, 16)
        return x

# 一维编码，使用全连接
class EdgeEncoder_1(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z, edge_label_index):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        x = torch.cat([src, dst], dim=1)
        return x

# 选边
class ChooseLabeledEdge(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, edge_index, mask):
        return edge_index[:, mask]

# 边预测，用卷积
class LP(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=1)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=1)
        self.fc1 = nn.Linear(16 * 2 * 2, 16)
        self.fc2 = nn.Linear(16, 2)

    def num_flat_features(self, x):
        # 计算展平成一维向量时的元素个数（除去批次维度）
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        x = F.max_pool2d(F.elu(self.conv1(x)), (2, 2))
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.max_pool2d(F.elu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.elu(self.fc1(x))
        x = self.fc2(x)
        return x

# 边预测。
class LP_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(32, 16)
        self.w2 = nn.Linear(16, 2)

    def forward(self, x):
        x = F.elu(self.w1(x))
        x = self.w2(x)
        return x

class MyModel2(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = GNN()
        self.edge_encoder = EdgeEncoder_1()
        self.mask = ChooseLabeledEdge()
        self.lp = LP_1()

    def forward(self, graph, mask):
        x = self.encoder.forward(graph.x, graph.edge_index)
        # x = self.edge_encoder.forward(x, graph.edge_index)
        selected_edge_index = self.mask.forward(graph.edge_index, mask)
        # x = self.mask.forward(x, mask)
        x = self.edge_encoder.forward(x, selected_edge_index)
        x = self.lp.forward(x)
        return x
    
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = GNN()
        self.edge_encoder = EdgeEncoder()
        self.mask = ChooseLabeledEdge()
        self.lp = LP()

    def forward(self, graph, mask):
        x = self.encoder.forward(graph.x, graph.edge_index)
        selected_edge_index = self.mask.forward(graph.edge_index, mask)
        x = self.edge_encoder.forward(x, selected_edge_index)
        x = self.lp.forward(x)
        return x
    
