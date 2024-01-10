import torch
import torch.optim as optim
import random, os
import numpy as np
from src.data_processing.data_loader import load_features
import pandas as pd

from src.data_processing.data_loader import load_pickle
from src.models.model import MyModel_2
from src.training.trainer import train, evaluate, predict


#% 加载数据
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


graph, train_mask, val_mask, test_mask = load_pickle("data/processed/graph_with_mask.pkl")

graph, train_mask, val_mask, test_mask = graph.to(device), train_mask.to(device), val_mask.to(device), test_mask.to(device)

train_labels = torch.cat((torch.ones(int(train_mask.sum().item() / 2), dtype=torch.long),
                          torch.zeros(int(train_mask.sum().item() / 2), dtype=torch.long)), dim=0).to(device)
val_labels = torch.cat((torch.ones(int(val_mask.sum().item() / 2), dtype=torch.long),
                        torch.zeros(int(val_mask.sum().item() / 2), dtype=torch.long)), dim=0).to(device)
test_labels = torch.cat((torch.ones(int(test_mask.sum().item() / 2), dtype=torch.long),
                         torch.zeros(int(test_mask.sum().item() / 2), dtype=torch.long)), dim=0).to(device)


##% 加载模型
model = MyModel_2().to(device)
model.load_state_dict(torch.load("model.pt"))
model.eval()

with torch.no_grad():
    node_embeddings = model.encoder(graph.x, graph.edge_index)
print(node_embeddings)
np.save('node_embeddings.npy', node_embeddings.detach().cpu().numpy())
