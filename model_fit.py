import os
import random

import numpy as np
import torch
import torch.optim as optim
from src.data_processing.data_loader import load_pickle
from src.models.model import MyModel, MyModel2
from src.training.trainer import evaluate, predict, train
from torch_geometric.data import Data

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

os.chdir(os.path.dirname(os.path.abspath(__file__)))

seed = 421
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def graph_to_undirect(graph, train_mask, val_mask, test_mask):
    edge_index = graph.edge_index
    edge_index = torch.cat([edge_index, torch.flip(edge_index, [0])], dim=1)
    graph.edge_index = edge_index
    if graph.y is not None:
        y = graph.y
        y = torch.cat([y, y], dim=0)
        graph.y = y
    train_mask = torch.cat([train_mask, torch.zeros(train_mask.shape[0]).bool()], dim=0)
    val_mask = torch.cat([val_mask, torch.zeros(val_mask.shape[0]).bool()], dim=0)
    test_mask = torch.cat([test_mask, torch.zeros(test_mask.shape[0]).bool()], dim=0)

    return graph, train_mask, val_mask, test_mask

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

graph, train_mask, val_mask, test_mask = load_pickle("data/processed/graph_with_mask.pkl")

graph, train_mask, val_mask, test_mask = graph.to(device), train_mask.to(device), val_mask.to(device), test_mask.to(device)

print("data load success")
print("graph: ", graph)
print("num_train: ", train_mask.sum().item())
print("num_val: ", val_mask.sum().item())
print("num_test: ", test_mask.sum().item())

train_labels = torch.cat((torch.ones(int(train_mask.sum().item() / 2), dtype=torch.long),
                          torch.zeros(int(train_mask.sum().item() / 2), dtype=torch.long)), dim=0).to(device)
val_labels = torch.cat((torch.ones(int(val_mask.sum().item() / 2), dtype=torch.long),
                        torch.zeros(int(val_mask.sum().item() / 2), dtype=torch.long)), dim=0).to(device)
test_labels = torch.cat((torch.ones(int(test_mask.sum().item() / 2), dtype=torch.long),
                         torch.zeros(int(test_mask.sum().item() / 2), dtype=torch.long)), dim=0).to(device)


model = MyModel2().to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0001)

n_epochs = 10000
patience = 200
min_val_loss = float("inf")
counter = 0

best_test_acc = 0
best_test_rcl = 0
best_test_prc = 0
best_test_f1 = 0
best_test_auc = 0

best_train_acc = 0
for epoch in range(1, n_epochs + 1):
    train_loss, train_acc = train(model, graph, train_mask, train_labels, criterion, optimizer)
    val_loss, val_acc, val_rcl, val_prc, val_f1, val_auc = evaluate(model, graph, val_mask, val_labels, criterion)
    test_loss, test_acc, test_rcl, test_prc, test_f1, test_auc = evaluate(model, graph, test_mask, test_labels, criterion)


    print(
        f"Epoch: {epoch}, Train Loss: {train_loss}, Train acc: {train_acc} ; Val Loss: {val_loss}, Test acc: {test_acc}")

    if val_loss < min_val_loss:
        min_val_loss = val_loss
        best_test_acc = test_acc
        best_test_rcl = test_rcl
        best_test_prc = test_prc
        best_test_f1 = test_f1
        best_test_auc = test_auc
        best_train_acc = train_acc
        print("val loss decreased ({} --> {}). Saving model ...".format(min_val_loss, val_loss))
        torch.save(model.state_dict(), "model.pt")
        counter = 0

    else:
        counter += 1
        if counter >= patience:
            print("Early stopped at epoch: ", epoch)
            break


print("Best train acc: ", best_train_acc)
print("Best val loss: ", min_val_loss)
print("\n")
print("Best test acc: ", best_test_acc)
print("Best test rcl: ", best_test_rcl)
print("Best test prc: ", best_test_prc)
print("Best test f1: ", best_test_f1)
print("Best test auc: ", best_test_auc)


predictions = predict(model, graph)
print(f"Predictions: {predictions}")
print(f"Predictions shape: {predictions.shape}")
print(f"Predictions sum: {predictions.sum()}")

