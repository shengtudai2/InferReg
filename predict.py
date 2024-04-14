import torch
import torch.optim as optim
import random, os
import numpy as np
from src.data_processing.data_loader import load_features
import pandas as pd

from src.data_processing.data_loader import load_pickle
from src.models.model import MyModel2
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

##% 指定阈值,模型在测试集以及预测的结果的表现
criterion = torch.nn.CrossEntropyLoss()

def performance(threshold):
    test_loss, test_acc, test_rcl, test_prc, test_f1, test_auc = evaluate(model, graph, test_mask, test_labels, criterion, threshold=threshold)
    predictions = predict(model, graph, threshold=threshold)
    return test_acc, test_rcl, test_prc, test_f1, test_auc, predictions


results = []

for t in np.arange(0.5, 1, 0.01):
    test_acc, test_rcl, test_prc, test_f1, test_auc, predictions = performance(t)
    results.append([t, test_acc, test_rcl, test_prc, test_f1, test_auc])

results = pd.DataFrame(results, columns=['threshold', 'acc', 'recall', 'precision', 'f1', 'auc'])
# 画图
import matplotlib.pyplot as plt
plt.plot(results['threshold'], results['acc'], label='acc')
plt.plot(results['threshold'], results['recall'], label='recall')
plt.plot(results['threshold'], results['precision'], label='precision')
plt.plot(results['threshold'], results['f1'], label='f1')
# plt.plot(results['threshold'], results['auc'], label='auc')
plt.axhline(y=0.8, color='gray', linestyle='--')
plt.axvline(x=0.77, color='gray', linestyle='--')

plt.xlabel("threshold")
plt.ylabel('performance')

plt.legend()
plt.savefig('data/predicted/acc_recall_precision_f1_auc.svg', format='svg')

plt.show()


t = 0.5
test_acc, test_rcl, test_prc, test_f1, test_auc, predictions = performance(t)

gene_expression = pd.read_csv('data/raw/gene_expression.csv', header=0, index_col=0)
features, node_dict = load_features(gene_expression)
predicted_edges = graph.edge_index[:, predictions == 1]

# 将节点id转换为节点名称
id2name = {i: name for name, i in node_dict.items()}
edges = predicted_edges.cpu().numpy()

src2dst = {}
for src_id, dst_id in edges.T:
    src_name = id2name[src_id]
    dst_name = id2name[dst_id]
    if src_name not in src2dst:
        src2dst[src_name] = []
    src2dst[src_name].append(dst_name)

# 将src和dst写入文件
with open('data/predicted/at_edges.tsv', 'w') as f:
    for src_name, dst_list in src2dst.items():
        if len(dst_list) < 10:
             continue
        dst_str = ','.join(dst_list)
        f.write(f'{src_name}\t{dst_str}\n')
