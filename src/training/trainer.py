import torch
import torch.nn.functional as F
from src.utils.utils import (gpu_accuracy_score, gpu_auc_score, gpu_f1_score,
                             gpu_precision_score, gpu_recall_score)

torch.backends.cudnn.benchmark = True


def train(model, graph, mask, labels, criterion, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(graph, mask)
    loss = criterion(out, labels)
    # acc = accuracy_score(labels.cpu().numpy(), torch.argmax(out, dim=1).cpu().numpy())
    acc = gpu_accuracy_score(labels, torch.argmax(out, dim=1))
    loss.backward()
    optimizer.step()
    return loss.item(), acc


def weighted_cross_entropy(pred, target, weight):
    loss = torch.nn.functional.cross_entropy(pred, target)
    return loss * weight

def evaluate(model, graph, mask, labels, criterion, threshold=0.5):
    model.eval()

    with torch.no_grad():
        out = model(graph, mask)
        loss = criterion(out, labels)

        # 计算预测概率
        probabilities = torch.softmax(out, dim=1)


        # 使用新阈值进行类别预测
        predictions = torch.where(probabilities[:, 1] >= threshold, torch.tensor(1).to(probabilities.device),
                                  torch.tensor(0).to(probabilities.device))

        probabilities_np = probabilities[:, 1].cpu().numpy()

        # 使用新的预测类别标签计算各个评估指标
        # acc = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
        acc = gpu_accuracy_score(labels, predictions)
        rcl = gpu_recall_score(labels, predictions)
        prc = gpu_precision_score(labels, predictions)
        f1 = gpu_f1_score(labels, predictions)
        auc = gpu_auc_score(labels, probabilities_np)

    return loss.item(), acc, rcl, prc, f1, auc

def predict(model, graph, threshold=0.5):
    mask = torch.ones(graph.num_edges, dtype=torch.bool)
    model.eval()
    with torch.no_grad():
        out = model(graph, mask)

        # 计算预测概率
        probabilities = torch.softmax(out, dim=1)

        # 使用新阈值进行类别预测
        predictions = torch.where(probabilities[:, 1] >= threshold, torch.tensor(1).to(probabilities.device),
                                  torch.tensor(0).to(probabilities.device))
    return predictions


# 计算预测的概率
def predict_proba(model, unlabeled_data):
    model.eval()
    with torch.no_grad():
        out = model(unlabeled_data)
        predictions = F.softmax(out, dim=1)
        predict_label = torch.argmax(out, dim=1)
    return predictions, predict_label
