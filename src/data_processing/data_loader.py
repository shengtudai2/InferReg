import os
import pickle
import pandas as pd
import torch
from torch_geometric.data import Data

# 从不存在的边中随机采样
def sample_neg_edges(edge_index, pos_edge_index, num_nodes):
    src_nodes = pos_edge_index[:, 0].unique()
    pos_count = (pos_edge_index[:, 0] == src_nodes[:, None]).sum(dim=1)

    neg_edge_list = []
    for src, count in zip(src_nodes, pos_count):
        neighbors = edge_index[:, 1][edge_index[:, 0] == src]
        non_neighbors = torch.tensor(list(set(range(num_nodes)) - set(neighbors.tolist()) - {src.item()}))
        sampled_neg = non_neighbors[torch.randperm(non_neighbors.size(0))][:count]
        neg_edge_list.extend([(src.item(), tg.item()) for tg in sampled_neg])

    return torch.tensor(neg_edge_list, dtype=torch.long)

# 从存在的边中随机采样
def sample_neg_edges(edge_index, pos_edge_index, num_nodes):
    src_nodes = pos_edge_index[:, 0].unique()
    pos_count = (pos_edge_index[:, 0] == src_nodes[:, None]).sum(dim=1)

    neg_edge_list = []
    for src, count in zip(src_nodes, pos_count):
        print(src)
        neighbors = edge_index[:, 1][edge_index[:, 0] == src]
        pos_neighbors = pos_edge_index[:, 1][pos_edge_index[:, 0] == src]
        neg_neighbors = torch.tensor([n for n in neighbors if n not in pos_neighbors.tolist()])
        sampled_neg = neg_neighbors[torch.randperm(neg_neighbors.size(0))][:count]
        neg_edge_list.extend([(src.item(), tg.item()) for tg in sampled_neg])

    return torch.tensor(neg_edge_list, dtype=torch.long)

# 上述代码的优化版本
def sample_neg_edges(edge_index, pos_edge_index, num_nodes):
    src_nodes = pos_edge_index[:, 0].unique()

    # 创建一个邻接矩阵，将边的存在表示为 1
    adjacency_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
    adjacency_matrix[edge_index[:, 0], edge_index[:, 1]] = 1

    # 移除正边，得到负邻居矩阵
    adjacency_matrix[pos_edge_index[:, 0], pos_edge_index[:, 1]] = 0

    neg_edge_list = []
    for src in src_nodes:
        pos_count = (pos_edge_index[:, 0] == src).sum().item()

        # 获取负邻居，并对负邻居进行乱序
        neg_neighbors = torch.nonzero(adjacency_matrix[src], as_tuple=True)[0]

        neg_count = neg_neighbors.size(0)

        sampled_neg = neg_neighbors[torch.randperm(neg_neighbors.size(0))][:pos_count]

        # 将采样到的负边添加到 neg_edge_list 中
        neg_edges = torch.stack([torch.full((min(pos_count, neg_count),), src, dtype=torch.long), sampled_neg], dim=1)
        neg_edge_list.append(neg_edges)

    neg_edge_list = torch.vstack(neg_edge_list)
    return neg_edge_list

def save_pickle(dataset, file_name):
    """
    保存数据为二进制文件。
    """
    f = open(file_name, "wb")
    pickle.dump(dataset, f)
    f.close()

def load_pickle(file_name):
    """
    从二进制文件中加载数据。
    """
    f = open(file_name, "rb+")
    dataset = pickle.load(f)
    f.close()
    return dataset


def load_raw_data():
    gene_expression = pd.read_csv('../../data/raw/gene_expression.csv', header=0, index_col=0)
    all_edges = open('../../data/raw/edges.tsv', 'r').read().split('\n')
    pos_edges = open('../../data/raw/pos_edges.tsv', 'r').read().split('\n')

    return gene_expression, all_edges, pos_edges


def load_features(gene_expression):
    features = gene_expression.values
    features = torch.tensor(features, dtype=torch.float)

    node_dict = {node: i for i, node in enumerate(gene_expression.index)}

    with open('node_map.pkl', 'wb') as f:
        pickle.dump(node_dict, f)
        print("node_map.pkl saved.")

    return features, node_dict


def load_edges(edges, node_dict):
    edge_index = []
    for edge in edges:
        if edge == '':
            continue
        tf = edge.split('\t')[0]
        tgs = edge.split('\t')[1].split(',')
        for tg in tgs:
            edge_index.append((node_dict[tf], node_dict[tg]))
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    return edge_index


def sample_neg_edges(edge_index, pos_edge_index):
    # 获得边张量的大小，确定节点数量
    num_nodes = edge_index.max().item() + 1

    # 提取src节点
    src_nodes = pos_edge_index[:, 0].unique()

    # 计算每个src节点的正边数量
    pos_count = (pos_edge_index[:, 0] == src_nodes[:, None]).sum(dim=1)

    # 初始化负边张量
    neg_edge_index = torch.empty(0, 2, dtype=torch.long)

    # 遍历每个src节点
    for src, count in zip(src_nodes, pos_count):
        # 获取src节点所有的邻居
        neighbors = edge_index[:, 1][edge_index[:, 0] == src]

        # 计算邻居的补集
        non_neighbors = torch.tensor(list(set(range(num_nodes)) - set(neighbors.tolist()) - {src.item()}))

        # 在补集中随机采样负边
        sampled_neg = non_neighbors[torch.randperm(non_neighbors.size(0))][:count]

        # 将采样到的负边加入到负边张量中
        neg_edge_index = torch.cat(
            [neg_edge_index, torch.stack([torch.full_like(sampled_neg, src), sampled_neg], dim=1)], dim=0)

    return neg_edge_index


def split_edges(pos_edge_index, neg_edge_index):
    def shuffle_and_split(tensor, ratio=(0.8, 0.1, 0.1)):
        num_samples = tensor.shape[0]
        num_train = int(num_samples * ratio[0])
        num_val = int(num_samples * ratio[1])

        shuffled_indices = torch.randperm(num_samples)
        shuffled_tensor = tensor[shuffled_indices, :]

        train_data = shuffled_tensor[:num_train, :]
        val_data = shuffled_tensor[num_train:num_train + num_val, :]
        test_data = shuffled_tensor[num_train + num_val:, :]

        return train_data, val_data, test_data

    pos_train, pos_val, pos_test = shuffle_and_split(pos_edge_index)
    neg_train, neg_val, neg_test = shuffle_and_split(neg_edge_index)

    train_edges = torch.cat((pos_train, neg_train), dim=0)
    val_edges = torch.cat((pos_val, neg_val), dim=0)
    test_edges = torch.cat((pos_test, neg_test), dim=0)

    return train_edges, val_edges, test_edges


def split_train_val_test_edges(edge_index, pos_edge_index):
    neg_edge_index = sample_neg_edges(edge_index, pos_edge_index)
    all_edges = torch.cat((edge_index, neg_edge_index), dim=0)
    train_edge_index, val_edge_index, test_edge_index = split_edges(pos_edge_index, neg_edge_index)

    return all_edges, train_edge_index, val_edge_index, test_edge_index


def load_edge_labels(edge_index):
    y = torch.cat((torch.ones(int(edge_index.shape[0]/2), dtype=torch.long), torch.zeros(int(edge_index.shape[0]/2), dtype=torch.long)), dim=0)

    return y


def process_data(gene_expression, all_edges, pos_edges):
    features, node_dict = load_features(gene_expression)
    edge_index = load_edges(all_edges, node_dict)
    pos_edge_index = load_edges(pos_edges, node_dict)

    all_edges, train_edges, val_edges, test_edges = split_train_val_test_edges(edge_index, pos_edge_index)

    def edges_to_mask(all_edges, target_edges):
        device = "cuda"
        mask = torch.zeros(all_edges.shape[0], dtype=torch.bool).to(device)
        all_edges = all_edges.to(device)
        target_edges = target_edges.to(device)

        for i in range(target_edges.shape[0]):
            edge = target_edges[i].unsqueeze(0)
            matches = torch.all(all_edges == edge, dim=1)
            mask |= matches

        mask = mask.to("cpu")
        return mask

    train_mask = edges_to_mask(all_edges, train_edges)
    val_mask = edges_to_mask(all_edges, val_edges)
    test_mask = edges_to_mask(all_edges, test_edges)

    all_data = Data(x=features, edge_index=all_edges.t())

    return all_data, train_mask, val_mask, test_mask


if __name__ == "__main__":

    # Load raw data
    gene_expression, graph_structure, labeled_edges = load_raw_data()

    # Process data and split it into training and testing datasets
    all_data, train_mask, val_mask, test_mask= process_data(gene_expression, graph_structure, labeled_edges)

    os.makedirs('../../data/processed', exist_ok=True)
    save_pickle([all_data, train_mask, val_mask, test_mask], '../../data/processed/graph_with_mask.pkl')

