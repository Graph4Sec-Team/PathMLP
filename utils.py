import random
import torch
import numpy as np
from scipy import io
import os.path as osp

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

class DotDict(dict):
    """
    Makes a  dictionary behave like an object,with attribute-style access.
    """
    def __getattr__(self, name):
        try:
            return self[name]
        except:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name]=value


def set_seed(seed):
    import os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  # GPU
    torch.cuda.manual_seed_all(seed)  # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False  # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现


def compute_edge_index(indices):
    node_num, chain_num, depth_num = indices.shape[0], indices.shape[1], indices.shape[2]
    indices = indices.view(indices.shape[0], -1)
    index_count = 0
    for node in range(node_num):
        indice = indices[node].unique()
        index_count += len(indice)
    edge_index = torch.zeros(size=(2, index_count), dtype=torch.long).cuda()
    start_index = 0
    for node in range(node_num):
        indice = indices[node].unique()
        edge_index[0][start_index : start_index + len(indice)] = torch.tensor([node] * len(indice))
        edge_index[1][start_index : start_index + len(indice)] = indice
        start_index += len(indice)
    return edge_index


def multi_adj_precompute(x, adj, degree):
    for i in range(degree):
        x = torch.spmm(adj, x)
    return x


def reset_weight(model):
    reset_parameters = getattr(model, "reset_parameters", None)
    if callable(reset_parameters):
        model.reset_parameters()


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def random_split(data, num_classes, train_rate=0.48, val_rate=0.32):
    y_has_label = (data.y != -1).nonzero().contiguous().view(-1)
    num_nodes = y_has_label.shape[0]
    indices = torch.randperm(y_has_label.size(0))
    indices = y_has_label[indices]
    train_num = int(round(train_rate * num_nodes))
    val_num = int(round(val_rate * num_nodes))
    train_index = indices[:train_num]
    val_index = indices[train_num:train_num+val_num]
    test_index = indices[train_num+val_num:]
    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(val_index, size=data.num_nodes)
    data.test_mask = index_to_mask(test_index, size=data.num_nodes)
    return data