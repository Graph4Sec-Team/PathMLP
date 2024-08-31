import torch
import numpy as np
from scipy import io
import os.path as osp
from utils import DotDict
import torch_geometric.transforms as T
from torch_geometric.datasets import Amazon, Actor, WikipediaNetwork, Flickr, WebKB, Planetoid, LINKXDataset
from torch_geometric.data import InMemoryDataset, download_url, Data
from sklearn.preprocessing import label_binarize
from torch_geometric.io import read_npz


def load_npy(path, name):
    if name == 'NBA':
        x = torch.tensor(np.load(f'{path}/Nba/x.npy'), dtype=torch.float)
        y = np.load(f'{path}/Nba/y.npy')
        y = torch.tensor(y, dtype=torch.long)
        edge_index = torch.tensor(np.load(f'{path}/Nba/edge_index.npy'), dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=y)
        transform = T.Compose([T.ToUndirected()])
        data = transform(data)
        dataset = {
            "num_classes": 2,
            "num_node_features": data.x.shape[1]
        }
        dataset = DotDict(dataset)
        return dataset, data
    elif name == 'BGP':
        x = torch.tensor(np.load(f'{path}/Bgp/x.npy'), dtype=torch.float)
        y = np.load(f'{path}/Bgp/y.npy')
        y = torch.tensor(y, dtype=torch.long)
        edge_index = torch.tensor(np.load(f'{path}/Bgp/edge_index.npy'), dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=y)
        transform = T.Compose([T.ToUndirected()])
        data = transform(data)
        dataset = {
            "num_classes": 7,
            "num_node_features": data.x.shape[1]
        }
        dataset = DotDict(dataset)
        return dataset, data
    elif name == 'Electronics':
        x = torch.tensor(np.load(f'{path}/Electronics/x.npy'), dtype=torch.float)
        y = torch.tensor(np.load(f'{path}/Electronics/y.npy'), dtype=torch.long)
        edge_index = torch.tensor(np.load(f'{path}/Electronics/edge_index.npy'), dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=y)
        transform = T.Compose([T.ToUndirected()])
        data = transform(data)
        dataset = {
            "num_classes": 167,
            "num_node_features": data.x.shape[1]
        }
        dataset = DotDict(dataset)
        return dataset, data


def load_npz(path, name):
    if name in ['tolokers', 'questions']:
        data = np.load(osp.join(path, name+'.npz'))
        x = torch.tensor(data['node_features'], dtype=torch.float)
        y = torch.tensor(data['node_labels'], dtype=torch.float)
        edge_index = torch.tensor(data['edges'], dtype=torch.long)
        edge_index = edge_index.permute(1, 0)
        data = Data(x=x, y=y, edge_index=edge_index)
        transform = T.Compose([T.ToUndirected()])
        data = transform(data)
        dataset = {
            "num_classes": 2,
            "num_node_features": data.x.shape[1]
        }
        dataset = DotDict(dataset)
        return dataset, data
    elif name in ['roman_empire','amazon_ratings']:
        data = np.load(osp.join(path, name+'.npz'))
        x = torch.tensor(data['node_features'], dtype=torch.float)
        y = torch.tensor(data['node_labels'], dtype=torch.long)
        edge_index = torch.tensor(data['edges'], dtype=torch.long)
        edge_index = edge_index.permute(1, 0)
        data = Data(x=x, y=y, edge_index=edge_index)
        print(f"edge_index detals {data.edge_index.shape}")
        transform = T.Compose([T.ToUndirected()])
        data = transform(data)
        print(f"edge_index detals {data.edge_index.shape}")
        if name == "amazon_ratings":
            dataset = {
                "num_classes": 5,
                "num_node_features": data.x.shape[1]
            }
        else:
            dataset = {
                "num_classes": 18,
                "num_node_features": data.x.shape[1]
            }
        dataset = DotDict(dataset)
        return dataset, data
    elif name in ['chameleon_f', 'squirrel_f']:
        data = np.load(osp.join(path, name + 'iltered_directed.npz'))
        x = torch.tensor(data['node_features'], dtype=torch.float)
        y = torch.tensor(data['node_labels'], dtype=torch.long)
        edge_index = torch.tensor(data['edges'], dtype=torch.long)
        edge_index = edge_index.permute(1, 0)
        data = Data(x=x, y=y, edge_index=edge_index)
        dataset = {
            "num_classes": 5,
            "num_node_features": data.x.shape[1]
        }
        dataset = DotDict(dataset)
        return dataset, data
    elif name in ['cora_full']:
        data = read_npz(osp.join(path, name + '.npz'))
        data = Data(x=data.x, y=data.y, edge_index=data.edge_index)
        dataset = {
            "num_classes": len(data.y.unique()),
            "num_node_features": data.x.shape[1]
        }
        dataset = DotDict(dataset)
        return dataset, data


def load_mat(path, name):
    full_data = io.loadmat(path + '/' + name + '.mat')
    A, metadata = full_data['A'], full_data['local_info']
    edge_index = torch.tensor(np.array(A.nonzero()), dtype=torch.long)
    metadata = metadata.astype(np.int)
    y = metadata[:, 1] - 1  # gender label, -1 means unlabeled
    feature_vals = np.hstack(
        (np.expand_dims(metadata[:, 0], 1), metadata[:, 2:]))
    features = np.empty((A.shape[0], 0))
    for col in range(feature_vals.shape[1]):
        feat_col = feature_vals[:, col]
        feat_onehot = label_binarize(feat_col, classes=np.unique(feat_col))
        # print(feat_onehot)
        features = np.hstack((features, feat_onehot))
    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.long)
    data = Data(x=x, y=y, edge_index=edge_index)
    transform = T.Compose([T.ToUndirected()])
    data = transform(data)
    dataset = {
        "num_classes": 2,
        "num_node_features": data.x.shape[1]
    }
    dataset = DotDict(dataset)
    return dataset, data


def load_data(path, name):
    if name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root=path, name=name)
    elif name in ['computers', 'photo']:
        dataset = Amazon(root=path, name=name)
    elif name in ['film']:
        dataset = Actor(root=f'{path}/film')
    elif name in ['texas', 'cornell', 'wisconsin']:
        dataset = WebKB(root=path, name=name)
        data = dataset[0]
        return data, dataset
    elif name in ['NBA', 'BGP', 'Electronics']:
        dataset, data = load_npy(path, name)
        return data, dataset
    elif name in ['Penn94']:
        dataset, data = load_mat(path, name)
        return data, dataset
    elif name in ['roman_empire', 'amazon_ratings', 'tolokers', 'questions', 'chameleon_f', 'squirrel_f', 'cora_full']:
        dataset, data = load_npz(path, name)
        return data, dataset
    elif name in ['chameleon', 'squirrel']:
        preProcDs = WikipediaNetwork(
            root=path, name=name, geom_gcn_preprocess=False)
        dataset = WikipediaNetwork(
            root=path, name=name, geom_gcn_preprocess=True)
        data = dataset[0]
        data.edge_index = preProcDs[0].edge_index
        return data, dataset
    return dataset[0], dataset
