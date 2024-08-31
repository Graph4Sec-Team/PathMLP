import os
import argparse
import warnings
from utils import *
import torch.nn as nn
from tqdm import trange
from sklearn import metrics
from sampler import sample_paths
from model.PathMLP import PathMLP
from torch.optim import AdamW, Adam
from data.load_data import load_data
import torch_geometric.transforms as T
from torch_geometric.nn.conv.gcn_conv import gcn_norm


def train(data):
    model.train()
    optimizer.zero_grad()
    out = model(sampled_paths_indices, data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss, out


def val(data):
    model.eval()
    with torch.no_grad():
        out = model(sampled_paths_indices, data)
        if args.dataset in ['tolokers', 'questions']:
            pred, label, val_mask = out, data.y, data.val_mask
            label, pred , val_mask = label.cpu().numpy(), pred.cpu().numpy(), val_mask.cpu().numpy()
            auc = metrics.roc_auc_score(label[val_mask], pred[val_mask] )
            return auc
        else:
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            val_correct = pred[data.val_mask] == data.y[data.val_mask]  # Check against ground-truth labels.
            val_correct = int(val_correct.sum()) / int(data.val_mask.sum())  # Derive ratio of correct predictions.
            return val_correct


def test(data):
    model.eval()
    with torch.no_grad():
        out = model(sampled_paths_indices, data)
        if args.dataset in ['tolokers', 'questions']:
            pred, label, test_mask = out, data.y, data.test_mask
            label, pred, test_mask = label.cpu().numpy(), pred.cpu().numpy(), test_mask.cpu().numpy()
            auc = metrics.roc_auc_score(label[test_mask], pred[test_mask])
            return auc
        else:
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
            test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
            return test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ['cora', 'citeseer', 'pubmed', 'computers', 'photo', 'film', 'texas', 'cornell',
    # 'chameleon', 'squirrel', 'wisconsin', 'NBA', 'BGP', 'Electronics']
    # ['minesweeper', 'tolokers', 'questions', 'roman_empire','amazon_ratings']
    # ['chameleon_f', 'squirrel_f', 'Penn94, 'genius', 'cora_full']
    parser.add_argument('--dataset', '-D', type=str, default='NBA')
    parser.add_argument('--hidden', '-H', type=int, default=64)
    parser.add_argument('--path_hidden', '-rH', type=int, default=24)
    parser.add_argument('--lr', '-L', type=float, default=0.01)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--fdropout', type=float, default=0.5)
    parser.add_argument('--model', '-M', type=str, default='PathMLP', help="PathMLP, PathMLP+")
    parser.add_argument('--degree', type=int, default=2, help="1, 2")
    parser.add_argument('--beta', type=float, default=0.0, help="0.0, 0.3, 0.5")
    parser.add_argument('--path_attention', type=str, default="learnable", help="learnable gat_attention dot scale_dot cos / mean sum")
    parser.add_argument('--path_encoder', type=str, default="concat", help="concat, mean, sum")

    parser.add_argument('--path', type=str, default='../../PathMLP/data')
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--times', type=int, default=10, help="10 times exp for mean std")
    # ================= sampler ========================#
    parser.add_argument('--path_num', type=int, default=10)
    parser.add_argument('--depth', type=int, default=4)
    args = parser.parse_args()
    print(args)
    warnings.filterwarnings("ignore")

    # 1 load data
    data, dataset = load_data(args.path, args.dataset)
    print("=============================================================")
    print(f"load {args.dataset} successfully!!, details: {data}")
    # 3 load model
    print("=============================================================")
    print(f"load {args.model} successfully!!!")
    # 4 optimizer, loss function and other hypermeter
    print("=============================================================")
    print(f"load lr is {args.lr} , weight decay is {args.wd} , dropout is {args.dropout} , hidden size is {args.hidden}")
    # 5 train/val/test
    print("=============================================================")

    sampled_path = f"./path/{args.dataset}"
    sampled_paths = f"./path/{args.dataset}/depth_{args.depth}_path_num_{args.path_num}.npz"
    if os.path.exists(sampled_paths):
        sampled_paths_indices = np.load(sampled_paths)["indices"]
        sampled_paths_indices = torch.tensor(sampled_paths_indices).to(device)
        print("load sampled_paths_indices from npz success!!!")
    else:
        if not os.path.exists(sampled_path):
            os.makedirs(sampled_path)
        sampled_paths_indices = sample_paths(data, args)
        np.savez(sampled_paths, indices=sampled_paths_indices)
        sampled_paths_indices = torch.tensor(sampled_paths_indices).to(device)

    if args.model == "PathMLP+":
        edge_index, edge_weight = gcn_norm(data.edge_index)
        sparse_edge_index = torch.sparse_coo_tensor(edge_index, edge_weight)
        data.x = torch.cat((data.x, multi_adj_precompute(data.x, sparse_edge_index, args.degree)), dim=-1)
    data = data.to(device)

    # 5 train/val/test
    count = 1
    val_acc_list, test_acc_list = [], []
    for t in trange(args.times):
        best_val_acc = 0
        set_seed(t)

        # random split for dataset
        if args.dataset in ['roman_empire', 'amazon_ratings', 'minesweeper', 'tolokers', 'questions']:
            train_rate = 0.50
            val_rate = 0.25
        else:
            train_rate = 0.48
            val_rate = 0.32
        data = random_split(data, dataset.num_classes, train_rate, val_rate)

        if args.dataset in ['minesweeper', 'tolokers', 'questions', 'genius']:
            args.is_binary = True
            criterion = nn.BCELoss()
        else:
            args.is_binary = False
            criterion = nn.CrossEntropyLoss()

        # 3 load model
        model = PathMLP(dataset, data.x.shape[0], args).to(device)

        # 4 optimizer, loss function and other hypermeter
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

        # 5 early stopping
        model.apply(reset_weight)
        step_count = args.patience
        for ep in range(args.epochs):
            loss, out = train(data)
            val_acc = val(data)
            if val_acc > best_val_acc:
                step_count = args.patience
                best_val_acc = val_acc
                test_acc = test(data)
            else:
                step_count -= 1
            if step_count <= 0:
                break
        val_acc_list.append(best_val_acc)
        test_acc_list.append(test_acc)
    val_acc_list = torch.tensor(val_acc_list)
    test_acc_list = torch.tensor(test_acc_list)
    print(f"{args.dataset} valid acc is {100 * val_acc_list.mean().item():.2f} ± {100 * val_acc_list.std().item():.2f}")
    print(f"{args.dataset} test acc is {100 * test_acc_list.mean().item():.2f} ± {100 * test_acc_list.std().item():.2f}")
