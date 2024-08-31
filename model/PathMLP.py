import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F


class PathMLP(nn.Module):
    def __init__(self, dataset, num_nodes, args: argparse.Namespace):
        super(PathMLP, self).__init__()
        self.args = args
        self.act = F.gelu
        self.beta = args.beta
        self.leaky_relu = nn.LeakyReLU()
        if args.model == "PathMLP+":
            self.path_project = nn.Linear(2 * dataset.num_node_features, args.path_hidden)
            self.mlp_x = nn.Linear(2 * dataset.num_node_features, args.hidden)
        else:
            self.path_project = nn.Linear(dataset.num_node_features, args.path_hidden)
            self.mlp_x = nn.Linear(dataset.num_node_features, args.hidden)
        self.path_attention = nn.Parameter(torch.zeros(size=(num_nodes, args.path_num)))
        self.mlp_attention = nn.Linear(2 * args.hidden, 1)
        self.mlpA = nn.Linear(num_nodes, args.hidden)
        self.mlp_path = nn.Linear(args.depth * args.path_hidden, args.hidden)
        self.mlp_path_encoder = nn.Linear(args.path_hidden, args.hidden)
        if self.args.dataset in ['tolokers', 'questions']:
            self.mlp_end = nn.Linear(args.hidden, 1)
        else:
            self.mlp_end = nn.Linear(args.hidden, dataset.num_classes)

    def forward(self, index, data):
        x = data.x
        path_inputs = self.path_project(x)
        path_inputs = path_inputs[index]

        # channel 1
        x = self.mlp_x(x)
        x = F.dropout(self.act(x), p=self.args.dropout, training=self.training)

        # channel 2
        num_nodes, num_path, depth, hidden_size = path_inputs.shape
        if self.args.path_encoder == "concat":
            path_inputs = path_inputs.reshape(num_nodes, num_path, -1)
            path_inputs = self.mlp_path(path_inputs)
        elif self.args.path_encoder == "mean":
            path_inputs = torch.mean(path_inputs, dim=2)
            path_inputs = self.mlp_path_encoder(path_inputs)
        elif self.args.path_encoder == "sum":
            path_inputs = torch.sum(path_inputs, dim=2)
            path_inputs = self.mlp_path_encoder(path_inputs)
        path_inputs = F.dropout(self.act(path_inputs), p=self.args.dropout, training=self.training)
        # path attention -> learnable gat_attention dot scale_dot cos
        path_attention = []
        if self.args.path_attention not in ["mean", "sum"]:
            if self.args.path_attention == "learnable":
                path_attention = F.softmax(self.path_attention, dim=-1).unsqueeze(2)
            elif self.args.path_attention == "gat_attention":
                gat_x = x.unsqueeze(1).repeat(1, self.args.path_num, 1)
                path_attention = F.softmax(self.leaky_relu(self.mlp_attention(
                    torch.cat((path_inputs, gat_x), dim=-1))), dim=1)
            elif self.args.path_attention == "dot":
                path_attention = torch.sum(x.unsqueeze(1) * path_inputs, dim=2)
                path_attention = F.softmax(path_attention, dim=-1).unsqueeze(2)
            elif self.args.path_attention == "scale_dot":
                path_attention = torch.sum(x.unsqueeze(1) * path_inputs, dim=2) / \
                                 torch.sqrt(torch.tensor(self.args.hidden))
                path_attention = F.softmax(path_attention, dim=-1).unsqueeze(2)
            elif self.args.path_attention == "cos":
                path_attention = torch.cosine_similarity(x.unsqueeze(1), path_inputs, dim=2)
                path_attention = F.softmax(path_attention, dim=-1).unsqueeze(2)
            path_inputs = torch.sum(path_attention * path_inputs, dim=1)
        elif self.args.path_attention == "mean":
            path_inputs = torch.mean(path_inputs, dim=1)
        elif self.args.path_attention == "sum":
            path_inputs = torch.sum(path_inputs, dim=1)

        # channel 3
        edge_weight = torch.ones(size=(data.edge_index.shape[1],)).cuda()
        A = torch.sparse_coo_tensor(data.edge_index, edge_weight)
        A_x = self.mlpA(A)

        # aggregation
        path_x = path_inputs + x
        output = (1 - self.beta) * path_x + self.beta * A_x
        output = F.dropout(self.act(output), p=self.args.fdropout, training=self.training)
        output = self.mlp_end(output)
        if self.args.dataset in ['tolokers', 'questions']:
            output = F.sigmoid(output).squeeze()
        return output