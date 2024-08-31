import torch
import time, math


def init(data, args):
    top_k_dict = {}
    node_num = data.x.shape[0]
    edge_index = data.edge_index
    for node in range(node_num):
        if len(edge_index[1][edge_index[0] == node]):
            nodes_indice_neighbors = edge_index[1][edge_index[0] == node]
        else:
            nodes_indice_neighbors = torch.tensor([node, node])
        nodes_neighbors = data.x[nodes_indice_neighbors]  # 获取邻居特征
        node_x = data.x[node].unsqueeze(dim=1)  # 获取自身节点特征
        attention = torch.mm(nodes_neighbors, node_x).squeeze(dim=-1)  # 将自身节点和邻居节点计算相似度
        if len(attention) >= args.depth + 1:
            values, indices = torch.topk(attention, args.depth + 1)  # topk选取k个最相似节点
        else:
            values, indices = torch.topk(attention, len(attention))
            indices = torch.cat((indices, torch.tensor([indices[-1]] * (args.depth  + 1 - len(attention)))))
        top_k_dict[str(int(node))] = nodes_indice_neighbors[indices]  # indice映射
    return top_k_dict


def sampler(origin_node, node, order, sampled_node_list, sampled_node, top_k_dict, args):
    # 1 终止条件
    if order == args.depth:
        sampled_node_list = sampled_node_list[::-1]
        sampled_node_list.append(origin_node)
        sampled_node.append(sampled_node_list)
        return
    sampled_indices = top_k_dict[str(int(node))]
    if order == 5:
        length = 2
    elif order > 5:
        length = 1
    else:
        length = order + 1
    for indice in range(length):
        # 2.1 数据处理
        sampled_node_list.append(sampled_indices[indice])
        # 2.2 递归
        sampler(origin_node, sampled_indices[indice], order+1, sampled_node_list, sampled_node, top_k_dict, args)
        # 2.3 回溯
        sampled_node_list.pop(-1)


def sample_paths(data, args):
    start_time = time.time()
    sampled, sampled_node, sampled_node_list = [], [], []
    node_num = data.x.shape[0]
    top_k_dict = init(data, args)
    for node in range(node_num):
        sampler(node, node, 1, sampled_node_list, sampled_node, top_k_dict, args)
        sampled.append(sampled_node)
        sampled_node = []
    sampled = torch.tensor(sampled)
    if args.depth >= 5:
        length = math.factorial(5)
    else:
        length = math.factorial(args.depth)
    random_indices = torch.randperm(length)[0:args.path_num]
    end_time = time.time()
    print(f"sample all paths used time : {end_time - start_time}s")
    return sampled[:, random_indices, :]