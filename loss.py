import torch
import torch.nn as nn
from utils import get_graph_idx

EPS = 1e-6
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_one_graph_emb(input_x, start, end):

    if type(input_x) == list:
        return torch.cat([x[start:end] for x in input_x], dim=1)
    else:
        return input_x[start:end]
    

def sc_loss(s, pos_mask, alpha):
    '''
    - s: 	(n, n) pairwise similarity scores based on MMD 
    - pos_mask: (n, n) pairwise indicator of label equality (if y_i == y_j, 1)
    - alpha: trade-off parameter
    '''
    s_denominator = s*pos_mask + alpha * s*(1-pos_mask)
    log_denominator = (pos_mask.sum(1) * torch.log(torch.sum(s_denominator, dim=-1))).sum()
    log_denominator = log_denominator/pos_mask.sum()
    log_nominator = s[pos_mask.bool()].mean()
    return - log_nominator + log_denominator
    

def kl_loss(s):
    '''
    - s: 	(n, n) pairwise similarity scores based on MMD 
    '''
    loss = nn.KLDivLoss(reduction="batchmean")
    weight = s**2 / s.sum(0)
    return loss(s, (weight.t()/weight.sum(1)).t() )


def get_MDD(source, target, bandwidth:list):
    '''
    - source: one distribution
    - target: another distribution
    - bandwidth: a family (list) of gaussian kernal 
    '''
    total = torch.cat([source, target], dim=0)
    distance = torch.cdist(total, total, p=2) ** 2

    mmd_sup = torch.zeros(1).to(device)
    batch_size = int(source.size()[0])
    batch_size_2 = int(target.size()[0])

    for bw in bandwidth:
        kernels = torch.exp(-distance / bw)
        M = - torch.ones_like(kernels).to(device) / (batch_size*batch_size_2)
        M[:batch_size, :batch_size] = torch.ones(batch_size, batch_size).to(device) / (batch_size*batch_size)
        M[batch_size:, batch_size:] = torch.ones(batch_size_2, batch_size_2).to(device) / (batch_size_2*batch_size_2)
        mmd = torch.trace(kernels @ M)
        mmd_sup = torch.maximum(torch.clamp(mmd, min=0), mmd_sup)

    return torch.sqrt(mmd_sup + EPS)


def get_graph_metric(graph_i, graph_j, bandwidth, dis_gamma):
    mmd = get_MDD(graph_i, graph_j, bandwidth=bandwidth)   
    mmd_simi = torch.exp(- dis_gamma * mmd) # Transform distance to similarity
    return mmd_simi

def loss_fn(model, input_emb, data):

    num_graph = len(data)
    # 1. Get the graph index
    graph_idx = get_graph_idx(data)

    # 2. Build MMD graph kernel & Calculate loss
    mmd_kernel = torch.ones(num_graph, num_graph).to(device)
    mask_kernel = torch.ones((num_graph, num_graph)).to(device)
        
    count = 0
    if model.objective_fuc == 'UCL': k_neg = [(1,0,0),]*(num_graph//3)

    for i in range(num_graph):
        for j in range(i+1, num_graph):
            
            graph_i = get_one_graph_emb(input_emb, graph_idx[i], graph_idx[i+1])
            graph_j = get_one_graph_emb(input_emb, graph_idx[j], graph_idx[j+1])

            if (model.objective_fuc == "SCL") and (data.y[i] != data.y[j]):
                mask_kernel[i,j] = 0
                mask_kernel[j,i] = 0

            mmd_simi = get_graph_metric(graph_i, graph_j, model.bandwidth, model.dis_gamma)
            mmd_kernel[i, j] = mmd_kernel[j, i] = mmd_simi
            
            if (model.objective_fuc == 'UCL') and (k_neg[-1][0] > mmd_simi):
                k_neg[-1] = (mmd_simi, i, j)
                k_neg = sorted(k_neg, key=lambda x: x[0])

            count +=1

    if model.objective_fuc == 'KL':
        train_loss = kl_loss(mmd_kernel)
    elif model.objective_fuc == 'SCL':
        train_loss = sc_loss(mmd_kernel, mask_kernel, model.alpha)
    elif model.objective_fuc == 'UCL':
        for (s, i, j) in k_neg:
            mask_kernel[i, j] = 0
            mask_kernel[j, i] = 0
        train_loss = sc_loss(mmd_kernel, mask_kernel, model.alpha)
    else:
        raise ValueError("Objective is not defined. (KL, SCL or UCL)")
        
    return train_loss, mmd_kernel