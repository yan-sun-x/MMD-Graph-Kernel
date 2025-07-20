import torch
from torch.nn import Linear, Parameter, ModuleList, BatchNorm1d, LayerNorm
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from mmdgk.loss import loss_fn, get_graph_metric, get_one_graph_emb
from mmdgk.utils import get_graph_idx, eva_clustering, eva_svc, train_test_svc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GCNConvVanilla(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 3-4: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)

        return out
    

class NodeNorm(torch.nn.Module):
    def __init__(self, nn_type="n", unbiased=False, eps=1e-5, power_root=2):
        super(NodeNorm, self).__init__()
        self.unbiased = unbiased
        self.eps = eps
        self.nn_type = nn_type
        self.power = 1 / power_root

    def forward(self, x):
        if self.nn_type == "n":
            mean = torch.mean(x, dim=1, keepdim=True)
            std = (
                torch.var(x, unbiased=self.unbiased, dim=1, keepdim=True) + self.eps
            ).sqrt()
            x = (x - mean) / std
        elif self.nn_type == "v":
            std = (
                torch.var(x, unbiased=self.unbiased, dim=1, keepdim=True) + self.eps
            ).sqrt()
            x = x / std
        elif self.nn_type == "m":
            mean = torch.mean(x, dim=1, keepdim=True)
            x = x - mean
        elif self.nn_type == "srv":  # squre root of variance
            std = (
                torch.var(x, unbiased=self.unbiased, dim=1, keepdim=True) + self.eps
            ).sqrt()
            x = x / torch.sqrt(std)
        elif self.nn_type == "pr":
            std = (
                torch.var(x, unbiased=self.unbiased, dim=1, keepdim=True) + self.eps
            ).sqrt()
            x = x / torch.pow(std, self.power)
        return x

    def __repr__(self):
        original_str = super().__repr__()
        components = list(original_str)
        nn_type_str = f"nn_type={self.nn_type}"
        components.insert(-1, nn_type_str)
        new_str = "".join(components)
        return new_str


def get_normalization(norm_type, num_channels=None):
    if norm_type is None:
        norm = None
    elif norm_type == "batch":
        norm = BatchNorm1d(num_features=num_channels)
    elif norm_type == "node_n":
        norm = NodeNorm(nn_type="n")
    elif norm_type == "node_v":
        norm = NodeNorm(nn_type="v")
    elif norm_type == "node_m":
        norm = NodeNorm(nn_type="m")
    elif norm_type == "node_srv":
        norm = NodeNorm(nn_type="srv")
    elif norm_type.find("node_pr") != -1:
        power_root = norm_type.split("_")[-1]
        power_root = int(power_root)
        norm = NodeNorm(nn_type="pr", power_root=power_root)
    elif norm_type == "layer":
        norm = LayerNorm(normalized_shape=num_channels)
    else:
        raise NotImplementedError
    return norm


class GCNConv(MessagePassing):
    def __init__(self, in_channels=None, out_channels=None):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix. 
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)

        # Step 6: Apply a final bias vector.
        out += self.bias

        return out


class MMD_GCN(torch.nn.Module):
    def __init__(self, gcn_input_dim, gcn_num_layers, dis_gamma, bandwidth, \
                alpha=1, normalization='node_m', encoder_equal_dim=True, \
                objective_fuc='KL', only_node_attr=False):
        super().__init__()
        self.gcn_num_layers = gcn_num_layers
        self.dis_gamma = dis_gamma
        self.bandwidth = bandwidth
        self.alpha = alpha
        self.objective_fuc = objective_fuc
        self.only_node_attr = only_node_attr

        if type(encoder_equal_dim) == list:
            dim_list = encoder_equal_dim
        else:
            dim_list = [gcn_input_dim for _ in range(gcn_num_layers+1)]

        self.gcn_list = ModuleList([GCNConv(dim_list[i], dim_list[i+1]).to(device) for i in range(gcn_num_layers)])
        if normalization is not None:
            self.normalization = True
            self.bns = ModuleList([torch.nn.BatchNorm1d(dim) for dim in dim_list[1:]])
            self.node_norm = get_normalization(norm_type=normalization)
        else:
            self.normalization = False

    def forward(self, x, edge_index):
        for layer in range(self.gcn_num_layers):
            x = self.gcn_list[layer](x, edge_index)
            
            if self.normalization:
                x = self.node_norm(x)
                x = self.bns[layer](x)
        return x


def get_pairwise_simi(model, g1, edge_index1, g2, edge_index2):

    g1 = model(g1, edge_index1)
    g2 = model(g2, edge_index2)
    mmd_simi, mmd_dis = get_graph_metric(g1, g2, model.bandwidth, model.dis_gamma)

    return mmd_simi, mmd_dis


def train_one_epoch(model, optimizer, dataloader, max_norm):

    running_loss = 0.
    mmd_kernel_list = []
    for data in dataloader:

        data = data.to(device)
        if model.only_node_attr:
            input_x = data.node_attr    # option 1 [only node attr]
        else:
            input_x = data.x            # option 2 [node attr; node label]

        # 1. Pass through encoder and get node emb
        input_emb = model(input_x, data.edge_index)
        # 2. Calculate the loss, kernels and its gradients
        train_loss, mmd_kernel = loss_fn(model, input_emb, data)
        train_loss.backward()
        mmd_kernel_list.append(mmd_kernel.cpu().detach().numpy())
        # 3. Adjust learning weights
        if max_norm is not None: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm) # 3.1 Perform gradient clipping
        optimizer.step()
        
        # 4. Gather data
        running_loss += train_loss.item()

    return running_loss, mmd_kernel_list


def validation_stage(model, dataloader):

    running_vloss = 0.0
    for data in dataloader:
        data = data.to(device)
        if model.only_node_attr:
            input_x = data.node_attr
        else:
            input_x = data.x
        input_emb = model(input_x, data.edge_index)
        vali_loss, _, _ = loss_fn(model, input_emb, data)
        running_vloss += vali_loss.item()

    return running_vloss


def get_train_test_matrix(model, train_data, test_data):

    train_data = train_data.to(device)
    train_num_graph = len(train_data)
    train_graph_idx = get_graph_idx(train_data)
    train_x = train_data.node_attr if model.only_node_attr else train_data.x
    train_emb = model(train_x, train_data.edge_index)

    test_data = test_data.to(device)
    test_num_graph = len(test_data)
    test_graph_idx = get_graph_idx(test_data)
    test_x = test_data.node_attr if model.only_node_attr else test_data.x
    test_emb = model(test_x, test_data.edge_index)
    
    test_mmd_kernel = torch.ones(test_num_graph, train_num_graph).to(device) # similarty
    for i in range(test_num_graph):
        for j in range(train_num_graph):
            graph_i = get_one_graph_emb(test_emb, test_graph_idx[i], test_graph_idx[i+1])
            graph_j = get_one_graph_emb(train_emb, train_graph_idx[j], train_graph_idx[j+1])
            mmd_simi = get_graph_metric(graph_i, graph_j, model.bandwidth, model.dis_gamma)
            test_mmd_kernel[i, j] = mmd_simi

    return test_mmd_kernel.cpu().data.numpy()


@torch.no_grad()
def test_stage(model, train_loader, test_data, mmd_kernel_list):

    model.eval()
    for i, train_data in enumerate(train_loader):
        mmd_kernel = mmd_kernel_list[i]

        if test_data is None:
            # Evaluation 1: Spectral Clustering
            results_clu = eva_clustering(mmd_kernel, train_data.y.cpu().data.numpy())
            # Evaluation 2: SVC
            results_svc = eva_svc(mmd_kernel, train_data.y.cpu().data.numpy())

        else:
            # Evaluation 1: Spectral Clustering
            results_clu = eva_clustering(mmd_kernel, train_data.y.cpu().data.numpy())
            
            test_mmd_kernel = get_train_test_matrix(model, train_data, test_data)
            # Evaluation 2: SVC
            results_svc = train_test_svc(mmd_kernel, train_data.y.cpu().data.numpy(), \
                                        test_mmd_kernel, test_data.y.cpu().data.numpy())
            
    results_clu.update(results_svc)
    return results_clu