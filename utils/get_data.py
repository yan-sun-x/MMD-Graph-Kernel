import torch
import numpy as np
import os.path as osp
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset

global_path = "./data"


def loadDS(DS:str, batch_size:int, num_dataloader = 3, random_seed=2023, ave_num_nodes = 10):

    class MyCustomTransform():
        '''
        Add two properties (split from data.x):
        - data.node_attr: Node feature matrix with shape [num_nodes, num_node_attribute]
        - data.node_label: Node feature matrix with shape [num_nodes, num_node_label]
        '''
        def __call__(self, data):
            num_node = int(np.random.normal(ave_num_nodes, 2))
            data.x = data.x[torch.randint(len(data.x), (num_node,)),:]
            data.edge_index = torch.from_numpy(np.random.random_integers(0, num_node-1, size=(2, num_node**2//3)))
            data.node_attr = data.x[:, :dataset.num_node_attributes]
            data.node_label = data.x[:, dataset.num_node_attributes:]
            return data

    path = osp.join(global_path, DS)
    dataset = TUDataset(path, name=DS, use_node_attr=True).shuffle()
    dataset.transform = MyCustomTransform() # add two properties
        
    if num_dataloader == 1:

        if batch_size == -1: batch_size = len(dataset)
        return ((DataLoader(dataset, batch_size=batch_size),), \
                dataset.num_features, \
                dataset.num_node_attributes)
    
    elif num_dataloader == 2:
        
        # Creating data indices for training and validation splits:
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        np.random.seed(random_seed)
        np.random.shuffle(indices)

        split = int(np.floor(0.8 * dataset_size))
        train_indices, vali_indices = indices[:split], indices[split:]

        train_dataset = dataset[train_indices]
        vali_dataset = dataset[vali_indices]

        train_batch_size = len(train_dataset) if batch_size == -1 else batch_size
        vali_batch_size = len(vali_dataset) #if batch_size == -1 else batch_size
        
        return ((DataLoader(train_dataset, batch_size=train_batch_size), \
                DataLoader(vali_dataset, batch_size=vali_batch_size), \
                ), dataset.num_features, dataset.num_node_attributes)

    else:
        # Creating data indices for training and validation splits:
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        np.random.seed(random_seed)
        np.random.shuffle(indices)

        train_split = int(np.floor(0.8 * dataset_size))
        test_split = int(np.floor(0.9 * dataset_size))
        train_indices, val_indices, test_indices = indices[:train_split], indices[train_split:test_split], indices[test_split:]

        train_dataset = dataset[train_indices]
        vali_dataset = dataset[val_indices]
        test_dataset = dataset[test_indices]

        train_batch_size = len(train_dataset) if batch_size == -1 else batch_size
        vali_batch_size = len(vali_dataset) if batch_size == -1 else batch_size
        test_batch_size = len(test_dataset) #if batch_size == -1 else batch_size
        
        return ((DataLoader(train_dataset, batch_size=train_batch_size), \
                DataLoader(vali_dataset, batch_size=vali_batch_size), \
                DataLoader(test_dataset, batch_size=test_batch_size), \
                ), dataset.num_features, dataset.num_node_attributes)



def get_graph_idx(data):
    # Get the index for list of graph embedding
    graph_idx = [0]
    graph_idx += [len(data[i].node_attr) for i in range(len(data))]
    graph_idx = np.cumsum(np.array(graph_idx))
    return graph_idx


def get_edge_idx(data):
    # Get the index for list of edge embedding
    edge_idx = [0]
    edge_idx += [data[i].edge_index.shape[1] for i in range(len(data))]
    edge_idx = np.cumsum(np.array(edge_idx))
    return edge_idx