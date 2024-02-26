import torch
from models import GCNConvVanilla, MMD_GCN, train_one_epoch, validation_stage, test_stage
from loss import get_graph_metric
from utils import loadDS, get_graph_idx, eva_clustering, eva_svc
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def MMDGK(args):

    param_dict = vars(args)
    dataloader, _, _ = loadDS(args.dataname, batch_size=-1, num_dataloader=1, random_seed=2023) # batch_size = -1 : all data in one batch
    train_loader = dataloader[0]
    model = GCNConvVanilla()
    timestamp = datetime.now().strftime("Y%m%d_%H%M%S")
    writer = SummaryWriter('runs_vanilla/{}_trainer_{}'.format(args.dataname, timestamp))

    mmd_kernel_list = []
    for data in train_loader:

        data = data.to(device)
        label = data.y.cpu().detach().numpy()
        num_graph = len(data)

        if args.only_node_attr:
            input_x = data.node_attr    # option 1 [only node attr]
        else:
            input_x = data.x            # option 2 [node attr; node label]

        pbar = tqdm(range(1, args.gcn_num_layer + 1))
        for layer in pbar:
            # 1. Pass through encoder and get node emb
            input_x = model(input_x, data.edge_index)
            # 2. Get the graph index
            graph_idx = get_graph_idx(data)
            # 3. Build MMD graph kernel & Calculate loss
            mmd_kernel = torch.ones(num_graph, num_graph).cuda()
            for i in range(num_graph):
                graph_i = input_x[graph_idx[i]:graph_idx[i+1]]
                for j in range(i+1, num_graph):
                    graph_j = input_x[graph_idx[j]:graph_idx[j+1]]
                    mmd_simi = get_graph_metric(graph_i, graph_j, args.bandwidth, args.dis_gamma)
                    mmd_kernel[i, j] = mmd_kernel[j, i] = mmd_simi

            # 4. Evaluate the graph kernel by graph classification accuracy & spectral clustering
            mmd_kernel = mmd_kernel.cpu().detach().numpy()
            mmd_kernel_list.append(mmd_kernel)
            # 4.1 Spectral Clustering
            results = eva_clustering(mmd_kernel, label)
            # 4.2: SVC
            results_svc = eva_svc(mmd_kernel, label)
            results.update(results_svc)
            
            writer.add_scalars('Scores', results, layer)
            pbar.set_description(f'Layer {layer:>3} | {results}')
            writer.flush()

    writer.add_hparams(param_dict, results)
    writer.close() 

    if len(mmd_kernel_list) == 1: # batch_size==-1
        return mmd_kernel_list[0]
    else:
        return mmd_kernel_list
    


def deep_MMDGK(args):

    param_dict = vars(args)
    dataloader, num_features, num_node_attributes = loadDS(args.dataname, batch_size=-1, num_dataloader=1, random_seed=2023) # batch_size = -1 : all data in one batch
    
    if len(dataloader) == 3: 
        train_loader = dataloader[0]
        validation_loader = dataloader[1]
        test_data = dataloader[2][0]
    elif len(dataloader) == 2:
        train_loader = dataloader[0]
        validation_loader = None 
        test_data = dataloader[1][0]
    else:
        train_loader = dataloader[0]
        validation_loader = None
        test_data = None
    
    # select dimensions of layers
    gcn_input_dim = num_node_attributes if args.only_node_attr else num_features
    # initial GCN model and optimizer
    model = MMD_GCN(gcn_input_dim, args.gcn_num_layer, args.dis_gamma, args.bandwidth,\
                    args.alpha, args.normalization, args.encoder_equal_dim, \
                    args.objective_fuc, args.only_node_attr).to(device)
    optimizer = torch.optim.Adam(model.parameters(), \
                                lr=args.step_size,        \
                                weight_decay=args.weight_decay)
    timestamp = datetime.now().strftime("Y%m%d_%H%M%S")
    writer = SummaryWriter('runs/{}_trainer_{}'.format(args.dataname, timestamp))

    print('============== Training ==============')
    best_vloss = 1_000_000.
    pbar = tqdm(range(1, args.epochs + 1))
    for epoch in pbar:
        model.train()
        tloss, mmd_kernel_list = train_one_epoch(model, optimizer, train_loader, args.max_norm)
        
        model.eval()
        if validation_loader is not None:
            vloss = validation_stage(model, validation_loader)
            pbar.set_description(f'Epoch {epoch:>3} | Train Loss: {tloss:.4f} | Validation Loss: {vloss:.4f}')
            writer.add_scalars('Training vs. Validation Loss',
                         {'Training' : tloss, 'Validation' : vloss},
                         epoch)
        else:
            vloss = 1_000_000.
            pbar.set_description(f'Epoch {epoch:>3} | Train Loss: {tloss:.4f}')
            writer.add_scalars('Training Loss', {'Training' : tloss}, epoch)

        test_results = test_stage(model, train_loader, test_data, mmd_kernel_list)
        writer.add_scalars('Scores', test_results, epoch)
        writer.flush()

        # Track best performance, and save the model's state
        if (vloss < best_vloss) | (epoch==args.epochs):
            best_vloss = vloss
            model_path = './history/model_{}_{}_{}'.format(args.dataname, timestamp, epoch)
            torch.save(model.state_dict(), model_path)

    writer.add_hparams(param_dict, test_results)
    writer.close()

    if len(mmd_kernel_list) == 1: # batch_size==-1
        return mmd_kernel_list[0]
    else:
        return mmd_kernel_list