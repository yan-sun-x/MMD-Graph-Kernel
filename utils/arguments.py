import argparse
import json

def arg_parse():
    parser = argparse.ArgumentParser(description='Run MMDGK GCN model', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', type=str, default="vanilla", help="type of graph kernel (vanilla or deep)")
    # 1. Data
    parser.add_argument('--dataname', '-d', type=str, default="MUTAG", help='Name of the dataset')
    parser.add_argument('--only_node_attr', '-ona', type=bool, default=False, help='Type of the input')
    # 2. Model
    parser.add_argument('--gcn_num_layer', '-gnl', type=int, default=2, help='Number of layer in GCN')
    parser.add_argument('--objective_fuc', '-of', type=str, default="UCL", help='Objective function(KL; SCL; UCL)')
    parser.add_argument('--alpha', '-alpha', type=float, default=1e0, help='Hyper-parameter in objective function')
    parser.add_argument('--normalization', '-norm', type=str, default=None, help='Normalization in GCN')
    parser.add_argument('--encoder_equal_dim', '-eed', type=bool, default=True, help='Whether GCN has equal dim')
    # 3. MMD
    parser.add_argument('--dis_gamma', '-gamma', type=float, default=1e0, help='MMD->Simi: s = exp(-gamma * d)')
    parser.add_argument('--bandwidth', '-bd', type=json.loads, default=[1e0, 1e1], help='MMD->Simi: k = exp(-x/bw)')
    # 4. Optimization
    parser.add_argument('--step_size', '-lr', type=float, default=1e-1, help='learning rate')
    parser.add_argument('--weight_decay', '-wd', type=float, default=0., help='weight decay')
    parser.add_argument('--max_norm', '-mn', type=float, default=1e-4, help='max norm')
    parser.add_argument('--epochs', '-epochs', type=int, default=300, help='Number of epochs')

    args = parser.parse_args()
    return args