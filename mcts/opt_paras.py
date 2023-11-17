import os
import time
import torch
import torch.multiprocessing as mp

from args import parser
from agent import Agent
from utils import load_data
from model.policy import TSPNetwork
from model.PolicyNet import PolicyNet
from hyperopt import hp, fmin, rand, tpe, space_eval
from analysis import statistic

os.environ["OMP_NUM_THREADS"] = "1"


def objective(virtual_loss):
    args = parser.parse_args()
    args.virtual_loss = virtual_loss
    torch.manual_seed(args.seed)
    if args.gpu_ids == -1:
        args.gpu_ids = [-1]
    else:
        torch.cuda.manual_seed(args.seed)
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass

    test_graphs = load_data("data/{}/tsp{}.txt".format(args.graph_type, args.graph_size), args.graph_size)
    # network = TSPNetwork(node_dim=args.node_dim, edge_dim=args.edge_dim, embed_dim=args.embed_dim,
    #                      hidden_dim=args.hidden_dim, graph_size=args.graph_size, layer=args.n_layer)
    network = PolicyNet(node_dim=args.node_dim, edge_dim=args.edge_dim, embed_dim=args.embed_dim,
                        hidden_dim=args.hidden_dim, graph_size=args.graph_size, layer=args.n_layer)

    if args.load:
        saved_state = torch.load(
            "{}/{}/tsp{}.pth".format(args.load_model_dir, args.graph_type, args.graph_size),
            map_location=lambda storage, loc: storage)['state_dict']
        network.load_paras(saved_state)
        print("Load model successfully ~~")

    processes = []
    tasks_num = len(test_graphs) // args.n_worker
    extra_num = len(test_graphs) % args.n_worker
    print(args.n_worker)
    for idx in range(args.n_worker):
        if idx == args.n_worker - 1:
            graphs = test_graphs[idx * tasks_num: (idx + 1) * tasks_num + extra_num]
        else:
            graphs = test_graphs[idx * tasks_num: (idx + 1) * tasks_num]

        agent = Agent(idx, args, graphs, network)
        p = mp.Process(target=agent.run)
        p.start()
        processes.append(p)
        time.sleep(0.1)

    for p in processes:
        time.sleep(0.1)
        p.join()

    mean_path_len = statistic()
    with open('paras.txt', 'a') as f:
        f.write("{} {}\n".format(virtual_loss, mean_path_len))
    return mean_path_len


if __name__ == '__main__':
    mp.set_start_method('spawn')
    for i in [3, 4, 5, 9, 11, 19]:
        print(objective(i))
