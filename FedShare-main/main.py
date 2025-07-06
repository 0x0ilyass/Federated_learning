import torch 
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

import copy
import numpy as np
import random
import time
from tqdm import trange

from utils.distribute import uniform_distribute, train_dg_split
from utils.sampling import iid, noniid
from utils.options import args_parser
from src.update import ModelUpdate
from src.nets import MLP, CNN_v1, CNN_v2
from src.strategy import FedAvg, FedAdam, FedProx, SGD_local
from src.test import test_img

writer = SummaryWriter()

if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Dataset loading (same as before)
    if args.dataset == 'mnist':
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans)
    elif args.dataset == 'cifar':
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans)
    else:
        exit('Error: unrecognized dataset')

    dg = copy.deepcopy(dataset)
    dataset_train = copy.deepcopy(dataset)
    dg_idx, dataset_train_idx = train_dg_split(dataset, args)

    if args.dataset == 'mnist':
        dg.data, dataset_train.data = dataset.data[dg_idx], dataset.data[dataset_train_idx]
        dg.targets, dataset_train.targets = dataset.targets[dg_idx], dataset.targets[dataset_train_idx]
    else:
        dg.data, dataset_train.data = dataset.data[dg_idx], dataset.data[dataset_train_idx]
        dg.targets = [dataset[i][1] for i in dg_idx]
        dataset_train.targets = [dataset[i][1] for i in dataset_train_idx]

    dict_users = iid(dataset_train, args.num_users) if args.sampling == 'iid' else noniid(dataset_train, args)

    img_size = dataset_train[0][0].shape
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNN_v2(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNN_v1(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = np.prod(img_size)
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')

    print(net_glob)
    net_glob.train()
    w_glob = net_glob.state_dict()

    # Init FedShare
    initialization_stage = ModelUpdate(args=args, dataset=dataset, idxs=set(dg_idx))
    w_glob, _ = initialization_stage.train(local_net=copy.deepcopy(net_glob).to(args.device),
                                            net=copy.deepcopy(net_glob).to(args.device))
    net_glob.load_state_dict(w_glob)

    prev_moments = {'w_glob_prev': copy.deepcopy(w_glob)}

    share_idx = uniform_distribute(dg, args)

    # Init metrics
    total_comm = 0
    round_times = []

    for iter in trange(args.rounds):
        start_time = time.time()
        w_locals = [] if not args.all_clients else [w_glob for _ in range(args.num_users)]

        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            # Simulate robustness issue: skip client with small prob or add noise
            if np.random.rand() < 0.05:
                print(f"Round {iter}: Client {idx} failed (simulated).")
                continue  

            local = ModelUpdate(args=args, dataset=dataset, idxs=set(list(dict_users[idx]) + share_idx))
            w, loss = local.train(local_net=copy.deepcopy(net_glob).to(args.device),
                                  net=copy.deepcopy(net_glob).to(args.device))

            # Optional: inject noise (simulate Byzantine client)
            if np.random.rand() < 0.02:
                for k in w.keys():
                    w[k] += torch.randn_like(w[k]) * 0.1
                print(f"Round {iter}: Client {idx} sent noisy weights (simulated).")

            w_locals.append(copy.deepcopy(w))

        # Communication overhead
        comm_size = sum(p.numel() * 4 for p in net_glob.parameters())  # 4 bytes per float32
        total_comm += comm_size * len(w_locals)

        # Aggregation
        if args.fed == 'avg':
            w_glob = FedAvg(w_locals, args)
        elif args.fed == 'adam':
            w_glob, prev_moments = FedAdam(w_locals, args, prev_moments)
        elif args.fed == 'prox':
            w_glob = FedProx(w_locals, prev_moments['w_glob_prev'], args)
        elif args.fed == 'sgd_local':
            w_glob = SGD_local(w_locals, args)
        else:
            exit('Error: unrecognized federated strategy')

        net_glob.load_state_dict(w_glob)

        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        round_time = time.time() - start_time
        round_times.append(round_time)

        if args.debug:
            print(f"Round {iter}: accuracy={acc_test:.4f}, loss={loss_test:.4f}, time={round_time:.2f}s, comm={comm_size * len(w_locals) / 1e6:.2f} MB, total comm={total_comm / 1e6:.2f} MB")

        if args.tsboard:
            writer.add_scalar(f"Test accuracy:Share{args.dataset}, {args.fed}", acc_test, iter)
            writer.add_scalar(f"Test loss:Share{args.dataset}, {args.fed}", loss_test, iter)
            writer.add_scalar(f"Comm bytes:Share{args.dataset}, {args.fed}", total_comm, iter)
            writer.add_scalar(f"Round time:Share{args.dataset}, {args.fed}", round_time, iter)

    writer.close()
    print("Training complete.")
    print(f"Final test accuracy: {acc_test:.4f}")
    print(f"Final test loss: {loss_test:.4f}")
    print(f"Total communication: {total_comm / 1e6:.2f} MB")
    print(f"Average round time: {np.mean(round_times):.2f} sec")
