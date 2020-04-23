import argparse

import os
import sys
import torch
import torch.distributed as dist
import torch.optim as optim

from math import ceil
from torch.multiprocessing import Process

sys.path.append("../")
sys.path.append("../../")

from ec2.trainer import Trainer
from ec2.data_partition import partition_yfcc100m
from pytorch_model.linear import LogisticRegression


validation_ratio = .1


def dist_is_initialized():
    if dist.is_available():
        if dist.is_initialized():
            return True
    return False


def run(args):
    """ Distributed Synchronous SGD Example """
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    torch.manual_seed(1234)

    f_id_start = args.rank * args.num_files
    f_id_end = f_id_start + args.num_files
    f_path_list = ["{}/{}".format(args.root, i) for i in range(f_id_start, f_id_end)]
    print("read file {}".format(f_path_list))
    train_loader, test_loader = partition_yfcc100m(f_path_list, args.features, args.pos_tag,
                                                   args.batch_size, validation_ratio)
    num_batches = ceil(len(train_loader.dataset) / float(args.batch_size))

    model = LogisticRegression(args.features, args.classes)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

    trainer = Trainer(model, optimizer, train_loader, test_loader, device)

    trainer.fit(args.epochs, is_dist=dist_is_initialized())


def init_processes(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


def run_local():
    size = 2
    processes = []
    for rank in range(size):
        p = Process(target=init_processes, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', type=str, default='gloo', help='Name of the backend to use.')
    parser.add_argument(
        '-i',
        '--init-method',
        type=str,
        default='tcp://127.0.0.1:23456',
        help='URL specifying how to initialize the package.')
    parser.add_argument('-s', '--world-size', type=int, default=1, help='Number of processes participating in the job.')
    parser.add_argument('-r', '--rank', type=int, default=0, help='Rank of the current process.')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--features', type=int, default=4096)
    parser.add_argument('--classes', type=int, default=2)
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--root', type=str, default='data')
    parser.add_argument('--num-files', type=int, default=1)
    parser.add_argument('--pos-tag', type=str, default="animal")
    parser.add_argument('--no-cuda', action='store_true')
    args = parser.parse_args()
    print(args)

    if args.world_size > 1:
        dist.init_process_group(backend=args.backend, init_method=args.init_method, world_size=args.world_size, rank=args.rank)

    run(args)


if __name__ == '__main__':
    main()
