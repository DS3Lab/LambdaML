import time

import numpy as np

import torch
from torchvision import datasets, transforms
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from data_loader.LibsvmDataset import DenseLibsvmDataset2
from data_loader.YFCCLibsvmDataset import DenseLibsvmDataset
from data_loader.LibsvmDataset import SparseLibsvmDataset

from random import Random


def dist_is_initialized():
    if dist.is_available():
        if dist.is_initialized():
            return True
    return False


class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


def partition_mnist(batch_size, path, download=True):
    """ Partitioning MNIST """
    train_dataset = datasets.MNIST(path, train=True, download=download,
                                   transform=transforms.Compose([transforms.ToTensor(),
                                                                 transforms.Normalize((0.1307, ), (0.3081, ))]))
    test_dataset = datasets.MNIST(path, train=False, download=download,
                                   transform=transforms.Compose([transforms.ToTensor(),
                                                                 transforms.Normalize((0.1307,), (0.3081,))]))
    size = dist.get_world_size()
    bsz = int(batch_size / float(size))
    train_partition_sizes = [1.0 / size for _ in range(size)]
    train_partition = DataPartitioner(train_dataset, train_partition_sizes)
    train_partition = train_partition.use(dist.get_rank())
    train_loader = DataLoader(
        train_partition, batch_size=bsz, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, bsz, test_loader


def partition_cifar10(batch_size, path, download=True):
    """ Partitioning MNIST """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = datasets.CIFAR10(path, train=True, download=download, transform=transform_train)
    test_dataset = datasets.CIFAR10(path, train=False, download=download, transform=transform_test)
    size = 1
    rank = 0
    if dist_is_initialized():
        size = dist.get_world_size()
        rank = dist.get_rank()
    bsz = int(batch_size / float(size))
    train_partition_sizes = [1.0 / size for _ in range(size)]
    train_partition = DataPartitioner(train_dataset, train_partition_sizes)
    train_partition = train_partition.use(rank)
    train_loader = DataLoader(
        train_partition, batch_size=bsz, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, bsz, test_loader


def partition_higgs(batch_size, file_name, validation_ratio):
    parse_start = time.time()
    f = open(file_name).readlines()
    dataset = DenseLibsvmDataset2(f, 30)
    print("parse data cost {} s".format(time.time() - parse_start))

    preprocess_start = time.time()
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_ratio * dataset_size))
    random_seed = 42
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size,
                                               sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              sampler=valid_sampler)

    print("preprocess data cost {} s".format(time.time() - preprocess_start))
    return train_loader, batch_size, test_loader


def partition_yfcc100m(file_list, n_features, pos_tag, batch_size, validation_ratio):
    parse_start = time.time()
    f = open(file_list[0]).readlines()
    dataset = DenseLibsvmDataset(f, n_features, pos_tag)
    if len(file_list) > 1:
        for file_name in file_list[1:]:
            f = open(file_name).readlines()
            dataset.add_more(f)

    total_count = dataset.__len__()
    pos_count = 0
    for i in range(total_count):
        if dataset.__getitem__(i)[1] == 1:
            pos_count += 1
    print("{} positive observations out of {}".format(pos_count, total_count))

    print("parse data cost {} s".format(time.time() - parse_start))

    preprocess_start = time.time()
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_ratio * dataset_size))
    random_seed = 42
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size,
                                               sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              sampler=valid_sampler)

    print("preprocess data cost {} s".format(time.time() - preprocess_start))
    return train_loader, test_loader


def partition_agaricus(batch_size, train_file, test_file):
    train_dataset = SparseLibsvmDataset(train_file, 127)
    test_dataset = SparseLibsvmDataset(test_file, 127)

    size = dist.get_world_size()
    bsz = 1 if batch_size == 1 else int(batch_size / float(size))
    train_partition_sizes = [1.0 / size for _ in range(size)]
    train_partition = DataPartitioner(train_dataset, train_partition_sizes)
    train_partition = train_partition.use(dist.get_rank())
    train_loader = DataLoader(
        train_partition, batch_size=bsz, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True)
    return train_partition, train_loader, bsz, test_loader


def partition_sparse(file, num_feature):
    train_dataset = SparseLibsvmDataset(file, num_feature)
    size = 1
    rank = 0
    if dist_is_initialized():
        size = dist.get_world_size()
        rank = dist.get_rank()
    train_partition_sizes = [1.0 / size for _ in range(size)]
    train_partition = DataPartitioner(train_dataset, train_partition_sizes)
    train_partition = train_partition.use(rank)
    return train_partition

