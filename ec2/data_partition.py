from torchvision import datasets, transforms
import torch.distributed as dist
from torch.utils.data import DataLoader

from random import Random


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