import math
import numpy as np
import torch
import torch.distributed as dist

from data_loader.LibsvmDataset import SparseLibsvmDataset


def dist_is_initialized():
    if dist.is_available():
        if dist.is_initialized():
            return True
    return False


class Loss(object):

    def __init__(self):
        self.sum = 0
        self.count = 0

    def __str__(self):
        return '{:.6f}'.format(self.average)

    @property
    def average(self):
        return self.sum / self.count

    def update(self, value, number):
        self.sum += value.item() * number
        self.count += number


class Accuracy(object):

    def __init__(self):
        self.correct = 0
        self.count = 0

    def __str__(self):
        return '{:.2f}%'.format(self.accuracy * 100)

    @property
    def accuracy(self):
        return self.correct / self.count

    def update(self, output, target):
        pred = 1 if output.item() >= 0 else -1

        self.correct += 1 if pred == target else 0
        self.count += 1


class SparseSVM(object):
    """ Based on http://eprints.pascal-network.org/archive/00004062/01/ShalevSiSr07.pdf
    """

    def __init__(self, _train_set, _test_set, _n_input, _epochs, _lr, _batch_size, _regularization_strength):
        self.train_set = _train_set
        self.test_set = _test_set
        self.num_train = self.train_set.__len__()
        self.num_test = self.test_set.__len__()
        self.n_input = _n_input
        self.weights = torch.zeros(_n_input, 1)
        self.num_epochs = _epochs
        self.lr = _lr
        self.batch_size = _batch_size
        self.cur_index = 0
        self.regularization_strength = _regularization_strength

    def next_batch(self, batch_idx):
        start = batch_idx * self.batch_size
        end = min((batch_idx + 1) * self.batch_size, self.num_train)
        ins = [ts[0] for ts in self.train_set[start:end]]
        label = [ts[1] for ts in self.train_set[start:end]]
        return ins, label

    def get_batch(self):
        ins_list = []
        label_list = []
        num = 0
        while num < self.batch_size:
            ins, label = self.train_set.__getitem__(self.cur_index)
            ins_list.append(ins)
            label_list.append(label)
            num += 1
            self.cur_index += 1
            if self.cur_index >= self.num_train:
                self.cur_index = 0
        return ins_list, label_list

    def one_epoch(self, iteration, is_dist=dist_is_initialized()):
        batch_ins, batch_label = self.next_batch(iteration)

        train_loss = Loss()
        train_acc = Accuracy()
        eta = 1.0 / (self.lr * (1+iteration))

        for i in range(len(batch_ins)):
            ascent = batch_label[i] * float(torch.sparse.mm(batch_ins[i], self.weights))
            if (ascent < 1.0 and batch_label[i] != 0.0):
                self.weights = (1 - self.lr*eta)* self.weights + eta * batch_label[i] * batch_ins[i].t()
            else:
                self.weights = (1 - self.lr*eta)* self.weights
            prediction = torch.sparse.mm(batch_ins[i], self.weights)
            loss = batch_label[i] / (1 + np.exp(prediction))
            # print(prediction)
            train_loss.update(loss, 1)
            train_acc.update(prediction, batch_label[i])
        # print(f"train loss:{train_loss}, train_acc:{train_acc}")
        return train_loss, train_acc

    def train(self):
        num_batches = math.floor(self.num_train / self.batch_size)
        for epoch in range(self.num_epochs):
            epoch_loss = 0
            for batch_idx in range(num_batches):
                epoch_loss, epoch_acc = self.one_epoch(epoch)
            test_loss, test_acc = self.evaluate()
            print("epoch[{}]: train loss = {}, train accuracy = {}, "
                  "test loss = {}, test accuracy = {}".format(epoch, epoch_loss, epoch_acc, test_loss, test_acc))
            self.cur_index = 0
            print(self.weights.numpy().flatten())

    def evaluate(self):
        test_loss = Loss()
        test_acc = Accuracy()
        for i in range(self.num_test):
            ins, label = self.test_set.__getitem__(i)
            y = self.predict(ins)
            loss = self.loss(y, label)
            test_loss.update(loss, 1)
            test_acc.update(y, label)
        return test_loss, test_acc

if __name__ == "__main__":
    train_file = "../dataset/agaricus_127d_train.libsvm"
    test_file = "../dataset/agaricus_127d_test.libsvm"
    train_data = SparseLibsvmDataset(train_file, 127)
    test_data = SparseLibsvmDataset(test_file, 127)
    svm = SparseSVM(train_data, test_data, 127, 20, 0.01, 10, 0.2)
    svm.fit()
