import time
import urllib.parse
import logging
import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler

from sync.sync_grad import *
from sync.sync_reduce_scatter import reduce_scatter_epoch, delete_expired_merged

from data_loader.LibsvmDataset import DenseDatasetWithLines

from utils.constants import Prefix
from storage.storage import S3Storage
from communicator.communicator import S3Communicator

from model import linear_models

# prefix
# w_prefix = "w_"
# b_prefix = "b_"
# tmp_w_prefix = "tmp_w_"
# tmp_b_prefix = "tmp_b_"


def handler(event, context):
    start_time = time.time()

    # dataset setting
    dataset = event['dataset']
    file = event['file']
    bucket = event['bucket_name']
    n_features = event['n_features']
    n_classes = event['n_classes']
    n_workers = event['n_workers']
    worker_index = event['rank']
    tmp_bucket = event['tmp_bucket']
    merged_bucket = event['merged_bucket']
    algo = event['algo']

    # hyper-parameter
    learning_rate = event['lr']
    batch_size = event['batch_size']
    n_epochs = event['n_epochs']
    valid_ratio = event['valid_ratio']

    shuffle_dataset = True
    random_seed = 100

    print('bucket = {}'.format(bucket))
    print("file = {}".format(file))
    print('number of workers = {}'.format(n_workers))
    print('worker index = {}'.format(worker_index))
    print('algorithm = {}'.format(algo))

    storage = S3Storage("s3")
    communicator = S3Communicator(storage, tmp_bucket, merged_bucket,
                                  n_workers, worker_index)

    # Read file from s3
    read_start = time.time()
    lines = storage.load(file, bucket).read().decode('utf-8').split("\n")
    print("read data cost {} s".format(time.time() - read_start))

    parse_start = time.time()
    dataset = DenseDatasetWithLines(lines, n_features)
    print("parse data cost {} s".format(time.time() - parse_start))

    preprocess_start = time.time()
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(valid_ratio * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size,
                                               sampler=train_sampler)
    n_train_batch = len(train_loader)
    validation_loader = torch.utils.data.DataLoader(dataset,
                                                    batch_size=batch_size,
                                                    sampler=valid_sampler)
    print("preprocess data cost {} s, dataset size = {}"
          .format(time.time() - preprocess_start, dataset_size))

    model = linear_models.get_model(algo, n_features, n_classes)

    # Loss and Optimizer
    # Softmax is internally computed.
    # Set parameters to be updated.
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    train_start = time.time()
    # Training the Model
    for epoch in range(n_epochs):
        epoch_start = time.time()
        epoch_loss = 0
        for batch_index, (items, labels) in enumerate(train_loader):
            # print("------worker {} epoch {} batch {}------".format(worker_index, epoch, batch_index))
            batch_start = time.time()
            items = Variable(items.view(-1, num_features))
            labels = Variable(labels)

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = model(items)
            loss = criterion(outputs, labels)
            epoch_loss += loss.data
            loss.backward()
            optimizer.step()

            # print('Epoch: [%d/%d], Step: [%d/%d], Time: %.4f s, Loss: %.4f, batch cost %.4f s'
            #        % (epoch + 1, n_epochs, batch_index + 1, n_train_batch,
            #           time.time() - train_start, loss.data, time.time() - batch_start))

        w = model.linear.weight.data.numpy()
        w_shape = w.shape
        b = model.linear.bias.data.numpy()
        b_shape = b.shape
        w_and_b = np.concatenate((w.flatten(), b.flatten()))
        epoch_cal_time = time.time() - epoch_start

        epoch_sync_start = time.time()
        postfix = str(epoch)
        w_and_b_merge = communicator.reduce_epoch(w_and_b, postfix)
        w_merge = w_and_b_merge[:w_shape[0] * w_shape[1]].reshape(w_shape) / float(n_workers)
        b_merge = w_and_b_merge[w_shape[0] * w_shape[1]:].reshape(b_shape[0]) / float(n_workers)
        model.linear.weight.data = torch.from_numpy(w_merge)
        model.linear.bias.data = torch.from_numpy(b_merge)
        epoch_sync_time = time.time() - epoch_sync_start

        if worker_index == 0:
            communicator.delete_expired_epoch(epoch)

        # Test the Model
        test_start = time.time()
        correct = 0
        total = 0
        test_loss = 0
        for items, labels in validation_loader:
            items = Variable(items.view(-1, n_features))
            labels = Variable(labels)
            outputs = model(items)
            test_loss += criterion(outputs, labels).data
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        test_time = time.time() - test_start

        print('Epoch: [%d/%d], Step: [%d/%d], Time: %.4f, Loss: %.4f, epoch cost %.4f, '
              'batch cost %.4f s: calculation cost = %.4f s, synchronization cost %.4f s, test cost %.4f s, '
              'accuracy of the model on the %d test samples: %d %%, loss = %f'
              % (epoch + 1, n_epochs, batch_index + 1, n_train_batch,
                 time.time() - train_start, epoch_loss.data, time.time() - epoch_start,
                 time.time() - batch_start, epoch_cal_time, epoch_sync_time, test_time,
                 len(val_indices), 100 * correct / total, test_loss / total))

    if worker_index == 0:
        clear_bucket(tmp_bucket)
        clear_bucket(merged_bucket)

    end_time = time.time()
    print("Elapsed time = {} s".format(end_time - start_time))
