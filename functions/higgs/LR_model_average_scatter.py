import time
import urllib.parse
import logging
import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler

from sync.sync_grad import *
from sync.sync_reduce_scatter import reduce_scatter_epoch, delete_expired_merged

from model.LogisticRegression import LogisticRegression
from data_loader.LibsvmDataset import DenseLibsvmDataset2
from sync.sync_meta import SyncMeta

# lambda setting
file_bucket = "higgs-libsvm"
tmp_bucket = "tmp-grads"
merged_bucket = "merged-params"
local_dir = "/tmp"
w_prefix = "w_"
b_prefix = "b_"
tmp_w_prefix = "tmp_w_"
tmp_b_prefix = "tmp_b_"

# algorithm setting
num_features = 30
num_classes = 2
learning_rate = 0.1
batch_size = 10000
num_epochs = 10
validation_ratio = .2
shuffle_dataset = True
random_seed = 42


def handler(event, context):
    start_time = time.time()
    original_bucket = event['Records'][0]['s3']['bucket']['name']
    bucket = file_bucket
    key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')

    print('bucket = {}'.format(bucket))
    print('original bucket = {}'.format(original_bucket))
    print('key = {}'.format(key))

    key_splits = key.split("_")
    worker_index = int(key_splits[0])
    num_worker = int(key_splits[1])
    sync_meta = SyncMeta(worker_index, num_worker)
    print("synchronization meta {}".format(sync_meta.__str__()))

    # read file from s3
    file = get_object(bucket, key).read().decode('utf-8').split("\n")
    print("read data cost {} s".format(time.time() - start_time))

    parse_start = time.time()
    dataset = DenseLibsvmDataset2(file, num_features)
    print("parse data cost {} s".format(time.time() - parse_start))

    preprocess_start = time.time()
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_ratio * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size,
                                               sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset,
                                                    batch_size=batch_size,
                                                    sampler=valid_sampler)

    print("preprocess data cost {} s".format(time.time() - preprocess_start))

    model = LogisticRegression(num_features, num_classes)

    # Loss and Optimizer
    # Softmax is internally computed.
    # Set parameters to be updated.
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    train_start = time.time()
    # Training the Model
    for epoch in range(num_epochs):
        epoch_start = time.time()
        for batch_index, (items, labels) in enumerate(train_loader):
            print("------worker {} epoch {} batch {}------".format(worker_index, epoch, batch_index))
            batch_start = time.time()
            items = Variable(items.view(-1, num_features))
            labels = Variable(labels)

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = model(items)
            loss = criterion(outputs, labels)
            loss.backward()
            #print("forward and backward cost {} s".format(time.time()-batch_start))
            optimizer.step()

            print('Epoch: [%d/%d], Step: [%d/%d], Time: %.4f, Loss: %.4f, batch cost %.4f s'
                    % (epoch + 1, num_epochs, batch_index + 1, len(train_indices) / batch_size,
                       time.time() - train_start, loss.data, time.time() - batch_start))

        w = model.linear.weight.data.numpy()
        w_shape = w.shape
        b = model.linear.bias.data.numpy()
        b_shape = b.shape
        #print("weight before sync shape = {}, values = {}".format(w.shape, w))
        #print("bias before sync shape = {}, values = {}".format(b.shape, b))
        w_and_b = np.concatenate((w.flatten(), b.flatten()))
        print("Epoch {} calculation cost = {} s".format(epoch, time.time() - epoch_start))

        sync_start = time.time()
        postfix = str(epoch)
        w_and_b_merge = reduce_scatter_epoch(w_and_b, tmp_bucket, merged_bucket, num_worker, worker_index, postfix)
        w_merge = w_and_b_merge[:w_shape[0]*w_shape[1]].reshape(w_shape) / float(num_worker)
        b_merge = w_and_b_merge[w_shape[0]*w_shape[1]:].reshape(b_shape[0]) / float(num_worker)
        model.linear.weight.data = torch.from_numpy(w_merge)
        model.linear.bias.data = torch.from_numpy(b_merge)
        # print("weight after sync = {}".format(model.linear.weight.data.numpy()[0][:5]))
        # print("bias after sync = {}".format(model.linear.bias.data.numpy()))
        print("Epoch {} synchronization cost {} s".format(epoch, time.time() - sync_start))

        if worker_index == 0:
            delete_expired_merged(merged_bucket, epoch)

        # Test the Model
        correct = 0
        total = 0
        test_loss = 0
        for items, labels in validation_loader:
            items = Variable(items.view(-1, num_features))
            labels = Variable(labels)
            outputs = model(items)
            test_loss += criterion(outputs, labels).data
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

        print('Time = %.4f, accuracy of the model on the %d test samples: %d %%, loss = %f'
              % (time.time() - train_start, len(val_indices), 100 * correct / total, test_loss))

    if worker_index == 0:
        clear_bucket(tmp_bucket)
        clear_bucket(merged_bucket)

    end_time = time.time()
    print("Elapsed time = {} s".format(end_time - start_time))