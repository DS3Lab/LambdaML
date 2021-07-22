import os
import time
import math

import numpy as np
import pickle
import json

import torch
from torch.autograd import Variable
import torch.nn.functional as F

import boto3

from utils.constants import Prefix, MLModel, Optimization, Synchronization
from storage import S3Storage
from communicator import S3Communicator

from model import deep_models
from utils.metric import Accuracy, Average


def handler(event, context):
    start_time = time.time()

    # dataset setting
    train_file = event['train_file']
    test_file = event['test_file']
    data_bucket = event['data_bucket']
    n_features = event['n_features']
    n_classes = event['n_classes']
    n_workers = event['n_workers']
    worker_index = event['worker_index']
    tmp_bucket = event['tmp_bucket']
    merged_bucket = event['merged_bucket']
    cp_bucket = event['cp_bucket']

    # training setting
    model_name = event['model']
    optim = event['optim']
    sync_mode = event['sync_mode']
    assert model_name.lower() in MLModel.Deep_Models
    assert optim.lower() in Optimization.All
    assert sync_mode.lower() in Synchronization.All

    # hyper-parameter
    learning_rate = event['lr']
    batch_size = event['batch_size']
    n_epochs = event['n_epochs']
    start_epoch = event['start_epoch']
    run_epochs = event['run_epochs']

    function_name = event['function_name']

    print('data bucket = {}'.format(data_bucket))
    print("train file = {}".format(train_file))
    print("test file = {}".format(test_file))
    print('number of workers = {}'.format(n_workers))
    print('worker index = {}'.format(worker_index))
    print('model = {}'.format(model_name))
    print('optimization = {}'.format(optim))
    print('sync mode = {}'.format(sync_mode))
    print('start epoch = {}'.format(start_epoch))
    print('run epochs = {}'.format(run_epochs))

    print("Run function {}, round: {}/{}, epoch: {}/{} to {}/{}"
          .format(function_name, int(start_epoch/run_epochs) + 1, math.ceil(n_epochs / run_epochs),
                  start_epoch + 1, n_epochs, start_epoch + run_epochs, n_epochs))

    storage = S3Storage()
    communicator = S3Communicator(storage, tmp_bucket, merged_bucket, n_workers, worker_index)

    # download file from s3
    local_dir = "/tmp"
    read_start = time.time()
    storage.download(data_bucket, train_file, os.path.join(local_dir, train_file))
    storage.download(data_bucket, test_file, os.path.join(local_dir, test_file))
    print("download file from s3 cost {} s".format(time.time() - read_start))

    train_set = torch.load(os.path.join(local_dir, train_file))
    test_set = torch.load(os.path.join(local_dir, test_file))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    print("read data cost {} s".format(time.time() - read_start))

    random_seed = 100
    torch.manual_seed(random_seed)

    device = 'cpu'
    net = deep_models.get_models(model_name).to(device)

    # Loss and Optimizer
    # Softmax is internally computed.
    # Set parameters to be updated.
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

    # load checkpoint model if it is not the first round
    if start_epoch != 0:
        checked_file = 'checkpoint_{}.pt'.format(start_epoch - 1)
        storage.download(cp_bucket, checked_file, os.path.join(local_dir, checked_file))
        checkpoint_model = torch.load(os.path.join(local_dir, checked_file))

        net.load_state_dict(checkpoint_model['model_state_dict'])
        optimizer.load_state_dict(checkpoint_model['optimizer_state_dict'])
        print("load checkpoint model at epoch {}".format(start_epoch - 1))

    for epoch in range(start_epoch, min(start_epoch + run_epochs, n_epochs)):

        train_loss, train_acc = train_one_epoch(epoch, net, train_loader, optimizer, worker_index, n_workers,
                                                communicator, optim, sync_mode)
        test_loss, test_acc = test(epoch, net, test_loader)

        print('Epoch: {}/{},'.format(epoch + 1, n_epochs),
              'train loss: {}'.format(train_loss),
              'train acc: {},'.format(train_acc),
              'test loss: {}'.format(test_loss),
              'test acc: {}.'.format(test_acc), )

    if worker_index == 0:
        storage.clear(tmp_bucket)
        storage.clear(merged_bucket)

    # training is not finished yet, invoke next round
    if epoch < n_epochs - 1:
        checkpoint_model = {
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss.average
        }

        checked_file = 'checkpoint_{}.pt'.format(epoch)

        if worker_index == 0:
            torch.save(checkpoint_model, os.path.join(local_dir, checked_file))
            storage.upload(cp_bucket, checked_file, os.path.join(local_dir, checked_file))
            print("checkpoint model at epoch {} saved!".format(epoch))

        print("Invoking the next round of functions. round: {}/{}, start epoch: {}, run epoch: {}"
              .format(int((epoch + 1) / run_epochs) + 1, math.ceil(n_epochs / run_epochs),
                      epoch + 1, run_epochs))
        lambda_client = boto3.client('lambda')
        payload = {
            'train_file': event['train_file'],
            'test_file': event['test_file'],
            'data_bucket': event['data_bucket'],
            'n_features': event['n_features'],
            'n_classes': event['n_classes'],
            'n_workers': event['n_workers'],
            'worker_index': event['worker_index'],
            'tmp_bucket': event['tmp_bucket'],
            'merged_bucket': event['merged_bucket'],
            'cp_bucket': event['cp_bucket'],
            'model': event['model'],
            'optim': event['optim'],
            'sync_mode': event['sync_mode'],
            'lr': event['lr'],
            'batch_size': event['batch_size'],
            'n_epochs': event['n_epochs'],
            'start_epoch': epoch + 1,
            'run_epochs': event['run_epochs'],
            'function_name': event['function_name']
        }
        lambda_client.invoke(FunctionName=function_name,
                             InvocationType='Event',
                             Payload=json.dumps(payload))

    end_time = time.time()
    print("Elapsed time = {} s".format(end_time - start_time))


# Train
def train_one_epoch(epoch, net, train_loader, optimizer, worker_index, n_workers,
                    communicator, optim, sync_mode):
    assert isinstance(communicator, S3Communicator)
    net.train()

    epoch_start = time.time()

    epoch_cal_time = 0
    epoch_comm_time = 0

    train_acc = Accuracy()
    train_loss = Average()

    # record the architecture of the network
    grads_shape = []
    grads_length = []
    for param in net.parameters():
        grads_shape.append(param.data.numpy().shape)
        l = 1
        for i in param.data.numpy().shape:
            l *= i
        grads_length.append(l)
        grad_dtype = param.data.numpy().dtype

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        batch_start = time.time()
        outputs = net(inputs)
        loss = F.cross_entropy(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        batch_cal_time = time.time() - batch_start
        batch_comm_time = 0

        # flatten the gradient into a 1-d numpy array
        grads_vector = np.empty(sum(grads_length), dtype=grad_dtype)
        curr = 0
        for i, param in enumerate(net.parameters()):
            grads_vector[curr:curr+grads_length[i]] = np.ravel(param.grad.data.numpy())
            curr += grads_length[i]

        batch_cal_time = time.time() - batch_start
        epoch_cal_time += batch_cal_time

        batch_comm_start = time.time()
        postfix = "{}_{}".format(epoch, batch_idx)
        if sync_mode == "reduce":
            merged_grads_vec = communicator.reduce_batch(grads_vector, postfix)
        elif sync_mode == "reduce_scatter":
            merged_grads_vec = communicator.reduce_scatter_batch(grads_vector, postfix)
        else:
            raise ValueError("Synchronization mode has to be either reduce or reduce_scatter")

        # reconstruct the gradient from the 1-d vector
        curr = 0
        for i, param in enumerate(net.parameters()):
            curr_grad = merged_grads_vec[curr:curr+grads_length[i]].reshape(grads_shape[i]) / n_workers
            param.grad.data = torch.from_numpy(curr_grad)
            curr += grads_length[i]

        batch_comm_time = time.time() - batch_comm_start
        print("one {} round cost {} s".format(sync_mode, batch_comm_time))
        epoch_comm_time += batch_comm_time

        train_acc.update(outputs, targets)
        train_loss.update(loss.item(), inputs.size(0))

        if batch_idx % 10 == 0:
            print("Epoch: [{}], Batch: [{}], train loss: {}, train acc: {}, batch cost {} s, "
                  "cal cost {} s, comm cost {} s"
                  .format(epoch + 1, batch_idx + 1, train_loss, train_acc, time.time() - batch_start,
                          batch_cal_time, batch_comm_time))

    if worker_index == 0:
        delete_start = time.time()
        communicator.delete_expired_batch(epoch, batch_idx)
        epoch_comm_time += time.time() - delete_start

    print("Epoch {} has {} batches, cost {} s, cal time = {} s, comm time = {} s"
          .format(epoch + 1, batch_idx, time.time() - epoch_start, epoch_cal_time, epoch_comm_time))

    return train_loss, train_acc


def test(epoch, net, test_loader):
    # global best_acc
    net.eval()
    test_loss = Average()
    test_acc = Accuracy()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            outputs = net(inputs)

            loss = F.cross_entropy(outputs, targets)

            test_loss.update(loss.item(), inputs.size(0))
            test_acc.update(outputs, targets)

    return test_loss, test_acc
