import time
import urllib.parse
import logging
import numpy as np

import torch
from torch.autograd import Variable
from torch.nn import Parameter
from torch.utils.data.sampler import SubsetRandomSampler

from pytorch_model.cifar10 import MobileNet
from data_loader.LibsvmDataset import DenseLibsvmDataset2

from s3.download_file import download_file

from thrift_ps.ps_service import ParameterServer
from thrift_ps.ps_service.ttypes import Model, Update, Grad, InvalidOperation
from thrift_ps.client import ps_client

from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol

from thrift_ps import constants


# algorithm setting
learning_rate = 0.1
batch_size = 200
num_epochs = 1
random_seed = 42


training_file = 'training.pt'
test_file = 'test.pt'

def handler(event, context):
    start_time = time.time()
    bucket = event['data_bucket']
    worker_index = event['rank']
    num_worker = event['num_workers']
    key = event['key']

    print('bucket = {}'.format(bucket))
    print('number of workers = {}'.format(num_worker))
    print('worker index = {}'.format(worker_index))


    # Set thrift connection
    # Make socket
    transport = TSocket.TSocket(constants.HOST, constants.PORT)
    # Buffering is critical. Raw sockets are very slow
    transport = TTransport.TBufferedTransport(transport)
    # Wrap in a protocol
    protocol = TBinaryProtocol.TBinaryProtocol(transport)
    # Create a client to use the protocol encoder
    t_client = ParameterServer.Client(protocol)
    # Connect!
    transport.open()

    # test thrift connection
    ps_client.ping(t_client)
    print("create and ping thrift server >>> HOST = {}, PORT = {}"
          .format(constants.HOST, constants.PORT))


    #bucket = "cifar10dataset"

    print('data_bucket = {}\n worker_index:{}\n num_worker:{}\n key:{}'.format(bucket, worker_index, num_worker, key))



    # read file from s3
    readS3_start = time.time()
    train_path = download_file(bucket, key)
    trainset = torch.load(train_path)
    test_path = download_file(bucket,test_file)
    testset = torch.load(test_path)

    print("read data cost {} s".format(time.time() - readS3_start))
    preprocess_start = time.time()
    batch_size = 200
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    device = 'cpu'
    print("preprocess data cost {} s".format(time.time() - preprocess_start))

    model = MobileNet()
    model = model.to(device)

    # Loss and Optimizer
    # Softmax is internally computed.
    # Set parameters to be updated.
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    # register model
    model_name = "mobilenet"
    parameter_shape = []
    parameter_length = []
    model_length = 0
    for param in model.parameters():
        tmp_shape = 1
        parameter_shape.append(param.data.numpy().shape)
        for w in param.data.numpy().shape:
            tmp_shape *=w
        parameter_length.append(tmp_shape)
        model_length += tmp_shape
    model_length = 1
    ps_client.register_model(t_client, worker_index, model_name, model_length, num_worker)
    ps_client.exist_model(t_client, model_name)
    print("register and check model >>> name = {}, length = {}".format(model_name, model_length))

    # Training the Model
    train_start = time.time()
    iter_counter = 0
    for epoch in range(num_epochs):
        epoch_start = time.time()
        for batch_index, (inputs, targets) in enumerate(train_loader):
            print("------worker {} epoch {} batch {}------"
                  .format(worker_index, epoch, batch_index))
            batch_start = time.time()
            # Forward + Backward + Optimize
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            # flatten and concat gradients of weight and bias
            weights = np.zeros((1))
            for parma in model.parameters():
                weights = np.concatenate((weights,param.grad.data.numpy().flatten()))

            weights = np.delete(weights,0)
            weigths = np.zeros((1))
            # push gradient to PS
            sync_start = time.time()
            print(ps_client.can_push(t_client, model_name, iter_counter, worker_index))
            print(ps_client.push_grad(t_client, model_name, weights, LEARNING_RATE, iter_counter, worker_index))
            print(ps_client.can_pull(t_client, model_name, iter_counter+1, worker_index))      # sync all workers

            # pull latest model
            ps_client.can_pull(t_client, model_name, iter_counter, worker_index)
            latest_model = ps_client.pull_model(t_client, model_name, iter_counter, worker_index)
            pos = 0
            for layer_index, param in enumerate(model.parameters()):
                param.grad.data = Variable(torch.from_numpy(np.asarray(latest_model[pos:pos+parameter_length[layer_index]]).reshape(parameter_shape[layer_index])))
                pos += parameter_length[layer_index]

            sync_time = time.time() - sync_start


            print('Epoch: [%d/%d], Step: [%d/%d] >>> Time: %.4f, Loss: %.4f, epoch cost %.4f, '
                  'batch cost %.4f s: cal cost %.4f s and communication cost %.4f s'
                  % (epoch + 1, num_epochs, batch_index + 1, len(train_indices) / batch_size,
                     time.time() - train_start, loss.data, time.time() - epoch_start,
                     time.time() - batch_start, cal_time, sync_time))
            iter_counter += 1

            test(epoch,model,test_loader,criterion,device)
            optimizer.step()
def test(epoch, net, testloader, criterion, device):
    # global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #     % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    print("Accuracy of epoch {} on test set is {}".format(epoch, acc))
