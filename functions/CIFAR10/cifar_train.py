import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import boto3
import time

from sync.sync_meta import SyncMeta
from models import *
from training_test import train, test


local_dir = "/tmp"
# local_dir = "/tmp"
# w_prefix = "w_"
# b_prefix = "b_"
# w_grad_prefix = "w_grad_"
# b_grad_prefix = "b_grad_"
# w_weights_prefix = "w_wei_"
# b_weights_prefix = "b_wei_"

# dataset setting
training_file = 'training.pt'
test_file = 'test.pt'

# sync up mode
sync_mode = 'grad_avg'
sync_step = 1

# learning algorithm setting
learning_rate = 0.01
batch_size = 32
num_epochs = 1

s3 = boto3.resource('s3')

def handler(event, context):

    startTs = time.time()
    bucket = event['data_bucket']
    worker_index = event['rank']
    num_worker = event['num_workers']
    key = 'training_{}.pt'.format(worker_index)
    print('data_bucket = {}\n worker_index:{}\n num_worker:{}\n key:{}'.format(bucket, worker_index, num_worker, key))

    sync_meta = SyncMeta(worker_index, num_worker)
    print("synchronization meta {}".format(sync_meta.__str__()))

    # read file from s3
    readS3_start = time.time()
    s3.Bucket(bucket).download_file(key, os.path.join(local_dir, training_file))
    s3.Bucket(bucket).download_file(test_file, os.path.join(local_dir, test_file))
    print("read data cost {} s".format(time.time() - readS3_start))

    trainset = torch.load(os.path.join(local_dir, training_file))
    testset = torch.load(os.path.join(local_dir, test_file))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    device = 'cpu'
    # best_acc = 0  # best test accuracy
    # start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    #Model
    print('==> Building model..')
    # net = VGG('VGG19')
    # net = ResNet18()
    net = ResNet50()
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    # net = ShuffleNetV2(1)
    # net = EfficientNetB0()
    
    print("Model: ResNet50")

    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    for epoch in range(num_epochs):
        train(epoch, net, trainloader, optimizer, criterion, device, worker_index, num_worker, sync_mode, sync_step)
        test(epoch, net, testloader, criterion)

