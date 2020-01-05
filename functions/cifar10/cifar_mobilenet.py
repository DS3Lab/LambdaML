import os
import torch
import torch.nn as nn
import torch.optim as optim
import boto3
import time

from pytorch_model.cifar10 import MobileNet
from .train_test import train, test

local_dir = "/tmp"

# dataset setting
training_file = 'training.pt'
test_file = 'test.pt'

# sync up mode
sync_mode = 'grad_avg'

# learning algorithm setting
learning_rate = 0.01
batch_size = 32
num_epochs = 1

s3 = boto3.resource('s3')


def handler(event, context):
    start_time = time.time()
    bucket = event['data_bucket']
    worker_index = event['rank']
    num_worker = event['num_workers']
    key = 'training_{}.pt'.format(worker_index)
    print('data_bucket = {}\n worker_index:{}\n num_worker:{}\n key:{}'.format(bucket, worker_index, num_worker, key))

    # read file from s3
    readS3_start = time.time()
    s3.Bucket(bucket).download_file(key, os.path.join(local_dir, training_file))
    s3.Bucket(bucket).download_file(test_file, os.path.join(local_dir, test_file))
    print("read data cost {} s".format(time.time() - readS3_start))

    train_set = torch.load(os.path.join(local_dir, training_file))
    test_set = torch.load(os.path.join(local_dir, test_file))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    device = 'cpu'
    # best_acc = 0  # best test accuracy
    # start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Model
    print('==> Building model..')
    # net = VGG('VGG19')
    # net = ResNet18()
    #net = ResNet50()
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    # net = ShuffleNetV2(1)
    # net = EfficientNetB0()

    print("Model: MobileNet")

    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    for epoch in range(num_epochs):
        train(epoch, net, train_loader, optimizer, criterion, device,
              worker_index, num_worker, sync_mode)
        test(epoch, net, test_loader, criterion, device)
