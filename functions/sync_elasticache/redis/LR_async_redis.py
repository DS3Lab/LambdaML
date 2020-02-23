import time
import urllib.parse
import logging
import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler

from elasticache.Redis.set_object import hset_object
from elasticache.Redis.counter import hcounter
from elasticache.Redis.get_object import hget_object
from elasticache.Redis.__init__ import redis_init
from s3.get_object import get_object
from s3.put_object import put_object
from sync.sync_grad_redis import *

from model.LogisticRegression import LogisticRegression
from data_loader.LibsvmDataset import DenseLibsvmDataset2
from sync.sync_meta import SyncMeta

# lambda setting

grad_bucket = "higgs-grads"
model_bucket = "higgs-updates"
local_dir = "/tmp"
w_prefix = "w_"
b_prefix = "b_"
w_grad_prefix = "w_grad_"
b_grad_prefix = "b_grad_"

# algorithm setting

learning_rate = 0.1
batch_size = 32
num_epochs = 2
validation_ratio = .2
shuffle_dataset = True
random_seed = 42




def handler(event, context):
    
    startTs = time.time()
    bucket = event['bucket']
    key = event['name']
    num_features = event['num_features']
    num_classes = event['num_classes']
    redis_location = event['redis']
    endpoint = redis_init(redis_location)
    print('bucket = {}'.format(bucket))
    print('key = {}'.format(key))
  
    key_splits = key.split("_")
    worker_index = int(key_splits[0])
    #num_worker = int(key_splits[1])
    num_worker = event['num_files']

    batch_size = 10000
    batch_size = int(np.ceil(batch_size/num_worker))
    
    sync_meta = SyncMeta(worker_index, num_worker)
    print("synchronization meta {}".format(sync_meta.__str__()))
    
    # read file(dataset) from s3
    file = get_object(bucket, key).read().decode('utf-8').split("\n")
    print("read data cost {} s".format(time.time() - startTs))
    parse_start = time.time()
    dataset = DenseLibsvmDataset2(file, num_features)
    preprocess_start = time.time()
    print("libsvm operation cost {}s".format(parse_start - preprocess_start))
   
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    print("dataset size = {}".format(dataset_size))
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
    
    
    
    # Training the Model
    for epoch in range(num_epochs):
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
            
            w_grad = model.linear.weight.grad.data.numpy()
            b_grad = model.linear.bias.grad.data.numpy()
            print("w_grad before merge = {}".format(w_grad[0][0:5]))
            print("b_grad before merge = {}".format(b_grad))
            
            #asynchronization / shuffle starts from that every worker writes their gradients of this batch and epoch
            import sys
            print("model size = {}".format((sys.getsizeof(w_grad.tobytes())+sys.getsizeof(b_grad.tobytes()))/1024/1024))
            #upload individual gradient
            hset_object(endpoint, model_bucket, w_grad_prefix , w_grad.tobytes())
            hset_object(endpoint, model_bucket, b_grad_prefix , b_grad.tobytes())
            tmp_w_dtype = w_grad.dtyep()
            tmp_b_dtype = b_grad.dtyep()
            tmp_w_shape = w_grad.shape()
            tmp_b_shape = b_grad.shape()
            #randomly get one gradient from others. (Asynchronized)
            w_grad = np.fromstring(hget_object(endpoint, model_bucket, w_grad_prefix),dtype = tmp_w_dtype).reshape(tmp_w_shape)
            b_grad = np.fromstring(hget_object(endpoint, model_bucket, b_grad_prefix),dtype = tmp_b_dtype).reshape(tmp_b_shape)	
            model.linear.weight.grad = Variable(torch.from_numpy(w_grad))
            model.linear.bias.grad = Variable(torch.from_numpy(b_grad))

            print("w_grad after merge = {}".format(model.linear.weight.grad.data.numpy()[0][:5]))
            print("b_grad after merge = {}".format(model.linear.bias.grad.data.numpy()))
            optimizer.step()

            print("batch cost {} s".format(time.time() - batch_start))
            #report train loss and test loss for every mini batch
            if (batch_index + 1) % 1 == 0:
                print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f'
                      % (epoch + 1, num_epochs, batch_index + 1, len(train_indices) / batch_size, loss.data))
    		
            # Test the Model
            correct = 0
            total = 0
            for items, labels in validation_loader:
                items = Variable(items.view(-1, num_features))
                outputs = model(items)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()

            print('Accuracy of the model on the %d test samples: %d %%' % (len(val_indices), 100 * correct / total))

    endTs = time.time()
    print("elapsed time = {} s".format(endTs - startTs))