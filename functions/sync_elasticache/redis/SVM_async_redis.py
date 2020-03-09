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

from pytorch_model.DenseSVM import DenseSVM, MultiClassHingeLoss
from data_loader.LibsvmDataset import DenseLibsvmDataset2
from sync.sync_meta import SyncMeta


# lambda setting

grad_bucket = "async-grads"
model_bucket = "async-updates"
local_dir = "/tmp"
w_prefix = "w_"
b_prefix = "b_"
w_grad_prefix = "w_grad_"
b_grad_prefix = "b_grad_"

# algorithm setting

learning_rate = 0.1
batch_size = 100000
num_epochs = 55
validation_ratio = .2
shuffle_dataset = True
random_seed = 42




def handler(event, context):
    
    startTs = time.time()
    bucket = event['bucket']
    key = event['name']
    num_features = event['num_features']
    num_classes = event['num_classes']
    elasti_location = event['elasticache']
    endpoint = redis_init(elasti_location)
    print('bucket = {}'.format(bucket))
    print('key = {}'.format(key))
    model_bucket = event['model_bucket']
    grad_bucket = event['grad_bucket']
    key_splits = key.split("_")
    worker_index = int(key_splits[0])
    #num_worker = int(key_splits[1])
    num_worker = event['num_files']

    batch_size = 100000
    batch_size = int(np.ceil(batch_size/num_worker))
    
    torch.manual_seed(random_seed)
    
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
    
    model = DenseSVM(num_features, num_classes)
    

    # Loss and Optimizer
    # Softmax is internally computed.
    # Set parameters to be updated.
    criterion = MultiClassHingeLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    train_loss = []
    test_loss = []
    test_acc = []
    total_time = 0
    # Training the Model
    epoch_start = time.time()
    for epoch in range(num_epochs):
        tmp_train = 0
        for batch_index, (items, labels) in enumerate(train_loader):
            #batch_start = time.time()
            print("------worker {} epoch {} batch {}------".format(worker_index, epoch, batch_index))
            items = Variable(items.view(-1, num_features))
            labels = Variable(labels)

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = model(items)
            loss = criterion(outputs, labels)
            loss.backward()
            
            w = model.linear.weight.data.numpy()
            b = model.linear.bias.data.numpy()
            file_postfix = "{}_{}".format(batch_index,epoch)
            #asynchronization / shuffle starts from that every worker writes their gradients of this batch and epoch
            #upload individual gradient
            hset_object(endpoint, model_bucket, w_prefix, w.tobytes())
            hset_object(endpoint, model_bucket, b_prefix, b.tobytes())
            
            time.sleep(0.0001)#
            #randomly get one gradient from others. (Asynchronized)
            w_new = np.fromstring(hget_object(endpoint, model_bucket, w_prefix),dtype = w.dtype).reshape(w.shape)
            b_new = np.fromstring(hget_object(endpoint, model_bucket, b_prefix),dtype = b.dtype).reshape(b.shape)	
            model.linear.weight.data = torch.from_numpy(w_new)
            model.linear.bias.data = torch.from_numpy(b_new)
            optimizer.step()
            
            #report train loss and test loss for every mini batch
            if (batch_index + 1) % 1 == 0:
                print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f'
                      % (epoch + 1, num_epochs, batch_index + 1, len(train_indices) / batch_size, loss.data))
            tmp_train += loss.item()
        total_time += time.time()-epoch_start
        train_loss.append(tmp_train)
        
        tmp_test,tmp_acc = test(model,validation_loader,criterion)
        test_loss.append(tmp_test)
        test_acc.append(tmp_acc)
        epoch_start = time.time()
        
    print("total time = {}".format(total_time))
    endTs = time.time()
    print("elapsed time = {} s".format(endTs - startTs))
    loss_record = [test_loss,test_acc,train_loss,total_time]
    put_object("async-model-loss","async-loss{}".format(worker_index),pickle.dumps(loss_record))

def test(model,testloader,criterion):
    # Test the Model
    correct = 0
    total = 0
    total_loss = 0
    count = 0
    with torch.no_grad():
        for items,labels in testloader:
            outputs = model(items)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            loss = criterion(outputs,labels)
            total_loss+=loss.data
            count = count+1
    return total_loss/count, float(correct)/float(total)*100
