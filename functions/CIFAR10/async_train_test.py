import numpy as np
import time
import pickle
import torch
from torch.autograd import Variable
import torch.nn.functional as F

from s3.list_objects import list_bucket_objects
from s3.get_object import get_object, get_object_or_wait
from s3.put_object import put_object

from sync.sync_neural_network import *
from sync.sync_meta import SyncMeta


merged_bucket = "merged-value"
tmp_bucket = "tmp-value"

weights_prefix = 'w_'
gradients_prefix = 'g_'


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
        with torch.no_grad():
            pred = output.argmax(dim=1)
            correct = pred.eq(target).sum().item()

        self.correct += correct
        self.count += output.size(0)
        
class Average(object):

    def __init__(self):
        self.sum = 0
        self.count = 0

    def __str__(self):
        return '{:.6f}'.format(self.average)

    @property
    def average(self):
        return self.sum / self.count

    def update(self, value, number):
        self.sum += value * number
        self.count += number


# Training
def async_train(epoch, net, trainloader, optimizer, device, worker_index, num_worker, sync_mode, sync_step):
    
    net.train()
    
    epoch_start = time.time()
    
    epoch_sync_time = 0
    num_batch = 0
    
    train_acc = Accuracy()
    train_loss = Average()

    for batch_idx, (inputs, targets) in enumerate(trainloader):

        # print("------worker {} epoch {} batch {}------".format(worker_index, epoch+1, batch_idx+1))
        batch_start = time.time()

        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = F.cross_entropy(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        # get local model
        weights = [param.data.numpy() for param in net.parameters()]
        # push local model
        put_object(tmp_bucket, 'latest_model', pickle.dumps(weights))
        # pull the latest model
        new_model = get_object_or_wait(tmp_bucket, 'latest_model', 0.1).read()
        new_model_np = pickle.loads(new_model)
        # update local model
        for layer_index, param in enumerate(net.parameters()):
            param.data = torch.from_numpy(new_model_np[layer_index])
                    
        
        train_acc.update(outputs, targets)
        train_loss.update(loss.item(), inputs.size(0))
        
        if num_batch%10 == 0:
            print("Epoch {} Batch {} training Loss:{}, Acc:{}".format(epoch+1, num_batch, train_loss, train_acc))
        
        num_batch += 1
        
    epoch_time = time.time() - epoch_start
    print("Epoch {} has {} batches, time = {} s, sync time = {} s, cal time = {} s".format(epoch+1, num_batch, epoch_time, epoch_sync_time, epoch_time - epoch_sync_time))
    
    return train_loss, train_acc



def test(epoch, net, testloader, device):
    # global best_acc
    net.eval()
    test_loss = Average()
    test_acc = Accuracy()
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = net(inputs)
            
            loss = F.cross_entropy(outputs, targets)
            
            test_loss.update(loss.item(), inputs.size(0))
            test_acc.update(outputs, targets)
            
    
    return test_loss, test_acc