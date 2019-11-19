import numpy as np
import time
import pickle
import torch
from torch.autograd import Variable

from s3.list_objects import list_bucket_objects
from s3.get_object import get_object
from s3.put_object import put_object

from sync.sync_neural_network import *
from sync.sync_meta import SyncMeta


merged_bucket = "merged-value-2"
tmp_bucket = "tmp-value-2"

weights_prefix = 'w_'
gradients_prefix = 'g_'

# Training
def train(epoch, net, trainloader, optimizer, criterion, device, worker_index, num_worker, sync_mode, sync_step):
    # print('\nEpoch: %d' % epoch)
    net.train()
    # train_loss = 0
    # correct = 0
    # total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):

        print("------worker {} epoch {} batch {}------".format(worker_index, epoch+1, batch_idx+1))
        batch_start = time.time()

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        print("forward and backward cost {} s".format(time.time()-batch_start))

        if sync_mode == 'model_avg':

            # apply local gradient to local model
            optimizer.step()
            # average model
            if (batch_idx+1) % sync_step == 0:
                
                sync_start = time.time()
                
                # get current weights
                weights = [param.data.numpy() for param in net.parameters()]
                # print("[Worker {}] Weights before sync = {}".format(worker_index, weights[0][0]))

                # upload updated weights to S3
                put_object(tmp_bucket, weights_prefix + str(worker_index), pickle.dumps(weights))
                
                file_postfix = "{}_{}".format(epoch, batch_idx)
                if worker_index == 0:
                    # merge all workers
                    merged_value = \
                        merge_all_workers(tmp_bucket, num_worker, weights_prefix)

                    # upload merged value to S3
                    put_merged(merged_bucket, merged_value,
                                        weights_prefix, file_postfix)

                    delete_expired(merged_bucket, epoch, batch_idx, weights_prefix)
                    
                else:
                    # get merged value from S3
                    merged_value = get_merged(merged_bucket, weights_prefix, file_postfix)
                
                # print("[Worker {}] Weights after sync = {}".format(worker_index, merged_value[0][0]))
                
                # update the model with averaged model
                for layer_index, param in enumerate(net.parameters()):
                    param.data = torch.nn.Parameter(torch.from_numpy(merged_value[layer_index]))
                
                
                # weights = [param.data.numpy() for param in net.parameters()]
                # print("[Worker {}] Weights after sync = {}".format(worker_index, weights[0][0]))
                  
                    
                print("synchronization cost {} s".format(time.time() - sync_start))
                
        if sync_mode == 'grad_avg':
            
            sync_start = time.time()
            
            gradients = [param.grad.data.numpy() for param in net.parameters()]
            # print("[Worker {}] Gradients before sync = {}".format(worker_index, gradients[0][0]))
            
            
            put_object_start = time.time()
            
            put_object(tmp_bucket, gradients_prefix + str(worker_index), pickle.dumps(gradients))
            
            print("write local gradients cost {} s".format(time.time() - put_object_start))
            
            
            
                
            file_postfix = "{}_{}".format(epoch, batch_idx)
            if worker_index == 0:
                # merge all workers
                
                merged_value_start = time.time()
                
                merged_value = \
                    merge_all_workers(tmp_bucket, num_worker, gradients_prefix)
                    
                print("merged_value cost {} s".format(time.time() - merged_value_start))
                
                
                
                put_merged_start = time.time()
                # upload merged value to S3
                put_merged(merged_bucket, merged_value,
                                    gradients_prefix, file_postfix)
                                    
                print("put_merged cost {} s".format(time.time() - put_merged_start))                   
                

                # delete_expired(merged_bucket, epoch, batch_idx, gradients_prefix)
                
            else:
                
                read_merged_start = time.time()
                # get merged value from S3
                merged_value = get_merged(merged_bucket, gradients_prefix, file_postfix)
                
                print("read_merged cost {} s".format(time.time() - read_merged_start))
                
            # print("[Worker {}] Gradients after sync = {}".format(worker_index, merged_value[0][0]))
              
                
            for layer_index, param in enumerate(net.parameters()):
                param.grad = Variable(torch.from_numpy(merged_value[layer_index]))
                
            # gradients = [param.grad.data.numpy() for param in net.parameters()]
            # print("[Worker {}] Gradients after sync = {}".format(worker_index, gradients[0][0]))
                
            print("synchronization cost {} s".format(time.time() - sync_start))
            
            if worker_index == 0:
                delete_expired(merged_bucket, epoch, batch_idx, gradients_prefix)
                
            optimizer.step()
            

            
        # train_loss += loss.item()
        # _, predicted = outputs.max(1)
        # total += targets.size(0)
        # correct += predicted.eq(targets).sum().item()
        print("batch cost {} s".format(time.time() - batch_start))
        if (batch_idx + 1) % 1 == 0:
            print('Epoch: {}, Step: {}, Loss:{}'.format(epoch+1, batch_idx+1, loss.data))


def test(epoch, net, testloader, criterion):
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