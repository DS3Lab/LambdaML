import numpy as np
import time
import pickle
import torch
from torch.autograd import Variable

from sync.sync_reduce_scatter import *
from sync.sync_meta import SyncMeta

merged_bucket = "merged-value"
tmp_bucket = "tmp-value"

weights_prefix = 'w_'
gradients_prefix = 'g_'


# Training
def train(epoch, net, train_loader, optimizer, criterion, device,
          worker_index, num_worker, sync_mode):
    # print('\nEpoch: %d' % epoch)
    epoch_start = time.time()
    net.train()
    train_loss = 0
    n_correct = 0
    n_total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        #print("------worker {} epoch {} batch {}------".format(worker_index, epoch + 1, batch_idx + 1))
        batch_start = time.time()

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        print("forward and backward cost {} s".format(time.time() - batch_start))

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        n_total += targets.size(0)
        n_correct += predicted.eq(targets).sum().item()

        if sync_mode == 'grad_avg':
            sync_start = time.time()
            # get gradients and flatten it to a 1-D array
            param_dic = {}
            for index, param in enumerate(net.parameters()):
                param_dic[index] = [param.grad.data.numpy().size, param.grad.data.numpy().shape]
                if index == 0:
                    flattened_param = param.grad.data.numpy().flatten()
                else:
                    flattened_param = np.concatenate((flattened_param, param.grad.data.numpy().flatten()))

            # merge gradients
            file_postfix = "{}_{}".format(epoch, batch_idx)
            merged_value = reduce_scatter_batch(flattened_param, tmp_bucket, merged_bucket,
                                                num_worker, worker_index, file_postfix)

            # update the model gradients by layers
            offset = 0
            for layer_index, param in enumerate(net.parameters()):
                layer_size = param_dic[layer_index][0]
                layer_shape = param_dic[layer_index][1]
                layer_value = merged_value[offset: offset + layer_size].reshape(layer_shape)
                param.grad = Variable(torch.from_numpy(layer_value))
                offset += layer_size

            if worker_index == 0:
                delete_expired_merged(merged_bucket, epoch, batch_idx)
            print("synchronization cost {} s".format(time.time() - sync_start))

            #gradients = [param.grad.data.numpy() for param in net.parameters()]
            #print("[Worker {}] Gradients after sync = {}".format(worker_index, gradients[0][0]))

            optimizer.step()

        print('Epoch: {}, Batch: {}, Loss: {}, cost {} s'
              .format(epoch + 1, batch_idx + 1, loss.data, time.time() - batch_start))

    if sync_mode == 'model_avg':

        # apply local gradient to local model
        optimizer.step()
        # average model
        sync_start = time.time()

        # get current weights
        weights = [param.data.numpy() for param in net.parameters()]

        # upload updated weights to S3
        put_object(tmp_bucket, weights_prefix + str(worker_index), pickle.dumps(weights))

        file_postfix = str(epoch)
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

    if (batch_idx + 1) % 1 == 0:
        print('Epoch[{}] cost {} s, Train Loss: {}, Train Accuracy: {}%'
              .format(epoch + 1, time.time() - epoch_start, train_loss.data, 100.0 * n_correct / n_total))


def test(epoch, net, test_loader, criterion, device):
    test_start = time.time()
    net.eval()
    test_loss = 0
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            n_total += targets.size(0)
            n_correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #     % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    print('Test time = %.4f, accuracy of the model on the %d test samples: %d %%, loss = %f'
          % (time.time() - test_start, n_total, 100 * n_correct / n_total, test_loss))
