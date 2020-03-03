import torch
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np




class DenseSVM(torch.nn.Module):
    def __init__(self, _num_features, _num_classes):
        super(DenseSVM, self).__init__()
        self.num_class = _num_classes
        self.num_features = _num_features
        self.linear = torch.nn.Linear(_num_features, _num_classes)
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


class MultiClassHingeLoss(torch.nn.Module):
    def forward(self,pred,target):
        loss_value = 0
        for i in range(pred.shape[0]):
            indices = torch.arange(pred.shape[1])
            indices = indices[indices!=target[i]]
            tmp_loss = torch.index_select(pred[i,:],0, indices)-pred[i,target[i]]+1
            tmp_hinge = torch.cat((torch.zeros((1)),tmp_loss))
            loss_value += torch.max(tmp_hinge,0)[0]
        return loss_value/target.shape[0]
