import torch
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np



class LogisticRegression(torch.nn.Module):
    def __init__(self, _num_features, _num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(_num_features, _num_classes)

    # torch.nn.CrossEntropyLoss includes softmax, so we don't need sigmoid here?
    def forward(self, x):
        # y_pred = torch.sigmoid(self.linear(x))
        y_pred = self.linear(x)
        return y_pred


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LogisticRegression(28,2).to(device)

    summary(model, (2,28))   