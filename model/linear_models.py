import torch


def get_model(algo, n_features, n_classes):
    if algo == "lr":
        return LogisticRegression(n_features, n_classes)
    elif algo == "svm":
        return SVM(n_features, n_classes)
    else:
        raise Exception("algorithm {} is not supported, should be lr or svm"
                        .format(algo))


class LogisticRegression(torch.nn.Module):

    def __init__(self, _num_features, _num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(_num_features, _num_classes)

    # torch.nn.CrossEntropyLoss includes softmax, don't need sigmoid here
    def forward(self, x):
        # y_pred = torch.sigmoid(self.linear(x))
        y_pred = self.linear(x)
        return y_pred


class SVM(torch.nn.Module):

    def __init__(self, _num_features, _num_classes):
        super(SVM, self).__init__()
        self.linear = torch.nn.Linear(_num_features, _num_classes)

    # torch.nn.CrossEntropyLoss includes softmax, don't need sigmoid here
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
