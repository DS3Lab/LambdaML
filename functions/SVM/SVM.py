

import torch
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler



if __name__ == "__main__":
    import sys
    sys.path.append('/Users/liuyue/LambdaML')
    from pytorch_model.DenseSVM import DenseSVM, BinaryClassHingeLoss
    from data_loader.LibsvmDataset import DenseLibsvmDataset
    num_features = 128
    num_class = 2
    train_file = "/Users/liuyue/LambdaML/dataset/agaricus_127d_train.libsvm"
    test_file = "/Users/liuyue/LambdaML/dataset/agaricus_127d_test.libsvm"
    train_dataset = DenseLibsvmDataset(train_file,num_features)
    validation_dataset = DenseLibsvmDataset(test_file,num_features)
    random_seed = 42
    batch_size = 1000
    torch.manual_seed(random_seed)
    learning_rate = 0.1

    #print(train_dataset.shape)
    m = torch.mean(train_dataset,1)
    std = torch.std(train_dataset,1)
    train_dataset = (train_dataset-m)/std
    validation_dataset = (validation_dataset-m)/std

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size)
    validation_loader = torch.utils.data.DataLoader(validation_dataset,
                                                    batch_size=batch_size)


    model = DenseSVM(num_features,num_class)
    #criterion = torch.nn.modules.MultiLabelMarginLoss()
    #criterion = torch.nn.modules.MultiLabelMarginLoss()
    criterion = BinaryClassHingeLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    for batch_index, (items, labels) in enumerate(train_loader):
        items = Variable(items.view(-1, num_features))
        labels = Variable(labels)
        # Forward + Backward + Optimize
        optimizer.zero_grad()

        outputs = model(items)
        print(outputs.data)
        _, predicted = torch.max(outputs.data, 1)
        #print("predict = {}".format(predicted))
        """
        padding = (torch.ones(outputs.shape)*(-1)).long()
        padding[:,labels] = 1
        loss = criterion(outputs,padding)
        """
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
    print(loss.item())
