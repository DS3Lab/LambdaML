#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 20:45:18 2019

@author: liuyue
"""

import torch
import boto3
from torch.utils.data import DataLoader, sampler

bucket = "test123data"
file_name = "iris.pt"

def accuracy(X, y):
    correct = 0
    for i in range(len(y)):
        y_predicted = int(np.sign((torch.dot(w, torch.Tensor(X[i])) - b).detach().numpy()[0]))
        if y_predicted == y[i]: correct += 1
    return float(correct)/len(y)

def get_data():
    """
    s3 = boto3.client('s3')
    return(s3.get_object(Bucket = bucket, Key = file_name))
    """
    return('iris.pt')
    
def handler(event=1,context=2):
    #get dataset
    iris = np.array(torch.load(get_data()))
    x_vals = iris[:,1:5]
    y_vals = np.array([1 if y == 0 else -1 for y in iris[:,0]])
    
    [num,col] = x_vals.shape

    
    #split training and test sets
    test_indices =  torch.arange(int(num/10)).numpy()
    train_indices = torch.tensor(list(set(torch.arange(len(x_vals)).numpy()) - set(test_indices)))
    X_train = x_vals[train_indices]
    X_test = x_vals[test_indices]
    y_train = y_vals[train_indices]
    y_test = y_vals[test_indices]
    
    batch_size = 100
    
    #initializing the model
    dim = len(X_train[0])
    w = torch.autograd.Variable(torch.rand(dim),requires_grad=True)
    b = torch.autograd.Variable(torch.rand(1),requires_grad=True)
    
    #training
    step_size = 1e-3
    num_epochs = 500
    minibatch_size = 20
    for epoch in range(num_epochs):
        inds = [i for i in range(X_train.shape[0])]
        sampler = sampler.RandomSampler(data_source = inds)
        inds = DataLoader(dataset = inds, sampler=sampler)

        for i in inds:
            #cost measurement for SVM max(0,1-ywx)
            L = max(0, 1 - y_train[i] * (torch.dot(w, torch.Tensor(X_train[i])) - b))
            #cost measurement for linear regression (y-wx)^2
            #L = (y_train[inds[i]]-(torch.dot(w, torch.Tensor(X_train[inds[i]])) - b))**2
    
            if L != 0: # if the loss is zero, Pytorch leaves the variables as a float 0.0, so we can't call backward() on it
                L.backward()
                w.data -= step_size * w.grad.data # step
                b.data -= step_size * b.grad.data # step
                w.grad.data.zero_()
                b.grad.data.zero_()
        #print(w.data)
    
    

    print([w.detach().numpy(),b.detach().numpy()])
    print('train accuracy', accuracy(X_train, y_train))
    print('test accuracy', accuracy(X_test, y_test))



handler()