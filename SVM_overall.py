{\rtf1\ansi\ansicpg1252\cocoartf1671\cocoasubrtf500
{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww10800\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 #!/usr/bin/env python3\
# -*- coding: utf-8 -*-\
"""\
Created on Thu Oct  3 20:45:18 2019\
\
@author: liuyue\
"""\
\
import torch\
import boto3\
import numpy as np\
from torch.utils.data import DataLoader, sampler\
from io import BytesIO\
\
object_key = "iris.pt"\
bucket_name = "test123dataset"\
\
def accuracy(X, y, model):\
    correct = 0\
    _,col = X.shape\
    w = model[0:col]\
    b = model[-1]\
    for i in range(len(y)):\
        y_predicted = np.sign((torch.dot(w, torch.Tensor(X[i])) - b).detach().numpy())\
        if y_predicted == y[i]: correct += 1\
    return float(correct)/len(y)\
\
def get_data(object_key):\
    \
    client = boto3.client('s3')\
    csv_obj = client.get_object(Bucket=bucket_name, Key=object_key)\
    body = csv_obj['Body'].read()\
    return(BytesIO(body))\
    \
def handler(event,context):\
    #get dataset\
    iris = torch.load(get_data(object_key))\
    x_vals = iris[:,1:5]\
    y_vals = np.array([1 if y == 0 else -1 for y in iris[:,0]])\
    \
    [num,col] = x_vals.shape\
    print(num)\
    print(col)\
    \
    #split training and test sets\
    test_indices =  torch.arange(int(num/10)).numpy()\
    train_indices = torch.tensor(list(set(torch.arange(len(x_vals)).numpy()) - set(test_indices)))\
    X_train = x_vals[train_indices]\
    X_test = x_vals[test_indices]\
    y_train = y_vals[train_indices]\
    y_test = y_vals[test_indices]\
    \
    batch_size = 100\
    \
    #initializing the model\
    dim = len(X_train[0])\
    w = torch.autograd.Variable(torch.rand(dim),requires_grad=True)\
    b = torch.autograd.Variable(torch.rand(1),requires_grad=True)\
    \
    #training\
    step_size = 1e-4\
    num_epochs = 50 #running out of memeory when >=3\
    #minibatch_size = 20\
    for epoch in range(num_epochs):\
        inds = [i for i in range(X_train.shape[0])]\
        Sampler = sampler.RandomSampler(data_source = inds)\
        inds = DataLoader(dataset = inds, sampler=Sampler)\
        for i in inds:\
            #cost measurement for SVM max(0,1-ywx)\
            regularizer = (torch.sum(w**2)+b**2)*0.1\
            L = max(0, 1 - y_train[i] * (torch.dot(w, torch.Tensor(X_train[i])) - b))+regularizer\
            #cost measurement for linear regression (y-wx)^2\
            #L = (y_train[inds[i]]-(torch.dot(w, torch.Tensor(X_train[inds[i]])) - b))**2\
    \
            if L != 0: # if the loss is zero, Pytorch leaves the variables as a float 0.0, so we can't call backward() on it\
                L.backward()\
                w.data -= step_size * w.grad.data # step\
                b.data -= step_size * b.grad.data # step\
                w.grad.data.zero_()\
                b.grad.data.zero_()\
    \
    model = torch.Tensor(np.hstack((w.detach().numpy(),b.detach().numpy())))\
    print(model)\
    print('train accuracy', accuracy(X_train, y_train, model))\
    print('test accuracy', accuracy(X_test, y_test, model))\
    \
        \
    \
\
\
\
}