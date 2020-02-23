{\rtf1\ansi\ansicpg1252\cocoartf1671\cocoasubrtf500
{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww10800\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 #!/usr/bin/env python3\
# -*- coding: utf-8 -*-\
"""\
Created on Thu Oct  8 22:58:01 2019 \
\
@author: liuyue\
"""\
import boto3\
from torch import load, save, Tensor, rand\
from torch.utils.data import DataLoader, sampler\
from torch.autograd import Variable\
from numpy import hstack\
\
object_key = "iris.pt"\
bucket_name = "test123dataset"\
\
class s3_handler():\
    def __init__(self):\
        self.client = boto3.client('s3')\
        \
    def get_data(self, object_key, bucket_name):\
        client = self.client\
        download_path = '/tmp/\{\}'.format(object_key)\
        client.download_file(bucket_name, object_key, download_path)\
        self.bucket = bucket_name\
        return(download_path)\
        \
    def send_data(self, key, upload_path):\
        # Put the object\
        client = self.client\
        client.upload_file(upload_path, self.bucket, key )\
        return True\
    \
       \
\
def handler(event,context):\
    s3 = s3_handler()\
    iris = load(s3.get_data(object_key, bucket_name))\
    [num, col] = iris.shape\
    index = 0\
    batch_size = 25\
    inds = [i for i in range(iris.shape[0])]\
    Sampler = sampler.RandomSampler(data_source = inds)\
    inds = [i for i in DataLoader(dataset = inds, sampler=Sampler)]\
    for i in range(0,num,batch_size):\
        if i+batch_size>=num:\
            temp = iris[inds[i:num],:]\
        else:\
            temp = iris[inds[i:i+batch_size],:]\
        file_name = "svm_"+str(index)+".pt"\
        upload_path = '/tmp/\{\}'.format(file_name)\
        save(temp,upload_path)\
        s3.send_data(file_name,upload_path)\
        index = index + 1\
    \
    w = Variable(rand(col-1),requires_grad=True)\
    b = Variable(rand(1),requires_grad=True)\
    target_model = Tensor(hstack((w.detach().numpy(),b.detach().numpy())))\
    state = \{'index':0, 'model':target_model,'cost':0\}\
    file_name = "target_model.pt"\
    upload_path = '/tmp/\{\}'.format(file_name)\
    save(state,upload_path)\
    s3.send_data(file_name,upload_path)\
    \
    }