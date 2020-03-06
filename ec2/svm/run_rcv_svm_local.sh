#!/bin/bash

world_size=$1
lr=$2
batch_size=$3

source /home/ubuntu/envs/pytorch/bin/activate
nohup python3.6 /home/ubuntu/LambdaML/ec2/svm/rcv_svm.py --init-method tcp://172.31.42.34:24000 --rank 0 --batch-size $batch_size -l $lr --world-size $world_size --train-file /home/ubuntu/dataset/rcv --no-cuda > log_local.txt 2>&1 &
