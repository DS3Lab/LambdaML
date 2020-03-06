#!/bin/bash

world_size=$1
# echo $world_size
nr_cluster=$2

source /home/ubuntu/envs/pytorch/bin/activate
nohup python3.6 /home/ubuntu/LambdaML/ec2/rcv_kmeans.py --init-method tcp://172.31.44.193:24000 --rank 0 --communication all-reduce -k $nr_cluster --world-size $world_size --train-file /home/ubuntu/dataset/rcv --no-cuda > log.txt 2>&1 &
