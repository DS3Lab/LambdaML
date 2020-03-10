#!/bin/bash

world_size=$1
nr_cluster=$2
# master_ip=$3

train_path="/home/ubuntu/dataset/rcv${world_size}"

if [ $world_size == 1 ]
then
  source /home/ubuntu/envs/pytorch/bin/activate
  nohup python3.6 /home/ubuntu/LambdaML/ec2/kmeans/rcv_kmeans.py --rank 0 --communication all-reduce -k $nr_cluster --world-size $world_size --train-file /home/ubuntu/dataset/rcv --no-cuda > "/home/ubuntu/log/kmeans_single_${nr_cluster}.txt" 2>&1 &
else
  for ((i=0; i<world_size; i++)); do
    source /home/ubuntu/envs/pytorch/bin/activate
    nohup python3.6 /home/ubuntu/LambdaML/ec2/kmeans/rcv_kmeans.py --rank $i --communication all-reduce -k $nr_cluster --world-size $world_size --train-file "${train_path}/${i}_${world_size}" --no-cuda > "/home/ubuntu/log/kmeans_${i}_t${world_size}_k${nr_cluster}.txt" 2>&1 &
  done
fi
