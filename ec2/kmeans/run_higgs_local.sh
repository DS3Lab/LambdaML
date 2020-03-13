#!/bin/bash

world_size=$1
nr_cluster=$2
# master_ip=$3

train_path="/home/ubuntu/dataset/higgs${world_size}"

source /home/ubuntu/envs/pytorch/bin/activate
if [ $world_size == 1 ]
then
  nohup python3.6 /home/ubuntu/LambdaML/ec2/kmeans/higgs_kmeans.py --rank 0 --communication all-reduce -k $nr_cluster --world-size 1 --train-file /home/ubuntu/dataset/higgs --no-cuda >> "/home/ubuntu/log/kmeans_higgs_single_${nr_cluster}.txt" 2>&1 &
else
  for ((i=1; i<$world_size; i++)); do
    nohup python3.6 /home/ubuntu/LambdaML/ec2/kmeans/higgs_kmeans.py --rank $i --communication all-reduce -k $nr_cluster --world-size $world_size --train-file "${train_path}/${i}_${world_size}" --no-cuda >> "/home/ubuntu/log/higgs_kmeans_r${i}_w${world_size}_k${nr_cluster}.txt" 2>&1 & 
  done
fi
