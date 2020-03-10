#!/bin/bash

world_size=$1
lr=$2
batch_size=$3
master_ip=$4

for ((i=0; i<world_size; i++)); do
  source /home/ubuntu/envs/pytorch/bin/activate
  nohup python3.6 /home/ubuntu/LambdaML/ec2/svm/rcv_svm.py --init-method "tcp://${master_ip}" --rank $i --batch-size $batch_size -l $lr --world-size $world_size --train-file /home/ubuntu/dataset/rcv --no-cuda > "/home/ubuntu/log/svm_local_${i}_${world_size}_${batch_size}_${lr}.txt" 2>&1 &
done
