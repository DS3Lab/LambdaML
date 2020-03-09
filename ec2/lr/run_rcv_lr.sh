#!/bin/bash

world_size=$1
lr=$2
batch_size=$3
master_ip=$4
cluster_name=$5

train_path="/home/ubuntu/dataset/rcv${world_size}"

for ((i=0; i<world_size; i++)); do
        if [ $i == 0 ]
        then
                source /home/ubuntu/envs/pytorch/bin/activate
                nohup python3.6 /home/ubuntu/LambdaML/ec2/lr/rcv_lr.py --init-method "tcp://${master_ip}" --rank 0 --batch-size $batch_size -l $lr --world-size $world_size --train-file "${train_path}/0_${world_size}" --no-cuda > "/home/ubuntu/log/lr_${i}_${world_size}_${batch_size}_${lr}.txt" 2>&1 &
        else
                ssh "${cluster_name}-node00${i}" "source /home/ubuntu/envs/pytorch/bin/activate; cd /home/ubuntu/LambdaML/ec2/lr/; nohup python3.6 rcv_lr.py --init-method tcp://172.31.42.34:24000 --rank $i --batch-size $batch_size -l $lr --world-size $world_size --train-file "${train_path}/${i}_${world_size}" --no-cuda > /home/ubuntu/log/lr_${i}_${world_size}_${batch_size}_${lr}.txt 2>&1 &"
        fi
done

