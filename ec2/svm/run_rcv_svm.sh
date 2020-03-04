#!/bin/bash

world_size=$1
# echo $world_size
nr_cluster=$2
lr=$3
batch_size=$4

for ((i=0; i<world_size; i++)); do
        if [ $i == 0 ]
        then
                source /home/ubuntu/envs/pytorch/bin/activate
                nohup python3.6 /home/ubuntu/LambdaML/ec2/svm/rcv_svm.py --init-method tcp://172.31.42.34:24000 --rank 0 --batch-size $batch_size -l $lr --world-size $world_size --train-file /home/ubuntu/data/rcv --no-cuda > log${i}_${nr_cluster}.txt 2>&1 &
        else
                ssh sparse-node00$i "source /home/ubuntu/envs/pytorch/bin/activate; cd /home/ubuntu/LambdaML/ec2/svm/; nohup python3.6 rcv_svm.py --init-method tcp://172.31.42.34:24000 --rank $i --batch-size $batch_size -l $lr --world-size $world_size --train-file /home/ubuntu/data/rcv --no-cuda > log${i}_${nr_cluster}.txt 2>&1 &"
        fi
done