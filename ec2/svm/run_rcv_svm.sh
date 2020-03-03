#!/bin/bash

world_size=$1
# echo $world_size
nr_cluster=$2

for ((i=0; i<world_size; i++)); do
        if [ $i == 0 ]
        then
                source /home/ubuntu/envs/pytorch/bin/activate
                nohup python3.6 /home/ubuntu/fl/LambdaML/ec2/svm/rcv_svm.py --init-method tcp://172.31.44.193:24000 --rank 0 --batch-size 1000 -lr 0.01 --world-size $world_size --train-file /home/ubuntu/data/rcv --no-cuda > log${i}_${nr_cluster}.txt 2>&1 &
        else
                ssh cluster20-node00$i "source /home/ubuntu/envs/pytorch/bin/activate; cd /home/ubuntu/fl/LambdaML/ec2/svm/; nohup python3.6 rcv_svm.py --init-method tcp://172.31.44.193:24000 --rank $i --batch-size 1000 -lr 0.01 --world-size $world_size --train-file /home/ubuntu/data/rcv --no-cuda > log${i}_${nr_cluster}.txt 2>&1 &"
        fi
done

