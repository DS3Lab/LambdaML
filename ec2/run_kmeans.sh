#!/bin/bash

world_size=$1
# echo $world_size
features=$2

for ((i=0; i<$world_size; i++)); do
        if [ $i == 0 ]
        then
                source /home/ubuntu/envs/pytorch/bin/activate
                python3.6 /home/ubuntu/code/LambdaML/ec2/agaricus_kmeans.py --init-method tcp://172.31.1.2:24000 --rank 0 -k 20 --features 127 --world-size $world_size --train-file /home/ubuntu/code/LambdaML/dataset/agaricus_127d_train.libsvm --test-file /home/ubuntu/code/LambdaML/dataset/agaricus_127d_test.libsvm --no-cuda &
        else
                ssh slave$i "source /home/ubuntu/envs/pytorch/bin/activate; python3.6 /home/ubuntu/code/LambdaML/ec2/agaricus_kmeans.py --init-method tcp://172.31.1.2:24000 --rank $i -k 20 --features 127 --world-size $world_size --train-file /home/ubuntu/code/LambdaML/dataset/agaricus_127d_train.libsvm --test-file /home/ubuntu/code/LambdaML/dataset/agaricus_127d_test.libsvm --no-cuda " &
        fi
done
