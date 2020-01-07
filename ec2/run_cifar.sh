#!/bin/bash

world_size=$1
# echo $world_size

batch_size_local=$2
# echo $batch_size_local
# batch_size is the global batch size
batch_size_global=`expr $world_size \* $batch_size_local`

for ((i=0; i<$world_size; i++)); do
        if [ $i == 0 ]
        then
                source /home/ubuntu/envs/pytorch/bin/activate
                python3.6 /home/ubuntu/code_2.0/LambdaML/ec2/cifar10.py --init-method tcp://172.31.0.225:24000 --rank 0 --world-size $world_size --root /mnt --no-cuda --batch-size $batch_size_global &
        else
                ssh cluster-node00$i "source /home/ubuntu/envs/pytorch/bin/activate; cd /home/ubuntu/code_2.0/LambdaML/ec2/; python3.6 /home/ubuntu/code_2.0/LambdaML/ec2/cifar10.py --init-method tcp://172.31.0.225:24000 --rank $i --world-size $world_size --root /mnt --no-cuda --batch-size $batch_size_global" &
        fi
done

# if more than 10 nodes
# for ((i=0; i<$world_size; i++)); do
#         if [ $i -eq 0 ]; then
#             source /home/ubuntu/envs/pytorch/bin/activate
#             python3.6 /home/ubuntu/code_2.0/LambdaML/ec2/cifar10.py --init-method tcp://172.31.44.193:24000 --rank 0 --world-size $world_size --root /mnt --no-cuda --batch-size $batch_size_global &
#         elif [ $i -le 9 ]; then
#             ssh cluster20-node00$i "source /home/ubuntu/envs/pytorch/bin/activate; cd /home/ubuntu/code_2.0/LambdaML/ec2/; python3.6 /home/ubuntu/code_2.0/LambdaML/ec2/cifar10.py --init-method tcp://172.31.44.193:24000 --rank $i --world-size $world_size --root /mnt --no-cuda --batch-size $batch_size_global" &
#         else
#         	ssh cluster20-node0$i "source /home/ubuntu/envs/pytorch/bin/activate; cd /home/ubuntu/code_2.0/LambdaML/ec2/; python3.6 /home/ubuntu/code_2.0/LambdaML/ec2/cifar10.py --init-method tcp://172.31.44.193:24000 --rank $i --world-size $world_size --root /mnt --no-cuda --batch-size $batch_size_global" &
#         fi
# done
