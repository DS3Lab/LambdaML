#!/bin/bash
# ./run_svm_higgs_local_16.sh 16 6250 0.01 100 172.31.41.172:24000

world_size=$1

batch_size=$2

learning_rate=$3

epochs=$4

master_ip=$5

for ((i=0; i<$world_size; i++)); do
  source /home/ubuntu/envs/pytorch/bin/activate
  #cd /home/ubuntu/code/lambda/ec2
  log_file=$i"_"$world_size".log"
  rm -f logs/$log_file
  nohup python higgs_svm_avg_grad.py --init-method tcp://$master_ip --world-size $world_size --rank $i --root /home/ubuntu/dataset/higgs/ --no-cuda --batch-size $batch_size --learning-rate $learning_rate --epochs $epochs > logs/$log_file 2>&1 &
done
