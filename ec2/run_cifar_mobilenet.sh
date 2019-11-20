rank=$1
world_size=$2
batch_size=$3

source ~/envs/pytorch/bin/activate

python cifar10_mobilenet_avg_grad.py --init-method tcp://172.31.1.2:24000 --rank $rank --world-size $world_size --root /home/ubuntu/code/data --no-cuda --batch-size $batch_size

