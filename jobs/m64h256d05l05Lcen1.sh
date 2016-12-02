
#!/bin/bash
#PBS -q isi
#PBS -l walltime=300:00:00
#PBS -l gpus=2:shared

source $NLGHOME/sh/init_tensorflow.sh
cd /home/nlg-05/xingshi/lstm/tensorflow/recsys/lstm/

data_part=/home/nlg-05/xingshi/lstm/tensorflow/recsys/data/data_part
data_full=/home/nlg-05/xingshi/lstm/tensorflow/recsys/data/data_full
train_dir=/home/nlg-05/xingshi/lstm/tensorflow/recsys/train/

python run.py --data_dir $data_part --batch_size 64 --size 256 --keep_prob 0.5 --learning_rate 0.5 --n_epoch 20 --loss ce --fulldata False --num_layers 1 --train_dir ${train_dir}/m64h256d05l05Lcen1
