#!/bin/bash

# default values
dfh=32
dflr=1
dfte=False
dfn=1000
dfrec=False

# hyper-parameters
h=${1:-$dfh}
lr=${2:-$dflr}
te=${3:-$dfte}
n=${4:-$dfn}
rec=${5:-$dfrec}

if [ $# -ne 5 ]
then 
		echo "Number of arguments should be 5!"
		echo "Usage: bash run_hmf.sh [model_size (e.g. 32)] [learning-rate (e.g. 1)] [test (True or False)] [num_epoch (e.g. 50)] [recommend (True or False)]"
		if [ $# -gt 5 ]
		then
				exit
		fi
		echo "Run with default values"
fi

if [ ! -d "./cache" ]
then
		mkdir ./cache
fi
if [ ! -d "./train" ]
then
		mkdir ./train
fi

cd ../hmf/

python run_hmf.py --dataset ml1m --raw_data ../examples/dataset/ --data_dir ../examples/cache/ml1m --train_dir ../examples/train/hmf_ml1m_h${h}lr${lr}te${te} --dataset ml1m --raw_data ../examples/dataset/ --item_vocab_size 3100  --vocab_min_thresh 1 --steps_per_checkpoint 300 --loss ce --learning_rate ${lr} --size $h --n_epoch $n --test ${te} --recommend ${rec}

echo 'finished!'
