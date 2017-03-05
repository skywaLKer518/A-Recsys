#!/bin/bash

# default values
dfh=64
dflr=1
dfrec=False

# hyper-parameters
h=${1:-$dfh}
lr=${2:-$dflr}
rec=${3:-$dfrec}

if [ $# -ne 3 ]
then 
		echo "Number of arguments should be 3!"
		echo "Usage: bash run_hmf.sh [model_size (e.g. 32)] [learning-rate (e.g. 1)] [recommend (True or False)]"
		if [ $# -gt 3 ]
		then
				echo "too many arguments. exit."
				exit
		fi
		echo "Not enough arguments. Run with default values"
fi

if [ ! -d "./cache" ]
then
		mkdir ./cache
fi
if [ ! -d "./train" ]
then
		mkdir ./train
fi

cd ../lstm/

python run.py --dataset ml1m --raw_data ../examples/dataset/ --data_dir ../examples/cache/ml1m --train_dir ../examples/train/lstm_ml1m_h${h}lr${lr} --dataset ml1m --raw_data ../examples/dataset/ --item_vocab_size 3100  --vocab_min_thresh 1 --steps_per_checkpoint 300 --loss ce --learning_rate ${lr} --size $h  --recommend ${rec}


echo 'finished!'
