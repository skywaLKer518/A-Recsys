#!/bin/bash

# default values
dfh=32
dflr=1
dw=5
dni=3
dfte=False
dfn=1000
dfrec=False
dmodel=cbow
dskips=3
# hyper-parameters
h=${1:-$dfh}
lr=${2:-$dflr}
w=${3:-$dw}
ni=${4:-$dni}
te=${5:-$dfte}
n=${6:-$dfn}
rec=${7:-$dfrec}

if [ $# -ne 7 ]
then 
		echo "Number of arguments should be 5!"
		echo "Usage: bash run_w2v.sh [model_size (e.g. 32)] [learning-rate (e.g. 1)] [window size (e.g. 5)] [ni (e.g. 3)] [test (True or False)] [num_epoch (e.g. 50)] [recommend (True or False)]"
		if [ $# -gt 7 ]
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

cd ../word2vec/

python run_w2v.py --model ${dmodel} --dataset ml1m --raw_data ../examples/dataset/ --data_dir ../examples/cache/ml1m --train_dir ../examples/train/${dmodel}_ml1m_h${h}lr${lr}w${w}ni${ni}n${n}te${te} --dataset ml1m --raw_data ../examples/dataset/ --item_vocab_size 3100  --vocab_min_thresh 1 --steps_per_checkpoint 300 --loss ce --learning_rate ${lr} --size $h --n_epoch $n --skip_window $w --ni ${ni} --num_skips ${dskips} --test ${te} --recommend ${rec}

echo 'finished!'
