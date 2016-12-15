from os import listdir, mkdir, path, rename
from os.path import isfile, join
import sys

head="""
#!/bin/bash

hostname
cd /nfs/isicvlnas01/users/liukuan/recsys/mf/
source /nfs/isicvlnas01/share/SGE_ROOT/default/common/settings.sh
export PATH="/nfs/isicvlnas01/share/SGE_ROOT/bin/linux-x64:/nfs/isicvlnas01/share/SGE_ROOT/bin/linux-x64:/nfs/gold/liukuan/anaconda2/bin:/nfs/isicvlnas01/share/anaconda/bin/:/usr/local/bin:/sbin:/usr/sbin:/usr/local/sbin:/usr/local/cuda/bin:/bin:/usr/bin"
export LD_LIBRARY_PATH="/nfs/isicvlnas01/share/cudnn-7.5-linux-x64-v5.0-ga/lib64/:/usr/local/lib:/usr/local/cuda/lib64:/nfs/isicvlnas01/share/intel/compilers_and_libraries_2016.1.150/linux/mkl/lib/intel64/:/nfs/isicvlnas01/share/SGE_ROOT/lib/linux-x64/:/nfs/isicvlnas01/share/boost_1_55_0/lib/:/nfs/isicvlnas01/share/nccl/lib:/nfs/isicvlnas01/share/opencv-2.4.9/lib/:/nfs/isicvlnas01/share/torch/install/lib:/nfs/isicvlnas01/share/cudnn-6.5-linux-x64-v2/"
echo $PATH
echo $LD_LIBRARY_PATH

echo $SGE_GPU
export CUDA_VISIBLE_DEVICES=$SGE_GPU

data_part=/nfs/isicvlnas01/users/liukuan/recsys/mf/data_part
data_full=/nfs/isicvlnas01/users/liukuan/recsys/mf/data_full
train_dir=/nfs/isicvlnas01/users/liukuan/recsys/mf/
log_dir=/nfs/isicvlnas01/users/liukuan/recsys/mf/log/

__cmd__
"""
import re

def main(ta=1):
  assert(ta == 1)  
  src_dir = './models_ta' + str(ta) + '_mf_drops/'

  dirs = [join(src_dir, f) for f in listdir(src_dir) if f.startswith('drop')]
  cmd_all = '--recommend True --batch_size 64 --size 32 --n_sampled 1024 --log ' + 'log_rec' + str(ta) + '_drop '

  cmd_all += '--data_dir $data_part --ta 1 '
  cmd_all = '/nfs/isicvlnas01/share/anaconda/bin/python go2.py ' + cmd_all


  cmds = [cmd_all + ' --train_dir ' + f for f in dirs]

  cmd = '\n'.join(cmds)
  content = head.replace("__cmd__", cmd)
  fn = "rec_ta{}_drop.sh".format(ta)
  f = open(fn, 'w')
  f.write(content)
  f.close()

ta = int(sys.argv[1])
main(ta)

