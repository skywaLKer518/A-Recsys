import os
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

def main():
    
    def data_dir(val):
        return "", "--data_dir {}".format(val)
    
    def train_dir(val):
        return "", "--train_dir {}".format(val)

    def batch_size(val):
        return "m{}".format(val), "--batch_size {}".format(val)

    def size(val):
        return "h{}".format(val), "--size {}".format(val)

    def dropout(val):
        return "d{}".format(val), "--keep_prob {}".format(val)

    def learning_rate(val):
        return "l{}".format(val), "--learning_rate {}".format(val)
    
    def n_resample(val):
        return "n_re{}".format(val), "--n_resample {}".format(val)
    def n_sampled(val):
        return "n_s{}".format(val), "--n_sampled {}".format(val)

    def loss(val):
        return "L{}".format(val), "--loss {}".format(val)

    def fulldata(val):
        return "", "--fulldata {}".format(val)

    def num_layers(val):
        return "n{}".format(val), "--num_layers {}".format(val)
    
    def logfile(val):
        return "", "--log {}".format(val)

    def ckpt(val):
        return "t{}".format(val), "--steps_per_checkpoint {}".format(val)

    funcs = [data_dir, batch_size, size, dropout, learning_rate, loss, ckpt, n_sampled, n_resample]
    
    paras = [
                ["$data_part", 64, 32, 1, 8, "warp", 4000, 1024, 100,],             
                ["$data_part", 64, 32, 0.95, 8, "warp", 4000, 1024, 100,],
                ["$data_part", 64, 32, 0.9, 8, "warp", 4000, 1024, 100,],
                ["$data_part", 64, 32, 0.8, 8, "warp", 4000, 1024, 100,],
                ["$data_part", 64, 32, 0.5, 8, "warp", 4000, 1024, 100,],
                ["$data_part", 64, 32, 0.3, 8, "warp", 4000, 1024, 100,],
             ]

    def get_name_cmd(para):
        name = "drop_"
        cmd = []
        for func, para in zip(funcs,para):
            n, c = func(para)
            name += n
            cmd.append(c)
            
        name = name.replace(".",'')
        n, c = train_dir("${train_dir}/"+name)
        cmd.append(c)
        n, c = logfile("${log_dir}/"+name)
        cmd.append(c)

        cmd = " ".join(cmd)
        return name, cmd

    for para in paras:
        name, cmd = get_name_cmd(para)
        cmd = "/nfs/isicvlnas01/share/anaconda/bin/python go2.py " + cmd
        fn = "../jobs/{}.sh".format(name)
        f = open(fn,'w')
        content = head.replace("__cmd__",cmd)
        f.write(content)
        f.close()
        

if __name__ == "__main__":
    main()

    

    
    
