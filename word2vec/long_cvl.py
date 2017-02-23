import os
import sys

head="""
#!/bin/bash

hostname
cd /nfs/isicvlnas01/users/liukuan/recsys/word2vec/
source /nfs/isicvlnas01/share/SGE_ROOT/default/common/settings.sh
export PATH="/nfs/isicvlnas01/share/SGE_ROOT/bin/linux-x64:/nfs/isicvlnas01/share/SGE_ROOT/bin/linux-x64:/nfs/gold/liukuan/anaconda2/bin:/nfs/isicvlnas01/share/anaconda/bin/:/usr/local/bin:/sbin:/usr/sbin:/usr/local/sbin:/usr/local/cuda/bin:/bin:/usr/bin"
export LD_LIBRARY_PATH="/nfs/isicvlnas01/share/cudnn-7.5-linux-x64-v5.0-ga/lib64/:/usr/local/lib:/usr/local/cuda/lib64:/nfs/isicvlnas01/share/intel/compilers_and_libraries_2016.1.150/linux/mkl/lib/intel64/:/nfs/isicvlnas01/share/SGE_ROOT/lib/linux-x64/:/nfs/isicvlnas01/share/boost_1_55_0/lib/:/nfs/isicvlnas01/share/nccl/lib:/nfs/isicvlnas01/share/opencv-2.4.9/lib/:/nfs/isicvlnas01/share/torch/install/lib:/nfs/isicvlnas01/share/cudnn-6.5-linux-x64-v2/"
echo $PATH
echo $LD_LIBRARY_PATH

echo $SGE_GPU
export CUDA_VISIBLE_DEVICES=$SGE_GPU

data_ml_part=/nfs/isicvlnas01/users/liukuan/recsys/data/data_ml_part
data_ml=/nfs/isicvlnas01/users/liukuan/recsys/data/data_ml
data_part=/nfs/isicvlnas01/users/liukuan/recsys/data/data_part
data_full=/nfs/isicvlnas01/users/liukuan/recsys/data/data_full
train_dir=/nfs/isicvlnas01/users/liukuan/recsys/train/
log_dir=/nfs/isicvlnas01/users/liukuan/recsys/train/log/

__cmd__
"""

def main():
    
    def data_dir(val):
        return "", "--data_dir {}".format(val)
    
    def train_dir(val):
        return "", "--train_dir {}".format(val)

    def batch_size(val):
        return "m{}".format(val), "--batch_size {}".format(val)

    def nonlinear(val):
        return "nl{}".format(val), "--nonlinear {}".format(val)

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

    def hidden_size(val):
        return "hs{}".format(val), "--hidden_size {}".format(val)

    def dataset(val):
        return "{}".format(val), "--dataset {}".format(val)

    def item_vocab_size(val):
        return "v{}".format(val), "--item_vocab_size {}".format(val)

    def ta(val):
        return "ta{}".format(val), "--ta {}".format(val)
    
    def n_i(val):
        if val >= 0:
            return 'ni{}'.format(val), '--ni {}'.format(val)
        elif val == -1:
            return 'ni-neg', '--ni {}'.format(val)

    def n_skip(val):
        return 'skip{}'.format(val), '--num_skips {}'.format(val)

    def win(val):
        return 'w{}'.format(val), '--skip_window {}'.format(val)        

    # whether or not to use separte embedding for input/output items
    def use_out_items(val):
        if val:
            return "oT", "--use_sep_item True"
        else:
            return "oF", "--use_sep_item False"


    funcs = [dataset, data_dir, batch_size, size, dropout, learning_rate, loss, ckpt, item_vocab_size, ta, n_i, n_skip, win, use_out_items]
    
    # ml
    # template = ["ml", "$data_ml", 64, 32, 0.5, 0.1, 'warp', 4000, 13000, 1024, 100, 0, 'random']
    # paras = []
    # _lr = [0.1, 0.2, 0.3, 0.5, 1, 2]
    # _size = [32]
    # for s in _size:
    #     for lr in _lr:
    #         temp = list(template)
    #         temp[3] = s
    #         temp[5] = lr
    #         paras.append(temp)


    # template = ["xing", "$data_part", 64, 32, 0.5, 3, 'ce', 4000, 50000, 1, -1, 3, 5, False]
    # _lr = [10, 8, 5, 2, 1, 0.5, 0.3]
    # _ni = [1, 2]
    # paras = []
    # for lr in _lr:
    #     for n in _ni:
    #         temp = list(template)
    #         temp[5] = lr
    #         temp[10] = n
    #         paras.append(temp)

    # template = ["xing", "$data_full", 64, 32, 0.5, 3, 'ce', 4000, 50000, 0, -1, 3, 5, False]
    # _lr = [10, 8, 5, 2, 1, 0.5, 0.3]
    # _ni = [1, 2]
    # paras = []
    # for lr in _lr:
    #     for n in _ni:
    #         temp = list(template)
    #         temp[5] = lr
    #         temp[10] = n
    #         paras.append(temp)
    
    # template = ["ml", "$data_ml", 64, 32, 0.5, 3,   'ce',   4000, 13000, 0, -1, 3, 5, False]
    # _lr = [10, 8, 5, 2, 1, 0.5, 0.3, 0.1]
    # _ni = [1, 2]
    # paras = []
    # for lr in _lr:
    #     for n in _ni:
    #         temp = list(template)
    #         temp[5] = lr
    #         temp[10] = n
    #         paras.append(temp)

    # template = ["xing", "$data_part", 64, 32, 0.5, 3, 'ce', 4000, 50000, 1, 1, 4, 5, False]
    # _lr = [5, 3]
    # _ni = [1, 2]
    # _w = [4, 6, 8]
    # paras = []
    # for lr in _lr:
    #     for w in _w:
    #         for n in _ni:
    #             temp = list(template)
    #             temp[5] = lr
    #             temp[12] = w
    #             temp[10] = n
    #             paras.append(temp)

    template = ["xing", "$data_part", 64, 64, 0.5, 3, 'ce', 4000, 50000, 1, -1, 3, 5, False]
    _lr = [10, 8, 5, 2, 1, 0.5, 0.3]
    _ni = [1, 2]
    paras = []
    for lr in _lr:
        for n in _ni:
            temp = list(template)
            temp[5] = lr
            temp[10] = n
            paras.append(temp)


    def get_name_cmd(para):
        name = ""
        cmd = []
        for func, para in zip(funcs,para):
            n, c = func(para)
            name += n
            cmd.append(c)
            
        name = name.replace(".",'')
        n, c = train_dir("${train_dir}/"+name)
        cmd.append(c)

        cmd = " ".join(cmd)
        return name, cmd

    # train
    cmds = ''
    # name = 'oFxing1h32'
    # name = 'oFxing0h32'
    # name = 'oFml0h32'
    # name = 'oFxing1h32windows'
    name = 'oFxing1h64'
    for para in paras:
        _, cmd = get_name_cmd(para)
        cmd = "/nfs/isicvlnas01/share/anaconda/bin/python run_sg.py " + cmd
        cmds += '\n\n' + cmd

    fn = "../jobs/sg_long_{}.sh".format(name)
    f = open(fn,'w')
    content = head.replace("__cmd__",cmds)
    f.write(content)
    f.close()
        
    # recommend
    batch_job_name = 'recommend_' + name
    cmds = ''
    for para in paras:
        name, cmd = get_name_cmd(para)
        cmd = "/nfs/isicvlnas01/share/anaconda/bin/python run_sg.py " + cmd + ' --recommend True'
        cmds += cmd + '\n'
        
    fn = "../jobs/sg_{}.sh".format(batch_job_name)
    f = open(fn,'w')
    content = head.replace("__cmd__",cmds)
    f.write(content)
    f.close()

if __name__ == "__main__":
    main()

    

    
    
