import os
import sys

head_cvl="""
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
data_ml1m=/nfs/isicvlnas01/users/liukuan/recsys/data/data_ml1m/
data_yelp=/nfs/isicvlnas01/users/liukuan/recsys/data/yelp/
train_dir=/nfs/isicvlnas01/users/liukuan/recsys/train/
log_dir=/nfs/isicvlnas01/users/liukuan/recsys/train/log/

__cmd__
"""

head_hpck="""
#!/bin/bash
#PBS -l walltime=23:59:59
#PBS -l nodes=1:ppn=16:gpus=2:shared
#PBS -M kuanl@usc.edu -p 1023


source /usr/usc/cuda/8.0/setup.sh
cd /home/rcf-proj/pn3/kuanl/recsys/word2vec/

data_part=/home/rcf-proj/pn3/kuanl/recsys/data/data_part/
data_full=/home/rcf-proj/pn3/kuanl/recsys/data/data_full/
data_full07=/home/rcf-proj/pn3/kuanl/recsys/data/data_full0.7/
data_full03=/home/rcf-proj/pn3/kuanl/recsys/data/data_full0.3/
data_ml=/home/rcf-proj/pn3/kuanl/recsys/data/data_ml/
data_ml1m=/home/rcf-proj/pn3/kuanl/recsys/data/data_ml1m/
data_yelp=/home/rcf-proj/pn3/kuanl/recsys/data/yelp/
train_dir=/home/rcf-proj/pn3/kuanl/recsys/train/

__cmd__
"""
def main(acct=0):
    
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

    def after40(val):
        if val:
            return 'gt40', '--after40 True'
        else:
            return '', '--after40 False'

    def output_feat(val):
        return 'oa{}'.format(val), '--output_feat {}'.format(val)

    def test(val):
        if val:
            return 't_t', '--test True'
        else:
            return '', ''

    def epoch(val):
        return 'n{}'.format(val), '--n_epoch {}'.format(val)

    # separate embedding
    # funcs = [dataset, data_dir, batch_size, size, dropout, learning_rate, loss, ckpt, item_vocab_size, ta, n_i, n_skip, win, use_out_items]
    # template = ["xing", "$data_full", 64, 32, 0.5, 3, 'ce', 4000, 50000, 0, 1, 3, 5, True]

    # after 40
    # funcs = [dataset, data_dir, batch_size, size, dropout, learning_rate, loss, ckpt, item_vocab_size, ta, n_i, n_skip, win, use_out_items, after40]
    # template = ["xing", "$data_full", 64, 32, 0.5, 3, 'ce', 4000, 50000, 0, 1, 3, 5, True, True]    

    # after 40, ni > 1
    # funcs = [dataset, data_dir, batch_size, size, dropout, learning_rate, loss, ckpt, item_vocab_size, ta, n_i, n_skip, win, use_out_items, after40]
    # template = ["xing", "$data_full", 64, 32, 0.5, 3, 'ce', 4000, 50000, 0, 1, 3, 5, True, True]    

    # after 40, ni > 1, test
    # funcs = [dataset, data_dir, batch_size, size, dropout, learning_rate, loss, ckpt, item_vocab_size, ta, n_i, n_skip, win, use_out_items, after40, epoch, test]
    # template = ["xing", "$data_full", 64, 32, 0.5, 3, 'ce', 4000, 50000, 0, 1, 3, 5, True, True, 70, True]    

    # yelp
    funcs = [dataset, data_dir, batch_size, size, dropout, learning_rate, loss, ckpt, item_vocab_size, ta, n_i, n_skip, win, use_out_items, epoch, test]
    template = ["yelp", "$data_yelp", 64, 32, 0.5, 3, 'ce', 4000, 80000, 0, 2, 3, 8, False, 1000, False]

    # ml1m
    # funcs = [dataset, data_dir, batch_size, size, dropout, learning_rate, loss, ckpt, item_vocab_size, ta, n_i, n_skip, win, use_out_items, epoch, test]
    # template = ["ml1m", "$data_ml1m", 64, 32, 0.5, 1, 'ce', 4000, 3100, 0, 1, 3, 5, True, 94, True]    

    # output feature combination
    # funcs = [dataset, data_dir, batch_size, size, dropout, learning_rate, loss, ckpt, item_vocab_size, ta, n_i, n_skip, win, use_out_items, output_feat]
    # template = ["xing", "$data_full", 64, 32, 0.5, 3, 'ce', 4000, 50000, 0, 1, 3, 5, True, 2]    

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

    # ml_part
    # template = ["ml", "$data_ml_part", 64, 32, 0.5, 0.1, 'ce', 4000, 6000, 1024, 100, 1, 'random']


    # _lr = [0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 5, 8]
    # _size = [32]
    # for s in _size:
    #     for lr in _lr:
    #         temp = list(template)
    #         temp[3] = s
    #         temp[5] = lr
    #         paras.append(temp)

    # template = ["xing", "$data_part", 64, 32, 0.5, 3, 'ce', 4000, 50000, 1, -1, 3, 5, True]
    # _lr = [3]
    # _ni = [2,3,4,5,6]
    # paras = []
    # for n in _ni:
    #     for lr in _lr:
    #         temp = list(template)
    #         temp[5] = lr
    #         temp[10] = n
    #         paras.append(temp)

    # template = ["xing", "$data_part", 64, 32, 0.5, 3, 'ce', 4000, 50000, 1, -1, 3, 5, False]
    # _lr = [3]
    # _ni = [1,2,3,4,5]
    # paras = []
    # for n in _ni:
    #     for lr in _lr:
    #         temp = list(template)
    #         temp[5] = lr
    #         temp[10] = n
    #         paras.append(temp)

    # separate embedding
    # template = ["xing", "$data_full", 64, 32, 0.5, 3, 'ce', 4000, 50000, 0, 1, 3, 5, True]
    # _lr = [8, 5, 3, 1, 0.5, 0.3]
    # _ni = [1]
    # paras = []
    # for n in _ni:
    #     for lr in _lr:
    #         temp = list(template)
    #         temp[5] = lr
    #         temp[10] = n
    #         paras.append(temp)

    # yelp
    _lr = [1, 3]
    _ni = [1,2,3]
    _h = [32]

    # ml1m
    # _lr = [0.5]
    # _ni = [2]
    # _h = [32]
    paras = []
    for n in _ni:
        for lr in _lr:
            for h in _h:

                temp = list(template)
                temp[5] = lr
                temp[3] = h
                temp[10] = n
                paras.append(temp)

    # ml
    # funcs = [dataset, data_dir, batch_size, size, dropout, learning_rate, loss, ckpt, item_vocab_size, ta, n_i, n_skip, win, use_out_items]
    # template_ml = ["ml", "$data_ml", 64, 32, 0.5, 0.3, 'ce', 4000, 13000, 0, 1, 3, 5, True]    

    # _lr = [0.3, 0.5, 1]
    # _ni = [1, 2]
    # _h = [32]
    # paras = []
    # for n in _ni:
    #     for lr in _lr:
    #         for h in _h:

    #             temp = list(template_ml)
    #             temp[5] = lr
    #             temp[3] = h
    #             temp[10] = n
    #             paras.append(temp)

    def get_name_cmd(para):
        name = "sg2_"
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


    if acct == 0: # isicvl
        head = head_cvl
    elif acct == 1:
        head = head_hpck # xing hpc
    

    # train
    for para in paras:
        name, cmd = get_name_cmd(para)
        if acct == 0:
            cmd = "/nfs/isicvlnas01/share/anaconda/bin/python run_sg.py " + cmd
        else:
            cmd = "python run_sg.py " + cmd
        fn = "../jobs/{}.sh".format(name)
        f = open(fn,'w')
        content = head.replace("__cmd__",cmd)
        f.write(content)
        f.close()
        
    # recommend
    # batch_job_name = 'sg2_xing_recommend_h32_oT_g40_ni'
    # batch_job_name = 'sg2_ml_recommend_h32_oT_ni'
    # batch_job_name = 'sg2_xing_recommend_test1'
    # batch_job_name = 'sg2_yelp_recommend_oF_ni'
    # batch_job_name = 'sg2_ml1m_recommend_oT_ni'
    # batch_job_name = 'sg2_ml1m_recommend_test1'
    # batch_job_name = 'sg2_yelp_recommend_test1'
    # batch_job_name = 'sg2_yelp_recommend_oF_ni2'
    batch_job_name = 'sg2_yelp_recommend_oF_win'
    cmds = ''
    for para in paras:
        name, cmd = get_name_cmd(para)
        if acct == 0:
            cmd = "/nfs/isicvlnas01/share/anaconda/bin/python run_sg.py " + cmd + ' --recommend True'
        else:
            cmd = "python run_sg.py " + cmd + ' --recommend True'            
        cmds += cmd + '\n'
        
    fn = "../jobs/{}.sh".format(batch_job_name)
    f = open(fn,'w')
    content = head.replace("__cmd__",cmds)
    content = content.replace("23:59:59", "0:59:59")
    f.write(content)
    f.close()

if __name__ == "__main__":
    if len(sys.argv) > 1 and int(sys.argv[1]) == 1: # hpc kuan
        main(1)
    else:
        main()    

    

    
    
