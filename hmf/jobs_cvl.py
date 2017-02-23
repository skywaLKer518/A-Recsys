import os
import sys

head_cvl="""
#!/bin/bash

hostname
cd /nfs/isicvlnas01/users/liukuan/recsys/hmf/
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

head_hpc2="""
#!/bin/bash
#PBS -l walltime=23:59:59
#PBS -l nodes=1:ppn=16:gpus=2:shared
#PBS -M kuanl@usc.edu -p 1023


source /usr/usc/cuda/8.0/setup.sh
cd /home/rcf-proj/pn3/kuanl/recsys/hmf/

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

head_hpc3="""
#!/bin/bash
#PBS -q isi
#PBS -l walltime=300:00:00
#PBS -l nodes=1:ppn=16:gpus=2:shared


source /usr/usc/cuDNN/7.5-v5.1/setup.sh
source /usr/usc/cuda/8.0/setup.sh
# export CUDA_VISIBLE_DEVICES=$1
cd /home/nlg-05/xingshi/lstm/tensorflow/recsys/hmf/

data_part=/home/nlg-05/xingshi/lstm/tensorflow/recsys/data/data_part
data_full=/home/nlg-05/xingshi/lstm/tensorflow/recsys/data/data_full
data_full07=/home/nlg-05/xingshi/lstm/tensorflow/recsys/data/data_full0.7
data_full03=/home/nlg-05/xingshi/lstm/tensorflow/recsys/data/data_full0.3
data_ml=/home/nlg-05/xingshi/lstm/tensorflow/recsys/data/data_ml
data_ml1m=/home/nlg-05/xingshi/lstm/tensorflow/recsys/data/data_ml1m/
data_yelp=/home/nlg-05/xingshi/lstm/tensorflow/recsys/data/yelp/
train_dir=/home/nlg-05/xingshi/lstm/tensorflow/recsys/train/

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

    def sampling(val):
        if val == 'permute':
            return 'sa_p', '--sample_type {}'.format(val)
        elif val == 'random':
            return 'sa_r', '--sample_type {}'.format(val)

    def wFeat(val):
        if val:
            return '', ''
        else:
            return 'woF', '--use_user_feature False --use_item_feature False'

    def test(val):
        if val:
            return 't_t', '--test True'
        else:
            return '', ''

    def epoch(val):
        return 'n{}'.format(val), '--n_epoch {}'.format(val)

    # XING
    # funcs = [dataset, data_dir, batch_size, size, dropout, learning_rate, loss, ckpt, item_vocab_size, ta, epoch, test]
    # template = ["xing", "$data_full", 64, 32, 0.5, 20, 'ce', 4000, 50000, 0, 33, True]

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

    # template = ["xing", "$data_part", 64, 32, 0.5, 0.1, 'warp', 4000, 50000, 1024, 100, 1, 'random']
    
    funcs = [dataset, data_dir, batch_size, size, dropout, learning_rate, loss, ckpt, item_vocab_size, ta, epoch, test]
    template = ["ml1m", "$data_ml1m", 64, 32, 0.5, 0.1, 'warp', 2000, 3100, 0, 307, True]

    # yelp
    # funcs = [dataset, data_dir, batch_size, size, dropout, learning_rate, loss, ckpt, item_vocab_size,  ta]
    # template = ["yelp", "$data_yelp", 64, 32, 0.5, 0.1, 'ce', 4000, 80000, 0]

    # yelp no features
    # funcs = [dataset, data_dir, batch_size, size, dropout, learning_rate, loss, ckpt, item_vocab_size,  ta, wFeat]
    # template = ["yelp", "$data_yelp", 64, 32, 0.5, 0.1, 'ce', 4000, 80000, 0, False]

    # yelp test warp
    # funcs = [dataset, data_dir, batch_size, size, dropout, learning_rate, loss, ckpt, item_vocab_size,  ta, epoch, test]
    # template = ["yelp", "$data_yelp", 64, 32, 0.5, 0.1, 'ce', 4000, 80000, 0, 26, True]    
    
    _lr = [0.1]
    _s = [32]
    _loss = ['ce']
    paras = []
    for s in _s:
        for lr in _lr:
            for L in _loss:
                temp = list(template)
                temp[5] = lr
                temp[3] = s
                temp[6] = L
                paras.append(temp)


    def get_name_cmd(para):
        name = "hmf_"
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
        head = head_hpc2 # kuanl
    else:
        head = head_hpc3 # xing hpc

    # train
    for para in paras:
        name, cmd = get_name_cmd(para)
        if acct == 0:
            cmd = "/nfs/isicvlnas01/share/anaconda/bin/python run_hmf.py " + cmd
        else:
            cmd = "python run_hmf.py " + cmd

        fn = "../jobs/{}.sh".format(name)
        f = open(fn,'w')
        content = head.replace("__cmd__",cmd)
        f.write(content)
        f.close()
        
    # recommend
    # batch_job_name = 'xing_recommend_h32_compare_sampling'
    # batch_job_name = 'ml1m_recommend_ce_h_lr'
    # batch_job_name = 'ml1m_recommend_loss_h16_lr'
    # batch_job_name = 'yelp_hmf_recommend_ce_h32_lr'
    # batch_job_name = 'yelp_hmf_recommend_warp_h_lr'
    # batch_job_name = 'yelp_hmf_recommend_ce_h16_lr'
    # batch_job_name = 'yelp_hmf_recommend_ce_h_lr_wof'
    # batch_job_name = 'yelp_hmf_recommend_warp_test1'
    # batch_job_name = 'xing_hmf_recommend_warp_test1'
    # batch_job_name = 'xing_hmf_recommend_ce_test1'
    # batch_job_name = 'ml1m_hmf_recommend_warp_test1'
    batch_job_name = 'ml1m_hmf_recommend_ce_test1'

    cmds = ''
    for para in paras:
        name, cmd = get_name_cmd(para)
        if acct == 0:
            cmd = "/nfs/isicvlnas01/share/anaconda/bin/python run_hmf.py " + cmd + ' --recommend True'
        else:
            cmd = "python run_hmf.py " + cmd + ' --recommend True'
        cmds += cmd + '\n'
        
    fn = "../jobs/{}.sh".format(batch_job_name)
    f = open(fn,'w')
    content = head.replace("__cmd__",cmds)
    content = content.replace("23:59:59", "0:59:59")
    f.write(content)
    f.close()

if __name__ == "__main__":

    if len(sys.argv) > 1 and int(sys.argv[1]) == 1: # hpc Kuan
        main(1)
    elif len(sys.argv) > 1 and int(sys.argv[1]) == 2: # hpc xing
        main(2)
    elif len(sys.argv) > 1 and int(sys.argv[1]) == 3: # i80
        main(3)        
    else:
        main()    

    
    
