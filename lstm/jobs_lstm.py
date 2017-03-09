import os
import sys

head_hpc1_tf9="""
#!/bin/bash
#PBS -q isi
#PBS -l walltime=300:00:00
#PBS -l nodes=1:ppn=16:gpus=2:shared

source $NLGHOME/sh/init_tensorflow.sh

# export CUDA_VISIBLE_DEVICES=$1
cd /home/nlg-05/xingshi/lstm/tensorflow/recsys/lstm/

data_xing_mix=/home/nlg-05/xingshi/lstm/tensorflow/recsys/cache/xing_mix/
data_xing_het=/home/nlg-05/xingshi/lstm/tensorflow/recsys/cache/xing_het/
data_yelp_mix=/home/nlg-05/xingshi/lstm/tensorflow/recsys/cache/yelp_mix/
data_yelp_het=/home/nlg-05/xingshi/lstm/tensorflow/recsys/cache/yelp_het/

raw_xing=/home/nlg-05/xingshi/lstm/tensorflow/recsys/raw_data/xing/
raw_yelp=/home/nlg-05/xingshi/lstm/tensorflow/recsys/raw_data/yelp/

train_dir=/home/nlg-05/xingshi/lstm/tensorflow/recsys/train/

__cmd__
"""


head_hpc1="""
#!/bin/bash
#PBS -q isi
#PBS -l walltime=300:00:00
#PBS -l nodes=1:ppn=16:gpus=2:shared

source /usr/usc/cuDNN/7.5-v5.1/setup.sh
source /usr/usc/cuda/8.0/setup.sh

# export CUDA_VISIBLE_DEVICES=$1
cd /home/nlg-05/xingshi/lstm/tensorflow/recsys/lstm/

data_xing_mix=/home/nlg-05/xingshi/lstm/tensorflow/recsys/cache/xing_mix/
data_xing_het=/home/nlg-05/xingshi/lstm/tensorflow/recsys/cache/xing_het/
data_yelp_mix=/home/nlg-05/xingshi/lstm/tensorflow/recsys/cache/yelp_mix/
data_yelp_het=/home/nlg-05/xingshi/lstm/tensorflow/recsys/cache/yelp_het/

raw_xing=/home/nlg-05/xingshi/lstm/tensorflow/recsys/raw_data/xing/
raw_yelp=/home/nlg-05/xingshi/lstm/tensorflow/recsys/raw_data/yelp/

train_dir=/home/nlg-05/xingshi/lstm/tensorflow/recsys/train/

__cmd__
"""

head_hpc2="""
#!/bin/bash
#PBS -l walltime=23:59:59
#PBS -l nodes=1:ppn=16:gpus=2:shared
#PBS -M kuanl@usc.edu -p 1023


source /usr/usc/cuda/8.0/setup.sh
cd /home/rcf-proj/pn3/kuanl/recsys/lstm/

data_xing_mix=/home/rcf-proj/pn3/kuanl/recsys/cache/xing_mix/
data_xing_het=/home/rcf-proj/pn3/kuanl/recsys/cache/xing_het/
data_yelp_mix=/home/rcf-proj/pn3/kuanl/recsys/cache/yelp_mix/
data_yelp_het=/home/rcf-proj/pn3/kuanl/recsys/cache/yelp_het/

raw_xing=/home/rcf-proj/pn3/kuanl/recsys/raw_data/xing/
raw_yelp=/home/rcf-proj/pn3/kuanl/recsys/raw_data/yelp/

train_dir=/home/rcf-proj/pn3/kuanl/recsys/train/

__cmd__
"""

head_hpc3="""
#!/bin/bash

hostname
cd /nfs/isicvlnas01/users/liukuan/recsys/lstm/
source /nfs/isicvlnas01/share/SGE_ROOT/default/common/settings.sh
export PATH="/nfs/isicvlnas01/share/SGE_ROOT/bin/linux-x64:/nfs/isicvlnas01/share/SGE_ROOT/bin/linux-x64:/nfs/gold/liukuan/anaconda2/bin:/nfs/isicvlnas01/share/anaconda/bin/:/usr/local/bin:/sbin:/usr/sbin:/usr/local/sbin:/usr/local/cuda/bin:/bin:/usr/bin"
export LD_LIBRARY_PATH="/nfs/isicvlnas01/share/cudnn-7.5-linux-x64-v5.0-ga/lib64/:/usr/local/lib:/usr/local/cuda/lib64:/nfs/isicvlnas01/share/intel/compilers_and_libraries_2016.1.150/linux/mkl/lib/intel64/:/nfs/isicvlnas01/share/SGE_ROOT/lib/linux-x64/:/nfs/isicvlnas01/share/boost_1_55_0/lib/:/nfs/isicvlnas01/share/nccl/lib:/nfs/isicvlnas01/share/opencv-2.4.9/lib/:/nfs/isicvlnas01/share/torch/install/lib:/nfs/isicvlnas01/share/cudnn-6.5-linux-x64-v2/"
echo $PATH
echo $LD_LIBRARY_PATH

echo $SGE_GPU
export CUDA_VISIBLE_DEVICES=$SGE_GPU

data_xing_mix=/nfs/isicvlnas01/users/liukuan/recsys/cache/xing_mix/
data_xing_het=/nfs/isicvlnas01/users/liukuan/recsys/cache/xing_het/
data_yelp_mix=/nfs/isicvlnas01/users/liukuan/recsys/cache/yelp_mix/
data_yelp_het=/nfs/isicvlnas01/users/liukuan/recsys/cache/yelp_het/
raw_xing=/nfs/isicvlnas01/users/liukuan/recsys/raw_data/xing/
raw_yelp=/nfs/isicvlnas01/users/liukuan/recsys/raw_data/yelp/

train_dir=/nfs/isicvlnas01/users/liukuan/recsys/train/
log_dir=/nfs/isicvlnas01/users/liukuan/recsys/train/log/

__cmd__
"""

def rawdata(val):
    return "", "--raw_data {}".format(val)

def data_dir(val):
    return "{}".format(val[1]), "--data_dir {}".format(val[0])
    
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

def n_epoch(val):
    return "", "--n_epoch {}".format(val)

def loss(val):
    return "{}".format(val), "--loss {}".format(val)

def ta(val):
    return "Ta{}".format(val), "--ta {}".format(val)

def user_sample(val):
    if val == 1.0:
        return "", ""
    else:
        return "part{}".format(val), "--user_sample {}".format(val)

def num_layers(val):
    return "n{}".format(val), "--num_layers {}".format(val)

def L(val):
    return "L{}".format(val), "--L {}".format(val)

def N(val):
    return "", "--N {}".format(val)

def dataset(val):
    if val == 'xing':
        return "xing", "--dataset xing --after40 True"
    elif val == "ml":
        return "Ml", "--dataset ml --after40 False"
    elif val == 'ml1m':
        return 'ml1m', '--dataset ml1m --after40 False'
    elif val == 'yelp':
        return 'yelp', '--dataset yelp --after40 False'


def use_concat(val):
    if val:
        name = "Cc"
    else:
        name = "Mn"
    return name, "--use_concat {}".format(val)

def item_vocab_size(val):
    return "", "--item_vocab_size {}".format(val)

def fromScratch(val):
    if not val:
        return "","--fromScratch False"
    else:
        return "","--fromScratch True"

def no_user_id(val):
    if val:
        return 'Nouid', '--no_user_id True'
    else:
        return 'Uid', '--no_user_id False'

def wFeat(val):
    if val:
        return 'Wfeat', '--use_user_feature True --use_item_feature True'
    else:
        return 'Wofeat', '--use_user_feature False --use_item_feature False'
# whether or not to use separte embedding for input/output items
def use_out_items(val):
    if val:
        return "OT", "--use_sep_item True"
    else:
        return "OF", "--use_sep_item False"

def randseed(val):
    if val == "Ensem": # for ensemble
        return "seed",""
    else:
        return 'seed{}'.format(val), '--seed {}'.format(val)

def output_feat(val):
    return 'oa{}'.format(val), '--output_feat {}'.format(val)

def no_input_feat(val):
    if val:
        return 'Noi', '--no_input_item_feature True'
    else:
        return '', '--no_input_item_feature False'

def test(val):
    if val == False:
        return '', '--test False'
    else:
        return 't_t', '--test True'

def ensemble(val):
    if val == False:
        return "","--ensemble False"
    else:
        return "","--ensemble True"

def ensemble_suffix(val):
    if val == "":
        return "",""
    else:
        return "","--ensemble_suffix {}".format(val)

def combine(val):
    return "", "--combine_att {}".format(val)

############# Different Settings ###############

def setting1():
    # ensemble
    funcs = [data_dir, batch_size, size,       #3 
             dropout, learning_rate, n_epoch,  #6
             loss, ta, num_layers,             #9
             L, N, use_concat,                 #12
             dataset, item_vocab_size, fromScratch, #15
             no_user_id, use_out_items, wFeat, #18
             randseed, test, ensemble, #21
             combine, ensemble_suffix, rawdata]
    

    #ensemble xing full
    # template = [["$data_xing_het", "het"], 64, 256, 0.4, 0.5, 150, "ce", 0, 1, 30, "000",False,'xing',50000, False, True, False, True, 0, False, False, "het", "", "$raw_xing"]
    
    # ML-1m 
    # template = ["$data_ml1m", 64, 128, 0.5, 0.5, 200, "warp", 0, 1, 50, "001",False,'ml1m',3100, False, True, True, True, 0, True, False, ""]
    
    # # yelp, oF
    template = [["$data_yelp_het", "het"], 32, 128, 0.6, 0.5, 250, "ce", 0, 1, 30, "001", False, 'yelp', 80000, False, True, False, True, 0, True, False, "het", "", "$raw_yelp"]

    # XING with test, no feature or yes feature; not output (OF)
    # template = ["$data_full", 64, 256, 0.5, 0.5, 250, "ce", 0, 1, 30, "001",False,'xing',50000, False, True, False, True, 0, True, False, ""]
    params = []

    for _seed in xrange(1,9):
        temp = list(template)
        temp[18] = _seed
        params.append(temp)

    # for the ensemble script
    temp = list(template)
    temp[18] = "Ensem"
    temp[20] = True
    temp[22] = "1,2,3,4,5,6,7,8"
    params.append(temp)

    return funcs, params



def setting2():
    # w or wo feat
    # funcs = [data_dir, batch_size, size,       #3
    #          dropout, learning_rate, n_epoch,  #6
    #          loss, ta, num_layers,             #9
    #          L, N, use_concat,                 #12
    #          dataset, item_vocab_size, fromScratch, #15
    #          no_user_id, use_out_items, wFeat] #18


    # user sample
    # funcs = [data_dir, batch_size, size,       #3
    #          dropout, learning_rate, n_epoch,  #6
    #          loss, ta, num_layers,             #9
    #          L, N, use_concat,                 #12
    #          dataset, item_vocab_size, fromScratch, #15
    #          no_user_id, use_out_items, wFeat, #18
    #          output_feat, user_sample]

    # output_feat
    funcs = [dataset, data_dir, rawdata,        #3
             combine, batch_size, size,         #6
             dropout, learning_rate, n_epoch,  #9
             loss, num_layers, L,              #12
             N, use_concat, item_vocab_size,   #15
             fromScratch, no_user_id, use_out_items,      #18
             wFeat, output_feat, test]      #21

    # input item feat
    # funcs = [data_dir, batch_size, size,       #3
    #          dropout, learning_rate, n_epoch,  #6
    #          loss, ta, num_layers,             #9
    #          L, N, use_concat,                 #12
    #          dataset, item_vocab_size, fromScratch, #15
    #          no_user_id, use_out_items, wFeat, #18
    #          no_input_feat, output_feat, test, # 21
    #          rawdata, combine]

    # yelp
    # template = ["$data_yelp", 32, 256, 0.4, 1.0, 150, "ce", 0, 1, 30, "001", False, 'yelp', 80000, False, True, True, True, 1, True]

    # yelp, oF
    # template = [["$data_yelp_het", 'het'], 32, 256, 0.4, 1.0, 250, "ce", 0, 1, 30, "001", False, 'yelp', 80000, False, True, False, True, 1, True, "$raw_yelp", "het"]

    # yelp, oF new

    template = ['yelp', ["$data_yelp_mix", "mix"], "$raw_yelp", 
                 "mix", 10, 128, 
                 0.4, 0.5, 250, 
                 "ce", 1, 30, 
                 "001",False,80000, 
                 False, True, False, 
                 True, 1, True]
    # yelp, oT
    # template = ["$data_yelp", 32, 128, 0.6, 1.0, 250, "ce", 0, 1, 30, "001", False, 'yelp', 80000, False, True, True, True, 0, True]

    # # yelp, oT/oF, oa0
    # template = ["$data_yelp", 32, 128, 0.6, 1.0, 250, "ce", 0, 1, 30, "001", False, 'yelp', 80000, False, True, True, True, 0, True]

    # yelp, oT/oF, oa1, noi
    # template = [["$data_yelp_het", "het"], 32, 128, 0.5, 1.0, 250, "ce", 0, 1, 30, "001", False, 'yelp', 80000, False, True, True, True, True, 1, True, "$raw_yelp", "het"]

    # XING sep-out-feat, output_feat
    # template = ["$data_full", 64, 256, 0.4, 1.0, 150, "ce", 0, 1, 30, "001",False,'xing',50000, False, True, True, True, 0]    

    # part of users
    # template = ["$data_full", 64, 256, 0.4, 0.5, 150, "ce", 0, 1, 30, "001",False,'xing',50000, False, True, False, True, 1, 0.3]

    # no feat
    # template_ml = ["$data_ml", 64, 128, 0.5, 0.5, 150, "ce", 0, 1, 200, "001",False,'ml',13000, False, False, False, False]

    # use out item 
    # template_ml = ["$data_ml", 64, 128, 0.5, 0.5, 150, "ce", 0, 1, 200, "001",False,'ml',13000, False, True, True, False]

    # use out item 
    # template_ml = ["$data_ml", 64, 128, 0.5, 0.5, 150, "ce", 0, 1, 200, "001",False,'ml',13000, False, True, True, True]

    # use out item, output_feat
    # template_ml = ["$data_ml", 64, 128, 0.5, 0.5, 150, "ce", 0, 1, 200, "001",False,'ml',13000, False, True, True, True, 1]

    #no/use out item, output_feat, L=40, h 256, 512
    # template_ml = ["$data_ml", 64, 128, 0.5, 0.5, 150, "ce", 0, 1, 40, "001",False,'ml',13000, False, True, False, False, 1]
    # template_ml = ["$data_ml", 64, 128, 0.5, 0.5, 150, "ce", 0, 1, 40, "001",False,'ml',13000, False, True, True, False, 1]


    # XING, with test new, het
    # template = ['xing', ["$data_xing_het", "het"], "$raw_xing", 
    #              "het", 64, 256, 
    #              0.4, 0.5, 150, 
    #              "warp", 1, 30, 
    #              "001",False,50000, 
    #              False, True, False, 
    #              True, 1, True]
    # XING, with test new, mix
    # template = ['xing', ["$data_xing_mix", "mix"], "$raw_xing", 
    #              "mix", 32, 256, 
    #              0.4, 0.5, 150, 
    #              "ce", 1, 30, 
    #              "001",False,50000, 
    #              False, True, False, 
    #              True, 1, True]

    # XING, with test
    # template = ["$data_full", 64, 256, 0.4, 0.5, 150, "ce", 0, 1, 30, "001",False,'xing',50000, False, True, False, True, 1, True]
    
    # XING with test, no feature or yes feature; not output (OF)
    # template = [["$data_xing_het", "het"], 64, 256, 0.4, 0.5, 150, "ce", 0, 1, 30, "001",False,'xing',50000, False, True, False, True, 0, True]

    # XING with test, no feature or yes feature; separate output (OT)
    # template = ["$data_full", 64, 256, 0.5, 0.5, 150, "ce", 0, 1, 30, "001",False,'xing',50000, False, True, True, True, 0, True]

    # XING with test, yes feature; no input feature
    # template = ["$data_full", 64, 256, 0.5, 0.5, 150, "ce", 0, 1, 30, "001",False,'xing',50000, False, True, True, True, True, 1, True]

    # XING with test, oa1, tuning (smaller h)
    # template = ["$data_full", 64, 256, 0.4, 0.5, 150, "ce", 0, 1, 30, "001",False,'xing',50000, False, True, False, True, 0, True]

    # ML-1m 
    # template = ["$data_ml1m", 64, 256, 0.4, 1.0, 200, "warp", 0, 1, 50, "001",False,'ml1m',3100, False, True, True, True, 1, True]
    
    params = []

    
    # _h = [128] #xing
    _h = [128] #yelp
    _dropout = [0.6] #yelp
    # _dropout = [0.5] #xing
    _learning_rate = [1, 0.5]
    _oa = [1]
    # _oa = [3]
    # _us = [0.7, 0.3]
    # # # _seeds = [1, 2, 3, 4, 5, 6, 7, 8]
    # _loss = ['ce', 'warp']
    _of = [False]
    _wf = [True]
    for lr in _learning_rate:
        for h in _h:
            for oa in _oa:
                for dr in _dropout:
                    for of in _of:
                        temp = list(template)
                        temp[7] = lr
                        temp[17] = of
                        temp[5] = h
                        temp[19] = oa
                        # temp[0] = '$data_full' + str(us).replace('.', '')
                        temp[6] = dr
                        params.append(temp)
    
                    
    
    # for ml
    # _h = [512]
    # _oa = [1]
    # _dropout = [0.6]
    # _learning_rate = [1, 0.5]

    # # _h = [256, 512]
    # # _oa = [0, 1, 2]
    # # _dropout = [0.8, 0.6]
    # # _learning_rate = [1]
    # _nouser = [True]

    # for lr in _learning_rate:
    #     for oa in _oa:
    #         for dr in _dropout:
    #             for h in _h:
    #                 temp = list(template_ml)
    #                 temp[4] = lr
    #                 temp[18] = oa
    #                 temp[3] = dr
    #                 temp[2] = h
    #                 params.append(temp)

    return funcs, params

def main(setting, acct=0, tf_version=12):
    
    funcs, params = setting()
    
    if acct == 0 or acct == 3:
        if tf_version == 12:
            head = head_hpc1
        elif tf_version == 9:
            head = head_hpc1_tf9
    elif acct == 1:
        head = head_hpc2 # kuanl
    else:
        head = head_hpc3 # isicvl

    def get_name_cmd(para):
        name = "lstm"
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
    for para in params:
        name, cmd = get_name_cmd(para)
        if acct in [0, 1]:
            cmd = "python run.py " + cmd
        elif acct == 3:
            cmd = "python run.py " + cmd
            name = 'i80_' + name
        else:
            cmd = "/nfs/isicvlnas01/share/anaconda/bin/python run.py " + cmd
            
        fn = "../jobs/{}.sh".format(name)
        f = open(fn,'w')
        content = head.replace("__cmd__",cmd)
        f.write(content)
        f.close()

    # decode
    for para in params:
        name, cmd = get_name_cmd(para)
        name = name + ".decode"
        cmd += " --recommend True"
        if acct in [0, 1]:
            cmd = "python run.py " + cmd
        elif acct == 3:
            cmd = "python run.py " + cmd
            name = 'i80_' + name
        else:
            cmd = "/nfs/isicvlnas01/share/anaconda/bin/python run.py " + cmd

        fn = "../jobs/{}.sh".format(name)
        f = open(fn,'w')
        content = head.replace("__cmd__",cmd)
        content = content.replace("23:59:59", "0:59:59")
        f.write(content)
        f.close()
        

if __name__ == "__main__":
    settings = [setting1,setting2]
    settings_desc = ["ensemble","to organize"]

    if len(sys.argv) == 4:
        acct = int(sys.argv[1]) # 1=hpc; 2=isicvl; 3=isi80
        setting_id = int(sys.argv[2])
        tf_version = int(sys.argv[3])
        setting = settings[setting_id]
        main(setting,acct,tf_version)
    else:
        print "python jobs_lstm.py <acct> <setting_id> <tf_version>"
        print "<acct>: \n\t0=xingshi@hpc\n\t1=kuanl@hpc\n\t2=isicvl\n\t3=xingshi@isi80"
        print "setting_id:"
        for i in xrange(len(settings)):
            print "\t{} {}".format(i,settings_desc[i])

        print "tf_version: 9 or 12"
    

    
    
