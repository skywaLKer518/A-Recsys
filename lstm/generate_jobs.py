import os
import sys

head="""
#!/bin/bash
#PBS -q isi
#PBS -l walltime=300:00:00
#PBS -l nodes=1:ppn=16:gpus=2:shared

source $NLGHOME/sh/init_tensorflow.sh
cd /home/nlg-05/xingshi/lstm/tensorflow/recsys/lstm/

data_part=/home/nlg-05/xingshi/lstm/tensorflow/recsys/data/data_part
data_full=/home/nlg-05/xingshi/lstm/tensorflow/recsys/data/data_full
data_ml=/home/nlg-05/xingshi/lstm/tensorflow/recsys/data/data_ml
train_dir=/home/nlg-05/xingshi/lstm/tensorflow/recsys/train/

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
    
    def n_epoch(val):
        return "", "--n_epoch {}".format(val)

    def loss(val):
        return "{}".format(val), "--loss {}".format(val)

    def ta(val):
        return "Ta{}".format(val), "--ta {}".format(val)

    def num_layers(val):
        return "n{}".format(val), "--num_layers {}".format(val)
    
    def L(val):
        return "L{}".format(val), "--L {}".format(val)

    def N(val):
        return "", "--N {}".format(val)
    
    def dataset(val):
        if val == 'xing':
            return "", "--dataset xing"
        elif val == "ml":
            return "Ml", "--dataset ml --after40 False"

    def use_concat(val):
        if val:
            name = "Cc"
        else:
            name = "Mn"
        return name, "--use_concat {}".format(val)

    def item_vocab_size(val):
        if item_vocab_size == 50000:
            return "", ""
        else:
            return "", "--item_vocab_size {}".format(val)
    
    def fromScratch(val):
        if not val:
            return "","--fromScratch False"
        else:
            return "",""
            

    funcs = [data_dir, batch_size, size,       #0
             dropout, learning_rate, n_epoch,  #3
             loss, ta, num_layers,       #6
             L, N, use_concat,                 #9
             dataset, item_vocab_size, fromScratch]         #12
    
    template = ["$data_full", 64, 128, 0.5, 0.5, 150, "ce", 0, 1, 30, "001",False,'xing',50000, False]
    template_ml = ["$data_ml", 64, 128, 0.5, 0.5, 150, "ce", 0, 1, 200, "000",False,'ml',13000, True]
    params = []

    # for xing
    _h = [256]
    _dropout = [0.4,0.6]
    _learning_rate = [0.5, 1.0]
    for lr in _learning_rate:
        for dr in _dropout:
            for h in _h:
                temp = list(template)
                temp[4] = lr
                temp[3] = dr
                temp[2] = h
                params.append(temp)
    
    # for ml
    _h = [128]
    _dropout = [0.4,0.6,0.8]
    _learning_rate = [0.5, 1.0]
    for lr in _learning_rate:
        for dr in _dropout:
            for h in _h:
                temp = list(template_ml)
                temp[4] = lr
                temp[3] = dr
                temp[2] = h
                params.append(temp)
    





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
    for para in params:
        name, cmd = get_name_cmd(para)
        cmd = "python run.py " + cmd
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
        cmd = "python run.py " + cmd
        fn = "../jobs/{}.sh".format(name)
        f = open(fn,'w')
        content = head.replace("__cmd__",cmd)
        f.write(content)
        f.close()
        

if __name__ == "__main__":
    main()

    

    
    
