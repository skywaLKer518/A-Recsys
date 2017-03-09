# A-RecSys : a Tensorflow Toolkit for Implicit Recommendation Tasks

## A-RecSys
A-RecSys implements a set of implicit recommendation algorithms and is designed to be used in large scale recommendation settings. It extends traditional matrix factorization algorithms and focuses on attribute embedding and applying sequence models.


Currently the models and features supported by A-RecSys include,

#### Models
+ Hybrid matrix factorization model (with deep layer extensions)
+ Linear sequence models based on CBOW and skip-gram
+ LSTM-based seq2seq model


#### Features
+ Recommendation with implicit feedback
+ Heterogeneous attribute embedding
+ Objective functions include cross-entropy, WARP

## How to use
### input data
CSV-formated (sep=\t) input files include

	u.csv	: user file. user id and attribute values.
	i.csv: item file. item id and attribute values.
	obs_tr.csv: implicit feedback for training. First two columns are user-id, item-id. Third column (optional) is for timestamp. 
	obs_va.csv: implicit feedback for development.
	obs_te.csv: implicit feedback for testing.
	
**A example** (adapted from MovieLens 1m) is given at ./examples/dataset/

### train models
**Example scripts** are provided at ./examples/ to start running the code

To train hybrid matrix factorization model on provided MovieLens 1m dataset,

``` 
cd examples/
bash run_hmf.sh 32 1 False 100 False
```

To train lstm model,

``` 
cd examples/
bash run_lstm.sh 64 1 False
```

(Code has been tested on TF 0.8 and above.)

### recommend
You can switch to "recommend" mode from "training" by setting flag *recommend* to 'true'. In the above HMF example, it would be

``` 
cd examples/
bash run_hmf.sh 32 1 False 100 True
```

By default, the code generates a ground truth interaction file *res_T_test.csv* from *obs_te.csv*, and perform recommendation on all users that appear in *res_T_test.csv*; however, you can generate your own *res_T_test.csv* to narrow down user set on which recommendation is performed.

## Cite
Please cite the following if you find this helpful.

@article{liu2016temporal,
  title={Temporal Learning and Sequence Modeling for a Job Recommender System},
  author={Liu, Kuan and Shi, Xing and Kumar, Anoop and Zhu, Linhong and Natarajan, Prem},
  journal={arXiv preprint arXiv:1608.03333},
  year={2016}
}

## Feedback
Your comments and suggestions are more than welcome! We really appreciate that!

Kuan Liu kuanl@usc.edu
Xing Shi xingshi@usc.edu
