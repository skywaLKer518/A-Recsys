# A-RecSys
## Attribute Learning for Large Scale Recommender Systems - a Tensorflow Implementation

A-RecSys implements a set of recommendation algorithms that focuses on attribute learning and is designed to be used in large scale recommendation settings.

A-RecSys is under active development. The models/features it supports and plans to support include,

#### Models
+ Hybrid matrix factorization model with deep layer extensions
+ Linear sequence models based on skip-gram and cbow
+ LSTM-Rec sequence model


#### Features
+ Recommendation with implicit feedback
+ Recommendation with explicit feedback (for HMF)
+ Loss functions (including Cross-Entropy, batch-BPR, batch-Warp, sampled-batch-Warp)

#### Citation
Please cite the following if you find this helpful.

@article{liu2016temporal,
  title={Temporal Learning and Sequence Modeling for a Job Recommender System},
  author={Liu, Kuan and Shi, Xing and Kumar, Anoop and Zhu, Linhong and Natarajan, Prem},
  journal={arXiv preprint arXiv:1608.03333},
  year={2016}
}

#### Feedback
Please let us know your comments and suggestions. Thanks!
Kuan Liu kuanl@usc.edu
Xing Shi xingshi@usc.edu
