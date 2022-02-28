
The code repository for "LGLNN: Label Guided Graph Learning-Neural Network for Few-shot Learning", 
which is implemented based on EGNN[1]

Training

# ************************** miniImagenet, 5way 1shot *****************************
$ python3 train.py --dataset mini --num_ways 5 --num_shots 1 --transductive True

# ************************** miniImagenet, 5way 5shot *****************************
$ python3 train.py --dataset mini --num_ways 5 --num_shots 5 --transductive True

Evaluation
$ python3 eval.py --test_model D-mini_N-5_K-1_U-0_L-3_B-40_T-True

References
[1] J Kim et al. Edge-Labeling Graph Neural Network for Few-shot Learning.
[2] L Yang et al. DPGN: Distribution Propagation Graph Network for Few-shot Learning.
