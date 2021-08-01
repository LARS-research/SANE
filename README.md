# SANE-search-to-aggregate-neighborhood-for-gnn

### Overview
This is the code for our paper [Search to aggregate neighborhood for Graph Neural Networks](https://arxiv.org/abs/2104.06608), publised in ICDE 2021.
It is a differentiable architecture search for graph neural network (GNN).

### Requirements：
torch-cluster==1.5.7  
torch-geometric==1.6.3  
torch-scatter==2.0.6  
torch==1.6.0  
scikit-learn==0.21.3  
numpy==1.17.2  
hyperopt==0.2.5  
python==3.7.4

### Instructions to run the experiment
**Step 1.** Run the search process, given different random seeds.
(The Cora dataset is used as an example)
```
python train_search.py  --data Cora --fix_last True  --epochs 20
```
The results are saved in the directory `exp_res`, e.g., `exp_res/cora.txt`.

**Step 2.** Fine tune the searched architectures. You need specify the arch_filename with the resulting filename from Step 1.
```
python fine_tune.py --data Cora  --fix_last True   --hyper_epoch 50  --arch_filename exp_res/cora.txt   
```
Step 2 is a coarse-graind tuning process, and the results are saved in a picklefile in the directory `tuned_res`.

### Cite
Please kindly cite [our paper](https://arxiv.org/abs/2104.06608) if you use this code:

    @inproceedings{zhao2021search,
	  title={Search to aggregate neighborhood for graph neural network},
	  author={Zhao, Huan and Yao, Quanming and Tu, Weiwei},
	  booktitle={ICDE},
	  year={2021}
    }
    
### Acknowledgement
We thank Lanning Wei to help implement several experiments and further improve the codes in this work.

### Misc
If you have any questions about this project, you can open issues, thus it can help more people who are interested in this project. We will reply to your issues as soon as possible. You are also welcomed to reach us by zhaohuan@4paradigm.com

