## Differential Neural Metric(DRM) Code Appendix

This is (1) an implementation of model and (2) reproducible experiments for the paper.

 We compare DRM<sub>dot</sub> and DRM<sub>L2</sub> with the following baselines.

### Baselines
- SLIM: [Sparse LInear Methods for Top-N Recommender systems](http://glaros.dtc.umn.edu/gkhome/node/774)
- CDAE: [Collaborative Denoising AutoEncoder](https://dl.acm.org/doi/10.1145/2835776.2835837)
- BPR: [Bayesian Personalized Ranking for Matrix Factorization](https://arxiv.org/abs/1205.2618)
- WMF: [Weighted Matrix Factorization](http://yifanhu.net/PUB/cf.pdf)
- WARP: [Weighted Approximated Ranking Pairwise Matrix Factorization](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41534.pdf)
- CML: [Collaborative Metric Learning](http://www.cs.cornell.edu/~ylongqi/paper/HsiehYCLBE17.pdf)
- SQLRANK: [Stochastic Queueing List Ranking](https://arxiv.org/abs/1803.00114)
- SRRMF: [Square Loss Ranking Regularizer Matrix Factorization](https://github.com/HERECJ/recsys/tree/master/alg/discrete/SRRMF)

We did not implement SQLRANK and SRRMF, because there exist implementations by the original paper authors.
[The implementation of SQLRANK](https://github.com/wuliwei9278/SQL-Rank) is written in Julia and [the implementation of SRRMF](https://github.com/HERECJ/recsys/tree/master/alg/discrete/SRRMF) is written in Matlab.

Except these two models, we wrote scripts for training and evaluating baselines, and our models in Python.


### Environment Settings
We assume that you have installed Python using [Anaconda](https://docs.anaconda.com/anaconda/install/), and your environment is equipped with CUDA. It should be possible to use other Python distributions, but we did not tested.

We use PyTorch with Cuda as the backend for our ML implementations.
 * PyTorch (version: 1.5.0)
 * CUDA (version: 10.1)

For conda:
```
conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.1 -c pytorch
```

For usual python:
```
pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

will install appropriate PyTorch with cuda 10.1 version. Please refer [here](https://pytorch.org/get-started/locally/) and [here](https://pytorch.org/get-started/previous-versions/) to see appropriate pytorch version for your environment.


We use the packages as listed below(alphabetical order):
 * Cython
 * fastrand
 * implicit (https://github.com/benfred/implicit)
 * lightfm (https://github.com/lyst/lightfm)
 * numpy
 * scipy
 * sklearn
 * tqdm

External Libraries:
* PyTorch
  *
* SLIM (https://github.com/KarypisLab/SLIM)
  * Follow instruction guide in (https://github.com/KarypisLab/SLIM/)
  * It is required only when evaluating SLIM in our code.




From installation to how-to-use
---------------
We assume that you have installed python using [Anaconda](https://docs.anaconda.com/anaconda/install/) and your environment is equipped with CUDA. It should be possible to use other python distributions, but we did not tested.

If you are using Conda
```
conda create --name drm_test python=3.7.3
conda activate drm_test
```
or refer to `virtualenv`.

1. Install *packages*.
    conda create --name drm_test python=3.7.3

```bash
pip install -r requirements.txt
cd ~/code/eval/
python setup.py build_ext --inplace
cd ~/code/
```

We noted the instruction in the codes. You can use `-h` command.
```python
python <some_name>-pt.py
```

For example,
```python
python ours-pt.py --dataset_name=ml-20m --kk=50 --infer_dot=0
```

1. After running `python <some_name>-pt.py`,
```python
python <some_name>-fb.py
```

For example,
```python
python ours-fb.py --dataset_name=ml-20m --pos_sample=1 --infer_dot=0
```

2. If you need to check the result of training, you can read it with `torch`.

     `python`

    `>>> import torch`

    `>>> torch.load(open('<location>', 'rb'))`

    For example,

        >>> torch.load(open('saved_results/sketchfab/k=1', 'rb'))

    then the result is like this:
```json
        {'model': mfrec(
        (u_emb): Embedding(15563, 128)
        (i_emb): Embedding(28806, 128)
        ), 'eval_metric': 'recall', 'epoch': 150, 'lr': 0.05, 'dim': 128, 'alpha': 1.0, 'tau': 0.1, 'reg': 0.5, 'best': 0.27563380826315786}
```

#### Datasets
We use these datasets:

 * SketchFab: https://github.com/EthanRosenthal/rec-a-sketch
 * Epinion: https://www.cse.msu.edu/~tangjili/datasetcode/truststudy.htm
 * MovieLens-20M: https://grouplens.org/datasets/movielens/20m/
 * Melon: https://arena.kakao.com/c/7/data

#### Q & A
1. `ModuleNotFoundError: No module named 'models'` Error
   * Go to `~/code/`, and then run the `.py` file.
     For example, if your result files are in `~/saved/`, then run like this:

2. I have a `ModuleNotFoundError: No module named 'eval.rec_eval'`
   * Go to `code/eval`, and run the command below:
    `python setup.py build_ext --inplace`

3. I have a `ModuleNotFoundError: No module named 'SLIM'`
   * You need to install SLIM package directly from https://github.com/KarypisLab/SLIM.
