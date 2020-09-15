## Differentiable Ranking Metric(DRM) Code Appendix

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

For Conda:
```
conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.1 -c pytorch
```

For usual Python:
```
pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

will install appropriate PyTorch with cuda 10.1 version. Please refer [here](https://pytorch.org/get-started/locally/) and [here](https://pytorch.org/get-started/previous-versions/) to see appropriate PyTorch version for your environment.


We use the packages as listed below(alphabetical order):
 * Cython
 * fastrand
 * [implicit](https://github.com/benfred/implicit)
 * [lightfm](https://github.com/lyst/lightfm)
 * numpy
 * scipy
 * sklearn
 * tqdm

External Libraries:
* PyTorch
  *
* [SLIM](https://github.com/KarypisLab/SLIM)
  * Follow instruction guide in [here](https://github.com/KarypisLab/SLIM/).
  * It is required only when evaluating SLIM in our code.




From installation to how-to-use
---------------
We assume that you have installed Python using [Anaconda](https://docs.anaconda.com/anaconda/install/) and your environment is equipped with CUDA. It should be possible to use other Python distributions, but we did not tested.

If you are using Conda,
```
conda create --name drm_test python=3.7.3
conda activate drm_test
```
or refer to `virtualenv`.

1. Install *packages* and train datasets.
```bash
pip install -r requirements.txt
cd ~/code/eval/
python setup.py build_ext --inplace
cd ~/code/
```

2. Install *submodules* (implicit, SLIM, spotlight).
```bash
git submodule update --init --recursive
```

3. Preprocess the raw dataset. Use *Data preprocessing.ipynb*

4. Run `python <some_name>-pt.py`. 
We noted the instruction in the codes. You can use `-h` command to check instruction.
```python
python <some_name>-pt.py
```

For example,
```python
python ours-pt.py --dataset_name=ml-20m --kk=50 --infer_dot=0
```

5. After running `python <some_name>-pt.py`, run `python <some_name>-fb.py`.
```python
python <some_name>-fb.py
```

For example,
```python
python ours-fb.py --dataset_name=ml-20m --pos_sample=1 --infer_dot=0
```

6. If you need to check the result of training, you can read it with `Data statistics.ipynb`.


#### Datasets
We use these datasets:

 * SketchFab: https://github.com/EthanRosenthal/rec-a-sketch
 * Epinion: https://www.cse.msu.edu/~tangjili/datasetcode/truststudy.htm
 * MovieLens-20M: https://grouplens.org/datasets/movielens/20m/
 * Melon: https://arena.kakao.com/c/7/data

#### Q & A
1. I have a `ModuleNotFoundError: No module named 'models'` error.
   * Go to `~/code/`, and then run the `.py` file.

2. I have a `ModuleNotFoundError: No module named 'eval.rec_eval'` error.
   * Go to `code/eval`, and run the command `python setup.py build_ext --inplace`.

3. I have a `ModuleNotFoundError: No module named 'SLIM'` error.
   * You need to install SLIM package directly from https://github.com/KarypisLab/SLIM.
