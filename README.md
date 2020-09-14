Differential Neural Metric(DRM) Code Appendix
============================
This is the implementation for our paper.

We compare $\text{DRM}_{\text{dot}}$ and $\text{DRM}_{\text{L2}}$ and baseline implementations.

### Baselines compared
    - SLIM: Sparse Linear I.. model
    - CDAE: Collaborative Denoising AutoEncoder;
    - BPR: Bayesian Personalized Ranking for Matrix Factorization
    - WMF: Weighted Matrix Factorization
    - WARP: Weighted Approximated Ranking Pairwise Matrix Factorization
    - CML: Collaborative Metric Learning



These two baselines are not implemented in python;
[SQLRANK](https://github.com/wuliwei9278/SQL-Rank) written in Julia
[SRRMF](https://github.com/HERECJ/recsys/tree/master/alg/discrete/SRRMF) written in Matlab


Environment Settings
We assume that 
----------------
We use PyTorch with Cuda as the backend for our ML implementations.
 * PyTorch (version: 1.5.0)
 * CUDA (version: 10.1)

For our environments,
```
conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.1 -c pytorch
```
will install appropriate PyTorch with cuda 10.1 version. Please refer https://pytorch.org/get-started/previous-versions/ to see appropriate pytorch version for your environemnt.

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
We assume you are using Anaconda.

conda create --name drm_test python=3.7.3
conda activate drm_test


1. Install *packages*.
    conda create --name drm_test python=3.7.3
    `pip install -r requirements.txt` or

    `pip install <package>`

2.  `cd ~/code/eval/`

    `python setup.py build_ext --inplace`

3.  `cd ~/code/`

    `python <some_name>-pt.py`.

    We noted the instruction in the codes. You can use `-h` command.

    For example,

        python ours-pt.py --dataset_name=ml-20m --kk=50 --infer_dot=0

4. After running `python <some_name>-pt.py`,

    `python <some_name>-fb.py`.

    For example,

        python ours-fb.py --dataset_name=ml-20m --pos_sample=1 --infer_dot=0

5. If you need to check the result of training, you can read it with `torch`.

     `python`

    `>>> import torch`

    `>>> torch.load(open('<location>', 'rb'))`

    For example,

        >>> torch.load(open('saved_results/sketchfab/k=1', 'rb'))

    then the result is like this:

        {'model': mfrec(
        (u_emb): Embedding(15563, 128)
        (i_emb): Embedding(28806, 128)
        ), 'eval_metric': 'recall', 'epoch': 150, 'lr': 0.05, 'dim': 128, 'alpha': 1.0, 'tau': 0.1, 'reg': 0.5, 'best': 0.27563380826315786}




Datasets
--------------
We use these datasets:

 * MovieLens-20M: https://grouplens.org/datasets/movielens/20m/

 * SketchFab: https://github.com/EthanRosenthal/rec-a-sketch

 * Melon: https://arena.kakao.com/c/7/data

 * Epinion: https://www.cse.msu.edu/~tangjili/datasetcode/truststudy.htm

 * Pinterest: https://sites.google.com/site/xueatalphabeta/academic-projects



Q & A
----------
1. I have a `ModuleNotFoundError: No module named 'models'`
   * Go to `~/code/`, and then run the `.py` file.
     For example, if your result files are in `~/saved/`, then run like this:

     ` `

2. I have a `ModuleNotFoundError: No module named 'eval.rec_eval'`
   * Go to `code/eval`, and run the command below:

    `python setup.py build_ext --inplace`

3. I have a `ModuleNotFoundError: No module named 'SLIM'`
   * You need to install SLIM package directly from https://github.com/KarypisLab/SLIM.


4.
