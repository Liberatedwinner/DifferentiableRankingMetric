About Differential Neural Metric(DRM)
============================

This is the implementation for our paper

Two precision metric models: NP-WARP and NP-CML.

If you use our codes, please cite our paper.

Thank you very much.

Environment Settings
----------------
We use PyTorch as the backend, with CUDA.
 * PyTorch (version: 1.5.0)
 * CUDA (version: 10.1)

We use the packages as listed below(alphabetical order):
 * Cython
 * fastrand
 * implicit (https://github.com/benfred/implicit)
 * lightfm (https://github.com/lyst/lightfm)
 * numpy
 * scipy
 * sklearn
 * SLIM (https://github.com/KarypisLab/SLIM)
   * *Note*: You need to install SLIM directly.
 * tensorboardX
 * tqdm
 *
 *
 *
 *
 *


From installation to how-to-use
---------------
1. Install *packages*.

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
