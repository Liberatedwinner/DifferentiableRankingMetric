# distutils: language = c++
# cython: language_level=3

"""
https://github.com/benfred/implicit/blob/master/implicit/evaluation.pyx
"""

from tqdm.auto import tqdm
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
import cython
from cython.operator import dereference
from cython.parallel import parallel, prange
from libc.stdlib cimport malloc, free
from libc.string cimport memset
from libc.math cimport fmin

from libcpp.unordered_set cimport unordered_set


@cython.boundscheck(False)
def ranking_metrics_at_k(model, train_user_items, test_user_items, int K=10, int num_threads=1):

    """ Calculates ranking metrics for a given trained model
    Parameters
    ----------
    model : RecommenderBase
        The fitted recommendation model to test
    train_user_items : csr_matrix
        Sparse matrix of user by item that contains elements that were used
            in training the model
    test_user_items : csr_matrix
        Sparse matrix of user by item that contains withheld elements to
        test on
    K : int
        Number of items to test on
    num_threads : int, optional
        The number of threads to use for testing. Specifying 0 means to default
        to the number of cores on the machine. Note: aside from the ALS and BPR
        models, setting this to more than 1 will likely hurt performance rather than
        help.
    Returns
    -------
    float
        the calculated p@k
    """
    num_threads = 1
    if not isinstance(train_user_items, csr_matrix):
        train_user_items = train_user_items.tocsr()

    if not isinstance(test_user_items, csr_matrix):
        test_user_items = test_user_items.tocsr()
    cdef int users = test_user_items.shape[0], items = test_user_items.shape[1]
    cdef int u, i
    # precision
    cdef int relevant = 0, pr_div = 0, total = 0, rc_div = 0
    # map
    cdef double mean_ap = 0, ap = 0
    # ndcg
    cdef double[:] cg = (1.0 / np.log2(np.arange(2, K + 2)))
    cdef double[:] cg_sum = np.cumsum(cg)
    cdef double ndcg = 0, idcg
    # auc
    cdef double precision = 0, recall = 0
    cdef double hit, miss, num_pos_items, num_neg_items
    cdef int[:] test_indptr = test_user_items.indptr
    cdef int[:] train_indptr = train_user_items.indptr
    cdef int[:] test_indices = test_user_items.indices

    cdef int * ids
    cdef unordered_set[int] * likes

    with nogil, parallel(num_threads=1):
        ids = <int * > malloc(sizeof(int) * K)
        likes = new unordered_set[int]()
        try:
            for u in prange(users, schedule='guided'):
                # if we don't have any test items, skip this user
                if train_indptr[u+1] - train_indptr[u] < 3:
                    continue
                if test_indptr[u] == test_indptr[u+1]:
                    continue
                memset(ids, -1, sizeof(int) * K)

                with gil:
                    recs = model.recommend(u, train_user_items, N=K)
                    for i in range(len(recs)):
                        ids[i] = recs[i][0]
                likes.clear()
                for i in range(test_indptr[u], test_indptr[u+1]):
                    likes.insert(test_indices[i])

                rc_div = likes.size()
                ap = 0
                hit = 0
                miss = 0
                idcg = cg_sum[min(K, rc_div) - 1]
                num_pos_items = rc_div
                num_neg_items = items - num_pos_items

                for i in range(K):
                    if likes.find(ids[i]) != likes.end():
                        relevant += 1
                        hit += 1
                        ap += hit / float(i + 1)
                        ndcg += cg[i] / idcg
                    else:
                        miss += 1
                mean_ap += (ap / K)
                precision += (hit / K)
                recall += (hit / rc_div)
                total += 1
        finally:
            free(ids)
            del likes
    return {
        "precision": precision / float(total),
        "recall": recall / float(total),
        "map": mean_ap / float(total),
        "ndcg": ndcg / float(total)
    }


@cython.boundscheck(False)
def leave_k_eval(model, train_user_items, _test_user_items, int leavek, int K=10, int num_threads=1):
    num_threads = 1
    if not isinstance(train_user_items, csr_matrix):
        train_user_items = train_user_items.tocsr()
    cdef int[:, :] test_user_items = _test_user_items
    cdef int users = train_user_items.shape[0], items = train_user_items.shape[1]
    cdef int u, i
    # precision
    cdef int relevant = 0, pr_div = 0, total = 0, rc_div = 0
    # map
    cdef double mean_ap = 0, ap = 0
    # ndcg
    cdef double[:] cg = (1.0 / np.log2(np.arange(2, K + 2)))
    cdef double[:] cg_sum = np.cumsum(cg)
    cdef double ndcg = 0, idcg
    # auc
    cdef double precision = 0, recall = 0, avghit = 0
    cdef double hit, miss, num_pos_items, num_neg_items
    cdef int * ids
    cdef unordered_set[int] * likes

    with nogil, parallel(num_threads=1):
        ids = <int * > malloc(sizeof(int) * K)
        likes = new unordered_set[int]()
        try:
            for u in prange(users, schedule='guided'):
                # if we don't have any test items, skip this user
                if len(test_user_items[u]) == 0:
                    continue
                memset(ids, -1, sizeof(int) * K)

                with gil:
                    recs = model.rank_items(u, train_user_items, _test_user_items[u])
                    for i in range(K):
                        ids[i] = recs[i][0]
                    #progress.update(1)

                # mostly we're going to be blocked on the gil here,
                # so try to do actual scoring without it
                likes.clear()
                for i in range(leavek):
                    likes.insert(test_user_items[u][i])

                ap = 0
                hit = 0
                miss = 0
                pr_div = min(likes.size(), K)
                idcg = cg_sum[pr_div - 1]
                num_pos_items = likes.size()
                num_neg_items = items - num_pos_items

                for i in range(K):
                    if likes.find(ids[i]) != likes.end():
                        relevant += 1
                        hit += 1
                        ap += hit / float(i + 1)
                        ndcg += cg[i] / idcg
                    else:
                        miss += 1
                mean_ap += ap / float(pr_div)
                precision += hit / float(pr_div)
                total += 1
        finally:
            free(ids)
            del likes

    return {
        "precision": precision / float(total),
        "map": mean_ap / float(total),
        "ndcg": ndcg / float(total)
    }


@cython.boundscheck(False)
def ranking_metrics_with_users(model, train_user_items, test_user_items, int K=10, int num_threads=1):

    """ Calculates ranking metrics for a given trained model
    Parameters
    ----------
    model : RecommenderBase
        The fitted recommendation model to test
    train_user_items : csr_matrix
        Sparse matrix of user by item that contains elements that were used
            in training the model
    test_user_items : csr_matrix
        Sparse matrix of user by item that contains withheld elements to
        test on
    K : int
        Number of items to test on
    num_threads : int, optional
        The number of threads to use for testing. Specifying 0 means to default
        to the number of cores on the machine. Note: aside from the ALS and BPR
        models, setting this to more than 1 will likely hurt performance rather than
        help.
    Returns
    -------
    float
        the calculated p@k
    """
    precision, recall, mean_ap, ndcgs = [], [], [], []
    num_threads = 1
    if not isinstance(train_user_items, csr_matrix):
        train_user_items = train_user_items.tocsr()

    if not isinstance(test_user_items, csr_matrix):
        test_user_items = test_user_items.tocsr()
    cdef int users = test_user_items.shape[0], items = test_user_items.shape[1]
    cdef int u, i
    # precision

    cdef int relevant = 0, pr_div = 0, total = 0, rc_div = 0
    # map
    cdef unordered_set[int] set_of_users

    cdef double ap = 0

    # ndcg
    cdef double[:] cg = (1.0 / np.log2(np.arange(2, K + 2)))
    cdef double[:] cg_sum = np.cumsum(cg)
    cdef double ndcg = 0, idcg
    # auc

    cdef double hit, miss, num_pos_items, num_neg_items
    cdef int[:] test_indptr = test_user_items.indptr
    cdef int[:] train_indptr = train_user_items.indptr
    cdef int[:] test_indices = test_user_items.indices

    cdef int * ids
    cdef unordered_set[int] * likes


    with nogil:
        ids = <int * > malloc(sizeof(int) * K)
        likes = new unordered_set[int]()
        try:
            for u in range(users):
                # if we don't have any test items, skip this user
                #if train_indptr[u+1] - train_indptr[u] < 3:
                #    continue
                #if test_indptr[u] == test_indptr[u+1]:
                #    continue

                memset(ids, -1, sizeof(int) * K)

                with gil:
                    recs = model.recommend(u, train_user_items, N=K)
                    for i in range(len(recs)):
                        ids[i] = recs[i][0]
                likes.clear()
                for i in range(test_indptr[u], test_indptr[u+1]):
                    likes.insert(test_indices[i])
                rc_div = max(1, likes.size())
                ap = 0
                hit = 0
                miss = 0
                idcg = cg_sum[min(K, rc_div) - 1]
                num_pos_items = rc_div
                num_neg_items = items - num_pos_items
                ndcg = 0
                for i in range(K):
                    if likes.find(ids[i]) != likes.end():
                        relevant += 1
                        hit += 1
                        ap += hit / float(i + 1)
                        ndcg += cg[i] / idcg
                    else:
                        miss += 1
                with gil:
                    ndcgs.append(ndcg)
                    mean_ap.append(ap / K)
                    precision.append(hit / K)
                    recall.append(hit / rc_div)
                    total += 1
        finally:
            free(ids)
            del likes
    return {
        "precision": np.asarray(precision),
        "recall": np.asarray(recall),
        "map": np.asarray(mean_ap),
        "ndcg": np.asarray(ndcgs)
    }