import numpy as np


def mean_reciprocal_rank(rs):
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])

def precision_at_k(r, k):
    assert k >= 1
    r = np.asarray(r) != 0
    return np.sum(r[:k]) / k if r[k-1] == 1 else 0


def average_precision(r,k):
    out = [precision_at_k(r, j + 1) for j in range(k)]
    return np.mean(out)


def mean_average_precision(rs):
    return np.mean([average_precision(r) for r in rs])


def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum((2**r - 1) / np.log2(np.arange(2, r.size + 2)))
    return 0.


def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    print(dcg_at_k(r,k))
    return dcg_at_k(r, k) / dcg_max

def err_at_k(array,n):
    ERR, p = 0, 1
    for i in range(n):
        r = (2**array[i] - 1) / 32
        ERR += p * r / (i + 1)
        p *= 1 - r
    return ERR


if __name__ == "__main__":
    a = [5,5,0,0,0]
    print(r_precision(a))
    print(err_at_k(a, 5))