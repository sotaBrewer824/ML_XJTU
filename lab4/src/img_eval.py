from test_image_search import *
import mat73
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def yael_vecs_normalize(v, rval=0):
    # L2 Normalize and nan process

    # norm of each column
    # same as vnr = np.sqrt(np.sum(np.square(v), axis=0))
    vnr = np.linalg.norm(v, axis=0)

    # sparse multiplication to apply the norm
    vout = v / vnr  # broadcasting

    ko = np.argwhere(np.isnan(vout))[:, 1]
    np.unique(ko)
    vout[:, ko] = rval
    return vout


def my_nn(x, q, k):
    # Return the k nearest neighbors of a set of query vectors

    sim = q.T @ x
    dis, idx = kmin_or_kmax(sim, k)

    idx += np.array([1])
    return idx, dis


def kmin_or_kmax(sim, k):
    # Choose max k elements

    dis = np.sort(sim, axis=1)[:, ::-1]
    idx = np.argsort(sim, axis=1)[:, ::-1]
    dis = dis[:, :k].T
    idx = idx[:, :k].T

    return dis, idx


def compute_map(ranks, gnd):
    map = 0
    nq = 500
    aps = np.zeros((nq, 1))
    for i in range(nq):
        qgnd = gnd["ok"][i][0].reshape(-1)
        qgndj = gnd["junk"][i][0].reshape(-1)

        # positions of positive and junk images
        _, pos, _ = intersect_mtlb(ranks[:, i], qgnd) + np.array([1])
        # _, junk, _ = intersect_mtlb(ranks[:, i], qgndj)
        junk = np.array([1])
        pos = np.sort(pos)
        # junk = np.sort(junk)

        k = 0
        ij = 1
        if len(junk):
            # decrease positions of positives based on the number of junk images appearing before them
            ip = 1
            while ip <= np.size(pos):
                while ij <= len(junk) and pos[ip - 1] > junk[ij - 1]:
                    k += 1
                    ij += 1

                pos[ip - 1] -= k
                ip += 1

        ap = score_ap_from_ranks1(pos, len(qgnd))
        map += ap
        aps[i] = ap
    map /= nq

    return map, aps


def intersect_mtlb(a, b):
    """Realize MATLAB's `intersect(a, b)`"""

    a1, ia = np.unique(a, return_index=True)
    b1, ib = np.unique(b, return_index=True)
    aux = np.concatenate((a1, b1))
    aux.sort()
    c = aux[:-1][aux[1:] == aux[:-1]]
    return c, ia[np.isin(a1, c)], ib[np.isin(b1, c)]


def score_ap_from_ranks1(ranks, nres):
    # This function computes the AP for a query

    nimgranks = len(ranks)  # number of images ranked by the system

    ranks -= np.array([1])

    # accumulate trapezoids in PR-plot
    ap = 0
    recall_step = 1 / nres
    for j in range(nimgranks):
        rank = ranks[j]
        if rank == 0:
            precision_0 = 1.0
        else:
            precision_0 = j / rank

        precision_1 = (j + 1) / (rank + 1)
        ap += (precision_0 + precision_1) * recall_step / 2

    return ap


os.chdir(os.path.dirname(__file__))  # change the work directory

# import testing data psi(each column represent an image)
X = np.loadtxt("../data/psi.csv")

# import ground truth of testing data
data_dict = mat73.loadmat("../data/qidx.mat")
# (500, ) index of the images uesd to query
qidx = data_dict["qidx"].astype("int") - np.array([1])
data_dict = mat73.loadmat("../data/gnd.mat")
gnd = data_dict["gnd"]

Q = X[:, qidx]

d = 128  # dimension of a single SIFT descriptor
X = X[d:, :]  # remove front dth dimension
Q = Q[d:, :]  # remove front dth dimension

D = X.shape[0]

print("[ Results for varying powerlaw ]")
dout = D
for pw in [1.0, 0.7, 0.5, 0.3, 0.2, 0.0]:
    x = np.sign(X) * np.power(abs(X), pw)
    q = np.sign(Q) * np.power(abs(Q), pw)
    x = yael_vecs_normalize(x)
    q = yael_vecs_normalize(q)

    ranks, sim = my_nn(x, q, 1000)

    map, aps = compute_map(ranks, gnd)

    print(
        "Holidays    k = %d    d = %d    pw = %.2f    map = %.3f"
        % (kc, x.shape[0], pw, map)
    )
