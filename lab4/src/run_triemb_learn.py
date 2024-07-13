import mat73
import numpy as np
import time
import os

from test_image_search import *


def triemb_learn(vtrain, C):
    """return [Xmean, eigvec, eigval]"""

    nlearn = vtrain.shape[1]  # number of input vectors
    k = C.shape[1]  # number of support centroids
    d = vtrain.shape[0]  # input vector dimensionality
    D = k * d  # output dimensionality
    dout = D

    slicesize = 10000
    Xmean = np.zeros((D, 1)).astype("float32")

    Rx = np.zeros([nlearn, D])
    Xsum = np.zeros((D, 1))

    vtrain = vtrain.T
    C = C.T

    for i in range(0, nlearn, slicesize):
        # ********************************************************* #
        # CODING HERE 求取特征均值Xmean, Xmean.shape = (128*k, 1)    #
        # ********************************************************* #
        np.mean(vtrain)
        for j in range(kc):
            item = vtrain[i : i + slicesize] - C[j]
            item_normed = item / np.linalg.norm(item, axis=1).reshape(slicesize, 1)
            Xsum[j * d : j * d + d] += np.sum(item_normed, axis=0)
            Rx[i : i + slicesize, j * d : j * d + d] += item_normed

    Xmean = Xsum / nlearn
    Rx -= Xmean.reshape(1, D)
    covD = (Rx.T @ Rx) / nlearn

    eigval, eigvec = np.linalg.eig(covD)
    idx = eigval.argsort()[::-1]  # descending sort
    eigval = eigval[idx]
    eigvec = eigvec[:, idx]

    return Xmean, eigvec, eigval


os.chdir(os.path.dirname(__file__))  # change the work directory

timeStart = time.time()
data_dict = mat73.loadmat("../data/vtrain.mat")
# vtrain - (128, 5000404) , attention whether need transpose befor clustering
vtrain = data_dict["vtrain"]
print("* Descriptors loaded and processed in %.3fs" % (time.time() - timeStart))

timeStart = time.time()
# ************************************************************************** #
# CODING HERE 利用 k-means 聚类算法, 找出 kc 个聚类中心, C.shape = (128, kc)   #
# ************************************************************************** #
size = vtrain.shape[0]
centers = np.zeros((kc, 128), dtype=np.float32)
vtrain = vtrain.T
rand_points = np.random.randint(low=0, high=size, size=kc)
for k in range(kc):
    centers[k] = vtrain[rand_points[k]]
while True:
    ave = np.zeros((kc, 128), dtype=np.float32)
    counts = np.zeros(kc, dtype=np.int32)
    closest = np.argmin(
        np.linalg.norm(np.array([vtrain - centers[j] for j in range(kc)]), axis=2),
        axis=0,
    )
    for i, closest in enumerate(closest):
        ave[closest] += vtrain[i]
        counts[closest] += 1
    ave = ave / counts.reshape(-1, 1)
    if np.allclose(ave, centers):
        break
    centers = ave
centers = centers.T
vtrain = vtrain.T

print("* kmeans cluster centers processed in %.3fs" % (time.time() - timeStart))

timeStart = time.time()
Xmean, eigvec, eigval = triemb_learn(vtrain, centers)

eigval[-128:] = eigval[-129]  # make it more robust
Pemb = np.diag(np.power(eigval, -0.5)) @ eigvec.T  # projection matrix
print("* Embedding parameters learned in %.3fs" % (time.time() - timeStart))

np.savetxt("../data/C.csv", centers)
np.savetxt("../data/eigval.csv", eigval)
np.savetxt("../data/eigvec.csv", eigvec)
np.savetxt("../data/Xmean.csv", Xmean)
np.savetxt("../data/Pemb.csv", Pemb)
