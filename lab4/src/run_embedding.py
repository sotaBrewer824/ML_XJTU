import mat73
import numpy as np
import time
import os

from test_image_search import *


def desc_postprocess(x, desc_mean):
    """root SIFT, subtract mean, and L2 normalize"""
    nin = x.shape[1]

    x = np.power(x, 0.5)  # root SIFT
    x -= np.reshape(desc_mean, (x.shape[0], 1))  # broadcasting

    x = yael_vecs_normalize(x)  # L2 Normalize
    return x


def yael_vecs_normalize(v):
    """L2 Normalize"""

    # norm of each column
    vnr = np.linalg.norm(
        v, axis=0
    )  # same as vnr = np.sqrt(np.sum(np.square(v), axis=0))

    # sparse multiplication to apply the norm
    vout = v / vnr  # broadcasting

    return vout


def triemb_sumagg(X, C, Xmean):
    """Realize that each column vector in X is concatenated after making a difference with k cluster centers,
    and perform L2 normalization on each 128-dimension of the concatenated column vector separately, and return
    the processed result. Finally, sum aggregation."""

    n = X.shape[1]
    d = X.shape[0]
    kc = C.shape[1]
    D = d * kc

    # **************************************************************************** #
    # CODING HERE 实现对图像 X 的单个 SIFT 特征的三角化嵌入表示并进行简单的加和融合, 即 #
    # 输入 X.shape = (128, n), 输出 Y.shape = (128*kc, 1)                           #
    # **************************************************************************** #
    Y = np.zeros(D)
    X = X.T
    C = C.T
    for j in range(kc):
        Y[j * d : j * d + d] = np.sum(
            (X - C[j]) / np.linalg.norm(X - C[j], axis=1).reshape(n, 1), axis=0
        )
    Y = Y.reshape(D, 1)
    Y -= n * Xmean
    return Y


os.chdir(os.path.dirname(__file__))  # change the work directory

n = 1491  # number of image set
d = 128  # dimension of a single SIFT descriptor
dout = d * kc

# Allocate matrix to store all vector image representations
psi = np.zeros((dout, n))

imidx = 0

# import testing data
data_dict = mat73.loadmat("../data/X.mat")
X = data_dict["X"]  # (128, 4455091) SIFT descriptor of testing images
data_dict = mat73.loadmat("../data/cndes.mat")
cndes = data_dict["cndes"].astype(
    "int"
)  # (1492, ) SIFT descriptor slices of each image(total 1491). e.g. X[:, cndes[0]:cndes[1]] belongs to first image

# import training data
data_dict = mat73.loadmat("../data/desc_mean.mat")
desc_mean = data_dict["desc_mean"]  # (128, ) SIFT mean value from training data
centers = np.loadtxt("../data/C.csv")  # cluster center
Xmean = np.loadtxt("../data/Xmean.csv").reshape(-1, 1)
Pemb = np.loadtxt("../data/Pemb.csv")

timeStart = time.time()
for im in range(n):
    if imidx % 100 == 1:
        print("Process image %d" % (imidx))
    x = X[:, cndes[im] : cndes[im + 1]].astype("float32")

    x = desc_postprocess(x, desc_mean)  # Pre-processing of descriptors

    psi[:, imidx] = triemb_sumagg(x, centers, Xmean)

    imidx += 1
    if imidx % 100 == 1:
        print("Elapsed time is %.3fs" % (time.time() - timeStart))
print("* Image representation elapsed time total in %.3fs" % (time.time() - timeStart))

psi = Pemb @ psi  # projection matrix
np.savetxt("../data/psi.csv", psi)
