import time
from datetime import timedelta
import numpy as np
from scipy.io import loadmat
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, pairwise_distances
from sklearn.metrics import normalized_mutual_info_score
import matplotlib.pyplot as plt

from ProjSe import cls_projector_kern as projse

"""
The datasets, "Carcinom" and "GLIOMA", can be downloaded as mat files at: 
https://github.com/jundongl/scikit-feature/tree/master/skfeature/data
"""


# ==========================================

# select which data to run the experiments with
data = "carciom"
# data = "glioma"

datapath = ""  # path to the downloaded mat files


def load_carciom(datapath):

    print("== CARCIOM ==")

    data = loadmat(datapath+"Carcinom.mat")
    # print(data.keys())
    X = data["X"]
    Y = data["Y"]
    print(X.shape, Y.shape)  # (174, 9182) (174, 1)
    return X, Y


def load_glioma(datapath):
    print("== GLIOMA ==")

    data = loadmat(datapath+"GLIOMA.mat")
    # print(data.keys())
    X = data["X"]
    Y = data["Y"]
    print(X.shape, Y.shape)  # (50, 4434) (50, 1)
    return X, Y


if data == "glioma":
    X, y = load_glioma(datapath)
elif data == "carciom":
    X, y = load_carciom(datapath)

C = len(np.unique(y))


# ==========================================

def gamma_from_sigma(sigma):
    gamma = 1/(2*sigma**2)
    return gamma


def my_poly_k(X, Y=None):
    if Y is None:
        Y = np.copy(X)
    return polynomial_kernel(X, Y, degree=3, gamma=1, coef0=0)


def my_rbf_k(X, Y=None):
    if Y is None:
        Y = np.copy(X)
    gamma = gamma_from_sigma(np.mean(pairwise_distances(X)))
    return rbf_kernel(X, Y, gamma=gamma)


n_feats_in_clustering = [1, 5, 10, 15, 20, 25, 50, 75, 100, 125, 150, 175, 200, 225,  250, 275, 300]


def cluster_data(X, featinds, y):

    clusterer = KMeans(C, n_init=20)

    nmis = []
    for ii in n_feats_in_clustering:
        y_pred = clusterer.fit_predict(X[:, featinds[:ii]])
        # hmm how to assess the accuracy? is there an implementation for this already?
        nmis.append(normalized_mutual_info_score(y.ravel(), y_pred))

    return nmis


def cluster_data_repeats(X, featinds, y, n_repeats=20):

    nmis = []
    times = []
    for ii in n_feats_in_clustering:
        nmis_small = []
        times_small = []
        for repeat in range(n_repeats):
            t0 = time.process_time()
            clusterer = KMeans(C, random_state=repeat)
            y_pred = clusterer.fit_predict(X[:, featinds[:ii]])
            # hmm how to assess the accuracy? is there an implementation for this already?
            times_small.append(time.process_time()-t0)
            nmis_small.append(normalized_mutual_info_score(y.ravel(), y_pred))
        nmis.append(nmis_small)
        times.append(times_small)
    nmis = np.array(nmis)
    nmis_means = np.mean(nmis, axis=1)
    nmis_stds = np.std(nmis, axis=1)
    times = np.array(times)
    times_means = np.mean(times, axis=1)
    times_stds = np.std(times, axis=1)
    return nmis_means, nmis_stds, times_means, times_stds

# ==========================================

# do the feature selection on full data
algo = projse()
print("linear projse:")
t0 = time.process_time()
projse_linear = algo.full_cycle(X, X, 300)
print(timedelta(seconds=time.process_time()-t0))
print(projse_linear)
    
print("rbf projse:")
t0 = time.process_time()
algo = projse(my_rbf_k)
projse_rbf = algo.full_cycle(X, X, 300)
print(timedelta(seconds=time.process_time()-t0))
print(projse_rbf)
    
print("poly projse:")
t0 = time.process_time()
algo = projse(my_poly_k)
projse_poly = algo.full_cycle(X, X, 300)
print(timedelta(seconds=time.process_time()-t0))
print(projse_poly)

all_projse = {"lin":projse_linear, "poly":projse_poly, "rbf": projse_rbf}


# these are assessed with k-means clustering, repeated 20 times
# first full data for comparison
clusterer = KMeans(C, n_init=20)
full_cpred = clusterer.fit_predict(X)
nmi_full = normalized_mutual_info_score(y.ravel(), full_cpred)
all_full_res = []
for ii in range(20):
    clusterer = KMeans(C, random_state=ii)
    full_cpred = clusterer.fit_predict(X)
    all_full_res.append(normalized_mutual_info_score(y.ravel(), full_cpred))
mean_nmi_full = np.mean(all_full_res)
std_nmi_full = np.std(all_full_res)
print("full:", nmi_full)

# finally, calculate clustering results for feature sepection with projse, and plot them
import matplotlib.cm
cmap = matplotlib.cm.get_cmap('viridis')
c1 = cmap(0.3)
c2 = cmap(0.5)
c3 = cmap(0.9)
colors = [c1, c2, c3]

plt.figure(figsize=(4, 2.8))
# plt.axhline(nmi_full, c="k", label="full")
plt.errorbar([1, 300], [mean_nmi_full, mean_nmi_full], [std_nmi_full, std_nmi_full], c="k", label="full")
indx = 0
for name, projse_res in [("lin", projse_linear), ("poly",projse_poly), ("rbf",projse_rbf)]:
    # res = cluster_data(X, projse_res, y)
    # plt.plot(n_feats_in_clustering, res, c=colors[indx], label=name)
    mean_res, std_res, mean_times, std_times = cluster_data_repeats(X, projse_res, y)
    plt.errorbar(n_feats_in_clustering, mean_res, std_res, c=colors[indx], label=name)
    print(name)
    # print(res)
    print(data, name, "10, 300:", np.round(mean_res[1], 2), "(",np.round(std_res[1], 2), ") &",
          np.round(mean_res[-1], 2), "(", np.round(std_res[-1], 2), ")")
    # print(np.round(mean_times[1], 3), "("+str(np.round(std_times[1], 3))+")")
    # print(np.round(mean_times[-1], 3), "("+str(np.round(std_times[-1], 3))+")")
    indx+=1
plt.legend()
plt.ylabel("NMI")
plt.xlabel("#variables")
plt.tight_layout()

plt.show()
