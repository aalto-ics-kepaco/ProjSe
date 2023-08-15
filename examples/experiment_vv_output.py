import matplotlib.pyplot as plt
import matplotlib.cm
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, pairwise_distances
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC, SVC
import pickle


from ProjSe import cls_projector_kern as projse

"""
The datasets, Crop, NonInvasiveFetalECGThorax1 ("Thorax"),  and ShapesAll, can be downloaded at:
- http://www.timeseriesclassification.com/description.php?Dataset=Crop
- http://www.timeseriesclassification.com/description.php?Dataset=NonInvasiveFetalECGThorax1
- http://www.timeseriesclassification.com/description.php?Dataset=ShapesAll
"""

# ==========================================================

# select with which dataset the experiments should be run
dataset = "thorax"
dataset = "crop"
dataset = "shapes"

datapath = ""  # the path to downloaded data files

# ==========================================================


def my_poly_k(X, Y=None):
    if Y is None:
        Y = np.copy(X)
    return polynomial_kernel(X, Y, degree=3, gamma=1, coef0=0)


def my_rbf_k(X, Y=None):
    if Y is None:
        Y = np.copy(X)
    gamma = gamma_from_sigma(np.mean(pairwise_distances(X)))
    return rbf_kernel(X, Y, gamma=gamma)


def gamma_from_sigma(sigma):
    gamma = 1/(2*sigma**2)
    return gamma


def hsic(K1, K2):

    n = K1.shape[0]
    C = np.eye(n)- np.ones((n, n))/n
    K1c = np.dot(C, np.dot(K1, C))
    K2c = np.dot(C, np.dot(K2, C))

    return np.trace(np.dot(K1c, K2c))


def ka(K1, K2):
    n = K1.shape[0]
    C = np.eye(n)- np.ones((n, n))/n
    K1c = np.dot(K1, C)
    K2c = np.dot(K2, C)

    upstairs = np.trace(np.dot(K1c, K2c))
    downstairs = np.sqrt(np.trace(np.dot(K1c, K1c)))*np.sqrt(np.trace(np.dot(K2c, K2c)))
    return upstairs/downstairs


def cv_fit_and_pred_svm(X, y, Xtst):
    g = gamma_from_sigma(np.mean(pairwise_distances(X)))
    svm = SVC(gamma=g)
    svm.fit(X, y)
    preds = svm.predict(Xtst)
    return preds


def cv_fit_and_pred_lin_svm(X, y, Xtst):
    svm = LinearSVC()
    svm.fit(X, y)
    preds = svm.predict(Xtst)
    return preds


# ========================================================================================================


if dataset == "thorax":

    data = np.loadtxt(datapath+'NonInvasiveFetalECGThorax1_TRAIN.txt')
    X = data[:, 1:]
    y = np.array(data[:, 0]).astype(int)
    data = np.loadtxt(datapath+'NonInvasiveFetalECGThorax1_TEST.txt')
    Xtst = data[:, 1:]
    ytst = np.array(data[:, 0]).astype(int)
    
elif dataset == "shapes":
    data = np.loadtxt(datapath+'ShapesAll_TRAIN.txt')
    X = data[:, 1:]
    y = np.array(data[:, 0]).astype(int)
    data = np.loadtxt(datapath+'ShapesAll_TEST.txt')
    Xtst = data[:, 1:]
    ytst = np.array(data[:, 0]).astype(int)
    
elif dataset == "crop":
    data = np.loadtxt(datapath+'Crop_TRAIN.txt')
    X = data[:, 1:]
    y = np.array(data[:, 0]).astype(int)
    data = np.loadtxt(datapath+'Crop_TEST.txt')
    Xtst = data[:, 1:]
    ytst = np.array(data[:, 0]).astype(int)


dy = len(np.unique(y))

Y = np.eye(dy)[y-1, :]
Ytst = np.eye(dy)[ytst-1, :]

        
# standardise the data
Xmean = np.mean(X, axis=0)
Xstd = np.std(X, axis=0)
Xtst = (Xtst-Xmean)/Xstd
X = (X-Xmean)/Xstd

# -------------------------------------------------------------------------------------------

algo = projse()
projse_linear = algo.full_cycle(Y, X, Y.shape[1])
algo = projse(my_rbf_k)
projse_rbf = algo.full_cycle(Y, X, Y.shape[1])
algo = projse(my_poly_k)
projse_poly = algo.full_cycle(Y, X, Y.shape[1])

# -------------------------------------------------------------------------------------------
# HSIC & KA

try:
    with open(dataset+"_kas.pkl", "rb") as f:
        all_ka_res = pickle.load(f)

    # hsicLin_projse = all_ka_res["hsicLin_projse"]
    # hsicLin_projse_poly = all_ka_res["hsicLin_projse_poly"]
    # hsicLin_projse_rbf = all_ka_res["hsicLin_projse_rbf"]
    kaLin_projse = all_ka_res["kaLin_projse"]
    kaLin_projse_poly = all_ka_res["kaLin_projse_poly"]
    kaLin_projse_rbf = all_ka_res["kaLin_projse_rbf"]
    # hsicRBF_projse = all_ka_res["hsicRBF_projse"]
    # hsicRBF_projse_poly = all_ka_res["hsicRBF_projse_poly"]
    # hsicRBF_projse_rbf = all_ka_res["hsicRBF_projse_rbf"]
    kaRBF_projse = all_ka_res["kaRBF_projse"]
    kaRBF_projse_poly = all_ka_res["kaRBF_projse_poly"]
    kaRBF_projse_rbf = all_ka_res["kaRBF_projse_rbf"]
    print("loaded ka results :) ")
except FileNotFoundError:
    KYYlin = np.dot(Y, Y.T)
    KYYrbf = my_rbf_k(Y)

    # hsicLin_projse = []
    # hsicLin_projse_poly = []
    # hsicLin_projse_rbf = []
    kaLin_projse = []
    kaLin_projse_poly = []
    kaLin_projse_rbf = []
    # hsicRBF_projse = []
    # hsicRBF_projse_poly = []
    # hsicRBF_projse_rbf = []
    kaRBF_projse = []
    kaRBF_projse_poly = []
    kaRBF_projse_rbf = []
    for ff in range(1, Y.shape[1]+1):
        KXXrbf_projse = rbf_kernel(X[:, projse_linear[:ff]])
        KXXrbf_projse_poly = rbf_kernel(X[:, projse_poly[:ff]])
        KXXrbf_projse_rbf = rbf_kernel(X[:, projse_rbf[:ff]])
        KXXlin_projse = np.dot(X[:, projse_linear[:ff]], X[:, projse_linear[:ff]].T)
        KXXlin_projse_poly = np.dot(X[:, projse_poly[:ff]], X[:, projse_poly[:ff]].T)
        KXXlin_projse_rbf = np.dot(X[:, projse_rbf[:ff]], X[:, projse_rbf[:ff]].T)

        # hsicLin_projse.append(hsic(KXXlin_projse, KYYlin))
        # hsicLin_projse_poly.append(hsic(KXXlin_projse_poly, KYYlin))
        # hsicLin_projse_rbf.append(hsic(KXXlin_projse_rbf, KYYlin))
        # hsicRBF_projse.append(hsic(KXXrbf_projse, KYYrbf))
        # hsicRBF_projse_poly.append(hsic(KXXrbf_projse_poly, KYYrbf))
        # hsicRBF_projse_rbf.append(hsic(KXXrbf_projse_rbf, KYYrbf))
        kaLin_projse.append(ka(KXXlin_projse, KYYlin))
        kaLin_projse_poly.append(ka(KXXlin_projse_poly, KYYlin))
        kaLin_projse_rbf.append(ka(KXXlin_projse_rbf, KYYlin))
        kaRBF_projse.append(ka(KXXrbf_projse, KYYrbf))
        kaRBF_projse_poly.append(ka(KXXrbf_projse_poly, KYYrbf))
        kaRBF_projse_rbf.append(ka(KXXrbf_projse_rbf, KYYrbf))

    all_ka_res = {"kaLin_projse": kaLin_projse,
                  "kaLin_projse_poly": kaLin_projse_poly,
                  "kaLin_projse_rbf": kaLin_projse_rbf,
                  "kaRBF_projse": kaRBF_projse,
                  "kaRBF_projse_poly": kaRBF_projse_poly,
                  "kaRBF_projse_rbf": kaRBF_projse_rbf
                  }

    with open(dataset+"_kas.pkl", "wb") as f:
        pickle.dump(all_ka_res, f)

cmap = matplotlib.cm.get_cmap('viridis')
c1 = cmap(0.3)
c2 = cmap(0.5)
c3 = cmap(0.9)

plt.figure(figsize=(4, 3.2))
plt.plot(np.arange(1, Y.shape[1]+1), kaRBF_projse, c="k", label="rbf KA")
plt.plot(np.arange(1, Y.shape[1]+1), kaLin_projse, c="k", linestyle="--", label="lin KA")
plt.plot(np.arange(1, Y.shape[1]+1), kaLin_projse, c=c1, linestyle="--")#, label="lin KA")
plt.plot(np.arange(1, Y.shape[1]+1), kaLin_projse_poly,c=c2, linestyle="--")#, label="poly.ProjSe - linKA")
plt.plot(np.arange(1, Y.shape[1]+1), kaLin_projse_rbf, c=c3, linestyle="--")#, label="rbf.ProjSe - linKA")
plt.plot(np.arange(1, Y.shape[1]+1), kaRBF_projse, c=c1, label="lin.ProjSe")# - rbfKA")
plt.plot(np.arange(1, Y.shape[1]+1), kaRBF_projse_poly, c=c2, label="poly.ProjSe")# - rbfKA")
plt.plot(np.arange(1, Y.shape[1]+1), kaRBF_projse_rbf, c=c3, label="rbf.ProjSe")# - rbfKA")
plt.xlabel("#variables")
plt.ylabel("KA")
plt.title(dataset)
plt.legend(loc='lower right')
plt.tight_layout()

# -------------------------------------------------------------------------------------------
# SVM classification

# full set of features for comparison
try:
    with open(dataset + "_full_svmaccs.pkl",
              "rb") as f:
        all_full_res = pickle.load(f)
    full_rbf_acc = all_full_res["full_rbf_acc"]
    full_lin_acc = all_full_res["full_lin_acc"]
except:
    full_rbf_acc = accuracy_score(ytst, cv_fit_and_pred_svm(X, y, Xtst))
    full_lin_acc = accuracy_score(ytst, cv_fit_and_pred_lin_svm(X, y, Xtst))
    all_full_res = {"full_rbf_acc": full_rbf_acc,
                    "full_lin_acc": full_lin_acc}

    with open(dataset + "_full_svmaccs.pkl", "wb") as f:
        pickle.dump(all_full_res, f)

# second comparison: randomly selected features
try:
    with open(dataset +"_random_svmaccs.pkl", "rb") as f:
        all_random_res = pickle.load(f)

    all_random_feats_svm_accs = all_random_res["all_random_feats_svm_accs"]
    all_random_feats_linsvm_accs = all_random_res["all_random_feats_linsvm_accs"]

except:
    all_random_feats_svm_accs = []
    all_random_feats_linsvm_accs = []
    for ii in range(10):
        np.random.seed(ii)  # for reproducible results
        random_feats = np.random.permutation(X.shape[1])[:Y.shape[1]]

        random_feats_svm_accs = []
        random_feats_linsvm_accs = []
        for ff in range(1, Y.shape[1]+1):
            random_feats_svm_accs.append(accuracy_score(ytst, cv_fit_and_pred_svm(X[:, random_feats[:ff]], y, Xtst[:, random_feats[:ff]])))
            random_feats_linsvm_accs.append(accuracy_score(ytst, cv_fit_and_pred_lin_svm(X[:, random_feats[:ff]], y, Xtst[:, random_feats[:ff]])))
        all_random_feats_svm_accs.append(random_feats_svm_accs)
        all_random_feats_linsvm_accs.append(random_feats_linsvm_accs)

    all_random_feats_svm_accs = np.array(all_random_feats_svm_accs)
    all_random_feats_linsvm_accs = np.array(all_random_feats_linsvm_accs)

    all_random_res = {"all_random_feats_svm_accs": all_random_feats_svm_accs,
                      "all_random_feats_linsvm_accs": all_random_feats_linsvm_accs}

    with open(dataset + "_random_svmaccs.pkl", "wb") as f:
        pickle.dump(all_random_res, f)

# finally, with feature selection
try:
    with open(dataset+"_svmaccs.pkl", "rb") as f:
        all_res = pickle.load(f)
    projse_svm_accs = all_res["projse_svm_accs"]
    projse_linsvm_accs = all_res["projse_linsvm_accs"]
    projse_poly_svm_accs = all_res["projse_poly_svm_accs"]
    projse_poly_linsvm_accs = all_res["projse_poly_linsvm_accs"]
    projse_rbf_svm_accs = all_res["projse_rbf_svm_accs"]
    projse_rbf_linsvm_accs = all_res["projse_rbf_linsvm_accs"]

    print("loaded svm results :) ")
except:
    projse_svm_accs = []
    projse_linsvm_accs = []
    projse_poly_svm_accs = []
    projse_poly_linsvm_accs = []
    projse_rbf_svm_accs = []
    projse_rbf_linsvm_accs = []
    for ff in range(1, Y.shape[1]+1):
        projse_svm_accs.append(accuracy_score(ytst, cv_fit_and_pred_svm(X[:, projse_linear[:ff]], y, Xtst[:, projse_linear[:ff]])))
        projse_linsvm_accs.append(accuracy_score(ytst, cv_fit_and_pred_lin_svm(X[:, projse_linear[:ff]], y, Xtst[:, projse_linear[:ff]])))

        projse_poly_svm_accs.append(accuracy_score(ytst, cv_fit_and_pred_svm(X[:, projse_poly[:ff]], y, Xtst[:, projse_poly[:ff]])))
        projse_poly_linsvm_accs.append(accuracy_score(ytst, cv_fit_and_pred_lin_svm(X[:, projse_poly[:ff]], y, Xtst[:, projse_poly[:ff]])))

        projse_rbf_svm_accs.append(accuracy_score(ytst, cv_fit_and_pred_svm(X[:, projse_rbf[:ff]], y, Xtst[:, projse_rbf[:ff]])))
        projse_rbf_linsvm_accs.append(accuracy_score(ytst, cv_fit_and_pred_lin_svm(X[:, projse_rbf[:ff]], y, Xtst[:, projse_rbf[:ff]])))

    all_res = {"projse_svm_accs": projse_svm_accs,
               "projse_linsvm_accs": projse_linsvm_accs,
               "projse_poly_svm_accs": projse_poly_svm_accs,
               "projse_poly_linsvm_accs": projse_poly_linsvm_accs,
               "projse_rbf_svm_accs": projse_rbf_svm_accs,
               "projse_rbf_linsvm_accs": projse_rbf_linsvm_accs,
               }

    with open(dataset+"_svmaccs.pkl", "wb") as f:
        pickle.dump(all_res, f)


# plot the results

cr = "rosybrown"


plt.figure(figsize=(4,3.3))

plt.axhline(y=full_rbf_acc, c="k", label="rbfSVM")
plt.axhline(y=full_lin_acc, c="k", linestyle="--", label="linSVM")

plt.errorbar(np.arange(1, Y.shape[1]+1), np.mean(all_random_feats_svm_accs, axis=0),
             np.std(all_random_feats_svm_accs, axis=0),
             c=cr, label="random")# - rbfSVM")
plt.errorbar(np.arange(1, Y.shape[1]+1), np.mean(all_random_feats_linsvm_accs, axis=0),
             np.std(all_random_feats_linsvm_accs, axis=0),
             c=cr, linestyle="--")#, label="random - linSVM")

plt.plot(np.arange(1, Y.shape[1]+1), projse_svm_accs, c=c1, label="lin.ProjSe")# - rbfSVM")
plt.plot(np.arange(1, Y.shape[1]+1), projse_linsvm_accs, c=c1, linestyle="--")#, label="lin.ProjSe - linSVM")

plt.plot(np.arange(1, Y.shape[1]+1), projse_poly_svm_accs, c=c2, label="poly.ProjSe")# - rbfSVM")
plt.plot(np.arange(1, Y.shape[1]+1), projse_poly_linsvm_accs, c=c2, linestyle="--")#, label="poly.ProjSe - linSVM")

plt.plot(np.arange(1, Y.shape[1]+1), projse_rbf_svm_accs, c=c3, label="rbf.ProjSe")# - rbfSVM")
plt.plot(np.arange(1, Y.shape[1]+1), projse_rbf_linsvm_accs, c=c3, linestyle="--")#, label="rbf.ProjSe - linSVM")

plt.legend()
plt.xlabel("#variables")
plt.ylabel("accuracy")
plt.tight_layout()

plt.show()
