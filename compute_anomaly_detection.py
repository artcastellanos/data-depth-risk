import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm
from sklearn.neighbors import LocalOutlierFactor

from sklearn.metrics import roc_auc_score

from Depths import *

import time
import multiprocessing as mp


def time_util(old_t, new_t):
    duration = new_t - old_t
    dur_min = duration/60.0
    print("--- %s seconds ---" % duration)
    print("--- %s min ---" % dur_min)

def ocsvm_auc(X, y, gamma="auto"):
    clf = svm.OneClassSVM(kernel="rbf", gamma=gamma, nu=0.1)
    clf.fit(X)
    scores = clf.decision_function(X)
    scores = scores - scores.min()
    if scores.max() > 0:
        scores /= scores.max() 
    return roc_auc_score(y, 1-scores)

def ocsvm_auc_best(X, y, gammas=[1.]):
    return max([ocsvm_auc(X, y, g) for g in gammas])

def lof_auc(X, y, n_neighbors=10):
    clf = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.1)
    clf.fit(X)
    scores = -clf.negative_outlier_factor_
    scores /= scores.max()
    return roc_auc_score(y, scores)

def lof_auc_best(X, y, n_neighbors=[5]):
    return max([lof_auc(X, y, n) for n in n_neighbors])

def compute_best_auc(method, X, y, params):
    if method == "ocsvm":
        return ocsvm_auc_best(X, y, params)
    if method == "lof":
        return lof_auc_best(X, y, params)


def logdepth_auc(X, y, alpha):
    lr = LogisticRegression(class_weight='balanced',C=1/alpha,max_iter=1000)
    n = X.shape[0]
    ys = np.ones(n+1)
    ys[-1] = -1
    kwargs = {'ys':ys,'lr':lr}
    ds = depths(X,X,LogDepth,**kwargs)
    scores = 1-ds
    return roc_auc_score(y, scores)

def depth_auc(depth_func, X, y):
    ds = depth_func(X,X)
    scores = 1-ds
    return roc_auc_score(y, scores)



if __name__ == '__main__':
    mp.freeze_support()

    methods = ["lof"]#["ocsvm"]#["ParallelGPDepth"]#["ParallelMLPDepth"]#["ParallelRFDepth"]#["ParallelLogDepth"]#
    bold_methods = [m.upper() for m in methods]

    dataset_names = ["45_wine","14_glass","21_Lymphography","40_vowels","29_Pima","4_breastw","38_thyroid","2_annthyroid","28_pendigits"]
    normalize = False#True
    data_prefix = "datasets/"#/opt/anaconda3/lib/python3.8/site-packages/adbench/datasets/Classical/"
    data_suffix = ".npz"

    auc_scores = dict()

    lof_neighbors = [5, 10, 15, 20, 30]

    for data_name in dataset_names:
        data_path = data_prefix + data_name + data_suffix
        data = np.load(data_path, allow_pickle=True)
        X, y = data['X'], data['y']
        gammas = np.array([1/np.median(cdist(X,X)**2)])

        scores = []
        for method in methods:

            params = gammas
            if method == "lof":
                params = lof_neighbors

            print(data_name)
            old_t = time.time()
            s = compute_best_auc(method, X, y, params)#depth_auc(ParallelGPDepth, X, y)# ! Change method here with depth
            new_t = time.time()
            scores.append(s)
            print(f"{method}: {s}")
            time_util(old_t, new_t)
        auc_scores[data_name] = scores
        
    df = pd.DataFrame(auc_scores, index=bold_methods).T
    bold_methods_string = ''
    for m  in bold_methods:
        bold_methods_string = bold_methods_string + '_' + m
    df.to_csv("results/anomaly_detection_auc"+ bold_methods_string + ".csv")
    with open("results/anomaly_detection_auc"+ bold_methods_string + ".tex", "w") as fh:
        df.style.format(precision=2).to_latex(fh)

            

