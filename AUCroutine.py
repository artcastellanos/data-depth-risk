import numpy as np
from sklearn.datasets import fetch_openml
from joblib import Memory
from time_util import time_util
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, RocCurveDisplay, roc_curve
import matplotlib.pyplot as plt
import joblib
#Depths
from Depths import ParallelLogDepth, ParallelSVMDepth


def paramdict_toString(param_dict):
    params = ''
    for (x,y) in param_dict.items():
        params += x + '_' + str(y) + '_'
    return params

def make_classes(X,y,y_label):
    inclass =  X[y==y_label]
    outclass =  X[y!=y_label]
    return inclass, outclass

def load(dataname):
    memory = Memory('./tmp')
    fetch_openml_cached = memory.cache(fetch_openml)
    print("Loading..."+dataname)
    old_t = time.time()
    data = fetch_openml_cached(dataname)
    new_t = time.time()
    time_util(old_t, new_t)
    X = data.data.to_numpy()
    y = data.target.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)
    return X_train, X_test, y_train, y_test

def take(X,num=None):
    if num==None:
        return X
    else:
        return X[:num]

def anomaly_data(inclass_train,inclass_test,outclass_test,numX=None,numZ_normal=None,numZ_anormal=None):
    X =  take(inclass_train,numX)
    Z_normal = take(inclass_test,numZ_normal)
    y_normal = np.ones(Z_normal.shape[0])
    Z_anormal = take(outclass_test,numZ_anormal)
    y_anormal = np.zeros(Z_anormal.shape[0])#-np.ones(Z_anormal.shape[0])
    Z = np.vstack((Z_normal,Z_anormal))
    y_true = np.concatenate((y_normal,y_anormal))
    return X,Z,y_true

def auc(y_true,scores,dataname,y_class):
    auc_score = roc_auc_score(y_true, scores)
    print("AUC",auc_score)
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot().figure_
    plt.title(dataname +" class " + str(y_class)+ " AUC "+str(auc_score))
    return auc_score, roc_display

def ParallelLogDepthroutine(X,Z,lamb=1,max_iter=1000,seed=None):
    old_t = time.time()
    depths_scores = ParallelLogDepth(Z,X,lamb=lamb,max_iter=max_iter)
    new_t = time.time()
    print("Depth computation in")
    time_util(old_t, new_t)
    return depths_scores

def ParallelSVMDepthroutine(X,Z,lamb=1,max_iter=1000,seed=None):
    old_t = time.time()
    depths_scores = ParallelSVMDepth(Z,X,lamb=lamb)
    new_t = time.time()
    print("Depth computation in")
    time_util(old_t, new_t)
    return depths_scores

def meta(method,dataname, y_class, X_train, X_test, y_train, y_test, lamb=1,max_iter=1000,numX=None,numZ_normal=None,numZ_anormal=None,seed=None):
    params = {
        'seed':seed,
        'y_class':y_class,
        'lamb':lamb ,
        'max_iter':max_iter,
        'numX':numX,
        'numZ_normal':numZ_normal,
        'numZ_anormal':numZ_anormal}
    inclass_train,_ = make_classes(X_train,y_train,y_class)
    inclass_test,outclass_test = make_classes(X_test,y_test,y_class)
    X,Z,y_true = anomaly_data(inclass_train,inclass_test,outclass_test,numX,numZ_normal,numZ_anormal)
    depths = method['func'](X,Z,lamb=lamb,max_iter=max_iter,seed=seed)
    auc_score, roc_display = auc(y_true,depths,dataname,y_class)
    figpath = "fig/AUC/" + method['name'] + dataname + '_' + paramdict_toString(params) + ".pdf"
    roc_display.savefig(figpath,dpi=700)
    return auc_score

if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    method = {'name':'ParallelSVMDepth','func':ParallelSVMDepthroutine}#{'name':'ParallelLogDepth','func':ParallelLogDepthroutine}
    dataname = 'mnist_784'#'Fashion-MNIST'
    y_class = '1'
    lamb = 1#
    max_iter = 1000
    numX = 100#None#1000#
    numZ_normal = None
    numZ_anormal = None

    params = {'seed':seed,
        'y_class':y_class,
        'lamb':lamb ,
        'max_iter':max_iter,
        'numX':numX,
        'numZ_normal':numZ_normal,
        'numZ_anormal':numZ_anormal}
    
    print(method['name'],"lamb",lamb,"numX",numX,"numZ_normal",numZ_normal,"numZ_anormal",numZ_anormal)

    y_list = [str(x) for x in range(10)]
    seed_list = range(10)#[0]#

    for y_class in y_list:
        params['y_class'] = y_class
        print("class",y_class)

        for seed in seed_list:
            np.random.seed(seed)
            print("seed",seed)
            params['seed'] = seed

            X_train, X_test, y_train, y_test = load(dataname)

            paramsname = paramdict_toString(params)
            save_path = 'results/AUC/' + method['name'] + dataname + '_' + paramsname
            score = meta(method,dataname, y_class, X_train, X_test, y_train, y_test, lamb, max_iter, numX, numZ_normal, numZ_anormal,seed)
            joblib.dump(score, save_path)