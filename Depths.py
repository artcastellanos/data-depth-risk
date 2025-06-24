import numpy as np
from sklearn.linear_model import LogisticRegression
from multiprocessing import Pool
from functools import partial
from sklearn import svm
from sklearn.metrics.pairwise import rbf_kernel
from scipy.spatial.distance import cdist
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import log_loss
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier






def depths(points,data,depth_func,**kwargs):
    depths = []
    for point in points:
        depth = depth_func(point,data,**kwargs)
        depths.append(depth)
    return np.array(depths)


def balanced_log_loss(classif,X,z):
    n = X.shape[0]
    with np.errstate(divide='ignore'):
        log_pred_X = classif.predict_log_proba(X)[:, 1]
    log_pred_X = log_pred_X/(2*n)
    log_pred_z = classif.predict_log_proba(z.reshape(1,-1))[:, 0]
    return (-log_pred_X.mean()-log_pred_z/2).item()

def logdepth(z,Xs,ys,lr):
    X_total = np.vstack((Xs,z))
    clf = lr.fit(X_total,ys)
    return balanced_log_loss(clf,Xs,z)

def LogDepth(Zs,Xs,lamb=1,max_iter=1000):
    n = Xs.shape[0]
    lr = LogisticRegression(class_weight='balanced',C=1/(lamb*n),max_iter=max_iter)#,penalty=None)
    ys = np.ones(n+1)
    ys[-1] = 0
    kwargs = {'ys':ys,'lr':lr}
    depth_vals = depths(Zs,Xs,logdepth,**kwargs)
    return depth_vals

def chunk_points(points,k):
    n = points.shape[0]
    q = n//k

def ParallelLogDepth(Zs,Xs,lamb=1,max_iter=1000,k=4,chunk_size=100):
    f = partial(LogDepth, Xs=Xs, lamb=lamb,max_iter=max_iter)
    with Pool(processes=k) as pool:
        # Process data in chunks of 100
        results = pool.map(f, [Zs[i:i+chunk_size] for i in range(0, len(Zs), chunk_size)])
        # Flatten the results
        flattened = [item for sublist in results for item in sublist]
    return np.array(flattened)


def artisanal_hinge_loss(y_true,pred_decision):
    margin = y_true * pred_decision
    losses = 1 - margin
    np.clip(losses, 0, None, out=losses)
    return losses

def art_hinge_loss(y_true,pred_decision):
    losses = artisanal_hinge_loss(y_true,pred_decision)
    return losses.mean()


def SVMKernelDepthPrecomp(Z,X,y,lamb,kernel_func,gamma=1,class_weight='balanced',verbose=False,tol=1e-3,max_iter=1000,random_state=None):
    n = X.shape[0]
    m = Z.shape[0]
    C=1/lamb
    depths = np.empty(m)
    tot = np.concatenate((X,Z),0)
    if kernel_func=='rbf':
        K = rbf_kernel(tot,tot,gamma=gamma)
    elif kernel_func=='linear':
        K = np.inner(tot,tot)
    else:
        K = kernel_func(tot,tot)
    for i in range(m):
        index = list(range(n)) + [n+i]
        classif = svm.SVC(kernel='precomputed', class_weight=class_weight, C=C,verbose=verbose,tol=tol,max_iter=max_iter,random_state=random_state)
        classif.fit(K[np.ix_(index,index)], y)
        depths[i] = 1/2*(art_hinge_loss(y[-1].reshape(1,-1),classif.decision_function(K[np.ix_([n+i],index)]).reshape(1,-1))+art_hinge_loss(y[:-1],classif.decision_function(K[np.ix_(list(range(n)),index)])))
        ## could replace directly by value of y -1 and 1
    return depths

def SVMDepth(Zs, Xs,lamb = 1, gamma ='auto',kernel_func = 'rbf'):
    n = Xs.shape[0]
    ys = np.ones(n+1)
    ys[-1] = -1
    if gamma == 'auto':
        gamma = 1/np.median(cdist(Xs,Xs)**2)
    kwargs = {'y':ys,'lamb':lamb,'kernel_func':kernel_func,'gamma':gamma}
    depths = SVMKernelDepthPrecomp(Zs,Xs,**kwargs)
    return depths


def ParallelSVMDepth(Zs,Xs,lamb = 1, gamma ='auto',kernel_func = 'rbf',k=4,chunk_size=100):
    f = partial(SVMDepth, Xs=Xs, lamb=lamb,gamma = gamma,kernel_func = kernel_func)
    with Pool(processes=k) as pool:
        # Process data in chunks of 100
        results = pool.map(f, [Zs[i:i+chunk_size] for i in range(0, len(Zs), chunk_size)])
        # Flatten the results
        flattened = [item for sublist in results for item in sublist]
    return np.array(flattened)

def gpdepth(z,X,y,classif):
    n = X.shape[0]
    X_total = np.vstack((X,np.tile(z,(n,1))))
    classif.fit(X_total,y)
    y_pred = classif.predict_proba(X_total)[:, 1]
    return log_loss(y, y_pred)


def GPDepth(Zs,Xs,max_iter=1000,optimizer=None,lamb='auto',kernel='rbf'):
    n = Xs.shape[0]
    y = np.ones(2*n)
    y[n:] = 0
    if lamb == 'auto':
        lamb = np.quantile(cdist(Xs,Xs),0.25)
    if kernel == 'rbf':
        kernel = RBF(length_scale=lamb)
    classif = GaussianProcessClassifier(max_iter_predict=max_iter,optimizer=optimizer,kernel=kernel)
    kwargs = {'y':y,'classif':classif}
    depth_vals = depths(Zs,Xs,gpdepth, **kwargs)
    return depth_vals

def ParallelGPDepth(Zs,Xs,optimizer=None,lamb ='auto',kernel='rbf',k=4,chunk_size=100,max_iter=1000):
    f = partial(GPDepth, Xs=Xs, max_iter=max_iter,optimizer=optimizer,lamb=lamb,kernel='rbf')
    with Pool(processes=k) as pool:
        # Process data in chunks of 100
        results = pool.map(f, [Zs[i:i+chunk_size] for i in range(0, len(Zs), chunk_size)])
        # Flatten the results
        flattened = [item for sublist in results for item in sublist]
    return np.array(flattened)

def mlpdepth(z,X,y,classif):
    n = X.shape[0]
    X_total = np.vstack((X,np.tile(z,(n,1))))
    classif.fit(X_total,y)
    y_pred = classif.predict_proba(X_total)[:, 1]
    return log_loss(y, y_pred)

def MLPDepth(Zs,Xs,alpha=0.01,max_iter=1000, random_state=None,hidden_layer_sizes= (100,)):
    # alpha: Strength of the L2 regularization term. 
    # The L2 regularization term is divided by the sample size when added to the loss.
    n = Xs.shape[0]
    y = np.ones(2*n)
    y[n:] = 0
    classif = MLPClassifier(alpha=alpha*n, max_iter=max_iter, random_state=random_state, hidden_layer_sizes= hidden_layer_sizes)
    kwargs = {'y':y,'classif':classif}
    depth_vals = depths(Zs,Xs,gpdepth, **kwargs)
    return depth_vals

def ParallelMLPDepth(Zs,Xs,alpha=0.01,max_iter=1000, random_state=None,hidden_layer_sizes= (100,),k=4,chunk_size=100):
    f = partial(MLPDepth, Xs=Xs, alpha=alpha,max_iter=max_iter, random_state=random_state,hidden_layer_sizes=hidden_layer_sizes)
    with Pool(processes=k) as pool:
        # Process data in chunks of 100
        results = pool.map(f, [Zs[i:i+chunk_size] for i in range(0, len(Zs), chunk_size)])
        # Flatten the results
        flattened = [item for sublist in results for item in sublist]
    return np.array(flattened)

def rfdepth(z,X,y,classif):
    n = X.shape[0]
    X_total = np.vstack((X,z))
    classif.fit(X_total,y)
    with np.errstate(divide='ignore'):
        log_pred_X = classif.predict_log_proba(X)[:, 1]
    log_pred_X = log_pred_X/(2*n)
    log_pred_z = classif.predict_log_proba(z.reshape(1,-1))[:, 0]
    return -log_pred_X.mean()-log_pred_z/2


def RFDepth(Zs,Xs,n_estimators=1000,max_depth=2, random_state=None):
    n = Xs.shape[0]
    y = np.ones(n+1)
    y[-1] = 0
    classif = RandomForestClassifier(bootstrap=False,n_estimators=n_estimators,criterion='log_loss',max_depth=max_depth,class_weight='balanced', random_state=random_state)
    kwargs = {'y':y,'classif':classif}
    depth_vals = depths(Zs,Xs,rfdepth, **kwargs)
    return depth_vals

def ParallelRFDepth(Zs,Xs,n_estimators=1000,max_depth=2, random_state=None,k=4,chunk_size=100):
    f = partial(RFDepth, Xs=Xs, n_estimators=n_estimators,max_depth=max_depth, random_state=random_state)
    with Pool(processes=k) as pool:
        # Process data in chunks of 100
        results = pool.map(f, [Zs[i:i+chunk_size] for i in range(0, len(Zs), chunk_size)])
        # Flatten the results
        flattened = [item for sublist in results for item in sublist]
    return np.array(flattened)

