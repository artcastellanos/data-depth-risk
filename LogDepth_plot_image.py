from sklearn.metrics import roc_auc_score, RocCurveDisplay, roc_curve
import matplotlib.pyplot as plt
from Depths import ParallelLogDepth, balanced_log_loss

#to sort
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_digits
from sklearn.metrics import log_loss
import numpy as np
import matplotlib.pyplot as plt
from distributions import distribute
from time_util import time_util
import time
from sklearn.datasets import fetch_openml
from joblib import Memory

def LogDepth(z,Xs,ys,lr):
    # assumes ys are precomputed ones with length n+1 and last point is -1
    # assumes last point of Xs is the point of computation
    # lr assumed to be balanced class weight logistic regression
    X_total = np.vstack((Xs,z))
    clf = lr.fit(X_total,ys)
    return balanced_log_loss(clf,Xs,z)




if __name__ == '__main__':

    np.random.seed(0)

    memory = Memory('./tmp')
    fetch_openml_cached = memory.cache(fetch_openml)

    dataname = 'Fashion-MNIST'#'mnist_784'

    print("Loading..."+dataname)
    old_t = time.time()
    data = fetch_openml_cached(dataname)
    new_t = time.time()
    time_util(old_t, new_t)


    
    
    X = data.data.to_numpy()
    y = data.target.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    X = X_train
    y = y_train

    class_one = X[y=='1']
    class_two = X[y=='2']
    class_five = X[y=='5']
    class_all_but_one = X[y!='1']
    class_all_but_five = X[y!='5']

    class_five_test = X_test[y_test=='5']
    class_all_but_five_test = X_test[y_test!='5']
    class_one_test = X_test[y_test=='1']
    class_all_but_one_test = X_test[y_test!='1']



    lamb = 1
    max_iter = 1000

    Xs =  class_five[:1000]

    Z_normal = class_five_test[:1000]
    y_normal = np.ones(Z_normal.shape[0])
    Z_anormal = class_all_but_five_test[:1000]
    y_anormal = np.zeros(Z_anormal.shape[0])

    Zs = np.vstack((Z_normal,Z_anormal))
    y_true = np.concatenate((y_normal,y_anormal))#

    n = Xs.shape[0]
    lr = LogisticRegression(class_weight='balanced',C=1/lamb,max_iter=max_iter)
    ys = np.ones(n+1)
    ys[-1] = 0#-1
    kwargs = {'ys':ys,'lr':lr}
    old_t = time.time()
    depths = ParallelLogDepth(Zs,Xs)
    new_t = time.time()
    time_util(old_t, new_t)
    scores = depths

    inclass_depths = depths[:1000]
    outclass_depths = depths[1000:]
    in_max = np.argmax(inclass_depths)
    in_min = np.argmin(inclass_depths)
    out_max = np.argmax(outclass_depths)
    out_min = np.argmin(outclass_depths)

    fig,ax = plt.subplots()
    ax.imshow(Z_normal[in_max].reshape(28, 28), cmap='binary')
    ax.set_axis_off()

    fig,ax = plt.subplots()
    ax.imshow(Z_normal[in_min].reshape(28, 28), cmap='binary')
    ax.set_axis_off()

    fig,ax = plt.subplots()
    ax.imshow(Z_anormal[out_max].reshape(28, 28), cmap='binary')
    ax.set_axis_off()

    fig,ax = plt.subplots()
    ax.imshow(Z_anormal[out_min].reshape(28, 28), cmap='binary')
    ax.set_axis_off()


    for i,ind in enumerate([in_max,in_min]):
        lr = LogisticRegression(class_weight='balanced',C=1/(lamb*n),max_iter=max_iter)#,penalty=None)
        ys = np.ones(n+1)
        ys[-1] = 0#-1
        kwargs = {'ys':ys,'lr':lr}
        old_t = time.time()
        depth = LogDepth(Z_normal[ind],Xs,ys,lr)
        new_t = time.time()
        print("depth",depth)
        time_util(old_t, new_t)
        coeff = lr.coef_
        print("coeff",coeff)
        print("intercept",lr.intercept_)
        fig,ax = plt.subplots()
        ax.imshow(coeff.reshape(28, 28), cmap='binary')
        ax.set_axis_off()

    for i,ind in enumerate([out_max,out_min]):
        lr = LogisticRegression(class_weight='balanced',C=1/(lamb*n),max_iter=max_iter)#,penalty=None)
        ys = np.ones(n+1)
        ys[-1] = 0#-1
        kwargs = {'ys':ys,'lr':lr}
        old_t = time.time()
        depth = LogDepth(Z_anormal[ind],Xs,ys,lr)
        new_t = time.time()
        print("depth",depth)
        time_util(old_t, new_t)
        coeff = lr.coef_
        print("coeff",coeff)
        print("intercept",lr.intercept_)
        fig,ax = plt.subplots()
        ax.imshow(coeff.reshape(28, 28), cmap='binary')
        ax.set_axis_off()

    plt.show()
    



