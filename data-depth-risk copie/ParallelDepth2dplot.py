import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_blobs
from distributions import distribute
from Depths import *
from time_util import time_util
import time





if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    n = 400
    d = 2
    distrib = 'two_gaussians'
    X = distribute(distrib,n,d)

    
    old_t = time.time()
    depth_name = 'MLPDepth'#'GPDepth'#'RFDepth'#'SVMDepth'#
    depths = ParallelMLPDepth(X,X)#ParallelGPDepth(X,X)#ParallelRFDepth(X,X)#ParallelSVMDepth(X,X)#
    new_t = time.time()
    print("Depth computation in")
    time_util(old_t, new_t)
    print(depths)
    print(depths.shape)

    fig = plt.figure(figsize=(6.4,5.2))
    pp = plt.scatter(X[:,0],X[:,1],c=depths,s=30.0)
    plt.grid()
    cb_ax = fig.add_axes([0.93, 0.09, 0.021, 0.8])
    plt.colorbar(pp, cax = cb_ax)
    plt.grid()
    plt.savefig('fig/two_gaussians_mean3.5_n'+str(n)+'_seed'+str(seed)+'_grid_'+depth_name+'.pdf',dpi=700)
    plt.show()