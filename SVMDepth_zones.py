import numpy as np

from sklearn.datasets import make_blobs

from math import comb

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm

from Depths import ParallelSVMDepth





def SVMDepth_mesh(data,
                freq = [100, 100],
                xlim = None,
                ylim = None,
                lamb=1,
                kernel_func='rbf',
                gamma=1.0):#:0.5):#0.1):#
    
    # Prepare the depth-calculating arguments
    xs, ys = np.meshgrid(np.linspace(xlim[0], xlim[1], freq[0]), np.linspace(ylim[0], ylim[1], freq[1]))
    objects = np.c_[xs.ravel(), ys.ravel()]

    kwargs = {'lamb':lamb,'kernel_func':kernel_func,'gamma':gamma}
    zDepth = ParallelSVMDepth(objects,data,**kwargs)    # Shape the grid
    depth_grid = zDepth.reshape(xs.shape)
    print(depth_grid)
    return xs, ys, depth_grid


if __name__ == '__main__':
    seed = 0
    np.random.seed(seed)
    n = [20,180]
    centers = [[2, 2], [-1, -1]]
    X, _ = make_blobs(n_samples=n, centers=centers, cluster_std=1.0)#0.6)#0.5)#
    n = 200
    x_range = [-5,5]
    y_range = [-5,5]
    lamb = 1#0.1#10#
    xs, ys, scores_mesh = SVMDepth_mesh(X, freq = [100, 100], 
                                       xlim = x_range, ylim = y_range,lamb=lamb)
    # Introduce colors
    #levels = MaxNLocator(nbins = 100).tick_values(0, 1)
    levels = MaxNLocator(nbins = 100).tick_values(0, np.max(scores_mesh))
    cmap = plt.get_cmap('YlOrRd')
    norm = BoundaryNorm(levels, ncolors = cmap.N, clip = True)
    # Plot the color mesh
    scores_mesh_cut = np.copy(scores_mesh)
    scores_mesh_cut[scores_mesh_cut == 0] = float('nan') # white color for zero depth
    scores_mesh_cut = scores_mesh_cut[:-1,:-1]
    fig, ax = plt.subplots()
    ax.pcolormesh(xs, ys, scores_mesh_cut, cmap = cmap, norm = norm)
    ax.plot(X[:,0], X[:,1], '.', c = 'gray')
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.grid()
    ax.set_aspect(1)
    # Add contours
    contour_levels = np.round(np.array([np.quantile(scores_mesh,q) for q in np.array(range(5, 10))/10.]),3)
    print("contour_levels",contour_levels)
    fmt_lens = [str(l) for l in contour_levels[:]]

    fmt_lens_tmp = {}
    for i in range(len(fmt_lens)):
        fmt_lens_tmp[contour_levels[i]] = fmt_lens[i]
    contours = ax.contour(xs, ys, scores_mesh, levels = contour_levels, 
                        linewidths = [0.5], linestyles = ['solid'], 
                        colors = ['blue'])
    ax.clabel(contours, contour_levels, fontsize = 8, inline = True, fmt = fmt_lens_tmp)

    plt.show()

