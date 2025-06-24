from utils import gen_name
from distributions import distribute, get_densities
from scipy.stats import spearmanr, kendalltau
import numpy as np
import joblib
from save import saving
from functools import partial
from Depths import *

def depth_routine(depth_func,res_list,X):
    res = depth_func(X, X)
    res_list.append(res)


def save_routine(depth_name,res_list,densities_list,spearman_results,kendalltau_results,n,nruns,d,distrib,seed_num,prefix_folder):
    title_name = depth_name + gen_name(n=n,nruns=nruns,d=d,distrib=distrib,seed_num=seed_num)
    ##### Save
    ### Depths
    file_name = 'results' + title_name
    saving(res_list,file_name,prefix_folder=prefix_folder)
    ### Densities
    file_name = 'densities' + title_name
    saving(densities_list,file_name,prefix_folder=prefix_folder)    
    ### Spearman
    file_name = 'spearman' + title_name
    saving(spearman_results,file_name,prefix_folder=prefix_folder)
    ### Kendall tau
    file_name = 'kendalltau' + title_name
    saving(kendalltau_results,file_name,prefix_folder=prefix_folder)



def depth_loop(n,d,nruns,distrib,seed_num,prefix_folder):
    RFD_res_list = []
    MLPD_res_list = []
    SVMD_res_list = []
    GPD_res_list = []
    GPD_res_list = []
    densities_list = []
    ## 1st part: compute depths
    for i in range(nruns):
        X  = distribute(distrib,n,d)
        densities = get_densities(distrib,X)
        densities_list.append(densities)
        depth_routine(ParallelRFDepth,RFD_res_list,X)
        depth_routine(ParallelMLPDepth,MLPD_res_list,X)
        depth_routine(ParallelSVMDepth,SVMD_res_list,X)
        depth_routine(ParallelGPDepth,GPD_res_list,X)
    ## 2nd part: compute kendall_tau and spearman
    RFD_spearman_results, RFD_kendalltau_results = rankcorr_loop(RFD_res_list,densities_list)
    MLPD_spearman_results, MLPD_kendalltau_results = rankcorr_loop(MLPD_res_list,densities_list)
    SVMD_spearman_results, SVMD_kendalltau_results = rankcorr_loop(SVMD_res_list,densities_list)
    GPD_spearman_results, GPD_kendalltau_results = rankcorr_loop(GPD_res_list,densities_list)
    ## 3rd part: save
    save_routine('RFD',RFD_res_list,densities_list,RFD_spearman_results, RFD_kendalltau_results,n,nruns,d,distrib,seed_num,prefix_folder)
    save_routine('MLPD',MLPD_res_list,densities_list,MLPD_spearman_results, MLPD_kendalltau_results,n,nruns,d,distrib,seed_num,prefix_folder)
    save_routine('SVMD',SVMD_res_list,densities_list,SVMD_spearman_results, SVMD_kendalltau_results,n,nruns,d,distrib,seed_num,prefix_folder)
    save_routine('GPD',GPD_res_list,densities_list,GPD_spearman_results, GPD_kendalltau_results,n,nruns,d,distrib,seed_num,prefix_folder)


def rankcorr_loop(res_list,densities_list):
    spearman_results = []
    kendalltau_results = []

    for res,densities in zip(res_list,densities_list):
        ##### spearman r
        spearman_val = spearmanr(res,densities).correlation
        spearman_results.append(spearman_val)
        ##### kendall tau
        kendalltau_val = kendalltau(res,densities).correlation
        kendalltau_results.append(kendalltau_val)
    return spearman_results, kendalltau_results



if __name__ == '__main__':

    seed_num = 0
    np.random.seed(seed_num)
    prefix_folder='save/'


    n = 200

    distrib = 'two_gaussians'



    nruns = 10

    drange = range(2,10,2)
    for d in drange: 
        depth_loop(n,d,nruns,distrib,seed_num,prefix_folder)
        print("d =" + str(d) + " done")
                        





