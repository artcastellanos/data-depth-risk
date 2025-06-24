
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Iterable
from utils import gen_name
from save import loader

def flatten(lis):#https://stackoverflow.com/questions/17485747/how-to-convert-a-nested-list-into-a-one-dimensional-list-in-python
     for item in lis:
         if isinstance(item, Iterable) and not isinstance(item, str):
             for x in flatten(item):
                 yield x
         else:        
             yield item

def routine(rank_correlation_name,RFD_biglist,MLPD_biglist,SVMD_biglist,GPD_biglist):
    RFD_biglist = list(flatten(RFD_biglist))
    MLPD_biglist = list(flatten(MLPD_biglist))
    SVMD_biglist = list(flatten(SVMD_biglist))
    GPD_biglist = list(flatten(GPD_biglist))


    ## RFD
    RFD_data = pd.DataFrame(columns=["d",rank_correlation_name,"depth"])
    RFD_data["d"] = dlist
    RFD_data[rank_correlation_name] = np.array(RFD_biglist)
    RFD_data["depth"] = [r'$RFD$' for x in range(len(drange)*nruns)]

    ## MLPD
    MLPD_data = pd.DataFrame(columns=["d",rank_correlation_name,"depth"])
    MLPD_data["d"] = dlist
    MLPD_data[rank_correlation_name] =  np.array(MLPD_biglist)
    MLPD_data["depth"] = [r'$MLP\mathcal{D}$' for x in range(len(drange)*nruns)]    

    ## SVMD
    SVMD_data = pd.DataFrame(columns=["d",rank_correlation_name,"depth"])
    SVMD_data["d"] = dlist
    SVMD_data[rank_correlation_name] =  np.array(SVMD_biglist)
    SVMD_data["depth"] = [r'$SVM\mathcal{D}$' for x in range(len(drange)*nruns)]  

    ## GPD
    GPD_data = pd.DataFrame(columns=["d",rank_correlation_name,"depth"])
    GPD_data["d"] = dlist
    GPD_data[rank_correlation_name] =  np.array(GPD_biglist)
    GPD_data["depth"] = [r'$GPD$' for x in range(len(drange)*nruns)] 

    data = pd.concat([RFD_data,MLPD_data, SVMD_data, GPD_data])

    seaplot = sns.boxplot(data=data,x="d",y=rank_correlation_name,hue="depth",palette="Paired")
    return seaplot


def rank_loader(title_name,rankcorrtype='spearman',prefix_folder='save/'):
    ###### Loading
    file_name = rankcorrtype + title_name
    results = loader(file_name,prefix_folder=prefix_folder)
    return results

def main_rank_loading(depth_name,n,nruns,d,distrib,seed_num):
    title_name = depth_name + gen_name(n=n,nruns=nruns,d=d,distrib=distrib,seed_num=seed_num)
    
    spearman_results = rank_loader(title_name,rankcorrtype='spearman')
    kendalltau_results = rank_loader(title_name,rankcorrtype='kendalltau')

    res_list = loader('results'+title_name)
    densities_list = loader('densities'+title_name)
    return res_list,densities_list,spearman_results,kendalltau_results

def for_bxp_loop(depth_name,drange,n,nruns,distrib,seed_num):
    spearman_biglist = []
    kendalltau_biglist = []
    for d in drange:
        res_list, densities_list, spearman_results, kendalltau_results = main_rank_loading(depth_name,n,nruns,d,distrib,seed_num)
        spearman_biglist.append(spearman_results)
        kendalltau_biglist.append(kendalltau_results)
    return spearman_biglist,kendalltau_biglist

if __name__ == '__main__':
    seed_num = 0
    np.random.seed(seed_num)
    prefix_folder='save/'

    distrib = 'two_gaussians'#

    n = 200
    nruns = 10

    drange = range(2,10,2)

    params = {"legend.fontsize": 18,
          "axes.titlesize": 16,
          "axes.labelsize": 16,
          "xtick.labelsize": 13,
          "ytick.labelsize": 13,
          "pdf.fonttype": 42,
          "svg.fonttype": 'none'}

    spearman_biglist_RFD, kendalltau_biglist_RFD = for_bxp_loop('RFD',drange,n,nruns,distrib,seed_num)
    spearman_biglist_MLPD, kendalltau_biglist_MLPD= for_bxp_loop('MLPD',drange,n,nruns,distrib,seed_num)
    spearman_biglist_SVMD, kendalltau_biglist_SVMD = for_bxp_loop('SVMD',drange,n,nruns,distrib,seed_num)
    spearman_biglist_GPD, kendalltau_biglist_GPD = for_bxp_loop('GPD',drange,n,nruns,distrib,seed_num)


    plt.rcParams.update(params)


    dlist = [[x]*nruns for x in drange]
    dlist = list(flatten(dlist))

    sns.set(style="whitegrid", font_scale=2)

    fig, ax = plt.subplots(figsize=(15,8))
    spearplot = routine("Spearman rank correlation",spearman_biglist_RFD,spearman_biglist_MLPD,spearman_biglist_SVMD,spearman_biglist_GPD)
 
    figtitle = "Spearman rank correlation w.r.t. the true density for a bigaussian distribution n = 200"
    
    figpath = 'fig/'+figtitle+'.pdf'
    fig.savefig(figpath,dpi=700)
    fig, ax = plt.subplots(figsize=(15,8))
    kenplot = routine("Kendall tau rank correlation",kendalltau_biglist_RFD,kendalltau_biglist_MLPD,kendalltau_biglist_SVMD,kendalltau_biglist_GPD)
    figtitle = "Kendall tau rank correlation w.r.t. the true density for a bigaussian distribution n = 200"

    figpath = 'fig/'+figtitle+'.pdf'
    fig.savefig(figpath,dpi=700)
    plt.show()
