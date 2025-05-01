import os
import numpy as np
import pandas as pd
from pathlib import Path

def abm_load(ddir,idxs):
        
    """
    This function loads Gabor patch contrast discrimination task data from 
    individual participant data files
    
    Input
        ddir    : data directory
        idxs    : participant indices (list of integers)
   
    Output
        D       : data frame containing all data
    
    """
    
    D = pd.DataFrame(None)                                                      # data frame initialization                                   
    for i in idxs:                                                              # participant iterations
        fname       = ddir / f"sub-{str(i).zfill(3)}.csv"                       # filename
        D_i         = pd.read_csv(fname, sep = '\t', header = 0)                # data
        D           = D._append(D_i, ignore_index = True)                       # append data frame

    return D


def abm_description(sta):
    
    """ 
    This function evaluates a set of descriptive statistics.
        Input
            sta  : structure with attributes 
                .ddir : data directory
                .idxs : participant indices (list of integers)
                .K    : number of blocks per participant
                .T    : number of trials per block

        Output
            sta : input structure with additional attributes 
            
                 # reward statistics 
                .r_avg      : participant average reward obtained 
                .r_avg_mean : group mean average reward obtained 
                .r_avg_sd   : group SD average reward obtained 
                .r_avg_sem  : group SEM faverage reward obtained                            
             
                 # reward learning curve
                .r_lrn_x    : trial indices  x-axis
                .r_lrn      : participant-specific reward average learning curve
                .r_lrn_mean : group reward average learning curve
                .r_lrn_sd   : group reward average learning curve
                .r_lrn_sem  : group eward average learning curve
                 
  
           
    """
    # data loading    
    data        = abm_load(sta.ddir,sta.idxs)                                   # data frame  
    N           = len(sta.idxs)                                                 # number of participants
    K           = sta.K                                                         # number of blocks per participant
    T           = sta.T                                                         # number of trials per block

    # data analysis
    # -------------------------------------------------------------------------    
    r_avg       = np.full([N], np.nan)                                          # average reward
    r_lrn_x     = np.linspace(1, T, T)                                          # trial indices x axis
    r_lrn       = np.full([N, len(r_lrn_x)], np.nan)                            # average reward learning curve array initialization
    a_lrn_x     = np.linspace(1, T, T)                                          # trial indices x axis
    a_lrn_a     = np.full([K,T,N], np.nan)                                      # average expectation maximizing actions array initialization 

    
    # average reward and learning curve
    for p in range(N):
        data_p      = data[data['P'] == p]                                      # participant data set
        r_avg[p]    = data_p['r_t'].mean()                                      # average reward
        r_lrn[p]    = data_p.groupby(['t'])['r_t'].mean().tolist()              # participant-specific learning curve
       
    # average maximizing actions
    for p in range(N):
        for b in range(K):
            data_pb     = data[(data['P'] == p) & (data['k'] == b)]             # participant block data set
            if data_pb['s_t'].iloc[0] >= 0.5:                                   # if a_t = 1 is the expectation maximizing action
                a_lrn_a[b,:,p]  = data_pb['a_t']                                  # average expectation maximizing actions
            else:                                                               # if a_t = 0 is the expectation maximizing action
                a_lrn_a[b,:,p]  = 1 - data_pb['a_t']                              # average expectation maximizing actions

    a_lrn  = np.mean(a_lrn_a, axis = 0).T                                       # average trialwise maximizing actions                              
    a_avg  = np.mean(a_lrn, axis = 1)                                           # across trial average maximizing action


    # group averages, standard deviations, and standard errors of the mean
    r_avg_mean     = np.mean(r_avg)    
    r_avg_sd       = np.std(r_avg)                                            
    r_avg_sem      = r_avg_sd/np.sqrt(N)                                   
    r_lrn_mean     = np.mean(r_lrn, axis = 0)
    r_lrn_sd       = np.std(r_lrn, axis = 0) 
    r_lrn_sem      = r_lrn_sd/np.sqrt(N)
    a_avg_mean     = np.mean(a_avg)    
    a_avg_sd       = np.std(a_avg)                                            
    a_avg_sem      = a_avg_sd/np.sqrt(N)
    a_lrn_mean     = np.mean(a_lrn, axis = 0)
    a_lrn_sd       = np.std(a_lrn, axis = 0) 
    a_lrn_sem      = a_lrn_sd/np.sqrt(N)  

    # output structure specification
    # -------------------------------------------------------------------------
    sta.r_avg      = r_avg
    sta.r_avg_mean = r_avg_mean 
    sta.r_avg_sd   = r_avg_sd
    sta.r_avg_sem  = r_avg_sem
    sta.r_lrn_x    = r_lrn_x
    sta.r_lrn      = r_lrn
    sta.r_lrn_mean = r_lrn_mean
    sta.r_lrn_sd   = r_lrn_sd
    sta.r_lrn_sem  = r_lrn_sem
    sta.a_avg      = a_avg
    sta.a_avg_mean = a_avg_mean 
    sta.a_avg_sd   = a_avg_sd
    sta.a_avg_sem  = a_avg_sem
    sta.a_lrn_x    = a_lrn_x
    sta.a_lrn      = a_lrn
    sta.a_lrn_mean = a_lrn_mean
    sta.a_lrn_sd   = a_lrn_sd
    sta.a_lrn_sem  = a_lrn_sem

    # output specification
    return sta
