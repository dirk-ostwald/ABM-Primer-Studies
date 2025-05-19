import numpy as np                                                              # numpy library
import pandas as pd                                                             # pandas library
from pathlib import Path                                                        # path library

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
    This function evaluates a set of descriptive statistics on a contrast 
    discrimination task data set.
        Input
            sta  : structure with attributes 
                .ddir : data directory
                .idxs : participant indices (list of integers)
     
        Output
            sta : input structure with additional attributes 
            
                 # meta data
                .N          : number of participants 
                .a_trl      : participant number of trials
                .a_trl_mean : group mean number of trials
                .a_trl_sd   : group SD number of trials
                .a_trl_sem  : group SEM of mean number of trials
            
                 # perceptual action (ap) statistics 
                .a_val      : participant number of valid (= non-nan) ap
                .a_val_mean : group mean number of valid ap
                .a_val_sd   : group SD of number of valid ap
                .a_val_sem  : group SEM deviation of number of valid ap
                .a_cor      : participant number of correct valid ap
                .a_cor_mean : group mean number of correct valid ap
                .a_cor_sd   : group SD number of correct valid ap
                .a_cor_sem  : group SEM number of correct valid ap
                .a_acc      : participant fraction of correct valid ap/valid ap
                .a_acc_mean : group mean fraction of correct valid ap./valid ap
                .a_acc_sd   : group SD fraction of correct valid ap/valid ap
                .a_acc_sem  : group SEM fraction of correct valid ap/valid ap

                 # perceptual action psychometric functions
                .a_psy_x    : psychometric function x-axis values
                .a_psy      : participant-specific psychometric functions
                .a_psy_mean : group psychometric function values
                .a_psy_sd   : group psychometric function standard deviation
                .a_psy_sem  : group psychometric function SEM
                 
  
           
    """
    # data loading    
    data       = abm_load(sta.ddir,sta.idxs)                                    # data frame  
    N          = len(list(set(data['P'])))                                      # number of participants
    print(N)

    # perceptual action data analysis
    a_bins     = np.linspace(-1, 1, 18)                                         # normalized contrast difference bins 
    a_psy_x    = np.linspace(-1, 1, 17)                                         # normalized contrast difference x axis
    a_trl      = np.full([N], np.nan)                                           # number of trials    
    a_val      = np.full([N], np.nan)                                           # number of valid perceptual actions
    a_cor      = np.full([N], np.nan)                                           # number of correct valid perceptual actions
    a_psy      = np.full([N, len(a_psy_x)], np.nan)                             # psychometric functions
    
    # participant iterations
    for p in range(N):      
        data_p      = data[data['P'] == sta.idxs[p]]                            # participant data set
        a_trl[p]    = data_p.shape[0]                                           # number of trials in the data set
        a_val[p]    = a_trl[p] - data_p['a_t'].isna().sum()                     # number of valid trials in the data
        a_cor[p]    = np.sum(data_p['a_t'] == data_p['s_t'])                    # number of correct valid perceptual actions
        a_cut       = pd.cut(data_p['c_t'], a_bins)                             # contrast difference grouping into bins
        a_psy[p,:]  = data_p['a_t'].groupby([a_cut], observed = False).mean()   # participant-specific psychometric functions
    
 
    # sanity check
    if np.any(a_cor > a_val):
        print('Perceptual action accuracy data invalid')
    
    # perceptual action accuracy
    a_acc          = a_cor/a_val
    
    # group averages, standard deviations, and stanard errors of the mean
    a_trl_mean     = np.mean(a_trl)    
    a_trl_sd       = np.std(a_trl)                                            
    a_trl_sem      = a_trl_sd/np.sqrt(N)                                  
    a_val_mean     = np.mean(a_val)                                            
    a_val_sd       = np.std(a_val)                                             
    a_val_sem      = a_val_sd/np.sqrt(N)     
    a_cor_mean     = np.mean(a_cor)                                            
    a_cor_sd       = np.std(a_cor)                                             
    a_cor_sem      = a_cor_sd/np.sqrt(N)     
    a_acc_mean     = np.mean(a_acc)                                            
    a_acc_sd       = np.std(a_acc)                                             
    a_acc_sem      = a_acc_sd/np.sqrt(N)     
    a_psy_mean     = np.mean(a_psy, axis = 0)
    a_psy_sd       = np.std(a_psy, axis = 0) 
    a_psy_sem      = a_psy_sd/np.sqrt(N)
    
 
    # output structure specification
    # -------------------------------------------------------------------------
    # perceptual actions
    sta.a_trl      = a_trl
    sta.a_trl_mean = a_trl_mean 
    sta.a_trl_sd   = a_trl_sd
    sta.a_trl_sem  = a_trl_sem
    sta.a_val      = a_val
    sta.a_val_mean = a_val_mean 
    sta.a_val_sd   = a_val_sd
    sta.a_val_sem  = a_val_sem
    sta.a_cor      = a_cor
    sta.a_cor_mean = a_cor_mean 
    sta.a_cor_sd   = a_cor_sd
    sta.a_cor_sem  = a_cor_sem
    sta.a_acc      = a_acc
    sta.a_acc_mean = a_acc_mean 
    sta.a_acc_sd   = a_acc_sd
    sta.a_acc_sem  = a_acc_sem
    sta.a_psy_x    = a_psy_x
    sta.a_psy      = a_psy
    sta.a_psy_mean = a_psy_mean
    sta.a_psy_sd   = a_psy_sd
    sta.a_psy_sem  = a_psy_sem
   

    # output specification
    return sta