from abm_structure import abm_structure                                         # Python structure
from abm_abm import abm_abm                                                     # core ABM function    
import os                                                                       # OS functionality   
import pandas as pd                                                             # dataframes                        
import numpy as np                                                              # numpy              

# parameter structure specification
# -----------------------------------------------------------------------------
P           = abm_structure()                                                   # structure initialization
P.mode      = "gen"                                                             # generative mode
P.agent     = "A"                                                               # agent name                          
P.tau       = 0.001                                                               # action noise
P.K         = 10                                                                # number of blocks
P.T         = 40                                                               # number of trials per block
P.dir       = os.path.join(os.path.dirname(os.getcwd()), "Data\\Simulations")   # data directory
P.pid       = "A"                                                               # simulation ID
P.file      = os.path.join(P.dir, P.pid + ".csv")                               # filename   

# run validation
# -----------------------------------------------------------------------------
abm_abm(P)                                                                      # run validation

# # log likelihooding
# # -----------------------------------------------------------------------------
# theta       = np.linspace(0,0.5,100)                                             # parameter space
# llh_theta   = np.zeros(len(theta))                                              # log likelihood vector
# P           = abm_structure()                                                   # structure initialization   
# P.mode      = "llh"                                                             # log likelihood mode 
# P.agent     = "A"                                                               # agent name
# P.dir       = os.path.join(os.path.dirname(os.getcwd()), "Data\\Simulations")   # data directory
# P.pid       = "A"                                                               # simulation ID
# P.file      = os.path.join(P.dir, P.pid + ".csv")                               # filename   
# P.df        = pd.read_csv(P.file, sep = '\t')                                   # read data  
# P.K         = max(P.df.k) + 1                                                   # number of blocks
# P.T         = max(P.df.t) + 1                                                   # number of trials per block

# for i in np.arange(len(theta)):                                                 # parameter iterations
#     P.tau           = theta[i]                                                  # action noise    
#     llh_theta[i]    = abm_abm(P)                                                # run log likelihood
                                                                    

# # plot
# # -----------------------------------------------------------------------------         
# import matplotlib.pyplot as plt                                                 # matplotlib
# plt.plot(theta, llh_theta)                                                      # plot log likelihood
# plt.xlabel("tau")                                                               # x-axis label
# plt.ylabel("log likelihood")                                                    # y-axis label
# plt.title("Log likelihood")                                                     # title
# plt.show()