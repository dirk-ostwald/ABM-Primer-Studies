# general functionality
import os                                                                       # OS functionality      
import pandas as pd                                                             # pandas for data handling    
import numpy as np                                                              # numpy for numerical operations                               
import matplotlib.pyplot as plt                                                 # matplotlib for plotting

# local modules
from abm_structure import abm_structure                                         # Python structure
from abm_load import abm_load                                                   # data loading function
from abm_description import abm_description                                     # descriptive statistics function
from abm_abm import abm_abm                                                     # core ABM function    
from abm_llh import abm_llh                                                     # analytical log likelihood function of the ABM


# data generation
# -----------------------------------------------------------------------------
n           = 15                                                                 # number of data sets
idxs        = np.arange(1,n+1)                                                  # participant indices
sdir        = os.path.join(os.path.dirname(os.getcwd()), "Data\\Simulations")   # data directory
P           = abm_structure()                                                   # structure initialization
P.mode      = "gen"                                                             # generative mode
P.agent     = "A"                                                               # agent name
P.sigma     = 0.1                                                               # fluctuation standard deviation                                   
P.tau       = 0.1                                                               # action noise
P.K         = 1                                                                 # number of blocks
P.T         = 300                                                               # number of trials per block
P.dir       = sdir                                                              # data directory
for i in idxs:                                                                     # data set iterations
    P.pid       = "sub-" + str(i).zfill(3)                                      # simulation ID
    P.file      = os.path.join(P.dir, P.pid + ".csv")                           # filename   
    P           = abm_abm(P)                                                    # run simulation

# descriptive statistics
# -----------------------------------------------------------------------------     
sta         = abm_structure()                                                   # structure initialization
sta.sdir    = sdir                                                              # data directory
sta.idxs    = idxs                                                              # participant indices
sta         = abm_description(sta)                                              # run descriptive statistics                     


# # analytical evaluation of the log likelihood
# # ---------------------------------------------------------------------------
# tau         = np.linspace(0.1,0.9,30)                                            # parameter space
# llh_a       = np.zeros(len(tau))                                                # analytical log likelihood  
# dir         = os.path.join(os.path.dirname(os.getcwd()), "Data\\Simulations")   # data directory
# pid         = "A"                                                               # simulation ID
# file        = os.path.join(dir, pid + ".csv")                                   # filename   
# df          = pd.read_csv(file, sep = '\t')                                     # read data  
# c_t         = df.c_t                                                            # contrast difference data
# a_t         = df.a_t                                                            # action data
# llh         = abm_structure()                                                   # structure initialization   
# for i in np.arange(len(tau)):                                                   # parameter iterations
#     llh_a[i]  = abm_llh(np.array([0.1, tau[i]]), a_t, c_t)                      # run log likelihood

# # numerical evaluation of the log likelihood
# # ---------------------------------------------------------------------------
# llh_n       = np.zeros(len(tau))                                                # analytical log likelihood 
# P           = abm_structure()                                                   # structure initialization   
# P.mode      = "llh"                                                             # log likelihood mode 
# P.agent     = "A"                                                               # agent name
# P.df        = df 
# P.K         = max(P.df.k) + 1                                                   # number of blocks
# P.T         = max(P.df.t) + 1                                                   # number of trials per block
# P.n_s       = 100                                                                 # number of fluctuation samples       
# P.sigma     = 0.1                                                               # fluctuation standard deviation
# for i in np.arange(len(tau)):                                                   # parameter iterations
#     P.tau       = tau[i]                                                        # action noise    
#     llh_n[i]    = abm_abm(P)                                                    # run log likelihood

# print(llh_a)
# print(llh_n)

# # plot
# # -----------------------------------------------------------------------------         
# plt.plot(tau, llh_a, color="darkblue", linestyle="-", label="llh_a")            # Dark blue solid line
# plt.plot(tau, llh_n, color="lightblue", linestyle="--", label="llh_n")          # Light blue dashed line
# plt.xlabel("tau")  
# plt.ylabel("log likelihood")  
# plt.title("Log likelihood")  
# plt.ylim(-500, 100)  
# plt.legend()  # Add legend
# plt.show()

# # analytical evaluation of the log likelihood
# # ---------------------------------------------------------------------------
# sigma       = np.linspace(0.01,1,30)                                            # parameter space
# llh_a       = np.zeros(len(sigma))                                              # analytical log likelihood  
# dir         = os.path.join(os.path.dirname(os.getcwd()), "Data\\Simulations")   # data directory
# pid         = "A"                                                               # simulation ID
# file        = os.path.join(dir, pid + ".csv")                                   # filename   
# df          = pd.read_csv(file, sep = '\t')                                     # read data  
# c_t         = df.c_t                                                            # contrast difference data
# a_t         = df.a_t                                                            # action data
# llh         = abm_structure()                                                   # structure initialization   
# for i in np.arange(len(sigma)):                                                   # parameter iterations
#     llh_a[i]  = abm_llh(np.array([sigma[i], 0.3]), a_t, c_t)                      # run log likelihood

# # numerical evaluation of the log likelihood
# # ---------------------------------------------------------------------------
# llh_n       = np.zeros(len(sigma))                                                # analytical log likelihood 
# P           = abm_structure()                                                   # structure initialization   
# P.mode      = "llh"                                                             # log likelihood mode 
# P.agent     = "A"                                                               # agent name
# P.df        = df 
# P.K         = max(P.df.k) + 1                                                   # number of blocks
# P.T         = max(P.df.t) + 1                                                   # number of trials per block
# P.n_s       = 100                                                                 # number of fluctuation samples       
# P.tau       = 0.3                                                               # fluctuation standard deviation
# for i in np.arange(len(sigma)):                                                   # parameter iterations
#     P.sigma     = sigma[i]                                                        # action noise    
#     llh_n[i]    = abm_abm(P)                                                    # run log likelihood

# # plot
# # -----------------------------------------------------------------------------         
# plt.plot(sigma, llh_a, color="darkblue", linestyle="-", label="llh_a")            # Dark blue solid line
# plt.plot(sigma, llh_n, color="lightblue", linestyle="--", label="llh_n")          # Light blue dashed line
# plt.xlabel("sigma")  
# plt.ylabel("log likelihood")  
# plt.title("Log likelihood")  
# plt.ylim(-220, -200)  
# plt.legend()  # Add legend
# plt.show()
